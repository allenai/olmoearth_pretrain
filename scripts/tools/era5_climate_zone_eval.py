"""Climate-zone eval harness for the ERA5 climate-awareness experiments.

Motivation (2026-09-15 ERA5 discussion): if predicting ERA5 makes the pretrained
embeddings climate/weather aware, then scenes that share a climate should cluster
together -- i.e. the embedding space should recover "climate zones". This tool makes
that testable and comparable across runs, WITHOUT baking anything model-specific in.

Two stages:

  1. ``build-zones``: read a sample of ``era5_10`` from an ERA5-inclusive h5 dataset,
     reduce each scene to its 12-month climate-normal signature (72-dim = 12 months x
     6 vars, i.e. the intra-year PATTERN incl. seasonality, matching the era5clim arm's
     ``temporal_reduction="flatten"`` target), standardise, and KMeans into K
     data-driven climate zones. Writes ``sample_index -> (zone, signature)`` to an npz.
     (This is a data-driven proxy; pass ``--koppen-npz`` to score against a real
     Koppen-Geiger label per sample instead.)

  2. ``score``: given pooled embeddings for the SAME sample indices (an npz mapping
     ``sample_index -> embedding``; produce these with whatever inference harness you
     use for the run), report how climate-aware the embeddings are:
       * NMI / ARI between KMeans(embeddings, K) and the ERA5 zones (unsupervised),
       * a logistic-regression linear-probe accuracy + macro-F1 predicting the zone
         from the (frozen) embedding (train/test split),
       * same-zone vs different-zone mean cosine similarity (the "similar
         representations within a climate" check).
     Run it for the baseline and the ERA5 arms and compare -- and ALWAYS read it
     next to the PASTIS ps=1 / crop evals, since the whole point is the trade-off
     between climate-awareness and spatial detail.

Embedding npz format expected by ``score`` (both keys 1-D aligned by row):
    ``indices``: int array [N] of sample indices (matching build-zones' h5 sample ids)
    ``embeddings``: float array [N, D] of pooled (scene-level) embeddings

Usage:
    python -m scripts.tools.era5_climate_zone_eval build-zones \
        --h5-dir /.../cdl_era5_10_.../<n> --num-samples 20000 --k 16 \
        --out era5_zones.npz
    python -m scripts.tools.era5_climate_zone_eval score \
        --zones-npz era5_zones.npz --embeddings-npz run_pooled_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import h5py
import hdf5plugin  # noqa: F401  # registers the zstd HDF5 filter used by the datasets
import numpy as np

logger = logging.getLogger(__name__)

ERA5_BANDS = (
    "2m-temperature",
    "2m-dewpoint-temperature",
    "surface-pressure",
    "10m-u-component-of-wind",
    "10m-v-component-of-wind",
    "total-precipitation",
)
ERA5_NUM_BANDS = len(ERA5_BANDS)
ERA5_NUM_MONTHS = 12
ERA5_SIGNATURE_DIM = ERA5_NUM_MONTHS * ERA5_NUM_BANDS  # 12-month climate normal
ERA5_H5_KEY = "era5_10"
SAMPLE_GLOB_PREFIX = "sample_"


def _iter_sample_files(h5_dir: str, num_samples: int, stride: int) -> list[str]:
    """Collect up to ``num_samples`` ``sample_*.h5`` paths via a strided scandir.

    Uses ``os.scandir`` (lazy) with a stride rather than globbing+sorting the whole
    directory, which is prohibitively slow on the ~1M-file datasets.
    """
    paths: list[str] = []
    i = 0
    with os.scandir(h5_dir) as it:
        for entry in it:
            name = entry.name
            if not (name.startswith(SAMPLE_GLOB_PREFIX) and name.endswith(".h5")):
                continue
            i += 1
            if stride > 1 and (i % stride) != 0:
                continue
            paths.append(entry.path)
            if len(paths) >= num_samples:
                break
    return paths


def _sample_index_from_path(path: str) -> int:
    """``.../sample_1234.h5`` -> ``1234``."""
    base = os.path.basename(path)
    return int(base[len(SAMPLE_GLOB_PREFIX) : -len(".h5")])


def _era5_signature(arr: np.ndarray) -> np.ndarray | None:
    """Reduce a stored ERA5 array to its 12-month climate-normal signature ``[72]``.

    The signature is the full intra-year TRAJECTORY (12 months x 6 vars, time-major:
    ``[m0_c0..m0_c5, m1_c0..]``), NOT the annual mean -- seasonality (Mediterranean vs
    monsoon vs continental) is what actually separates climate zones, and this matches
    the ``temporal_reduction="flatten"`` supervision target the era5clim arm predicts.

    Shape-tolerant: the converter stacks 12 monthly (6, 1, 1) frames, and the loader
    may expose it as (72,), (72, 1, 1), (12, 6), or (12, 6, 1, 1). Anything folding to
    exactly (12, 6) is flattened to a length-72 vector. Non-12-month or NaN/empty tiles
    return ``None`` (dropped from the fit).
    """
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size != ERA5_SIGNATURE_DIM:
        return None
    if not np.isfinite(a).all():
        return None
    return a.reshape(ERA5_NUM_MONTHS, ERA5_NUM_BANDS).reshape(-1)


def build_zones(args: argparse.Namespace) -> None:
    """Build KMeans climate zones from a sample of ERA5 signatures."""
    from sklearn.cluster import KMeans

    paths = _iter_sample_files(args.h5_dir, args.num_samples, args.stride)
    logger.info("collected %d sample files", len(paths))

    indices: list[int] = []
    sigs: list[np.ndarray] = []
    missing = 0
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                if ERA5_H5_KEY not in f:
                    missing += 1
                    continue
                sig = _era5_signature(f[ERA5_H5_KEY][()])
        except (OSError, KeyError):
            missing += 1
            continue
        if sig is None:
            missing += 1
            continue
        indices.append(_sample_index_from_path(p))
        sigs.append(sig)

    if len(sigs) < args.k:
        raise SystemExit(
            f"only {len(sigs)} valid ERA5 signatures (< k={args.k}); "
            "raise --num-samples or lower --k"
        )
    signatures = np.asarray(sigs)  # [N, 72] (12 months x 6 vars, time-major)
    idx = np.asarray(indices, dtype=np.int64)

    # Standardise each of the 72 month-band dims (temp/pressure/precip live on wildly
    # different scales; this also weights every month equally so seasonality counts).
    mean = signatures.mean(axis=0)
    std = signatures.std(axis=0) + 1e-9
    standardized = (signatures - mean) / std

    km = KMeans(n_clusters=args.k, n_init=args.n_init, random_state=args.seed)
    zones = km.fit_predict(standardized).astype(np.int64)

    _, counts = np.unique(zones, return_counts=True)
    logger.info(
        "fit %d zones over %d scenes (%d missing/nan); cluster size min/med/max %d/%d/%d",
        args.k,
        len(sigs),
        missing,
        counts.min(),
        int(np.median(counts)),
        counts.max(),
    )
    # Report per-band stats averaged across the 12 months (mean is [72] = 12 x 6).
    band_mean = mean.reshape(ERA5_NUM_MONTHS, ERA5_NUM_BANDS).mean(axis=0)
    band_std = std.reshape(ERA5_NUM_MONTHS, ERA5_NUM_BANDS).mean(axis=0)
    for i, b in enumerate(ERA5_BANDS):
        logger.info(
            "  band %-24s raw mean %12.3f std %12.3f (avg over 12 months)",
            b,
            band_mean[i],
            band_std[i],
        )

    np.savez(
        args.out,
        indices=idx,
        zones=zones,
        signatures=signatures,
        band_mean=mean,
        band_std=std,
        centers=km.cluster_centers_,
        bands=np.array(ERA5_BANDS),
    )
    logger.info("wrote %s (indices, zones, signatures, centers)", args.out)


def _align(
    zone_idx: np.ndarray,
    zones: np.ndarray,
    emb_idx: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join zone labels and embeddings on sample index. Returns (zones, emb)."""
    zpos = {int(s): i for i, s in enumerate(zone_idx)}
    keep_z, keep_e = [], []
    for j, s in enumerate(emb_idx):
        i = zpos.get(int(s))
        if i is not None:
            keep_z.append(i)
            keep_e.append(j)
    if not keep_z:
        raise SystemExit("no overlapping sample indices between zones and embeddings")
    return zones[np.asarray(keep_z)], embeddings[np.asarray(keep_e)]


def score(args: argparse.Namespace) -> None:
    """Score embeddings against ERA5 climate zones (or Koppen labels)."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        adjusted_rand_score,
        f1_score,
        normalized_mutual_info_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    zf = np.load(args.zones_npz, allow_pickle=True)
    if args.koppen_npz:
        kf = np.load(args.koppen_npz, allow_pickle=True)
        zone_idx, zones = kf["indices"], kf["zones"].astype(np.int64)
        label_name = "koppen"
    else:
        zone_idx, zones = zf["indices"], zf["zones"].astype(np.int64)
        label_name = "era5-kmeans"

    ef = np.load(args.embeddings_npz, allow_pickle=True)
    emb_idx, embeddings = ef["indices"], np.asarray(ef["embeddings"], dtype=np.float64)

    zones_a, emb_a = _align(zone_idx, zones, emb_idx, embeddings)
    k = int(zones_a.max()) + 1
    n = len(zones_a)
    logger.info("aligned %d scenes on %s labels over %d zones", n, label_name, k)

    emb_std = StandardScaler().fit_transform(emb_a)

    # (1) Unsupervised agreement: cluster embeddings, compare partitions to zones.
    km = KMeans(n_clusters=k, n_init=args.n_init, random_state=args.seed)
    emb_clusters = km.fit_predict(emb_std)
    nmi = normalized_mutual_info_score(zones_a, emb_clusters)
    ari = adjusted_rand_score(zones_a, emb_clusters)

    # (2) Supervised linear probe: can a linear map read the zone off the embedding?
    x_tr, x_te, y_tr, y_te = train_test_split(
        emb_std,
        zones_a,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=zones_a if np.bincount(zones_a).min() >= 2 else None,
    )
    clf = LogisticRegression(max_iter=args.max_iter, multi_class="multinomial")
    clf.fit(x_tr, y_tr)
    y_hat = clf.predict(x_te)
    probe_acc = float((y_hat == y_te).mean())
    probe_f1 = float(f1_score(y_te, y_hat, average="macro"))
    chance = float(np.bincount(zones_a).max() / n)

    # (3) Same-zone vs different-zone cosine similarity (the "similar reps" check).
    normed = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(args.seed)
    m = min(args.sim_pairs, n * (n - 1) // 2)
    ii = rng.integers(0, n, size=m)
    jj = rng.integers(0, n, size=m)
    ok = ii != jj
    ii, jj = ii[ok], jj[ok]
    cos = (normed[ii] * normed[jj]).sum(axis=1)
    same = zones_a[ii] == zones_a[jj]
    same_cos = float(cos[same].mean()) if same.any() else float("nan")
    diff_cos = float(cos[~same].mean()) if (~same).any() else float("nan")

    report = {
        "label_source": label_name,
        "n_scenes": n,
        "n_zones": k,
        "unsupervised_nmi": round(nmi, 4),
        "unsupervised_ari": round(ari, 4),
        "linear_probe_acc": round(probe_acc, 4),
        "linear_probe_macro_f1": round(probe_f1, 4),
        "majority_class_chance": round(chance, 4),
        "same_zone_cos": round(same_cos, 4),
        "diff_zone_cos": round(diff_cos, 4),
        "cos_gap": round(same_cos - diff_cos, 4),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("wrote %s", args.out)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-zones", help="KMeans climate zones from ERA5 signatures")
    b.add_argument("--h5-dir", required=True, help="ERA5-inclusive h5 dataset dir")
    b.add_argument("--num-samples", type=int, default=20000)
    b.add_argument("--stride", type=int, default=1, help="take every Nth sample file")
    b.add_argument("--k", type=int, default=16, help="number of climate zones")
    b.add_argument("--n-init", type=int, default=8)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="era5_zones.npz")
    b.set_defaults(func=build_zones)

    s = sub.add_parser("score", help="score pooled embeddings against climate zones")
    s.add_argument("--zones-npz", required=True)
    s.add_argument(
        "--embeddings-npz", required=True, help="indices + embeddings arrays"
    )
    s.add_argument("--koppen-npz", default=None, help="optional real Koppen labels npz")
    s.add_argument("--test-size", type=float, default=0.3)
    s.add_argument("--n-init", type=int, default=8)
    s.add_argument("--max-iter", type=int, default=2000)
    s.add_argument("--sim-pairs", type=int, default=200000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=None, help="optional path to write the JSON report")
    s.set_defaults(func=score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
