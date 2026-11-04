"""Window probes: does an embedding carry what happened during a specific window?

PASTIS crop types are nearly season-invariant, so they cannot show whether a
retrieved window embedding knows its window. These probes can: their labels are
quantities of the window itself, computed from the raw Sentinel-2 data of the
cached samples, so an annual embedding cannot know which window it is asked
about and fresh window embeddings set the ceiling.

Two steps over a ``window_retrieval_cache.py`` cache (calendar-year samples):

- ``labels`` (CPU): for every cached (sample, window, pixel), statistics of the
  window's monthly NDVI ``(B08 - B04) / (B08 + B04)`` and NDWI
  ``(B03 - B08) / (B03 + B08)``, skipping months without S2: NDVI max (peak
  greenness), min (bare soil / harvest), amplitude (max - min), mean, the peak's
  position in the window (0 = first month, 1 = last), and NDWI max (standing
  water). Monthly layers are single low-cloud scenes, so clouds pull NDVI down:
  max is robust to them, min and amplitude are not. Written as
  ``labels_<chunk>.npz`` beside each cache chunk.
- ``probe`` (GPU): ridge regression from an embedding to each label, trained on
  the trainer's train split (same seed and fractions), regularization chosen on
  val, R^2 reported on test, overall and by window length, for:

  * ``floor``: the annual embedding ``z_T``;
  * ``floor+window``: ``z_T`` plus a one-hot of the window (a linear probe told
    which window, without any retriever);
  * ``retrieved``: ``h(z_T, window)`` from a trained retriever;
  * ``fresh``: ``z_t``, fresh inference on the window (the ceiling).

Usage::

    python scripts/tools/window_retrieval_probes.py labels --cache_dir $CACHE --workers 32
    python scripts/tools/window_retrieval_probes.py probe --cache_dir $CACHE --retriever_dir $R --out $OUT
    python scripts/tools/window_retrieval_probes.py spectrum --cache_dir $CACHE --retriever_dir $R --out $OUT

``spectrum`` (CPU or GPU, no labels needed) mirrors the PASTIS diagnostic on the
cache's val split: the floor's and the retrieved embedding's error against fresh
along each principal component of fresh, as error / fresh variance, in bins of
8 components (bin 0 = highest variance), overall and by window length.
"""

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

LABELS = (
    "ndvi_max",
    "ndvi_min",
    "ndvi_amplitude",
    "ndvi_mean",
    "ndvi_peak_position",
    "ndwi_max",
)
DEFAULT_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_"
    "worldpop_wri_canopy_height_map/1138828"
)
MONTHS = 12


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def monthly_indices(
    s2: np.ndarray, present: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Per-month NDVI and NDWI ``[H, W, T]`` from raw S2 ``[H, W, T, C]``; NaN where missing."""
    from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality

    order = Modality.SENTINEL2_L2A.band_order
    b03, b04, b08 = (
        s2[..., order.index(b)].astype(np.float32) for b in ("B03", "B04", "B08")
    )
    bad = (b03 == MISSING_VALUE) | (b04 == MISSING_VALUE) | (b08 == MISSING_VALUE)
    if present is not None:
        bad |= ~np.asarray(present, dtype=bool)[None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b08 - b04) / (b08 + b04)
        ndwi = (b03 - b08) / (b03 + b08)
    out = {}
    for name, v in (("ndvi", ndvi), ("ndwi", ndwi)):
        v = np.where(bad | ~np.isfinite(v), np.nan, v)
        out[name] = np.clip(v, -1, 1)
    return out


def chunk_labels(chunk_path: str, dataset) -> None:
    """Window-mean NDVI / NDWI for every (sample, window, pixel) of one cache chunk."""
    from olmoearth_pretrain.data.constants import Modality

    out_path = chunk_path.replace("shard", "labels_shard")
    if os.path.exists(out_path):
        return
    with np.load(chunk_path) as d:
        index, win, pix = d["index"], d["win"].astype(int), d["pix"].astype(int)
    n, K, P = len(index), win.shape[1], pix.shape[1]
    labels = {name: np.full((n, K, P), np.nan, dtype=np.float32) for name in LABELS}
    for i in range(n):
        # Fill absent timesteps the way the cache did, so the time axis is the
        # full stored axis the cached windows index into.
        sample_dict, present = dataset.read_h5_file(
            dataset._get_h5_file_path(int(index[i]))
        )
        sample, missing_modalities = dataset.fill_sample_with_missing_values(
            sample_dict, present
        )
        if Modality.SENTINEL2_L2A.name in missing_modalities:
            continue
        s2 = sample.as_dict()[Modality.SENTINEL2_L2A.name]
        monthly = monthly_indices(s2, present.get(Modality.SENTINEL2_L2A.name))
        rows, cols = pix[i, :, 0], pix[i, :, 1]
        ndvi, ndwi = monthly["ndvi"][rows, cols], monthly["ndwi"][rows, cols]  # [P, T]
        for k in range(K):
            s, length = win[i, k]
            v, w = ndvi[:, s : s + length], ndwi[:, s : s + length]
            has = np.isfinite(v).any(axis=1)
            with np.errstate(all="ignore"):
                vmax, vmin = np.nanmax(v, axis=1), np.nanmin(v, axis=1)
                labels["ndvi_max"][i, k] = vmax
                labels["ndvi_min"][i, k] = vmin
                labels["ndvi_amplitude"][i, k] = vmax - vmin
                labels["ndvi_mean"][i, k] = np.nanmean(v, axis=1)
                peak = np.nanargmax(np.where(np.isfinite(v), v, -np.inf), axis=1)
                labels["ndvi_peak_position"][i, k] = np.where(
                    has, peak / max(length - 1, 1), np.nan
                )
                labels["ndwi_max"][i, k] = np.nanmax(w, axis=1)
    np.savez(out_path, **labels)
    logger.info(f"wrote {out_path}")


def labels(args: argparse.Namespace) -> None:
    """Write labels for every chunk of the cache."""
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig

    dataset = OlmoEarthDatasetConfig(
        h5py_dir=args.h5py_dir, training_modalities=[Modality.SENTINEL2_L2A.name]
    ).build()
    chunks = sorted(glob.glob(os.path.join(args.cache_dir, "shard*.npz")))
    logger.info(f"{len(chunks)} chunks")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda c: chunk_labels(c, dataset), chunks))


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def load_labels(cache_dir: str) -> dict[str, torch.Tensor]:
    """Labels in the same chunk order as ``load_cache``."""
    chunks = sorted(glob.glob(os.path.join(cache_dir, "shard*.npz")))
    parts: dict[str, list] = {name: [] for name in LABELS}
    for c in chunks:
        with np.load(c.replace("shard", "labels_shard")) as d:
            for name in LABELS:
                parts[name].append(d[name])
    return {name: torch.from_numpy(np.concatenate(v)) for name, v in parts.items()}


def window_onehot(start: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """One-hot over the 78 contiguous windows of whole steps."""
    windows = [(s, ln) for ln in range(1, MONTHS + 1) for s in range(MONTHS - ln + 1)]
    index = {w: i for i, w in enumerate(windows)}
    ids = torch.tensor(
        [index[(int(s), int(ln))] for s, ln in zip(start.tolist(), length.tolist())]
    )
    return torch.nn.functional.one_hot(ids, len(windows)).float()


@torch.no_grad()
def features(d: dict, retriever, device: str) -> dict[str, torch.Tensor]:
    """Rows = every valid (sample, window, pixel): feature matrices and window fields."""
    from window_retrieval_train import retrieve

    s_idx, k_idx = torch.nonzero(d["valid"], as_tuple=True)
    n_pix = d["z_T"].shape[1]
    out: dict[str, list] = {
        "floor": [],
        "retrieved": [],
        "fresh": [],
        "start": [],
        "length": [],
        "rows": [],
    }
    for chunk in torch.arange(len(s_idx)).split(512):
        s, k = s_idx[chunk], k_idx[chunk]
        z = d["z_T"][s].float()  # [C, P, D]
        zt = d["z_w"][s, k].float()
        st, ln, sm = d["start"][s, k], d["length"][s, k], d["start_month"][s]
        rep = lambda t: t[:, None].expand(-1, n_pix).reshape(-1)  # noqa: E731
        flat_z = z.reshape(-1, z.shape[-1])
        pred = retrieve(
            retriever,
            flat_z.to(device),
            rep(sm).to(device),
            rep(st).to(device),
            rep(ln).to(device),
        ).cpu()
        out["floor"].append(flat_z.half())
        out["retrieved"].append(pred.half())
        out["fresh"].append(zt.reshape(-1, zt.shape[-1]).half())
        out["start"].append(rep(st))
        out["length"].append(rep(ln))
        out["rows"].append(
            torch.stack([rep(s), rep(k), torch.arange(n_pix).repeat(len(s))], dim=1)
        )
    return {key: torch.cat(v) for key, v in out.items()}


def ridge_fit(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form ridge with an unpenalized intercept; returns weights incl. bias."""
    Xb = torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], dim=1)
    A = Xb.T @ Xb
    reg = lam * torch.eye(A.shape[0], device=X.device, dtype=X.dtype)
    reg[-1, -1] = 0
    return torch.linalg.solve(A + reg, Xb.T @ y)


def predict(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Apply ridge weights."""
    return (
        torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], dim=1) @ w
    )


def mlp_fit(
    X: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    hidden: int = 256,
    max_epochs: int = 30,
    patience: int = 3,
) -> tuple[float, "callable"]:
    """One-hidden-layer MLP regressor on standardized inputs and target, early-stopped on val R^2."""
    mu, sd = X["train"].mean(0), X["train"].std(0).clamp_min(1e-6)
    ym, ys = y["train"].mean(), y["train"].std().clamp_min(1e-6)
    net = torch.nn.Sequential(
        torch.nn.Linear(X["train"].shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    ).to(X["train"].device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    def run(Xs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (
                torch.cat([net((c - mu) / sd).squeeze(-1) for c in Xs.split(65536)])
                * ys
                + ym
            )

    best, best_state, bad = float("-inf"), None, 0
    n = len(X["train"])
    for _ in range(max_epochs):
        net.train()
        for idx in torch.randperm(n, device=X["train"].device).split(8192):
            loss = (
                (
                    net((X["train"][idx] - mu) / sd).squeeze(-1)
                    - (y["train"][idx] - ym) / ys
                )
                ** 2
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        score = r2(run(X["val"]), y["val"])
        if score > best:
            best, best_state, bad = (
                score,
                {k: v.clone() for k, v in net.state_dict().items()},
                0,
            )
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return best, run


def r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Coefficient of determination."""
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def probe(args: argparse.Namespace) -> None:
    """Fit and score the four probes for each label."""
    from window_retrieval_train import build, load_cache, split, subset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_cache([args.cache_dir], quantize=True)
    data.update(load_labels(args.cache_dir))
    train, val, test = split(data, 0.1, 0.1, args.seed)
    k, n_pix = train["z_w"].shape[1], train["z_T"].shape[1]
    n_train = min(train["z_T"].shape[0], max(1, args.max_train_rows // (k * n_pix)))
    train = subset(train, torch.arange(n_train))
    meta = json.loads((Path(args.retriever_dir) / "metrics.json").read_text())
    retriever = build(argparse.Namespace(**meta["args"]), dim=data["z_T"].shape[-1])
    retriever.load_state_dict(
        torch.load(Path(args.retriever_dir) / "model.pt", map_location="cpu")
    )
    retriever = retriever.to(device).eval()
    logger.info(
        f"retriever {meta['args']['arch']} from {args.retriever_dir}; {n_train} train samples"
    )

    splits = {}
    for name, d in (("train", train), ("val", val), ("test", test)):
        f = features(d, retriever, device)
        r = f["rows"]
        f["labels"] = {lab: d[lab][r[:, 0], r[:, 1], r[:, 2]] for lab in LABELS}
        f["floor+window"] = torch.cat(
            [f["floor"].float(), window_onehot(f["start"], f["length"])], dim=1
        ).half()
        splits[name] = f
        logger.info(f"{name}: {len(r)} rows")

    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    probes = ("floor", "floor+window", "retrieved", "fresh")
    results: dict = {}
    for lab in LABELS:
        results[lab] = {}
        ok = {n: torch.isfinite(sp["labels"][lab]) for n, sp in splits.items()}
        for pr in probes:
            X = {n: sp[pr][ok[n]].float().to(device) for n, sp in splits.items()}
            y = {
                n: sp["labels"][lab][ok[n]].float().to(device)
                for n, sp in splits.items()
            }
            if args.readout == "mlp":
                val_r2, run = mlp_fit(X, y)
                best = ("mlp", val_r2)
                pred = run(X["test"])
            else:
                best = max(
                    (
                        (
                            lam,
                            r2(
                                predict(
                                    X["val"], ridge_fit(X["train"], y["train"], lam)
                                ),
                                y["val"],
                            ),
                        )
                        for lam in lambdas
                    ),
                    key=lambda t: t[1],
                )
                pred = predict(X["test"], ridge_fit(X["train"], y["train"], best[0]))
            length = splits["test"]["length"][ok["test"]].to(device)
            by_len = {
                int(L): r2(pred[length == L], y["test"][length == L])
                for L in length.unique().tolist()
            }
            results[lab][pr] = {
                "test_r2": r2(pred, y["test"]),
                "val_r2": best[1],
                "lambda": best[0],
                "by_length": by_len,
            }
            logger.info(
                f"{lab:5s} {pr:13s} test R2 {results[lab][pr]['test_r2']:.4f}  (val {best[1]:.4f}, {args.readout} {best[0]})"
            )
        lengths = sorted(results[lab]["fresh"]["by_length"])
        logger.info(
            f"{lab} test R2 by window length: " + "  ".join(f"L{L}" for L in lengths)
        )
        for pr in probes:
            logger.info(
                f"  {pr:13s} "
                + "  ".join(f"{results[lab][pr]['by_length'][L]:.3f}" for L in lengths)
            )
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "retriever_dir": args.retriever_dir,
                    "cache_dir": args.cache_dir,
                    "results": results,
                },
                fh,
                indent=2,
            )
        logger.info(f"wrote {args.out}")


SPECTRUM_BIN = 8
LENGTH_GROUPS = {"all": (1, 12), "L2-4": (2, 4), "L5-8": (5, 8), "L9-12": (9, 12)}


def spectrum(args: argparse.Namespace) -> None:
    """Error / fresh variance per principal component of fresh, floor vs retrieved."""
    from window_retrieval_train import build, load_cache, split, subset

    from olmoearth_pretrain.evals.embedding_transforms import (
        dequantize_embeddings,
        quantize_embeddings,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_cache([args.cache_dir], quantize=True)
    _, val, _ = split(data, 0.1, 0.1, args.seed)
    val = subset(val, torch.arange(min(args.max_samples, val["z_T"].shape[0])))
    meta = json.loads((Path(args.retriever_dir) / "metrics.json").read_text())
    retriever = build(argparse.Namespace(**meta["args"]), dim=data["z_T"].shape[-1])
    retriever.load_state_dict(
        torch.load(Path(args.retriever_dir) / "model.pt", map_location="cpu")
    )
    retriever = retriever.to(device).eval()
    f = features(val, retriever, device)
    # int8 round trip, as the PASTIS bake stores every variant.
    floor = f["floor"].float()
    retrieved = dequantize_embeddings(quantize_embeddings(f["retrieved"].float()))
    fresh = f["fresh"].float()
    logger.info(f"val: {len(fresh)} rows from {val['z_T'].shape[0]} samples")
    if args.center == "sample":
        # Rows come in blocks of one (sample, window) and its pixels: subtracting each
        # block's mean keeps only the within-tile variation.
        n_pix = val["z_T"].shape[1]

        def center(x: torch.Tensor) -> torch.Tensor:
            v = x.view(-1, n_pix, x.shape[-1])
            return (v - v.mean(1, keepdim=True)).view_as(x)

        floor, retrieved, fresh = center(floor), center(retrieved), center(fresh)
        logger.info("centered per (sample, window): within-tile variation only")

    fc = fresh - fresh.mean(0)
    eigvals, eigvecs = torch.linalg.eigh(fc.T @ fc / len(fc))
    eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
    results: dict = {}
    for group, (lo, hi) in LENGTH_GROUPS.items():
        m = (f["length"] >= lo) & (f["length"] <= hi)
        results[group] = {"rows": int(m.sum())}
        for name, v in (("floor", floor), ("retrieved", retrieved)):
            err = (v[m] - fresh[m]) @ eigvecs
            ratio = (err**2).mean(0) / eigvals.clamp_min(1e-12)
            results[group][name] = [
                float(ratio[i : i + SPECTRUM_BIN].mean())
                for i in range(0, len(ratio), SPECTRUM_BIN)
            ]
        logger.info(
            f"{group} ({results[group]['rows']} rows), error / fresh variance, bins of 8 PCs, high variance first:"
        )
        for name in ("floor", "retrieved"):
            logger.info(
                f"  {name:>10s} " + " ".join(f"{x:.3f}" for x in results[group][name])
            )
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "retriever_dir": args.retriever_dir,
                    "cache_dir": args.cache_dir,
                    "results": results,
                },
                fh,
                indent=2,
            )
        logger.info(f"wrote {args.out}")


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["labels", "probe", "spectrum"])
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--h5py_dir", default=DEFAULT_H5_DIR)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retriever_dir")
    parser.add_argument("--out")
    parser.add_argument("--max_train_rows", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--readout",
        choices=["ridge", "mlp"],
        default="ridge",
        help="probe: readout fit per label",
    )
    parser.add_argument(
        "--max_samples", type=int, default=500, help="spectrum: val samples used"
    )
    parser.add_argument(
        "--center",
        choices=["none", "sample"],
        default="none",
        help="spectrum: subtract each sample's mean first",
    )
    args = parser.parse_args()
    if args.command == "labels":
        labels(args)
    elif args.command == "spectrum":
        spectrum(args)
    else:
        probe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
