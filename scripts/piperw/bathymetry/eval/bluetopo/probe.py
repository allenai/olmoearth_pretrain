"""Linear depth probe against BlueTopo on 40 m tokens of shallow US pretraining samples.

Token label: mean BlueTopo depth over the token's 4x4 10 m pixels. Kept if every pixel is
valid and below 0 m, and the mean depth is 1-30 m; samples whose finest tile is coarser than
8 m are dropped. Folds: GroupKFold over 1-degree cells, so test tokens come from regions not
seen by the probe. Ridge (standardised features, alpha by GCV on the training fold) at several
training-set sizes. The out-of-fold predictions are then compared between arms with a
cluster bootstrap over cells.

Feature sets: each embedding file given, the 12-band log median (no encoder), and the
dataset's own sdb (formula output, token mean, only where valid; scored as a 1-d linear fit).

With --holdout, only samples in the 1-degree cells held out from BlueTopo training
(holdout_cells.json, see make_train_labels.py) are used.

  python probe.py name=emb/x.npy [name=emb/y.npy ...] [--labels labels.npz] [--out results.json] [--holdout]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
H5 = Path(
    "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)
P = 4
DEPTH = (1.0, 30.0)
MAX_TILE_RES = 8
SIZES = [1024, 16384, 200_000]
RESAMPLES = 3
FOLDS = 5
ALPHAS = np.logspace(-2, 5, 15)
DEPTH_BINS = [(1, 5), (5, 10), (10, 20), (20, 30)]


def blocks(a: np.ndarray) -> np.ndarray:
    """[N, 128, 128, ...] -> [N, 32, 32, 16, ...]: the pixels of each token."""
    n, h, w = a.shape[:3]
    a = a.reshape(n, h // P, P, w // P, P, *a.shape[3:])
    a = np.moveaxis(a, 2, 3)
    return a.reshape(n, h // P, w // P, P * P, *a.shape[5:])


def metrics(y, p) -> dict:
    """R², MAE, Spearman and per-depth-band MAE of predictions p against y."""
    out = {
        "n": int(len(y)),
        "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "mae": float(np.mean(np.abs(y - p))),
        "spearman": float(spearmanr(y, p).statistic),
    }
    for lo, hi in DEPTH_BINS:
        m = (y >= lo) & (y < hi)
        out[f"mae_{lo}_{hi}m"] = (
            float(np.mean(np.abs(y[m] - p[m]))) if m.any() else None
        )
    return out


def oof_predict(x, y, groups, size, seed) -> np.ndarray:
    """Out-of-fold ridge predictions with folds grouped by cell."""
    pred = np.full(len(y), np.nan)
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(FOLDS).split(x, y, groups):
        if len(tr) > size:
            tr = rng.choice(tr, size, replace=False)
        sc = StandardScaler().fit(x[tr])
        model = RidgeCV(alphas=ALPHAS).fit(sc.transform(x[tr]), y[tr])
        pred[te] = model.predict(sc.transform(x[te]))
    return pred


def cluster_bootstrap_diff(y, pa, pb, groups, n=1000, seed=0) -> dict:
    """R² and MAE of b minus a, resampling 1-degree cells."""
    rng = np.random.default_rng(seed)
    cells, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(cells))]
    d_r2, d_mae = [], []
    for _ in range(n):
        ix = np.concatenate(
            [members[c] for c in rng.integers(0, len(cells), len(cells))]
        )
        a, b = metrics(y[ix], pa[ix]), metrics(y[ix], pb[ix])
        d_r2.append(b["r2"] - a["r2"])
        d_mae.append(b["mae"] - a["mae"])
    full_a, full_b = metrics(y, pa), metrics(y, pb)
    return {
        "r2_diff": full_b["r2"] - full_a["r2"],
        "r2_diff_ci95": np.percentile(d_r2, [2.5, 97.5]).tolist(),
        "mae_diff": full_b["mae"] - full_a["mae"],
        "mae_diff_ci95": np.percentile(d_mae, [2.5, 97.5]).tolist(),
        "n_cells": int(len(cells)),
    }


def main() -> None:
    """Probe depth from frozen embeddings for each model and compare models pairwise."""
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--labels", default=str(HERE / "labels.npz"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arms)

    lab = np.load(args.labels)
    inp = np.load(args.labels.replace("labels", "inputs"))
    assert (lab["idx"] == inp["idx"]).all()
    keep_s = lab["res"] <= MAX_TILE_RES
    if args.holdout:
        ll = np.load(H5 / "latlon_distribution.npy")[lab["idx"]]
        sample_cells = np.floor(ll[:, 0]).astype(int) * 1000 + np.floor(
            ll[:, 1]
        ).astype(int)
        keep_s &= np.isin(
            sample_cells, json.loads((HERE / "holdout_cells.json").read_text())
        )
    idx = lab["idx"][keep_s]

    elev = blocks(lab["elev"][keep_s].astype(np.float32))
    ok = np.isfinite(elev).all(-1) & (elev < 0).all(-1)
    depth = -np.nanmean(elev, -1)
    ok &= (depth >= DEPTH[0]) & (depth <= DEPTH[1])
    s, r, c = np.nonzero(ok)
    y = depth[s, r, c]

    latlon = np.load(H5 / "latlon_distribution.npy")[idx]
    cell = (
        np.floor(latlon[:, 0]).astype(int) * 1000 + np.floor(latlon[:, 1]).astype(int)
    )[s]

    s2 = blocks(inp["s2_median"][keep_s].astype(np.float32))[s, r, c]  # [M, 16, 12]
    feats = {"bands": np.log(np.clip(np.nanmean(s2, 1), 1, None))}
    feats["bands"] = np.nan_to_num(feats["bands"], nan=0.0)
    for name, path in arms.items():
        feats[name] = np.load(path, mmap_mode="r")[keep_s][s, r, c].astype(np.float32)
    sdb = blocks(inp["sdb"][keep_s].astype(np.float32))[s, r, c]
    sdb_valid = np.isfinite(sdb).all(-1)
    sdb_mean = np.nanmean(np.where(np.isfinite(sdb), sdb, np.nan), -1)

    print(
        f"samples {len(idx)} (dropped {int((~keep_s).sum())} with tile res > {MAX_TILE_RES} m), "
        f"tokens {len(y)}, cells {len(np.unique(cell))}, sdb-valid tokens {int(sdb_valid.sum())}"
    )
    print(f"depth percentiles 5/50/95: {np.percentile(y, [5, 50, 95]).round(1)}")

    results = {
        "n_samples": int(len(idx)),
        "n_tokens": int(len(y)),
        "n_cells": int(len(np.unique(cell))),
        "n_sdb_valid_tokens": int(sdb_valid.sum()),
        "probe": {},
        "paired": {},
        "sdb_subset": {},
    }
    oof = {}
    for size in SIZES:
        for name, x in feats.items():
            runs = [oof_predict(x, y, cell, size, seed) for seed in range(RESAMPLES)]
            ms = [metrics(y, p) for p in runs]
            results["probe"][f"{name}@{size}"] = {
                k: float(np.mean([m[k] for m in ms])) if ms[0][k] is not None else None
                for k in ms[0]
            }
            oof[(name, size)] = np.mean(runs, 0)
            m = results["probe"][f"{name}@{size}"]
            print(
                f"{name:>10} n_train={size:>6}: R² {m['r2']:.3f}  MAE {m['mae']:.2f} m  "
                f"rho {m['spearman']:.3f}",
                flush=True,
            )

    names = list(arms)
    size = SIZES[-1]
    for i, a in enumerate(names):
        for b in names[i + 1 :] + ["bands"]:
            if b == a:
                continue
            key = f"{b} - {a} @{size}"
            results["paired"][key] = cluster_bootstrap_diff(
                y, oof[(a, size)], oof[(b, size)], cell
            )
            d = results["paired"][key]
            print(
                f"{key}: dR² {d['r2_diff']:+.3f} {np.round(d['r2_diff_ci95'], 3)}, "
                f"dMAE {d['mae_diff']:+.2f} m {np.round(d['mae_diff_ci95'], 2)}"
            )

    # sdb formula vs probes, on the tokens where the formula gives a depth
    m = sdb_valid
    if m.sum() > 100:
        sdb_fit = oof_predict(sdb_mean[m, None], y[m], cell[m], size, 0)
        results["sdb_subset"]["sdb_formula_raw"] = metrics(y[m], sdb_mean[m])
        results["sdb_subset"]["sdb_formula_linear_fit"] = metrics(y[m], sdb_fit)
        for name in ["bands", *names]:
            results["sdb_subset"][f"{name}@{size}"] = metrics(
                y[m], oof[(name, size)][m]
            )
        for k, v in results["sdb_subset"].items():
            print(
                f"[sdb-valid tokens] {k:>24}: R² {v['r2']:.3f}  MAE {v['mae']:.2f} m  rho {v['spearman']:.3f}"
            )

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
