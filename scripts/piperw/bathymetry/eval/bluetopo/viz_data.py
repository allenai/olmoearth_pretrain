"""Data and image tiles for the BlueTopo results page (viz/).

Refits the probe's ridge per arm on the same 1-degree-cell folds (200k training tokens, seed 0)
and predicts every token of the held-out samples, so example maps and stats are out-of-fold.

Example rule (fixed before looking at predictions): samples with >= 50% labelled tokens and a
10-90th percentile depth spread >= 4 m, at most one per region, drawn with seed 0.

With --holdout, only samples in the 1-degree cells held out from BlueTopo training are used
(as probe.py --holdout). --compare a,b picks the pair for the cell map (default: first two arms).

  python viz_data.py name=emb/x.npy [...] [--holdout] [--compare a,b]
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from probe import ALPHAS, DEPTH, DEPTH_BINS, FOLDS, H5, MAX_TILE_RES, blocks, metrics
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
OUT = HERE / "viz"
GEBCO = "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/gebco/gebco_2026.tif"
N_TRAIN = 200_000
DEPTH_MAX = 20.0
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DEPTH_CMAP = LinearSegmentedColormap.from_list("depth", BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a8a", "#b52f2f"]
)
EX_PER_REGION = 1
N_EXAMPLES = 6


def region(lat: float, lon: float) -> str:
    """Coarse region name for a sample location."""
    if lat > 50:
        return "Alaska"
    if lon < -150:
        return "Hawaii"
    if lon > -68 and lat < 20:
        return "Puerto Rico / USVI"
    if lon < -114:
        return "Pacific coast"
    if lat < 31 and lon < -81.5:
        return "Gulf of Mexico"
    if 41 < lat < 49.5 and -93 < lon < -75.5 and not (lon > -80 and lat < 42):
        return "Great Lakes"
    if lat < 35:
        return "Florida / SE Atlantic"
    return "Mid / N Atlantic"


def save_img(arr, path, cmap=None, vmin=None, vmax=None) -> None:
    """Save a 2D array as a PNG tile."""
    if cmap is None:
        plt.imsave(path, arr)
    else:
        rgba = cmap(np.clip((arr - vmin) / (vmax - vmin), 0, 1))
        rgba[~np.isfinite(arr)] = 0
        plt.imsave(path, rgba)


def main() -> None:
    """Write the results page data: per-cell metrics, example tiles and maps."""
    args = sys.argv[1:]
    holdout = "--holdout" in args
    compare = next(
        (a.split("=", 1)[1].split(",") for a in args if a.startswith("--compare=")),
        None,
    )
    arms = dict(a.split("=", 1) for a in args if not a.startswith("--"))
    OUT.mkdir(exist_ok=True)
    (OUT / "tiles").mkdir(exist_ok=True)
    lab, inp = np.load(HERE / "labels.npz"), np.load(HERE / "inputs.npz")
    keep_s = lab["res"] <= MAX_TILE_RES
    if holdout:
        ll = np.load(H5 / "latlon_distribution.npy")[lab["idx"]]
        cells_all = np.floor(ll[:, 0]).astype(int) * 1000 + np.floor(ll[:, 1]).astype(
            int
        )
        keep_s &= np.isin(
            cells_all, json.loads((HERE / "holdout_cells.json").read_text())
        )
    idx = lab["idx"][keep_s]
    elev10 = lab["elev"][keep_s].astype(np.float32)
    elev = blocks(elev10)
    ok = np.isfinite(elev).all(-1) & (elev < 0).all(-1)
    with np.errstate(invalid="ignore"):
        depth = -np.nanmean(elev, -1)
    ok &= (depth >= DEPTH[0]) & (depth <= DEPTH[1])
    s, r, c = np.nonzero(ok)
    y = depth[s, r, c]
    latlon = np.load(H5 / "latlon_distribution.npy")[idx]
    cell_s = np.floor(latlon[:, 0]).astype(int) * 1000 + np.floor(latlon[:, 1]).astype(
        int
    )
    cell = cell_s[s]
    regions_s = np.array([region(*ll) for ll in latlon.astype(float)])

    sdb10 = inp["sdb"][keep_s].astype(np.float32)
    sdb_tok = blocks(sdb10)
    sdb_valid_tok = np.isfinite(sdb_tok).all(-1)
    with np.errstate(invalid="ignore"):
        sdb_mean = np.nanmean(sdb_tok, -1)

    # sample-level folds identical to the probe's token-level GroupKFold
    fold_of_cell = {}
    for f, (_, te) in enumerate(GroupKFold(FOLDS).split(y, y, cell)):
        for g in np.unique(cell[te]):
            fold_of_cell[g] = f
    fold_s = np.array([fold_of_cell.get(g, -1) for g in cell_s])

    rng = np.random.default_rng(0)
    preds = {}
    feats_all = {"bands": None, **arms}
    s2 = inp["s2_median"][keep_s]
    for name, path in feats_all.items():
        if name == "bands":
            b = blocks(s2.astype(np.float32)).mean(3)
            x_all = np.nan_to_num(np.log(np.clip(b, 1, None)), nan=0.0)
        else:
            x_all = np.load(path, mmap_mode="r")[keep_s]
        pmap = np.full(depth.shape, np.nan, np.float32)
        for f in range(FOLDS):
            tr = np.flatnonzero(fold_s[s] != f)
            tr = rng.choice(tr, min(N_TRAIN, len(tr)), replace=False)
            xt = np.asarray(x_all[s[tr], r[tr], c[tr]], np.float32)
            sc = StandardScaler().fit(xt)
            model = RidgeCV(alphas=ALPHAS).fit(sc.transform(xt), y[tr])
            for k in np.flatnonzero(fold_s == f):
                xs = np.asarray(x_all[k], np.float32).reshape(-1, xt.shape[1])
                pmap[k] = model.predict(sc.transform(xs)).reshape(depth.shape[1:])
        preds[name] = pmap
        print(f"{name}: {metrics(y, pmap[s, r, c])}", flush=True)

    # sdb formula, 1-d linear fit, same folds, only where the formula is valid
    fmap = np.full(depth.shape, np.nan, np.float32)
    m_tok = sdb_valid_tok[s, r, c]
    for f in range(FOLDS):
        tr = np.flatnonzero((fold_s[s] != f) & m_tok)
        a, b0 = np.polyfit(sdb_mean[s[tr], r[tr], c[tr]], y[tr], 1)
        te = fold_s == f
        fmap[te] = np.where(sdb_valid_tok[te], a * sdb_mean[te] + b0, np.nan)

    names = ["bands", *arms]
    data = {
        "arms": names,
        "holdout": holdout,
        "n_samples": int(len(idx)),
        "n_tokens": int(len(y)),
        "n_cells": int(len(np.unique(cell))),
        "depth_bins": DEPTH_BINS,
    }

    data["overall"] = {n: metrics(y, preds[n][s, r, c]) for n in names}
    m = m_tok
    data["sdb_subset"] = {
        "n": int(m.sum()),
        "formula_raw": metrics(y[m], sdb_mean[s, r, c][m]),
        "formula_fit": metrics(y[m], fmap[s, r, c][m]),
        **{n: metrics(y[m], preds[n][s, r, c][m]) for n in names},
    }

    # predicted vs true: per 1 m bin, median and 10/90th percentiles
    edges = np.arange(1, 31)
    tb = np.digitize(y, edges) - 1
    curves = {}
    for n in names:
        p = preds[n][s, r, c]
        curves[n] = [
            [float(v) for v in np.percentile(p[tb == i], [10, 50, 90])]
            if (tb == i).sum() > 50
            else None
            for i in range(len(edges) - 1)
        ]
    pf = sdb_mean[s, r, c]
    curves["formula_raw"] = [
        [float(v) for v in np.percentile(pf[m & (tb == i)], [10, 50, 90])]
        if (m & (tb == i)).sum() > 50
        else None
        for i in range(len(edges) - 1)
    ]
    data["curves"] = {
        "bin_centres": (edges[:-1] + 0.5).tolist(),
        "series": curves,
        "token_counts": np.bincount(tb, minlength=len(edges) - 1)[
            : len(edges) - 1
        ].tolist(),
    }

    # per region and per cell
    reg_tok = regions_s[s]
    data["regions"] = []
    for g in sorted(set(reg_tok), key=lambda g: -(reg_tok == g).sum()):
        mm = reg_tok == g
        row = {
            "region": g,
            "n_tokens": int(mm.sum()),
            "n_samples": int(len(np.unique(s[mm]))),
            "median_depth": float(np.median(y[mm])),
        }
        for n in names:
            row[n] = metrics(y[mm], preds[n][s, r, c][mm])
        data["regions"].append(row)
    data["cells"] = []
    a_name, b_name = compare or (
        (list(arms)[0], list(arms)[1]) if len(arms) >= 2 else (None, None)
    )
    for g in np.unique(cell):
        mm = cell == g
        if mm.sum() < (500 if holdout else 2000):
            continue
        row = {
            "lat": float(latlon[cell_s == g, 0].mean()),
            "lon": float(latlon[cell_s == g, 1].mean()),
            "n_tokens": int(mm.sum()),
            "region": str(reg_tok[mm][0]),
        }
        for n in names:
            row[f"mae_{n}"] = float(np.mean(np.abs(y[mm] - preds[n][s, r, c][mm])))
        data["cells"].append(row)
    data["cell_compare"] = [a_name, b_name]

    # land backdrop for the cell map, from GEBCO (elevation >= 0), equirectangular
    bbox = (-170.0, 15.0, -64.0, 62.0)
    with rasterio.open(GEBCO) as src:
        win = rasterio.windows.from_bounds(*bbox, transform=src.transform)
        g = src.read(1, window=win, out_shape=(470, 1060))
    plt.imsave(
        OUT / "land.png", np.where(g >= 0, 1.0, 0.0), cmap="gray_r", vmin=0, vmax=4
    )
    data["map_bbox"] = bbox

    # examples
    lab_frac = ok.mean((1, 2))
    with np.errstate(invalid="ignore"):
        spread = np.nanpercentile(
            np.where(ok, depth, np.nan), 90, axis=(1, 2)
        ) - np.nanpercentile(np.where(ok, depth, np.nan), 10, axis=(1, 2))
    cand = np.flatnonzero((lab_frac >= 0.5) & (spread >= 4) & (fold_s >= 0))
    cand = rng.permutation(cand)
    picked, seen = [], {}
    for k in cand:
        g = regions_s[k]
        if seen.get(g, 0) < EX_PER_REGION:
            picked.append(k)
            seen[g] = seen.get(g, 0) + 1
        if len(picked) == N_EXAMPLES:
            break
    data["examples"] = []
    for k in picked:
        sid = int(idx[k])
        rgb = s2[k][..., [2, 1, 0]].astype(np.float32)
        rgb = np.clip(np.nan_to_num(rgb) / 2500.0, 0, 1) ** (1 / 1.6)
        plt.imsave(OUT / "tiles" / f"{sid}_rgb.png", rgb)
        bt = -elev10[k]
        bt[~(bt > 0)] = np.nan
        save_img(bt, OUT / "tiles" / f"{sid}_bluetopo.png", DEPTH_CMAP, 0, DEPTH_MAX)
        save_img(
            sdb10[k], OUT / "tiles" / f"{sid}_formula.png", DEPTH_CMAP, 0, DEPTH_MAX
        )
        ex = {
            "id": sid,
            "lat": float(latlon[k, 0]),
            "lon": float(latlon[k, 1]),
            "region": str(regions_s[k]),
            "labelled_frac": float(lab_frac[k]),
            "median_depth": float(np.nanmedian(np.where(ok[k], depth[k], np.nan))),
            "sdb_valid_frac": float(np.isfinite(sdb10[k]).mean()),
            "mae": {},
        }
        for n in names:
            p = np.where(ok[k], preds[n][k], np.nan)
            save_img(p, OUT / "tiles" / f"{sid}_{n}.png", DEPTH_CMAP, 0, DEPTH_MAX)
            save_img(
                p - np.where(ok[k], depth[k], np.nan),
                OUT / "tiles" / f"{sid}_{n}_err.png",
                DIV_CMAP,
                -6,
                6,
            )
            ex["mae"][n] = float(np.nanmean(np.abs(p - depth[k])[ok[k]]))
        data["examples"].append(ex)

    (OUT / "data.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}/data.json; examples {[e['id'] for e in data['examples']]}")


if __name__ == "__main__":
    main()
