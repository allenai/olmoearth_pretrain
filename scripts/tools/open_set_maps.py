r"""Per-pixel open-set maps on PASTIS val tiles, next to a Sentinel-2 capture and the labels.

For each scenario (a set of known PASTIS classes), each method is fit exactly as
in open_set_eval.py on the dump's train pixels (per-class sample; unlabeled pool
= the uniform per-window sample of train windows, natural class mix), its
threshold is set on the dump's known val pixels to keep ``--tpr`` of them, and
it is then applied to EVERY pixel of a few val windows. A window's row shows:
Sentinel-2 RGB (one monthly mosaic), ground truth, the ideal open-set map
(known classes in colour, everything else grey), then each method's map (a
known-class prediction where accepted, grey where rejected) and the first
method's score.

Windows are the val windows with the most even mix of known and unknown crop
pixels. Writes one PNG per (scenario, window) and stats.json to ``--out_dir``.

    python scripts/tools/open_set_maps.py --npz .../pastis_open_set_v2.npz --out_dir /outputs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rslearn.dataset import Dataset  # noqa: E402
from upath import UPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_pastis_open_set_pixels import (  # noqa: E402
    OE_DS,
    SRC_DS,
    read_layer,
    valid_mask,
)
from open_set_eval import (  # noqa: E402
    SINGLE_CLASS_SKIP,
    Context,
    cap_per_class,
    scores_for,
)

logger = logging.getLogger(__name__)

# PASTIS nomenclature (Garnot & Landrieu 2021)
CLASS_NAMES = [
    "Background",
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits, veg., flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
    "Void",
]
VOID = 19
REJECT = 20  # map value for "rejected / unknown"
PRODUCT_LAYER = {"olmoearth": "olmoearth_emb", "aef": "gse", "tessera_v2": "tessera_v2"}
SCENARIOS = {
    "wheat_only": (2,),
    "corn_only": (3,),
    "main5": (1, 2, 3, 4, 5),
}


def palette() -> ListedColormap:
    """Colours for labels 0..18, void (19, white) and rejected (20, grey)."""
    tab = plt.get_cmap("tab20").colors
    cols = [(0.15, 0.15, 0.15)] + [tab[i % 20] for i in range(18)]
    cols += [(1.0, 1.0, 1.0), (0.72, 0.72, 0.72)]
    return ListedColormap(cols)


def s2_rgb(window, month: int) -> np.ndarray | None:
    """(H, W, 3) RGB of one monthly S2 mosaic, or None if missing."""
    layer = f"sentinel2_l2a_mo{month:02d}"
    if not window.is_layer_completed(layer):
        return None
    for band_dir in Path(str(window.get_layer_dir(layer))).iterdir():
        bands = band_dir.name.split("_")
        if {"B02", "B03", "B04"} <= set(bands):
            # S2 layers hold several band-set geotiffs; read this one directly
            with rasterio.open(band_dir / "geotiff.tif") as src:
                arr = src.read()
            rgb = np.stack([arr[bands.index(b)] for b in ("B04", "B03", "B02")], -1)
            return np.clip(rgb.astype(np.float32) / 3000.0, 0, 1) ** 0.8
    return None


def main() -> None:
    """Fit, threshold, map and plot every scenario."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--product", default="olmoearth")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", default=["unlab_neg", "knn10", "msp"])
    parser.add_argument("--windows_per_scenario", type=int, default=4)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--train_cap", type=int, default=2000)
    parser.add_argument("--unlab_size", type=int, default=20000)
    parser.add_argument("--tpr", type=float, default=0.95)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = np.load(args.npz)
    split_of = d["window_split"]
    meta = json.loads(str(d["windows_json"]))
    x = d[f"pix_{args.product}"].astype(np.float32)
    y = d["pix_label"]
    ps = split_of[d["pix_window"]]
    ok = np.isfinite(x).all(axis=1)
    tr = np.nonzero(ok & (ps == "train"))[0]
    tr = tr[cap_per_class(y[tr], args.train_cap, rng)]
    va = np.nonzero(ok & (ps == "val"))[0]
    xu = d[f"uni_{args.product}"].astype(np.float32)
    uok = np.isfinite(xu).all(axis=1) & (split_of[d["uni_window"]] == "train")
    pool = xu[uok]
    x_unlab = pool[
        np.random.default_rng(args.seed + 1).choice(
            len(pool), min(args.unlab_size, len(pool)), replace=False
        )
    ]

    oe_ds = Dataset(UPath(OE_DS))
    src_ds = Dataset(UPath(SRC_DS))
    val_ids = [i for i, m in enumerate(meta) if m["split"] == "val"]
    logger.info(f"reading labels of {len(val_ids)} val windows")
    oe_windows = {(w.group, w.name): w for w in oe_ds.storage.get_windows()}
    src_windows = {(w.group, w.name): w for w in src_ds.storage.get_windows()}
    labels = {}
    for i in val_ids:
        w = oe_windows[(meta[i]["group"], meta[i]["name"])]
        labels[i] = read_layer(w, "label")[0]

    cmap = palette()
    stats = {}
    for scen in args.scenarios:
        ks = SCENARIOS[scen]

        # windows with the most even mix of known and unknown-crop pixels
        def mix(lab: np.ndarray) -> float:
            """min(known fraction, unknown-crop fraction)."""
            n = (lab != VOID).sum() or 1
            known = np.isin(lab, ks).sum() / n
            unk = (~np.isin(lab, ks) & (lab != VOID) & (lab != 0)).sum() / n
            return min(known, unk)

        chosen = sorted(val_ids, key=lambda i: -mix(labels[i]))[
            : args.windows_per_scenario
        ]
        methods = [
            m for m in args.methods if not (len(ks) == 1 and m in SINGLE_CLASS_SKIP)
        ]
        trk = tr[np.isin(y[tr], ks)]
        vak = va[np.isin(y[va], ks)]
        fitted = {}
        for m in methods:
            ctx = Context(
                xtr=x[trk],
                ytr=y[trk],
                xval=x[vak],
                yval=y[vak],
                x_unlab=x_unlab,
                xneg=x[:0],
                kmeans={},
                seed=args.seed,
                c=args.c,
            )
            f = scores_for(m, ctx)
            sv, _ = f(x[vak])
            fitted[m] = (f, float(np.quantile(sv, 1 - args.tpr)))
        stats[scen] = {"known": list(ks), "windows": {}}

        for i in chosen:
            key = (meta[i]["group"], meta[i]["name"])
            w = oe_windows[key]
            lab = labels[i]
            emb = read_layer(
                w if args.product == "olmoearth" else src_windows[key],
                PRODUCT_LAYER[args.product],
            )
            valid = valid_mask(emb, lab.shape)
            flat = emb.reshape(emb.shape[0], -1).T.astype(np.float32)
            ideal = np.where(np.isin(lab, ks), lab, REJECT)
            ideal[lab == VOID] = VOID
            maps, scores, wstats = {}, None, {}
            for m, (f, thr) in fitted.items():
                s, pred = f(flat)
                s, pred = s.reshape(lab.shape), pred.reshape(lab.shape)
                mp = np.where(s >= thr, pred, REJECT)
                mp[~valid] = VOID
                maps[m] = mp
                if scores is None:
                    scores = np.where(valid, s, np.nan)
                kn = np.isin(lab, ks) & valid
                unk = ~np.isin(lab, ks) & (lab != VOID) & valid
                wstats[m] = {
                    "known_kept": float((s[kn] >= thr).mean()) if kn.any() else None,
                    "known_correct": float(((s >= thr) & (pred == lab))[kn].mean())
                    if kn.any()
                    else None,
                    "unknown_rejected": float((s[unk] < thr).mean())
                    if unk.any()
                    else None,
                    "background_rejected": float((s[unk & (lab == 0)] < thr).mean())
                    if (unk & (lab == 0)).any()
                    else None,
                }
            stats[scen]["windows"][meta[i]["name"]] = wstats

            rgb = s2_rgb(src_windows[key], args.month)
            n_panels = 3 + len(maps) + 1
            fig, axes = plt.subplots(1, n_panels, figsize=(3.1 * n_panels, 3.6))
            panels = [(f"Sentinel-2 RGB (mo{args.month:02d})", rgb, None)]
            panels.append(("Ground truth", lab, cmap))
            panels.append(("Ideal: known vs unknown", ideal, cmap))
            for m, mp in maps.items():
                st = wstats[m]
                kk = st["known_kept"]
                ur = st["unknown_rejected"]
                panels.append(
                    (
                        f"{m}\nknown kept {kk:.0%} | unknown rej. {ur:.0%}"
                        if kk is not None and ur is not None
                        else m,
                        mp,
                        cmap,
                    )
                )
            for ax, (title, img, cm) in zip(axes, panels):
                if img is None:
                    ax.text(0.5, 0.5, "no S2 mosaic", ha="center", va="center")
                elif cm is None:
                    ax.imshow(img)
                else:
                    ax.imshow(
                        img, cmap=cm, vmin=0, vmax=REJECT, interpolation="nearest"
                    )
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            ax = axes[-1]
            first = methods[0]
            lo, hi = np.nanquantile(scores, [0.02, 0.98])
            im = ax.imshow(scores, cmap="viridis", vmin=lo, vmax=hi)
            ax.contour(
                np.nan_to_num(scores, nan=lo) >= fitted[first][1],
                levels=[0.5],
                colors="white",
                linewidths=0.5,
            )
            ax.set_title(f"{first} score (white = threshold)", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            present = sorted(set(np.unique(lab).tolist()) | {REJECT})
            handles = [
                Patch(
                    color=cmap(c / REJECT),
                    label=("Rejected / unknown" if c == REJECT else CLASS_NAMES[c])
                    + (" (known)" if c in ks else ""),
                )
                for c in present
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(6, len(handles)),
                fontsize=7,
                frameon=False,
            )
            fig.suptitle(
                f"{scen}: known = {', '.join(CLASS_NAMES[c] for c in ks)} | "
                f"{args.product} | window {meta[i]['name']}",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0.1, 1, 0.95))
            path = out_dir / f"{scen}_{meta[i]['name']}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            logger.info(f"wrote {path}: {wstats}")
    with open(out_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
