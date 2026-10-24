r"""How far does "similar embeddings share a class" hold? Neighbour purity vs cosine similarity.

Before propagating labels along embedding similarity (label spreading,
self-training), measure how often close neighbours actually share a class.
For a class-balanced sample of labelled query points, find their nearest
neighbours (cosine) in an unlabeled pool that follows the data's own class
mix, and report, for each similarity cutoff:

- coverage: share of queries with at least one neighbour above the cutoff;
- purity: share of those above-cutoff neighbours that have the query's class;
- per-class purity and the classes the impure neighbours come from.

Plus plain k-NN purity (k = 1, 5, 20) with no cutoff.

PASTIS (``--pastis_npz``, the open-set dump): pixel queries are the per-class
train sample, the pool is the uniform per-window train sample (void dropped);
region queries/pool are train segments. Neighbours from the query's own tile
are counted separately, since same-field pixels are trivially similar.
AEF eval tasks (``--aef_dump_root``): queries are val points, the pool is the
train split (one point per window, so no same-tile issue).

    python scripts/tools/neighbour_purity.py --pastis_npz .../pastis_open_set_v2.npz \
        --aef_dump_root .../aef_evals --out_dir /outputs
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CUTOFFS = [0.99, 0.98, 0.97, 0.95, 0.9, 0.8]
KS = [1, 5, 20]
K_MAX = 50
VOID = 19
PASTIS_NAMES = [
    "Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley",
    "Winter rapeseed", "Spring barley", "Sunflower", "Grapevine", "Beet",
    "Winter triticale", "Winter durum wheat", "Fruits, veg., flowers",
    "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal",
    "Sorghum",
]  # fmt: skip


def l2n(a: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm."""
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)


def knn_cosine(q: np.ndarray, pool: np.ndarray, k: int, chunk: int = 2048):
    """(similarities, indices) of the k most cosine-similar pool rows per query."""
    qn, pn = l2n(q), l2n(pool)
    sims, idxs = [], []
    for s in range(0, len(qn), chunk):
        sim = qn[s : s + chunk] @ pn.T
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        part = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-part, axis=1)
        sims.append(np.take_along_axis(part, order, axis=1))
        idxs.append(np.take_along_axis(idx, order, axis=1))
    return np.concatenate(sims), np.concatenate(idxs)


def purity_tables(
    yq: np.ndarray,
    sim: np.ndarray,
    ynb: np.ndarray,
    same_tile: np.ndarray | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Cutoff rows, k-NN rows and per-class/confusion rows for one setting."""
    same = ynb == yq[:, None]
    cut_rows, knn_rows, cls_rows = [], [], []
    valid = np.isfinite(sim)  # self matches are -inf
    masks = {"all neighbours": valid}
    if same_tile is not None:
        masks["other tiles only"] = valid & ~same_tile
    for scope, m in masks.items():
        for t in CUTOFFS:
            above = (sim >= t) & m
            has = above.any(axis=1)
            cut_rows.append(
                {
                    "scope": scope,
                    "cutoff": t,
                    "coverage": float(has.mean()),
                    "purity": float(same[above].mean()) if above.any() else None,
                    "n_pairs": int(above.sum()),
                    # class-averaged purity: each query class weighs the same
                    "purity_class_avg": float(
                        np.nanmean(
                            [
                                same[above & (yq[:, None] == c)].mean()
                                if (above & (yq[:, None] == c)).any()
                                else np.nan
                                for c in np.unique(yq)
                            ]
                        )
                    )
                    if above.any()
                    else None,
                }
            )
            if t in (0.98, 0.95, 0.9):
                for c in np.unique(yq):
                    a = above & (yq[:, None] == c)
                    if not a.any():
                        continue
                    wrong = ynb[a & ~same]
                    top = pd.Series(wrong).value_counts(normalize=True).head(3)
                    cls_rows.append(
                        {
                            "scope": scope,
                            "cutoff": t,
                            "class": int(c),
                            "coverage": float(a.any(axis=1)[yq == c].mean()),
                            "purity": float(same[a].mean()),
                            "top_confusions": {
                                int(k): round(float(v), 3) for k, v in top.items()
                            },
                        }
                    )
        for k in KS:
            # first k neighbours inside the scope
            rank = np.cumsum(m, axis=1)
            sel = m & (rank <= k)
            knn_rows.append(
                {
                    "scope": scope,
                    "k": k,
                    "purity": float(same[sel].mean()),
                    "purity_class_avg": float(
                        np.mean(
                            [
                                same[sel & (yq[:, None] == c)].mean()
                                for c in np.unique(yq)
                            ]
                        )
                    ),
                    # median similarity of the k-th in-scope neighbour
                    "median_sim_kth": float(
                        np.median(np.sort(np.where(sel, sim, -np.inf), axis=1)[:, -k])
                    ),
                }
            )
    return cut_rows, knn_rows, cls_rows


def run_pastis(npz_path: str, product: str, cap: int, rng) -> list[tuple[str, tuple]]:
    """PASTIS settings: pixel and region."""
    d = np.load(npz_path)
    split = d["window_split"]
    out = []
    for level, qp, pp in (("pixel", "pix", "uni"), ("region", "seg", "seg")):
        xq, yq, wq = (
            d[f"{qp}_{product}"].astype(np.float32),
            d[f"{qp}_label"],
            d[f"{qp}_window"],
        )
        xp, yp, wp = (
            d[f"{pp}_{product}"].astype(np.float32),
            d[f"{pp}_label"],
            d[f"{pp}_window"],
        )
        okq = np.isfinite(xq).all(1) & (split[wq] == "train") & (yq != VOID)
        okp = np.isfinite(xp).all(1) & (split[wp] == "train") & (yp != VOID)
        qi = np.nonzero(okq)[0]
        keep = []
        for c in np.unique(yq[qi]):
            idx = qi[yq[qi] == c]
            keep.append(rng.choice(idx, min(cap, len(idx)), replace=False))
        qi = np.concatenate(keep)
        pi = np.nonzero(okp)[0]
        sim, idx = knn_cosine(xq[qi], xp[pi], K_MAX + 1)
        nb = pi[idx]
        if qp == pp:  # a region is its own nearest neighbour; drop self matches
            notself = nb != qi[:, None]
            sim = np.where(notself, sim, -np.inf)
        same_tile = wp[nb] == wq[qi][:, None]
        out.append(
            (
                f"PASTIS {level}",
                (yq[qi], sim[:, : K_MAX + 1], yp[nb], same_tile),
            )
        )
        logger.info(f"PASTIS {level}: {len(qi)} queries, pool {len(pi)}")
    return out


def run_aef(root: str, product: str, cap: int, rng) -> list[tuple[str, tuple]]:
    """One setting per dumped AEF eval task: val queries against the train pool."""
    out = []
    for npz in sorted((Path(root) / product).glob("*.npz")):
        d = np.load(npz)
        xp, yp = d["train_embeddings"].astype(np.float32), d["train_labels"].astype(int)
        xq, yq = d["val_embeddings"].astype(np.float32), d["val_labels"].astype(int)
        keep = []
        for c in np.unique(yq):
            idx = np.nonzero(yq == c)[0]
            keep.append(rng.choice(idx, min(cap, len(idx)), replace=False))
        qi = np.concatenate(keep)
        k = min(K_MAX, len(xp) - 1)
        sim, idx = knn_cosine(xq[qi], xp, k)
        name = str(d["dataset"]).replace("_year_aligned", "")
        out.append((name, (yq[qi], sim, yp[idx], None)))
        logger.info(
            f"{name}: {len(qi)} queries, pool {len(xp)}, classes {len(np.unique(yp))}"
        )
    return out


def main() -> None:
    """Compute purity tables for every setting and write CSV/JSON."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pastis_npz")
    parser.add_argument("--aef_dump_root")
    parser.add_argument("--product", default="olmoearth")
    parser.add_argument("--query_cap", type=int, default=500)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = []
    if args.pastis_npz:
        settings += run_pastis(args.pastis_npz, args.product, args.query_cap, rng)
    if args.aef_dump_root:
        settings += run_aef(args.aef_dump_root, args.product, args.query_cap, rng)

    cut_all, knn_all, cls_all = [], [], []
    for name, (yq, sim, ynb, same_tile) in settings:
        c, k, cl = purity_tables(yq, sim, ynb, same_tile)
        for rows, acc in ((c, cut_all), (k, knn_all), (cl, cls_all)):
            for r in rows:
                acc.append({"setting": name, "n_classes": int(len(np.unique(yq))), **r})
    pd.DataFrame(cut_all).to_csv(out_dir / "purity_by_cutoff.csv", index=False)
    pd.DataFrame(knn_all).to_csv(out_dir / "purity_knn.csv", index=False)
    with open(out_dir / "purity_by_class.json", "w") as f:
        json.dump(cls_all, f, indent=1)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(pd.DataFrame(cut_all).round(3).to_string())
    print(pd.DataFrame(knn_all).round(3).to_string())
    pastis = [
        r for r in cls_all if r["setting"].startswith("PASTIS") and r["cutoff"] == 0.95
    ]
    for r in pastis:
        r["class"] = PASTIS_NAMES[r["class"]]
        r["top_confusions"] = {
            PASTIS_NAMES[k]: v for k, v in r["top_confusions"].items()
        }
    print(pd.DataFrame(pastis).to_string())


if __name__ == "__main__":
    main()
