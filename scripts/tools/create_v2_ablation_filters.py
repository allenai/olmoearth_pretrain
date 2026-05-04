"""Create v2 ablation filters based on data ablation analysis findings.

B: remove_boring — cut ocean, desert, homogeneous, uniform tiles
C: remove_boring_strict — stricter version of B
D: diverse_infra — only diverse land cover + infrastructure-rich
E: best_of_all — remove_boring + moderate diversity + infrastructure enrichment

Usage:
    python scripts/tools/create_v2_ablation_filters.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FILTERS_DIR = Path("ablation_filters")
OUTPUT_DIR = FILTERS_DIR / "v2"
SCORES_PATH = Path("v0_1_osm_sampling_scores.parquet")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(SCORES_PATH, columns=[
        "lc_entropy", "osm_feature_count", "spatial_autocorr",
        "abs_lat", "lc_frac_water", "lc_frac_bare_sparse",
        "spatial_homogeneity", "lc_num_classes",
    ])

    qg = set(np.load(FILTERS_DIR / "clean_pool.npy").tolist())
    is_qg = np.array([i in qg for i in range(len(scores))])

    # Base: remove boring content from QG pool
    base = (is_qg &
            (scores.lc_frac_water <= 0.5) &
            (scores.lc_frac_bare_sparse <= 0.5) &
            (scores.lc_entropy >= 0.3) &
            (scores.spatial_homogeneity < 0.85))

    experiments = {
        "remove_boring": {
            "mask": base,
            "description": "QG + remove >50% water, >50% bare, entropy<0.3, homogeneity>=0.85",
        },
        "remove_boring_strict": {
            "mask": (is_qg &
                     (scores.lc_frac_water <= 0.5) &
                     (scores.lc_frac_bare_sparse <= 0.5) &
                     (scores.lc_entropy >= 0.5) &
                     (scores.spatial_homogeneity < 0.80) &
                     (scores.lc_num_classes >= 3)),
            "description": "QG + remove >50% water, >50% bare, entropy<0.5, homogeneity>=0.80, classes<3",
        },
        "diverse_infra": {
            "mask": is_qg & (scores.lc_entropy > 1.06) & (scores.osm_feature_count >= 4),
            "description": "QG + lc_entropy>1.06 AND osm_feature_count>=4",
        },
        "best_of_all": {
            "mask": base & (scores.lc_entropy >= 0.7) & (scores.osm_feature_count >= 2.0),
            "description": "remove_boring + lc_entropy>=0.7 AND osm_feature_count>=2.0",
        },
    }

    manifest = {}
    print(f"{'name':25s} {'n':>10s} {'entropy':>10s} {'osm':>8s} {'autocorr':>10s} {'lat':>8s}")
    print("-" * 75)

    for name, exp in experiments.items():
        mask = exp["mask"]
        indices = np.where(mask)[0]
        np.save(OUTPUT_DIR / f"{name}.npy", indices)

        sub = scores.iloc[indices]
        manifest[name] = {
            "description": exp["description"],
            "n_samples": int(len(indices)),
            "lc_entropy_mean": float(sub.lc_entropy.mean()),
            "osm_feature_count_mean": float(sub.osm_feature_count.mean()),
            "spatial_autocorr_mean": float(sub.spatial_autocorr.mean()),
            "abs_lat_mean": float(sub.abs_lat.mean()),
            "pct_tropical": float((sub.abs_lat < 35).mean()),
        }
        print(f"{name:25s} {len(indices):10,} {sub.lc_entropy.mean():10.3f} {sub.osm_feature_count.mean():8.3f} {sub.spatial_autocorr.mean():10.3f} {sub.abs_lat.mean():8.1f}")

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved filters and manifest to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
