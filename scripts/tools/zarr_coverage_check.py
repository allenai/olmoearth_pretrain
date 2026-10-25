r"""Does the public OlmoEarth v1.3 geozarr cover the AEF eval tasks' locations and years?

For each eval dataset, samples windows and reads the store's 128-dim
embedding at the window centre for the window's target year, reporting the
window CRSs, the years asked for vs the years each UTM zone holds, and the
share of sampled windows with a valid (non-nodata) embedding. Run before
building landscape pools from the store for these tasks.

    uv run --with 'zarr>=3.1' --with google-crc32c --with aiohttp \
        python scripts/tools/zarr_coverage_check.py --zarr_url ...
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rslearn.dataset import Dataset
from upath import UPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_olmoearth_zarr_embeddings import ZARR_NODATA, ZarrReader  # noqa: E402

from olmoearth_pretrain.evals.embedding_materializer.providers import (  # noqa: E402
    get_target_year,
)

logger = logging.getLogger(__name__)

ROOT = "/weka/dfive-default/olmoearth/eval_datasets"
DATASETS = [
    "africa_crop_mask_year_aligned",
    "canada_crops_coarse_year_aligned",
    "canada_crops_fine_year_aligned",
    "descals_year_aligned",
    "ethiopia_crops_year_aligned",
    "glance_year_aligned",
    "lcmap_lu_year_aligned",
    "us_trees_year_aligned",
]


def main() -> None:
    """Sample windows per dataset and report store coverage."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zarr_url", required=True)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    reader = ZarrReader(args.zarr_url)

    for name in DATASETS:
        windows = Dataset(UPath(f"{ROOT}/{name}")).storage.get_windows()
        pick = [
            windows[i]
            for i in rng.choice(len(windows), min(args.n, len(windows)), replace=False)
        ]
        crs = Counter(w.projection.crs.to_epsg() for w in pick)
        res = Counter(
            (w.projection.x_resolution, w.projection.y_resolution) for w in pick
        )
        years = Counter(get_target_year(w) for w in pick)
        ok = bad = err = ok23 = 0
        errors: Counter = Counter()
        zone_years = {}
        for w in pick:
            epsg = w.projection.crs.to_epsg()
            try:
                emb, yrs, x0, y0 = reader._zone(epsg)
                zone_years[epsg] = yrs
                year = get_target_year(w)
                if year not in yrs:
                    raise ValueError(f"year {year} not in store")
                left, top, right, bottom = w.bounds
                c = (left + right) // 2 - round(x0 / 10)
                r = (top + bottom) // 2 + round(y0 / 10)
                v = np.asarray(emb[yrs.index(year), :, r, c])
                if (v == ZARR_NODATA).all():
                    bad += 1
                else:
                    ok += 1
                # 2023-2025 are global; 2017-2022 cover selected countries only
                if not (np.asarray(emb[yrs.index(2023), :, r, c]) == ZARR_NODATA).all():
                    ok23 += 1
            except Exception as e:  # report, keep going
                err += 1
                errors[f"{type(e).__name__}: {str(e)[:80]}"] += 1
        logger.info(
            f"{name}: {len(windows)} windows; sampled {len(pick)}: valid {ok}, nodata {bad}, "
            f"valid in 2023 {ok23}, "
            f"error {err}; CRS {dict(crs)}; res {dict(res)}; years {dict(years)}; "
            f"store years by zone {zone_years}; errors {dict(errors)}"
        )


if __name__ == "__main__":
    main()
