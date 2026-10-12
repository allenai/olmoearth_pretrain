r"""Select the open-set H5 samples that do NOT overlap an AlphaEarth supplemental eval.

The open-set label bank ingests the AEF supplemental eval datasets themselves
(``olmoearth_{africa_crop_mask,canada_crops_fine,descals_oil_palm,ethiopia_crops,
glance_land_cover,lcmap_land_use,us_tree_genus}``, all splits) as well as many of their
upstream sources (eddmaps, globalgeotree, cropharvest, glance training data, ...), so
the AEF evals (``all_evals.AEF_SUPPLEMENTAL_YEAR_ALIGNED``) are contaminated for any
model trained on the full bank. This script drops every H5 sample whose footprint
intersects an AEF eval window, purely spatially (no time check, so it is conservative):

* an H5 sample is the ``--half_extent_m`` square around its
  ``latlon_distribution.npy`` center (open-set: 128 px at 10 m, centered, so 640 m);
* an eval window is its ``metadata.json`` bounds in its own UTM projection (32 px at
  10 m); the ``_year_aligned`` windows are geometrically identical to their parents;
* a sample collides when the two squares intersect, tested in the eval window's CRS.

Two keep-lists are written per base subset (the full build, plus ``--base_filter``
files such as the HQ subset), for ``OlmoEarthDatasetConfig.filter_idx_file``:

* ``*_valtest_<count>.npy``: drops samples on ``eval_split`` val/test windows -- enough
  for train -> val/test probes and fine-tuning;
* ``*_allsplits_<count>.npy``: also drops samples on train windows. Needed for the
  v1.3 in-loop AEF evals: the kNN twins carry AEF's balanced-trial protocol, which
  pools ALL splits into its support / query draws.

Like ``open_set_hq/select_h5_indices.py`` the indices are positions in ONE H5 build, so
the names embed the build's sample count and the script must be re-run whenever
``open_set_base.OPEN_SET_H5_DIR`` changes.

Usage (from the repo root, weka mounted)::

    python scripts/official/v1_3/open_set_aef_clean/select_aef_clean_indices.py
    # Report only (no output), e.g. for the osm_sampling corpus (256 px grid tiles):
    python scripts/official/v1_3/open_set_aef_clean/select_aef_clean_indices.py \
        --h5_dir <osm_sampling h5 dir> --half_extent_m 1280 --report_only
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

# The experiment scripts live one directory up (open_set_base.OPEN_SET_H5_DIR).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_set_base import OPEN_SET_H5_DIR  # noqa: E402

from olmoearth_pretrain.internal.all_evals import (  # noqa: E402
    AEF_SUPPLEMENTAL_YEAR_ALIGNED,
)

logger = logging.getLogger("select_aef_clean_indices")

EVAL_DATASETS_ROOT = "/weka/dfive-default/olmoearth/eval_datasets"
FILTERS_DIR = "/weka/dfive-default/helios/dataset/open_set_dataset/filters"
# Base subsets to clean besides the full build: output prefix -> keep-list prefix.
DEFAULT_BASE_FILTERS = {"open_set_hq_v1_aef_clean": "open_set_hq_v1"}
EARTH_RADIUS_M = 6371000.0
SPLITS = {"valtest": {"val", "test"}, "allsplits": {"train", "val", "test"}}


def _read_window(path: str) -> dict:
    with open(os.path.join(path, "metadata.json")) as f:
        meta = json.load(f)
    projection = meta["projection"]
    assert projection["x_resolution"] == -projection["y_resolution"], projection
    b0, b1, b2, b3 = meta["bounds"]
    res = projection["x_resolution"]
    return dict(
        crs=projection["crs"],
        # Pixel bounds -> projection units (rslearn y_resolution is negative).
        cx=(b0 + b2) / 2 * res,
        cy=(b1 + b3) / 2 * -res,
        half=max(b2 - b0, b3 - b1) / 2 * res,
        eval_split=meta.get("options", {}).get("eval_split"),
    )


def load_eval_windows(names: tuple[str, ...] = AEF_SUPPLEMENTAL_YEAR_ALIGNED):
    """Every window (all splits) of the given eval datasets."""
    frames = []
    for name in names:
        root = os.path.join(EVAL_DATASETS_ROOT, name, "windows")
        paths = [
            os.path.join(root, group, window)
            for group in os.listdir(root)
            for window in os.listdir(os.path.join(root, group))
        ]
        with ThreadPoolExecutor(64) as executor:
            df = pd.DataFrame(list(executor.map(_read_window, paths)))
        df["dataset"] = name
        logger.info(f"{name}: {len(df)} windows")
        frames.append(df)
    windows = pd.concat(frames, ignore_index=True)
    assert windows.eval_split.isin(SPLITS["allsplits"]).all()
    return windows


def _unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
    )


def find_collisions(
    latlon: np.ndarray, windows: pd.DataFrame, half_extent_m: float
) -> pd.DataFrame:
    """(window_row, sample_idx) pairs whose footprints intersect."""
    tree = cKDTree(_unit_xyz(latlon[:, 0], latlon[:, 1]))
    pairs = []
    for crs, group in windows.groupby("crs"):
        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(group.cx.values, group.cy.values)
        # Chord radius covering both squares' corners, plus slack for projection skew.
        radius = (
            np.sqrt(2) * (half_extent_m + group.half.max()) + 100
        ) / EARTH_RADIUS_M
        candidates = tree.query_ball_point(_unit_xyz(lat, lon), radius)
        rows = np.repeat(group.index.values, [len(c) for c in candidates])
        if not len(rows):
            continue
        idx = np.concatenate([np.asarray(c, dtype=np.int64) for c in candidates])
        from_ll = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        sx, sy = from_ll.transform(latlon[idx, 1], latlon[idx, 0])
        lim = half_extent_m + windows.half.values[rows]
        hit = (np.abs(sx - windows.cx.values[rows]) < lim) & (
            np.abs(sy - windows.cy.values[rows]) < lim
        )
        pairs.append(pd.DataFrame({"window_row": rows[hit], "sample_idx": idx[hit]}))
    out = pd.concat(pairs, ignore_index=True)
    out["dataset"] = windows.dataset.values[out.window_row]
    out["eval_split"] = windows.eval_split.values[out.window_row]
    return out


def main() -> None:
    """Scan the H5 build and write (or report) the AEF-clean keep-lists."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5_dir", default=OPEN_SET_H5_DIR)
    parser.add_argument("--half_extent_m", type=float, default=640.0)
    parser.add_argument("--output_dir", default=FILTERS_DIR)
    parser.add_argument("--output_prefix", default="open_set_aef_clean")
    parser.add_argument("--report_only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    count = int(Path(args.h5_dir).name)
    latlon = np.load(os.path.join(args.h5_dir, "latlon_distribution.npy"))
    assert len(latlon) == count, (len(latlon), count)
    windows = load_eval_windows()
    pairs = find_collisions(latlon, windows, args.half_extent_m)

    bases = {args.output_prefix: np.arange(count, dtype=np.int64)}
    if args.h5_dir == OPEN_SET_H5_DIR:
        for prefix, base in DEFAULT_BASE_FILTERS.items():
            bases[prefix] = np.load(os.path.join(FILTERS_DIR, f"{base}_{count}.npy"))

    for split_name, splits in SPLITS.items():
        hits = pairs[pairs.eval_split.isin(splits)]
        colliding = np.unique(hits.sample_idx.values)
        in_split = windows.eval_split.isin(splits)
        frac_hit = (
            hits.groupby("dataset").window_row.nunique()
            / windows[in_split].groupby("dataset").size()
        )
        logger.info(
            f"{split_name}: {len(colliding)} / {count} samples collide "
            f"({len(colliding) / count:.2%}); fraction of eval windows hit:\n"
            f"{frac_hit.fillna(0).round(3).to_string()}"
        )
        for prefix, base in bases.items():
            keep = np.setdiff1d(base, colliding).astype(np.int64)
            path = os.path.join(args.output_dir, f"{prefix}_{split_name}_{count}.npy")
            logger.info(f"  {prefix}: keep {len(keep)} / {len(base)}")
            if not args.report_only:
                np.save(path, keep)
                logger.info(f"  wrote {path}")


if __name__ == "__main__":
    main()
