"""Find windows whose precomputed embedding layer contains NaN.

``mosaic_tiles_to_bounds`` (embedding_materializer/fetchers.py) returns None --
and so records a coverage gap -- only when NO pixel of a window is covered
(``if not filled.any()``). A window that the product's tiles cover *partially*
is written as a normal layer with NaN in the uncovered pixels. Because the
layer exists, ``required: true`` does not drop that window, and the NaNs reach
the probe, where sklearn raises ``ValueError: Input contains NaN``.

The distinction that decides the fix: the AEF-supplemental tasks are ps1
center-pixel tasks, so a NaN only reaches the feature matrix if it lands on the
labelled centre pixel. This reports both counts separately.

Usage::

    python scripts/tools/scan_embedding_nans.py \
        --ds_path /weka/dfive-default/olmoearth/eval_datasets/lcmap_lu_year_aligned \
        --layer tessera

    # then, to make the affected windows drop out of the eval cleanly:
    python scripts/tools/scan_embedding_nans.py --ds_path ... --layer tessera \
        --drop_center_nan --write

``--drop_center_nan`` deletes the layer directory for the offending windows, so
rslearn's required-input filtering removes them exactly as it removes the
fully-uncovered ones. It does NOT delete the window: other evals on the same
dataset that do not require this layer are unaffected.
"""

import argparse
import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rslearn.dataset import Dataset
from upath import UPath

logger = logging.getLogger(__name__)


def find_raster(layer_dir: UPath) -> UPath | None:
    """Return the GeoTIFF inside a materialized layer directory, if any.

    rslearn stores the raster under a band-set hash subdirectory, so this walks
    one level down rather than assuming a filename.
    """
    for band_dir in sorted(layer_dir.iterdir()):
        if not band_dir.is_dir():
            continue
        for candidate in sorted(band_dir.iterdir()):
            if candidate.suffix.lower() in (".tif", ".tiff"):
                return candidate
    return None


def scan_window(layer_dir: UPath) -> tuple[bool, bool, float] | None:
    """Report (any_nan, center_nan, nan_fraction) for one window's layer.

    Returns None when the layer has no readable raster.
    """
    raster = find_raster(layer_dir)
    if raster is None:
        return None
    with rasterio.open(raster) as src:
        arr = src.read()  # (bands, H, W)
    mask = np.isnan(arr)
    if not mask.any():
        return False, False, 0.0
    _, height, width = arr.shape
    center = mask[:, height // 2, width // 2]
    return True, bool(center.any()), float(mask.mean())


def main() -> int:
    """Scan a dataset's embedding layer for NaN and optionally drop the bad ones."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", required=True, help="Dataset root (weka_path).")
    parser.add_argument("--layer", default="tessera", help="Layer name to scan.")
    parser.add_argument(
        "--sample", type=int, default=0, help="Scan only the first N windows (0 = all)."
    )
    parser.add_argument(
        "--drop_center_nan",
        action="store_true",
        help="Delete the layer dir for windows whose CENTRE pixel is NaN.",
    )
    parser.add_argument(
        "--drop_any_nan",
        action="store_true",
        help="Delete the layer dir for windows with ANY NaN (stricter).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Threads reading rasters (IO bound; default 32).",
    )
    parser.add_argument(
        "--write", action="store_true", help="Actually delete. Default is a dry run."
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    windows = Dataset(ds_path).storage.get_windows()
    if args.sample:
        windows = windows[: args.sample]
    logger.info(
        f"scanning {len(windows)} windows for NaN in layer {args.layer!r} "
        f"({args.workers} workers)"
    )

    counts: Counter = Counter()
    to_drop: list[UPath] = []
    offenders: list[tuple[str, bool, float]] = []
    worst = 0.0
    done = 0

    def inspect(window: object) -> tuple[object, UPath, tuple | None | str]:
        """Read one window's layer; returns a sentinel string when absent."""
        layer_dir = (
            ds_path / "windows" / window.group / window.name / "layers" / args.layer
        )
        if not layer_dir.exists():
            return window, layer_dir, "no_layer"
        try:
            return window, layer_dir, scan_window(layer_dir)
        except Exception as e:  # a corrupt raster should not abort a 45k scan
            logger.warning(f"{window.group}/{window.name}: unreadable ({e})")
            return window, layer_dir, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for window, layer_dir, result in pool.map(inspect, windows):
            done += 1
            if result == "no_layer":
                counts["no_layer"] += 1
            elif result is None:
                counts["unreadable"] += 1
            else:
                any_nan, center_nan, fraction = result
                counts["scanned"] += 1
                if any_nan:
                    counts["any_nan"] += 1
                    worst = max(worst, fraction)
                    offenders.append(
                        (f"{window.group}/{window.name}", center_nan, fraction)
                    )
                if center_nan:
                    counts["center_nan"] += 1
                if (args.drop_center_nan and center_nan) or (
                    args.drop_any_nan and any_nan
                ):
                    to_drop.append(layer_dir)
            if done % 5000 == 0:
                logger.info(f"  {done}/{len(windows)} ...")

    scanned = counts["scanned"]
    logger.info("")
    logger.info(f"scanned          : {scanned}")
    logger.info(f"no layer         : {counts['no_layer']} (already a coverage gap)")
    logger.info(f"unreadable       : {counts['unreadable']}")
    logger.info(
        f"any NaN          : {counts['any_nan']}"
        + (f" ({counts['any_nan'] / scanned:.2%})" if scanned else "")
    )
    logger.info(
        f"centre-pixel NaN : {counts['center_nan']}"
        + (f" ({counts['center_nan'] / scanned:.2%})" if scanned else "")
        + "   <- the ones that actually reach a ps1 probe"
    )
    logger.info(f"worst window NaN fraction: {worst:.1%}")

    # With a handful of offenders the list itself is the useful artifact: it is
    # what makes the deletion auditable afterwards.
    if offenders and len(offenders) <= 200:
        logger.info("")
        logger.info("windows containing NaN (centre? / NaN fraction):")
        for name, center_nan, fraction in sorted(offenders):
            flag = "CENTRE" if center_nan else "      "
            logger.info(f"  {flag}  {fraction:6.1%}  {name}")
    elif offenders:
        logger.info(f"({len(offenders)} offenders -- too many to list)")

    if not (args.drop_center_nan or args.drop_any_nan):
        logger.info("")
        logger.info(
            "no --drop_* flag given; nothing to delete. Re-run with "
            "--drop_center_nan (or --drop_any_nan) plus --write to act."
        )
        return 0

    verb = "deleting" if args.write else "would delete"
    logger.info(f"{verb} {len(to_drop)} layer directories")
    if args.write:
        for layer_dir in to_drop:
            shutil.rmtree(layer_dir)
        logger.info(
            "done -- those windows now look like coverage gaps and will be "
            "filtered out of every eval that requires this layer. Re-run the "
            "affected evals; note the window set has changed, so numbers "
            "already recorded on this dataset are no longer on the same windows."
        )
    else:
        logger.info("dry run -- add --write to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
