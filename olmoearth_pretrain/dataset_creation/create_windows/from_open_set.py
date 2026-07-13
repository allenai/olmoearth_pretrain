"""Create open-set segmentation pretraining windows from the label bank.

For each label sample in each completed open-set dataset, this creates one rslearn
window that is 128x128 at 10 m/pixel, centered on the sample, with a per-sample time
range. It also writes the combined ``open_set`` (classification) or
``open_set_regression`` label layer into the window, remapping per-dataset local class
ids to the global ids from ``class_mapping.json``.

Windows are bound to the dataset's ``window_data_storage`` factory (expected to be
``PerLayerStorageFactory``) so that the later-materialized imagery and these label
layers are stored one file per layer.

Datasets in ``EXCLUDED_SLUGS`` (held-out evals) are skipped. If an ``--exclude_geojson``
of WGS84 polygons is given, any window whose footprint intersects a polygon is skipped
(used to remove PASTIS / yemen_crop val/test extents).

Imagery layers are NOT materialized here; run ``rslearn prepare/ingest/materialize`` on
the resulting dataset afterwards.
"""

import argparse
import functools
import json
import logging
import multiprocessing
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import StarImapUnorderedWrapper
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from shapely import STRtree
from upath import UPath

from olmoearth_pretrain.open_set_segmentation_data.assemble_classes import (
    DEFAULT_OUTPUT_PATH as DEFAULT_CLASS_MAPPING_PATH,
)
from olmoearth_pretrain.open_set_segmentation_data.io import (
    CLASS_NODATA,
    REGRESSION_NODATA,
    lonlat_to_utm_pixel,
)
from olmoearth_pretrain.open_set_segmentation_data.manifest import (
    OUTPUT_ROOT,
    load_registry,
)
from olmoearth_pretrain.open_set_segmentation_data.pretrain_constants import (
    EXCLUDED_SLUGS,
    OPEN_SET_GROUP,
    OPEN_SET_LAYER,
    OPEN_SET_NODATA,
    OPEN_SET_REGRESSION_LAYER,
    OPEN_SET_RESOLUTION,
    OPEN_SET_WINDOW_SIZE,
    REGRESSION_DATASET_ID_NODATA,
    REGRESSION_VALUE_MAX_OUT,
    REGRESSION_VALUE_MIN_OUT,
    REGRESSION_VALUE_NODATA,
)

# Raster format for the label layers. Small tiled blocks, matching the pretraining
# imagery format so decoded shapes/layouts are consistent.
LABEL_RASTER_FORMAT = GeotiffRasterFormat(block_size=32, always_enable_tiling=True)

logger = logging.getLogger(__name__)

# Band names for the label layers.
OPEN_SET_BANDS = ["class"]
OPEN_SET_REGRESSION_BANDS = ["dataset_id", "value"]

# Fallback time range for the rare sample that has no time_range in its metadata.
_FALLBACK_TIME_RANGE = (
    datetime(2016, 6, 1, tzinfo=UTC),
    datetime(2024, 6, 1, tzinfo=UTC),
)


@dataclass
class ClassLookup:
    """Fast lookups derived from class_mapping.json."""

    # slug -> {local_class_id: global_class_id}
    local_to_global: dict[str, dict[int, int]]
    # slug -> {dataset_id, value_range: [min, max]}
    regression: dict[str, dict]


def _parse_time_range(
    tr: list[str] | None,
) -> tuple[datetime, datetime]:
    """Parse an [iso, iso] time range, falling back to a default if absent."""
    if not tr:
        return _FALLBACK_TIME_RANGE
    return (datetime.fromisoformat(tr[0]), datetime.fromisoformat(tr[1]))


def load_class_lookup(class_mapping_path: str) -> ClassLookup:
    """Build fast lookups from a class_mapping.json path."""
    with UPath(class_mapping_path).open() as f:
        mapping = json.load(f)
    local_to_global: dict[str, dict[int, int]] = {}
    for c in mapping["open_set"]["classes"]:
        local_to_global.setdefault(c["slug"], {})[int(c["local_id"])] = int(
            c["global_id"]
        )
    regression: dict[str, dict] = {}
    for d in mapping["open_set_regression"]["datasets"]:
        regression[d["slug"]] = {
            "dataset_id": int(d["dataset_id"]),
            "value_range": d["value_range"],
        }
    return ClassLookup(local_to_global=local_to_global, regression=regression)


def load_exclusion_index(geojson_path: str | None) -> STRtree | None:
    """Load an exclusion GeoJSON (WGS84) into an STRtree of polygons, or None."""
    if not geojson_path:
        return None
    with UPath(geojson_path).open() as f:
        fc = json.load(f)
    geoms = [
        shapely.geometry.shape(feat["geometry"])
        for feat in fc.get("features", [])
        if feat.get("geometry") is not None
    ]
    if not geoms:
        return None
    return STRtree(geoms)


def _window_wgs84_box(
    projection: Projection, bounds: tuple[int, int, int, int]
) -> shapely.Geometry:
    """Return the window footprint as a WGS84 shapely polygon."""
    geom = STGeometry(projection, shapely.box(*bounds), None).to_projection(
        WGS84_PROJECTION
    )
    return geom.shp


def _centered_window_bounds(
    center_col: int, center_row: int, size: int
) -> tuple[int, int, int, int]:
    """Return pixel bounds for a size x size box centered on (center_col, center_row)."""
    half = size // 2
    return (
        center_col - half,
        center_row - half,
        center_col - half + size,
        center_row - half + size,
    )


def _sanitize_name(name: str) -> str:
    """Make a filesystem-safe window name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def normalize_regression(values: np.ndarray, value_range: list[float]) -> np.ndarray:
    """Linearly remap values from [min, max] to [MIN_OUT, MAX_OUT] as uint16.

    Out-of-range values are clipped. Callers mask nodata separately.
    """
    vmin, vmax = float(value_range[0]), float(value_range[1])
    span = vmax - vmin
    if span <= 0:
        # Degenerate range: map everything to the min output value.
        return np.full(values.shape, REGRESSION_VALUE_MIN_OUT, dtype=np.uint16)
    scale = REGRESSION_VALUE_MAX_OUT - REGRESSION_VALUE_MIN_OUT
    out = REGRESSION_VALUE_MIN_OUT + (values - vmin) / span * scale
    out = np.clip(np.round(out), REGRESSION_VALUE_MIN_OUT, REGRESSION_VALUE_MAX_OUT)
    return out.astype(np.uint16)


# ---------------------------------------------------------------------------
# Sample iteration
# ---------------------------------------------------------------------------


def iter_dense_samples(datasets_root: UPath, slug: str) -> Iterator[dict]:
    """Yield lightweight per-sample descriptors for a dense (locations/*.json) dataset.

    Only the sidecar JSON path is recorded here; the JSON itself is read later, in the
    worker (see ``_hydrate_sample``). This keeps the main-process scan to one directory
    listing per dataset instead of one (slow, network-latency-bound) file open per sample.
    """
    locations = datasets_root / slug / "locations"
    if not locations.exists():
        return
    for jp in sorted(locations.glob("*.json")):
        yield {
            "kind": "dense",
            "slug": slug,
            "sample_id": jp.stem,
            "json_path": str(jp),
        }


def iter_sparse_samples(datasets_root: UPath, slug: str) -> Iterator[dict]:
    """Yield per-sample descriptors for a sparse (points.geojson) dataset."""
    points_path = datasets_root / slug / "points.geojson"
    if not points_path.exists():
        return
    with points_path.open() as f:
        fc = json.load(f)
    for feat in fc.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        props = feat.get("properties", {})
        yield {
            "kind": "sparse",
            "slug": slug,
            "sample_id": str(props.get("id")),
            "lon": float(lon),
            "lat": float(lat),
            "label": props.get("label"),
            "time_range": props.get("time_range"),
        }


def iter_dataset_samples(datasets_root: UPath, slug: str) -> Iterator[dict]:
    """Yield all sample descriptors for a dataset (dense or sparse)."""
    dataset_dir = datasets_root / slug
    if (dataset_dir / "points.geojson").exists():
        yield from iter_sparse_samples(datasets_root, slug)
    else:
        yield from iter_dense_samples(datasets_root, slug)


def completed_slugs(excluded: frozenset[str] = EXCLUDED_SLUGS) -> list[str]:
    """Return completed, non-excluded dataset slugs (sorted)."""
    registry = load_registry()
    return sorted(
        e["slug"]
        for e in registry["datasets"]
        if e.get("status") == "completed" and e["slug"] not in excluded
    )


# ---------------------------------------------------------------------------
# Window + label creation
# ---------------------------------------------------------------------------


def _hydrate_sample(sample: dict, datasets_root: UPath) -> dict:
    """Read a dense sample's sidecar JSON (deferred from the scan to the worker).

    Sparse samples already carry all their fields (from the single points.geojson read
    in the main process) and are returned unchanged, as are dense samples that have
    already been hydrated.
    """
    if sample["kind"] != "dense" or "crs" in sample:
        return sample
    json_path = sample.get("json_path")
    p = (
        UPath(json_path)
        if json_path is not None
        else datasets_root
        / sample["slug"]
        / "locations"
        / f"{sample['sample_id']}.json"
    )
    with p.open() as f:
        meta = json.load(f)
    sample = dict(sample)
    sample["crs"] = meta["crs"]
    sample["pixel_bounds"] = meta["pixel_bounds"]
    sample["time_range"] = meta.get("time_range")
    return sample


def _window_geometry_for_sample(
    sample: dict,
) -> tuple[Projection, tuple[int, int, int, int], int, int]:
    """Compute (projection, window_bounds, center_col, center_row) for a sample."""
    if sample["kind"] == "dense":
        projection = Projection(
            CRS.from_string(sample["crs"]), OPEN_SET_RESOLUTION, -OPEN_SET_RESOLUTION
        )
        x0, y0, x1, y1 = sample["pixel_bounds"]
        center_col = (x0 + x1) // 2
        center_row = (y0 + y1) // 2
    else:
        projection, center_col, center_row = lonlat_to_utm_pixel(
            sample["lon"], sample["lat"]
        )
    bounds = _centered_window_bounds(center_col, center_row, OPEN_SET_WINDOW_SIZE)
    return projection, bounds, center_col, center_row


def _build_open_set_label(
    sample: dict,
    lookup: ClassLookup,
    datasets_root: UPath,
    projection: Projection,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Build the (1, H, W) uint16 classification label for a sample."""
    slug = sample["slug"]
    size = OPEN_SET_WINDOW_SIZE
    out = np.full((size, size), OPEN_SET_NODATA, dtype=np.uint16)
    local_map = lookup.local_to_global.get(slug, {})

    if sample["kind"] == "dense":
        raster = LABEL_RASTER_FORMAT.decode_raster(
            datasets_root / slug / "locations",
            projection,
            bounds,
            resampling=Resampling.nearest,
            fname=f"{sample['sample_id']}.tif",
            nodata_val=CLASS_NODATA,
        )
        src = raster.get_chw_array()[0]
        for local_id, global_id in local_map.items():
            out[src == local_id] = global_id
    else:
        label = sample["label"]
        if label is not None and int(label) in local_map:
            half = size // 2
            out[half, half] = local_map[int(label)]

    return out[np.newaxis, :, :]


def _build_regression_label(
    sample: dict,
    lookup: ClassLookup,
    datasets_root: UPath,
    projection: Projection,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Build the (2, H, W) uint16 regression label (band0=dataset_id, band1=value)."""
    slug = sample["slug"]
    size = OPEN_SET_WINDOW_SIZE
    info = lookup.regression[slug]
    dataset_id = info["dataset_id"]
    value_range = info["value_range"]

    band0 = np.full((size, size), REGRESSION_DATASET_ID_NODATA, dtype=np.uint16)
    band1 = np.full((size, size), REGRESSION_VALUE_NODATA, dtype=np.uint16)

    if sample["kind"] == "dense":
        raster = LABEL_RASTER_FORMAT.decode_raster(
            datasets_root / slug / "locations",
            projection,
            bounds,
            resampling=Resampling.nearest,
            fname=f"{sample['sample_id']}.tif",
            nodata_val=REGRESSION_NODATA,
        )
        src = raster.get_chw_array()[0].astype(np.float64)
        valid = (src != REGRESSION_NODATA) & np.isfinite(src)
        band1[valid] = normalize_regression(src[valid], value_range)
        band0[valid] = dataset_id
    else:
        label = sample["label"]
        if label is not None:
            half = size // 2
            band1[half, half] = normalize_regression(
                np.array([float(label)]), value_range
            )[0]
            band0[half, half] = dataset_id

    return np.stack([band0, band1], axis=0)


def _write_label_layer(
    window: Window,
    layer_name: str,
    bands: list[str],
    array: np.ndarray,
    nodata_value: int,
) -> None:
    """Write a label array to the window through its WindowDataStorage."""
    raster = RasterArray(
        chw_array=array,
        metadata=RasterMetadata(nodata_value=nodata_value),
    )
    with window.data.open_layer_writer(layer_name) as writer:
        writer.write_raster(
            bands,
            LABEL_RASTER_FORMAT,
            window.projection,
            window.bounds,
            raster,
            group_idx=0,
        )
    window.mark_layer_completed(layer_name)


def create_sample_window(
    dataset: Dataset,
    sample: dict,
    lookup: ClassLookup,
    datasets_root: UPath,
    exclusion: STRtree | None,
) -> str:
    """Create one window (+ label layer) for a sample. Returns a status string."""
    sample = _hydrate_sample(sample, datasets_root)
    projection, bounds, center_col, center_row = _window_geometry_for_sample(sample)

    if exclusion is not None:
        box = _window_wgs84_box(projection, bounds)
        if len(exclusion.query(box, predicate="intersects")) > 0:
            return "excluded"

    time_range = _parse_time_range(sample["time_range"])
    center_time = time_range[0] + (time_range[1] - time_range[0]) / 2
    example_id = f"{sample['slug']}_{sample['sample_id']}"
    name = _sanitize_name(example_id)
    options = {
        "crs": projection.crs.to_string(),
        "resolution": OPEN_SET_RESOLUTION,
        "col": int(center_col),
        "row": int(center_row),
        "time": center_time.isoformat(),
        "example_id": example_id,
        "source_slug": sample["slug"],
    }
    window = Window(
        storage=dataset.storage,
        group=OPEN_SET_GROUP,
        name=name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options=options,
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()

    is_regression = sample["slug"] in lookup.regression
    if is_regression:
        array = _build_regression_label(
            sample, lookup, datasets_root, projection, bounds
        )
        _write_label_layer(
            window,
            OPEN_SET_REGRESSION_LAYER,
            OPEN_SET_REGRESSION_BANDS,
            array,
            REGRESSION_VALUE_NODATA,
        )
    else:
        array = _build_open_set_label(sample, lookup, datasets_root, projection, bounds)
        _write_label_layer(
            window, OPEN_SET_LAYER, OPEN_SET_BANDS, array, OPEN_SET_NODATA
        )
    return "created"


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level, caches per-process state)
# ---------------------------------------------------------------------------


@functools.cache
def _get_dataset(ds_path: str) -> Dataset:
    return Dataset(UPath(ds_path))


@functools.cache
def _get_lookup(class_mapping_path: str) -> ClassLookup:
    return load_class_lookup(class_mapping_path)


@functools.cache
def _get_exclusion(geojson_path: str | None) -> STRtree | None:
    return load_exclusion_index(geojson_path)


def _process_sample_job(
    ds_path: str,
    class_mapping_path: str,
    exclude_geojson_path: str | None,
    datasets_root: str,
    sample: dict,
) -> str:
    dataset = _get_dataset(ds_path)
    lookup = _get_lookup(class_mapping_path)
    exclusion = _get_exclusion(exclude_geojson_path)
    return create_sample_window(
        dataset, sample, lookup, UPath(datasets_root), exclusion
    )


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Destination rslearn dataset path (must contain config.json)",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default=None,
        help="Label bank root with {slug}/... (default: OUTPUT_ROOT/datasets on weka)",
    )
    parser.add_argument(
        "--class_mapping",
        type=str,
        default=str(DEFAULT_CLASS_MAPPING_PATH),
        help="Path to class_mapping.json produced by assemble_classes",
    )
    parser.add_argument(
        "--exclude_geojson",
        type=str,
        default=None,
        help="Optional WGS84 GeoJSON of polygons; windows intersecting any are skipped",
    )
    parser.add_argument(
        "--slugs",
        type=str,
        default=None,
        help="Optional comma-separated list of dataset slugs to process (default: all completed)",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    datasets_root = (
        UPath(args.datasets_root) if args.datasets_root else OUTPUT_ROOT / "datasets"
    )

    if args.slugs:
        slugs = [s for s in args.slugs.split(",") if s and s not in EXCLUDED_SLUGS]
    else:
        slugs = completed_slugs()

    logger.info("Scanning %d datasets for samples...", len(slugs))
    jobs: list[dict] = []
    for slug in tqdm.tqdm(slugs, desc="scanning datasets"):
        n_before = len(jobs)
        for sample in iter_dataset_samples(datasets_root, slug):
            jobs.append(
                dict(
                    ds_path=args.ds_path,
                    class_mapping_path=args.class_mapping,
                    exclude_geojson_path=args.exclude_geojson,
                    datasets_root=str(datasets_root),
                    sample=sample,
                )
            )
        logger.info(
            "scanned %s: %d samples (running total %d)",
            slug,
            len(jobs) - n_before,
            len(jobs),
        )

    logger.info(
        "Creating windows for %d samples across %d datasets", len(jobs), len(slugs)
    )
    counts = {"created": 0, "excluded": 0}
    if args.workers <= 1:
        lookup = load_class_lookup(args.class_mapping)
        exclusion = load_exclusion_index(args.exclude_geojson)
        dataset = Dataset(UPath(args.ds_path))
        for job in tqdm.tqdm(jobs):
            status = create_sample_window(
                dataset, job["sample"], lookup, datasets_root, exclusion
            )
            counts[status] = counts.get(status, 0) + 1
    else:
        p = multiprocessing.Pool(args.workers)
        outputs = p.imap_unordered(StarImapUnorderedWrapper(_process_sample_job), jobs)
        for status in tqdm.tqdm(outputs, total=len(jobs)):
            counts[status] = counts.get(status, 0) + 1
        p.close()
        p.join()

    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
