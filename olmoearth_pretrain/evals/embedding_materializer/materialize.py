"""Materialize precomputed embedding products into eval datasets with rslearn.

Each product is an rslearn data source that reads directly from the published
rasters (no ingestion step). Materializing it into a dataset means declaring it
as a layer in the dataset's config.json, then running rslearn's own prepare and
materialize on that one layer. What this module adds on top of rslearn:

- the layer's time window is chosen so each window reads the annual product for
  the year its own time range is centred on (see ``request_time_offset``);
- a per-dataset manifest recording which windows carry the layer, which fell in
  coverage gaps and which failed. ``scripts/tools/register_embedding_products.py``
  reads it to decide when a dataset goes live.
"""

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from rslearn.data_sources import DataSource
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import materialize_window, prepare_dataset_windows
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec

logger = logging.getLogger(__name__)

# Backup of the pre-edit config.json, written once so re-runs keep the pristine
# copy. Shared with register_embedding_products.py, which edits the same file.
CONFIG_BACKUP_NAME = "config.json.pre_embedding_layers.bak"

# rslearn resolves a relative metadata_cache_dir against the dataset root, so
# the path stored in config.json works on any machine that mounts the dataset.
DEFAULT_METADATA_CACHE_DIR = "cache/aef_index"

# Transient reads over HTTP (e.g. S3 range requests behind AEF COGs) can fail
# in ways GDAL's own retry logic does not cover, such as a 206 response with a
# truncated body. rslearn retries the whole window this many times.
MATERIALIZE_RETRY_ATTEMPTS = 3
MATERIALIZE_RETRY_BACKOFF = timedelta(seconds=5)

# Length of the request window placed at each window's midpoint. Any positive
# span shorter than a day selects exactly one annual item.
REQUEST_DURATION = timedelta(seconds=1)


@dataclass(frozen=True)
class EmbeddingProduct:
    """A precomputed embedding product readable through an rslearn data source.

    Attributes:
        name: short product name used on the CLI and in the manifest filename.
        modality: the OlmoEarth modality; its name is the layer name and its
            band order names the bands.
        product_version: version identifier recorded in the manifest.
        data_source_class_path: the rslearn DataSource class. It must support
            direct materialization (the layer is written with ingest=False).
        nodata_value: the value the data source uses for pixels without data.
    """

    name: str
    modality: ModalitySpec
    product_version: str
    data_source_class_path: str
    nodata_value: float


AEF = EmbeddingProduct(
    name="aef",
    modality=Modality.GSE,
    # The AWS Open Data bucket serves v1; the data source dequantizes int8 to
    # float32 in [-1, 1] with -1.0 as nodata.
    product_version="v1",
    data_source_class_path=(
        "rslearn.data_sources.aws_google_satellite_embedding_v1."
        "GoogleSatelliteEmbeddingV1"
    ),
    nodata_value=-1.0,
)

PRODUCTS: dict[str, EmbeddingProduct] = {AEF.name: AEF}


def get_target_year(window: Window) -> int | None:
    """Return the product year a window should read: its time-range midpoint year.

    Args:
        window: the rslearn window.

    Returns:
        the year of the midpoint of the window's time range, or None if the
        window has no time range.
    """
    if window.time_range is None:
        return None
    start, end = window.time_range
    return (start + (end - start) / 2).year


def request_time_offset(windows: list[Window]) -> timedelta:
    """Pick the layer time_offset that makes each window read its midpoint year.

    rslearn sorts annual items by start time and keeps the first match, so a
    window spanning Sep 2022 - Sep 2023 would read 2022. Shifting the request
    to (start + offset, start + offset + REQUEST_DURATION) instead reads the
    year the window is centred on. time_offset is a single value per layer, so
    this uses half the median window length and checks that it lands in every
    window's midpoint year (leap years move the true midpoint by half a day).

    Args:
        windows: windows with a time range.

    Returns:
        the time_offset to write into the layer config.

    Raises:
        ValueError: if no single offset gives every window its midpoint year.
    """
    durations = sorted(w.time_range[1] - w.time_range[0] for w in windows)
    # Whole seconds, since config.json stores the offset as "<seconds>s".
    offset = timedelta(seconds=int(durations[len(durations) // 2].total_seconds() / 2))
    mismatched = [
        w for w in windows if (w.time_range[0] + offset).year != get_target_year(w)
    ]
    if mismatched:
        examples = ", ".join(_window_id(w) for w in mismatched[:5])
        raise ValueError(
            f"{len(mismatched)} windows would read a different year than their "
            f"time-range midpoint with time_offset={offset} (e.g. {examples}). "
            "Window lengths vary too much for one per-layer offset; materialize "
            "those window groups separately with --groups."
        )
    return offset


def build_layer_config(
    product: EmbeddingProduct,
    time_offset: timedelta,
    metadata_cache_dir: str = DEFAULT_METADATA_CACHE_DIR,
) -> dict[str, Any]:
    """Build the config.json layer entry for a product.

    Args:
        product: the embedding product.
        time_offset: offset from each window's start to its midpoint.
        metadata_cache_dir: where the data source caches its spatial index.

    Returns:
        the JSON-serializable rslearn layer config.
    """
    return {
        "type": "raster",
        "band_sets": [
            {
                "bands": list(product.modality.band_order),
                "dtype": "float32",
                "nodata_value": product.nodata_value,
            }
        ],
        "data_source": {
            "class_path": product.data_source_class_path,
            "init_args": {"metadata_cache_dir": metadata_cache_dir},
            "query_config": {"space_mode": "MOSAIC", "max_matches": 1},
            "time_offset": f"{int(time_offset.total_seconds())}s",
            "duration": f"{int(REQUEST_DURATION.total_seconds())}s",
            "ingest": False,
        },
    }


def write_layer_config(
    dataset_path: UPath, layer_name: str, layer_config: dict[str, Any]
) -> bool:
    """Set a layer entry in the dataset's config.json, backing it up once.

    Replaces an existing entry of the same name, such as a data-source-less
    layer declared for an earlier bake that wrote the rasters directly.

    Args:
        dataset_path: the rslearn dataset root.
        layer_name: the layer to set.
        layer_config: the layer entry.

    Returns:
        True if config.json changed.
    """
    config_path = dataset_path / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    if config.get("layers", {}).get(layer_name) == layer_config:
        return False

    backup = dataset_path / CONFIG_BACKUP_NAME
    if not backup.exists():
        with config_path.open("rb") as src, backup.open("wb") as dst:
            dst.write(src.read())
    config.setdefault("layers", {})[layer_name] = layer_config
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    logger.info(f"Set '{layer_name}' layer in {config_path}")
    return True


def _window_id(window: Window) -> str:
    """Return the "group/name" identifier of a window."""
    return f"{window.group}/{window.name}"


def _clear_layer(window: Window, layer_name: str) -> None:
    """Delete a window's materialized layer and its prepared items."""
    layer_dir = window.get_layer_dir(layer_name)
    if layer_dir.exists():
        shutil.rmtree(str(layer_dir))
    layer_datas = window.load_layer_datas()
    if layer_datas.pop(layer_name, None) is not None:
        window.save_layer_datas(layer_datas)


def materialize_product(
    dataset_path: UPath | str,
    product: EmbeddingProduct,
    overwrite: bool = False,
    workers: int = 1,
    groups: list[str] | None = None,
    log_every: int = 25,
    metadata_cache_dir: str = DEFAULT_METADATA_CACHE_DIR,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one embedding product into an rslearn eval dataset.

    Declares the product's layer in config.json, then runs rslearn prepare and
    materialize on that layer alone. Windows whose layer is already complete
    are skipped unless overwrite is set. Windows the product does not cover
    are prepared with no items, so rslearn writes nothing for them.

    Args:
        dataset_path: path to the rslearn dataset root.
        product: the embedding product to materialize.
        overwrite: whether to rewrite layers that already exist.
        workers: number of threads materializing windows concurrently.
        groups: optional list of window groups to restrict to.
        log_every: log progress every this many windows.
        metadata_cache_dir: where the data source caches its spatial index.
        cli_args: the CLI arguments used, recorded in the manifest.

    Returns:
        the provenance manifest dictionary (see ``build_manifest``).
    """
    dataset_path = UPath(dataset_path)
    layer_name = product.modality.name
    all_windows = Dataset(dataset_path).storage.get_windows(groups=groups)

    no_year = [w for w in all_windows if w.time_range is None]
    for window in no_year:
        logger.warning(f"Window {_window_id(window)} has no time range; skipping.")
    windows = [w for w in all_windows if w.time_range is not None]
    if not windows:
        raise ValueError(f"No windows with a time range in {dataset_path}")

    time_offset = request_time_offset(windows)
    layer_config = build_layer_config(product, time_offset, metadata_cache_dir)
    write_layer_config(dataset_path, layer_name, layer_config)
    dataset = Dataset(dataset_path, enabled_layers=[layer_name])
    logger.info(
        f"Materializing product '{product.name}' "
        f"(layer {layer_name}, version {product.product_version}, "
        f"time_offset {time_offset}) for {len(windows)} windows in {dataset_path}"
    )

    if overwrite:
        for window in windows:
            _clear_layer(window, layer_name)
    already_done = {_window_id(w) for w in windows if w.is_layer_completed(layer_name)}

    prepare_dataset_windows(
        dataset,
        windows,
        retry_max_attempts=MATERIALIZE_RETRY_ATTEMPTS,
        retry_backoff=MATERIALIZE_RETRY_BACKOFF,
    )

    layer_cfg = dataset.layers[layer_name]
    data_source: DataSource = layer_cfg.instantiate_data_source(dataset.path)
    tile_store = dataset.get_tile_store()

    def run_one(window: Window) -> None:
        """Materialize one window, logging instead of raising on failure.

        A window that still fails after rslearn's retries is left without a
        completed layer, so the manifest records it as failed and a re-run
        picks it back up.
        """
        try:
            materialize_window(
                window,
                dataset,
                data_source,
                tile_store,
                layer_name,
                layer_cfg,
                retry_max_attempts=MATERIALIZE_RETRY_ATTEMPTS,
                retry_backoff=MATERIALIZE_RETRY_BACKOFF,
            )
        except Exception:
            logger.exception(f"Window {_window_id(window)} failed; continuing")

    todo = [w for w in windows if _window_id(w) not in already_done]
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        for num_done, _ in enumerate(executor.map(run_one, todo), start=1):
            if num_done % log_every == 0:
                logger.info(f"Materialized {num_done}/{len(todo)} windows")

    manifest = build_manifest(
        product, windows, no_year, already_done, time_offset, cli_args
    )
    logger.info(
        f"Product '{product.name}': wrote {manifest['num_windows_written']} windows, "
        f"skipped {manifest['num_windows_skipped_existing']} existing, "
        f"{manifest['num_coverage_gaps']} coverage gaps, "
        f"{manifest['num_windows_without_year']} without a time range, "
        f"{manifest['num_windows_failed']} failed."
    )
    if manifest["num_windows_failed"]:
        logger.warning(
            f"Product '{product.name}': {manifest['num_windows_failed']} windows "
            "failed; re-run to retry them (completed layers are skipped)."
        )
    return manifest


def build_manifest(
    product: EmbeddingProduct,
    windows: list[Window],
    no_year: list[Window],
    already_done: set[str],
    time_offset: timedelta,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance manifest from the windows' state after a run.

    Each window is classified from what rslearn left on disk: a completed layer
    was written (or already there), a window prepared with no item groups is a
    coverage gap, and anything else failed.

    Args:
        product: the embedding product.
        windows: the windows that were materialized.
        no_year: windows skipped because they have no time range.
        already_done: ids of windows whose layer was complete before the run.
        time_offset: the layer's time_offset.
        cli_args: the CLI arguments used.

    Returns:
        JSON-serializable manifest dictionary.
    """
    layer_name = product.modality.name
    written, skipped, gaps, failed = [], [], [], []
    for window in windows:
        window_id = _window_id(window)
        if window_id in already_done:
            skipped.append(window_id)
        elif window.is_layer_completed(layer_name):
            written.append(window_id)
        else:
            layer_data = window.load_layer_datas().get(layer_name)
            if layer_data is not None and not layer_data.serialized_item_groups:
                gaps.append(window_id)
            else:
                failed.append(window_id)

    return {
        "product": product.name,
        "product_version": product.product_version,
        "modality": layer_name,
        "year_policy": "window_time_range_midpoint",
        "time_offset_seconds": time_offset.total_seconds(),
        "num_windows_written": len(written),
        "num_windows_skipped_existing": len(skipped),
        "num_coverage_gaps": len(gaps),
        "coverage_gaps": sorted(gaps),
        "num_windows_without_year": len(no_year),
        "windows_without_year": sorted(_window_id(w) for w in no_year),
        "num_windows_failed": len(failed),
        "windows_failed": sorted(failed),
        "cli_args": cli_args or {},
    }


def write_manifest(
    dataset_path: UPath | str, product_name: str, manifest: dict[str, Any]
) -> UPath:
    """Write the provenance manifest JSON next to the dataset.

    Args:
        dataset_path: path to the rslearn dataset root.
        product_name: short product name; used in the manifest filename.
        manifest: the manifest dictionary to serialize.

    Returns:
        the path the manifest was written to.
    """
    manifest_path = (
        UPath(dataset_path) / f"embedding_materializer_manifest_{product_name}.json"
    )
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")
    return manifest_path
