"""Orchestration for materializing embedding products into eval datasets."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from rslearn.dataset import Window
from upath import UPath

from olmoearth_pretrain.evals.embedding_materializer.fetchers import EmbeddingFetcher
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
    get_target_year,
)

logger = logging.getLogger(__name__)

# Per-window outcomes recorded while materializing.
STATUS_WRITTEN = "written"
STATUS_SKIPPED_EXISTING = "skipped_existing"
STATUS_COVERAGE_GAP = "coverage_gap"
STATUS_NO_YEAR = "no_year"


@dataclass
class MaterializeStats:
    """Tallies of per-window outcomes for one (dataset, product) run.

    Attributes:
        written: names (group/name) of windows whose layer was written.
        skipped_existing: windows skipped because the layer already existed.
        coverage_gaps: windows where the product had no data.
        no_year: windows with no time range when no --year override is given.
    """

    written: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    no_year: list[str] = field(default_factory=list)

    def record(self, status: str, window_id: str) -> None:
        """Record one window outcome.

        Args:
            status: one of the STATUS_* constants.
            window_id: the "group/name" identifier of the window.
        """
        status_to_list = {
            STATUS_WRITTEN: self.written,
            STATUS_SKIPPED_EXISTING: self.skipped_existing,
            STATUS_COVERAGE_GAP: self.coverage_gaps,
            STATUS_NO_YEAR: self.no_year,
        }
        status_to_list[status].append(window_id)


def _window_id(window: Window) -> str:
    """Return the "group/name" identifier of a window."""
    return f"{window.group}/{window.name}"


def _process_window(
    window: Window,
    fetcher: EmbeddingFetcher,
    provider: RslearnWindowProvider,
    year: int | None,
    overwrite: bool,
) -> str:
    """Fetch and write the embedding layer for one window.

    Args:
        window: the rslearn window to materialize.
        fetcher: the embedding product fetcher.
        provider: the window provider used for writing.
        year: optional fixed-year override.
        overwrite: whether to rewrite layers that already exist.

    Returns:
        one of the STATUS_* constants.
    """
    layer_name = fetcher.modality.name
    if not overwrite and provider.is_layer_written(window, layer_name):
        return STATUS_SKIPPED_EXISTING

    target_year = get_target_year(window, year)
    if target_year is None:
        logger.warning(
            f"Window {_window_id(window)} has no time range and no --year override; "
            "skipping."
        )
        return STATUS_NO_YEAR

    array = fetcher.fetch(window.bounds, window.projection, target_year)
    if array is None:
        return STATUS_COVERAGE_GAP

    provider.write_embedding(
        window,
        fetcher.modality,
        array,
        nodata_value=fetcher.nodata_value,
        layer_name=layer_name,
    )
    return STATUS_WRITTEN


def materialize_product(
    dataset_path: UPath | str,
    fetcher: EmbeddingFetcher,
    product_name: str,
    year: int | None = None,
    overwrite: bool = False,
    workers: int = 1,
    groups: list[str] | None = None,
    log_every: int = 25,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one embedding product into an rslearn eval dataset.

    For each window: fetch the product raster for the window's target year and
    write it as a raster layer named after the product's modality. Windows
    whose layer already exists are skipped unless overwrite is set; windows
    without product coverage are recorded as gaps.

    Args:
        dataset_path: path to the rslearn dataset root.
        fetcher: the embedding product fetcher.
        product_name: short product name (e.g. "aef", "tessera"), used in
            logs and the manifest.
        year: optional fixed year; if None, each window's time-range midpoint
            year is used.
        overwrite: whether to rewrite layers that already exist.
        workers: number of threads fetching/writing windows concurrently.
        groups: optional list of window groups to restrict to.
        log_every: log progress every this many windows.
        cli_args: the CLI arguments used, recorded in the manifest.

    Returns:
        the provenance manifest dictionary (see ``build_manifest``).
    """
    dataset_path = UPath(dataset_path)
    provider = RslearnWindowProvider(dataset_path, groups=groups)
    windows = provider.load_windows()
    logger.info(
        f"Materializing product '{product_name}' "
        f"(modality {fetcher.modality.name}, version {fetcher.product_version}) "
        f"for {len(windows)} windows in {dataset_path}"
    )

    stats = MaterializeStats()
    num_done = 0

    def run_one(window: Window) -> tuple[str, str]:
        """Process one window and return (status, window_id)."""
        return (
            _process_window(window, fetcher, provider, year, overwrite),
            _window_id(window),
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = executor.map(run_one, windows)
            for status, window_id in outcomes:
                stats.record(status, window_id)
                num_done += 1
                if num_done % log_every == 0:
                    logger.info(f"Processed {num_done}/{len(windows)} windows")
    else:
        for window in windows:
            status, window_id = run_one(window)
            stats.record(status, window_id)
            num_done += 1
            if num_done % log_every == 0:
                logger.info(f"Processed {num_done}/{len(windows)} windows")

    logger.info(
        f"Product '{product_name}': wrote {len(stats.written)} windows, "
        f"skipped {len(stats.skipped_existing)} existing, "
        f"{len(stats.coverage_gaps)} coverage gaps, "
        f"{len(stats.no_year)} without a target year."
    )
    return build_manifest(fetcher, product_name, year, stats, cli_args)


def build_manifest(
    fetcher: EmbeddingFetcher,
    product_name: str,
    year: int | None,
    stats: MaterializeStats,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance manifest for one materialization run.

    Args:
        fetcher: the embedding product fetcher used.
        product_name: short product name (e.g. "aef", "tessera").
        year: the fixed-year override, or None for per-window years.
        stats: the per-window outcome tallies.
        cli_args: the CLI arguments used.

    Returns:
        JSON-serializable manifest dictionary.
    """
    return {
        "product": product_name,
        "product_version": fetcher.product_version,
        "modality": fetcher.modality.name,
        "year_policy": (
            f"fixed:{year}" if year is not None else "window_time_range_midpoint"
        ),
        "num_windows_written": len(stats.written),
        "num_windows_skipped_existing": len(stats.skipped_existing),
        "num_coverage_gaps": len(stats.coverage_gaps),
        "coverage_gaps": sorted(stats.coverage_gaps),
        "num_windows_without_year": len(stats.no_year),
        "windows_without_year": sorted(stats.no_year),
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
