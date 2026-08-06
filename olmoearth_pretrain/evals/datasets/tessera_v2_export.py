r"""Produce the ``tessera_v2`` layer of an eval dataset with our own inference.

No precomputed Tessera v2 product is published (the geotessera bucket only
serves v1/v1.1), but the v2 student weights + inference code are public — see
docs/TesseraV2Inference.md. This module reproduces their pipeline on top of
rslearn instead of their bash + Rust preprocessing, for any rslearn eval
dataset:

1. ``write_fetch_config``: emit a standalone rslearn config carrying only the
   three all-scenes fetch layers from
   ``data/rslearn_dataset_configs/config_tessera_v2_fetch.json``, to be passed
   to prepare/materialize via ``--config``. They deliberately do NOT go into
   the dataset's own ``config.json`` (pastis_rslearn, built first, does carry
   them inline — hence the warning in its runbook that every other
   prepare/materialize on it must pass ``--enabled-layers``).
2. ``create_windows``: mirror every eval window into a fetch-only group whose
   time range is the calendar year Tessera products are defined over. The
   year comes from the eval window itself for the ``*_year_aligned`` datasets
   — they are already anchored to ``(Jan 1 Y, Jan 1 Y+1)`` — or from
   ``--year`` for datasets whose eval window is not a calendar year (pastis,
   whose eval window is Sept–Sept while its Tessera products are year-2019).
3. rslearn fetches every acquisition in that year via the ``*_all`` layers
   (``sentinel2_l2a_all`` incl. the SCL band,
   ``sentinel1_ascending_all``/``sentinel1_descending_all``), one item group
   per scene, chronological. Run prepare/materialize against the fetch config
   and the fetch group only::

    export DS_PATH=/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn
    export EXPORT="python -m olmoearth_pretrain.evals.datasets.tessera_v2_export"
    $EXPORT write_fetch_config --ds_path $DS_PATH
    $EXPORT create_windows --ds_path $DS_PATH --dataset pastis_rslearn
    rslearn dataset prepare --root $DS_PATH --group pastis_tessera_v2 \
        --config $DS_PATH/config_tessera_v2_fetch.json \
        --workers 16 --no-use-initial-job --retry-max-attempts 12 \
        --retry-backoff-seconds 2
    rslearn dataset materialize ... (same flags, --retry-backoff-seconds 60)

   For many concurrent CPU jobs use
   ``scripts/tools/launch_year_aligned_prepare.sh`` with
   ``LAYER_SET=tessera_v2_fetch``, which wraps the rslearn_projects launcher
   (https://github.com/allenai/rslearn_projects/tree/master/rslp/common).
4. ``infer``: per eval window, assemble the per-pixel year of observations in
   exactly the d-pixel form their pipeline produces (S2 uint16 DN in their
   band order + SCL-derived cloud mask; S1 converted with their
   ``(20*log10(raw) + 50) * 200 -> int16`` transform, asc/desc separate;
   per-scene DOYs), run the vendored v2 student
   (olmoearth_pretrain/evals/models/tessera/tessera_v2_{model,infer}.py), and
   write the (128, H, W) embedding as the ``tessera_v2`` raster layer of the
   EVAL window.
5. Wire it up so the ``tessera_v2_precomputed`` baseline can read it::

    python scripts/tools/wire_embedding_modalities.py \
        --datasets <name> --products tessera_v2 --required
    python scripts/tools/backfill_eval_registry_provenance.py

   ``infer`` writes the same manifest shape the embedding materializer does,
   so the wiring script's coverage gate works on tessera_v2 unchanged.

``--eval_ds_path`` lets step 4 read the fetched scenes from one dataset copy
and write the embedding layer into another. That is the normal case for the
``*_year_aligned`` datasets: the scenes are fetched in the staging copy under
``rslearn-eai/datasets/olmoearth_evals`` (which is where the imagery pipeline
runs), while the layer, the manifest and the wiring belong in the ingested
``olmoearth/eval_datasets`` copy that model.yaml points at. Both copies hold
the same windows on the same grids, so the layer is valid in either.

Weights: download from the Hugging Face ``geotessera`` org
(TESSERA-V-2.0-2B-{N,S,M,L} -> ckpt/student_{nano,small,medium,large}.pt) and
pass ``--checkpoint_path``. Use the same student size across datasets or the
numbers are not comparable; the size is recorded in the manifest's
``product_version``.
"""

import argparse
import itertools
import json
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rasterio.enums import Resampling
from rslearn.data_sources.data_source import Item
from rslearn.dataset import Window
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.embedding_materializer.materialize import write_manifest
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)

logger = logging.getLogger(__name__)

LAYER_NAME = Modality.TESSERA_V2.name

# Repo-checkout location of the fetch layer definitions, resolved relative to
# this file so the script works from any working directory.
FETCH_LAYERS_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "data/rslearn_dataset_configs/config_tessera_v2_fetch.json"
)
FETCH_CONFIG_NAME = "config_tessera_v2_fetch.json"

S2_LAYER = "sentinel2_l2a_all"
S1_LAYERS = ("sentinel1_ascending_all", "sentinel1_descending_all")
FETCH_LAYERS = (S2_LAYER, *S1_LAYERS)
# Band-set dirs as configured in config_tessera_v2_fetch.json.
S2_BAND_SETS = (
    ["B02", "B03", "B04", "B08"],
    ["B05", "B06", "B07", "B8A", "B11", "B12"],
)
SCL_BAND_SET = ["SCL"]
S1_BAND_SET = ["vv", "vh"]


class DatasetSpec:
    """Fetch-group settings for one eval dataset."""

    def __init__(
        self,
        fetch_group: str,
        eval_groups: list[str] | None = None,
        year: int | None = None,
    ) -> None:
        """Initialize a DatasetSpec.

        Args:
            fetch_group: window group the year-of-scenes fetch lives in.
            eval_groups: eval window groups, or None for "every group except
                the fetch group".
            year: calendar year to fetch for every window, or None to take
                each window's own (calendar-aligned) time range.
        """
        self.fetch_group = fetch_group
        self.eval_groups = eval_groups
        self.year = year


# Per-dataset defaults.
#
# pastis_rslearn is the original: its eval window is the Sept 2018 - Aug 2019
# agricultural year, but Tessera's published products for it are year-2019, so
# its fetch year is pinned. The *_year_aligned datasets were re-anchored to
# (Jan 1 Y, Jan 1 Y+1) by reanchor_year_aligned_dataset.py, so their own range
# is already the product year and is read per window -- which is what a
# multi-year dataset needs, since no single --year would be right for it.
DATASETS: dict[str, DatasetSpec] = {
    "pastis_rslearn": DatasetSpec(
        fetch_group="pastis_tessera_v2", eval_groups=["pastis"], year=2019
    ),
    "pastis_year_aligned": DatasetSpec(
        fetch_group="pastis_tessera_v2", eval_groups=["pastis"], year=2019
    ),
    "africa_crop_mask_year_aligned": DatasetSpec(
        fetch_group="africa_crop_mask_tessera_v2"
    ),
    "ethiopia_crops_year_aligned": DatasetSpec(fetch_group="ethiopia_crops_tessera_v2"),
}

# Tessera's S2 input channel order (see tessera_v2_model.S2_BAND_ORDER),
# expressed as indices into S2_BAND_SETS concatenated in config order
# (B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12).
_S2_CONCAT_ORDER = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
]
_TESSERA_S2_ORDER = [
    "B04",
    "B02",
    "B03",
    "B08",
    "B8A",
    "B05",
    "B06",
    "B07",
    "B11",
    "B12",
]
TESSERA_S2_INDICES = [_S2_CONCAT_ORDER.index(b) for b in _TESSERA_S2_ORDER]

# SCL classes their Rust stacker treats as invalid: nodata, saturated/defective,
# dark, cloud shadow, cloud medium/high probability. Everything else (incl.
# thin cirrus, class 10) is valid.
INVALID_SCL_CLASSES = (0, 1, 2, 3, 8, 9)


def resolve_spec(
    dataset: str | None, fetch_group: str | None = None, year: int | None = None
) -> DatasetSpec:
    """Pick a dataset's settings, with CLI overrides applied.

    Args:
        dataset: a key of DATASETS, or None to build the spec from the flags.
        fetch_group: overrides the preset's fetch group.
        year: overrides the preset's fetch year.

    Returns:
        the resolved DatasetSpec.

    Raises:
        SystemExit: if neither a known dataset nor a fetch group is given.
    """
    if dataset is not None and dataset not in DATASETS:
        raise SystemExit(
            f"Unknown --dataset {dataset!r}; known: {sorted(DATASETS)}. Pass "
            "--fetch_group (and --year if the eval windows are not calendar "
            "years) to run on a dataset with no preset."
        )
    if dataset is None:
        spec = DatasetSpec(fetch_group="")
    else:
        preset = DATASETS[dataset]
        spec = DatasetSpec(preset.fetch_group, preset.eval_groups, preset.year)
    if fetch_group is not None:
        spec.fetch_group = fetch_group
    if year is not None:
        spec.year = year
    if not spec.fetch_group:
        raise SystemExit("pass --dataset or --fetch_group")
    return spec


def scl_to_valid_mask(scl: np.ndarray) -> np.ndarray:
    """Per-pixel validity from the SCL band (1 = clear, 0 = cloud/nodata)."""
    return (~np.isin(scl, INVALID_SCL_CLASSES)).astype(np.uint8)


def s1_raw_to_tessera_units(raw: np.ndarray) -> np.ndarray:
    """Replicate their downloader's S1 transform, including int16 rounding.

    ``int16 = clip((20 * log10(raw) + 50) * 200, 0, 32767)`` with non-finite
    or non-positive raw values set to 0 (their missing-timestep sentinel).
    ``raw`` is the Planetary Computer sentinel-1-rtc pixel value, which their
    code passes into ``20*log10`` as-is.
    """
    out = np.zeros_like(raw, dtype=np.int16)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(raw) & (raw > 0)
        db = np.zeros_like(raw, dtype=np.float64)
        np.log10(raw, out=db, where=valid)
    db_scaled = (20.0 * db[valid] + 50.0) * 200.0
    out[valid] = np.clip(db_scaled, 0, 32767).astype(np.int16)
    return out.astype(np.float32)


def _acquisition_times(window: Window, layer_name: str) -> list[datetime]:
    """Per-item-group acquisition start times from the window's items.json."""
    layer_datas = window.load_layer_datas()
    if layer_name not in layer_datas:
        raise ValueError(
            f"window {window.group}/{window.name} has no prepared items for "
            f"layer {layer_name}; run rslearn dataset prepare first"
        )
    times = []
    for group in layer_datas[layer_name].serialized_item_groups:
        # space_mode INTERSECTS puts exactly one item per group.
        item = Item.deserialize(group[0])
        if item.geometry.time_range is None:
            raise ValueError(f"item {item.name} has no time range")
        times.append(item.geometry.time_range[0])
    return times


def _read_scenes(
    window: Window,
    layer_name: str,
    band_sets: tuple[list[str], ...],
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, np.ndarray]:
    """Read every completed scene of a layer onto the window grid.

    A layer that was prepared but matched zero items (e.g. no descending S1
    passes over the window) yields empty (0, H, W, C) / (0,) arrays — the
    Tessera inference handles missing sources. A layer with prepared items
    but no materialized scenes raises (materialization incomplete).

    Returns:
        (T, H, W, C) float32 array (band sets concatenated in the given
        order, coarser band sets resampled to the window resolution) and the
        (T,) int day-of-year array, chronologically sorted.
    """
    completed = sorted(
        group_idx
        for name, group_idx in window.list_completed_layers()
        if name == layer_name
    )
    times = _acquisition_times(window, layer_name)
    if not times:
        height = window.bounds[3] - window.bounds[1]
        width = window.bounds[2] - window.bounds[0]
        num_bands = sum(len(bands) for bands in band_sets)
        return (
            np.zeros((0, height, width, num_bands), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    raster_format = GeotiffRasterFormat()
    scenes = []
    for group_idx in completed:
        arrays = [
            raster_format.decode_raster(
                window.get_raster_dir(layer_name, bands, group_idx),
                window.projection,
                window.bounds,
                resampling=resampling,
            ).get_chw_array()
            for bands in band_sets
        ]
        scenes.append(np.concatenate(arrays, axis=0).transpose(1, 2, 0))
    if not scenes:
        raise ValueError(
            f"window {window.group}/{window.name} has no materialized "
            f"{layer_name} scenes"
        )
    doys = np.array(
        [times[group_idx].timetuple().tm_yday for group_idx in completed],
        dtype=np.int64,
    )
    order = np.argsort([times[group_idx] for group_idx in completed])
    stacked = np.stack(scenes, axis=0).astype(np.float32)[order]
    return stacked, doys[order]


def build_dpixel_inputs(fetch_window: Window) -> dict[str, np.ndarray]:
    """Assemble the tessera_v2_infer.encode_tile inputs for one window."""
    s2, s2_doys = _read_scenes(fetch_window, S2_LAYER, S2_BAND_SETS)
    if s2.shape[0] == 0:
        # Missing S1 passes can be real (orbit geometry); zero S2 scenes over
        # an eval window can only be a fetch-pipeline failure.
        raise ValueError(
            f"window {fetch_window.group}/{fetch_window.name} has zero "
            f"{S2_LAYER} scenes"
        )
    # SCL is categorical: nearest resampling, no averaging across classes.
    scl, scl_doys = _read_scenes(
        fetch_window, S2_LAYER, (SCL_BAND_SET,), resampling=Resampling.nearest
    )
    if not np.array_equal(s2_doys, scl_doys):
        raise ValueError("S2 band and SCL scene ordering diverged")
    s2_masks = scl_to_valid_mask(scl[..., 0])

    s1 = {}
    for layer in S1_LAYERS:
        raw, doys = _read_scenes(fetch_window, layer, (S1_BAND_SET,))
        s1[layer] = (s1_raw_to_tessera_units(raw), doys)

    return {
        "s2_bands": s2[..., TESSERA_S2_INDICES],
        "s2_doys": s2_doys,
        "s2_masks": s2_masks,
        "s1_asc_bands": s1[S1_LAYERS[0]][0],
        "s1_asc_doys": s1[S1_LAYERS[0]][1],
        "s1_desc_bands": s1[S1_LAYERS[1]][0],
        "s1_desc_doys": s1[S1_LAYERS[1]][1],
    }


def write_fetch_config(ds_path: str, out_path: str | None = None) -> UPath:
    """Write the standalone dataset config the fetch runs against.

    The fetch layers stay OUT of the dataset's own ``config.json`` and are
    passed to prepare/materialize via ``rslearn ... --config`` instead. That
    is what keeps them harmless: a run that forgets ``--enabled-layers``
    against a config carrying ``*_all`` would fetch a year of scenes for every
    eval window, and ingest overwriting config.json would silently drop them
    again. Nothing reads these layers through the dataset config —
    ``build_dpixel_inputs`` goes at the rasters via ``window.get_raster_dir``.

    ``storage`` and ``tile_store`` are copied from the dataset's real config,
    because rslearn instantiates the window storage from whatever config it is
    handed; only ``layers`` is replaced.

    Args:
        ds_path: rslearn dataset root the fetch will run against.
        out_path: where to write it; defaults to
            ``<ds_path>/config_tessera_v2_fetch.json`` (inert there — it is
            not named config.json).

    Returns:
        the path written.

    Raises:
        FileNotFoundError: if the dataset has no config.json.
    """
    config_path = UPath(ds_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json at {ds_path}")
    dataset_config = json.loads(config_path.read_text())

    fetch_config = {
        key: value
        for key, value in dataset_config.items()
        if key not in ("layers", "_comment")
    }
    fetch_config["layers"] = json.loads(FETCH_LAYERS_CONFIG.read_text())["layers"]

    target = UPath(out_path) if out_path else UPath(ds_path) / FETCH_CONFIG_NAME
    target.write_text(json.dumps(fetch_config, indent=2) + "\n")
    logger.info(
        f"Wrote {target} with layers {sorted(fetch_config['layers'])}; pass it as "
        f"`rslearn dataset prepare --root {ds_path} --config {target}`"
    )
    return target


def fetch_time_range(
    window: Window, year: int | None
) -> tuple[datetime, datetime] | str:
    """Calendar-year range to fetch for one eval window.

    Args:
        window: the eval window.
        year: a pinned year, or None to read the window's own range.

    Returns:
        the (Jan 1 Y, Jan 1 Y+1) range, or a string explaining why the
        window's own range cannot be read as a product year.
    """
    if year is None:
        if window.time_range is None:
            return "window has no time range"
        start, end = window.time_range
        if (start.month, start.day, end.month, end.day) != (1, 1, 1, 1):
            return f"window range {start.date()}..{end.date()} is not a calendar year"
        if end.year != start.year + 1:
            return f"window range {start.date()}..{end.date()} spans != 1 year"
        year = start.year
    return (
        datetime(year, 1, 1, tzinfo=UTC),
        datetime(year + 1, 1, 1, tzinfo=UTC),
    )


def eval_windows_of(provider: RslearnWindowProvider, spec: DatasetSpec) -> list[Window]:
    """Load the dataset's eval windows, excluding the fetch group."""
    return [w for w in provider.load_windows() if w.group != spec.fetch_group]


def check_names_unique(windows: list[Window]) -> None:
    """Fail if window names repeat across groups.

    The fetch group is flat, so ``infer`` matches fetch windows to eval
    windows by name alone.

    Raises:
        SystemExit: on any collision.
    """
    seen: dict[str, str] = {}
    collisions = [
        f"{seen[w.name]}/{w.name} vs {w.group}/{w.name}"
        for w in windows
        if seen.setdefault(w.name, w.group) != w.group
    ]
    if collisions:
        raise SystemExit(
            f"{len(collisions)} eval window names occur in more than one group, "
            f"e.g. {collisions[:3]}. Pass --eval_groups to restrict to one."
        )


def create_windows(ds_path: str, spec: DatasetSpec) -> None:
    """Mirror the eval windows into the calendar-year fetch group.

    Raises:
        SystemExit: if window names collide across groups, or if any window's
            fetch year cannot be determined.
    """
    provider = RslearnWindowProvider(UPath(ds_path), groups=spec.eval_groups)
    windows = eval_windows_of(provider, spec)
    check_names_unique(windows)

    ranges = {w.name: fetch_time_range(w, spec.year) for w in windows}
    bad = {name: r for name, r in ranges.items() if isinstance(r, str)}
    if bad:
        raise SystemExit(
            f"{len(bad)} of {len(windows)} windows have no usable product year, "
            f"e.g. {list(bad.items())[:3]}. Re-anchor the dataset first "
            "(scripts/tools/reanchor_year_aligned_dataset.py) or pass --year."
        )

    years: dict[int, int] = {}
    for eval_window in windows:
        time_range = ranges[eval_window.name]
        assert not isinstance(time_range, str)
        Window(
            storage=eval_window.storage,
            group=spec.fetch_group,
            name=eval_window.name,
            projection=eval_window.projection,
            bounds=eval_window.bounds,
            time_range=time_range,
        ).save()
        years[time_range[0].year] = years.get(time_range[0].year, 0) + 1
    logger.info(
        f"Created {len(windows)} {spec.fetch_group} windows; "
        f"years: {dict(sorted(years.items()))}"
    )


def infer(
    ds_path: str,
    checkpoint_path: str,
    spec: DatasetSpec,
    model_size: str,
    eval_ds_path: str | None = None,
    batch_pixels: int = 4096,
    device: str | None = None,
    overwrite: bool = False,
    read_workers: int = 8,
) -> None:
    """Run v2 student inference per window and write the tessera_v2 layer.

    Args:
        ds_path: dataset holding the materialized fetch group.
        checkpoint_path: student weights.
        spec: the dataset's fetch-group settings.
        model_size: student size, recorded in the manifest.
        eval_ds_path: dataset to write the layer and manifest into; defaults
            to ds_path. Use it to fetch in the staging copy and write into the
            ingested eval copy (they hold the same windows on the same grids).
        batch_pixels: pixels per forward pass.
        device: torch device string; default cuda when available.
        overwrite: re-run windows that already carry the layer.
        read_workers: threads assembling d-pixel inputs.

    Raises:
        SystemExit: if an eval window has no matching fetch window.
    """
    from olmoearth_pretrain.evals.models.tessera.tessera_v2_infer import encode_tile
    from olmoearth_pretrain.evals.models.tessera.tessera_v2_model import load_model

    torch_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(checkpoint_path, device=torch_device)

    out_ds_path = eval_ds_path or ds_path
    provider = RslearnWindowProvider(UPath(out_ds_path), groups=spec.eval_groups)
    eval_windows = eval_windows_of(provider, spec)
    check_names_unique(eval_windows)
    fetch_provider = RslearnWindowProvider(UPath(ds_path), groups=[spec.fetch_group])
    fetch_windows = {w.name: w for w in fetch_provider.load_windows()}
    missing = [w.name for w in eval_windows if w.name not in fetch_windows]
    if missing:
        raise SystemExit(
            f"{len(missing)} eval windows have no {spec.fetch_group} window in "
            f"{ds_path} (e.g. {missing[:3]}); run create_windows first"
        )

    def read_one(
        eval_window: Window,
    ) -> tuple[Window, dict[str, Any] | None, bool]:
        """Load one window's inputs -> (window, inputs or None, failed)."""
        if not overwrite and provider.is_layer_written(eval_window, LAYER_NAME):
            return eval_window, None, False
        try:
            inputs = build_dpixel_inputs(fetch_windows[eval_window.name])
        except Exception:
            logger.exception(f"window {eval_window.name}: reading inputs failed")
            return eval_window, None, True
        return eval_window, inputs, False

    def bounded_map(pool: ThreadPoolExecutor, windows: list[Window]) -> Any:
        """pool.map with bounded read-ahead.

        pool.map submits every window upfront and buffers all completed
        results until consumed — each holds ~200MB of scene arrays, so if
        reads outpace inference the host runs out of memory. Keep at most
        2 x read_workers windows in flight instead.
        """
        pending: deque = deque()
        window_iter = iter(windows)
        for window in itertools.islice(window_iter, 2 * read_workers):
            pending.append(pool.submit(read_one, window))
        while pending:
            yield pending.popleft().result()
            for window in itertools.islice(window_iter, 1):
                pending.append(pool.submit(read_one, window))

    written = 0
    skipped = 0
    failed: list[str] = []
    # Overlap raster reads (I/O-bound) with GPU inference.
    with ThreadPoolExecutor(max_workers=read_workers) as pool:
        for eval_window, inputs, read_failed in bounded_map(pool, eval_windows):
            if read_failed:
                failed.append(eval_window.name)
                continue
            if inputs is None:
                skipped += 1
                continue
            try:
                embedding = encode_tile(
                    model,
                    batch_pixels=batch_pixels,
                    device=torch_device,
                    **inputs,
                )
            except Exception:
                logger.exception(f"window {eval_window.name}: inference failed")
                failed.append(eval_window.name)
                continue
            provider.write_embedding(
                eval_window,
                Modality.TESSERA_V2,
                embedding.transpose(2, 0, 1),
                nodata_value=float("nan"),
            )
            written += 1
            if written % 50 == 0:
                logger.info(f"{written}/{len(eval_windows)} windows written")

    # Same key names the embedding materializer writes, so the coverage gate in
    # wire_embedding_modalities.py reads this manifest unchanged. There is no
    # coverage-gap category here: we run the model ourselves, so a window
    # either succeeds or lands in windows_failed.
    write_manifest(
        UPath(out_ds_path),
        LAYER_NAME,
        {
            "product": LAYER_NAME,
            "product_version": f"v2-{model_size}",
            "checkpoint_path": checkpoint_path,
            "year_policy": (
                f"fixed:{spec.year}"
                if spec.year is not None
                else "window_time_range_calendar_year"
            ),
            "num_windows_written": written,
            "num_windows_skipped_existing": skipped,
            "num_coverage_gaps": 0,
            "coverage_gaps": [],
            "num_windows_without_year": 0,
            "windows_without_year": [],
            "num_windows_failed": len(failed),
            "windows_failed": sorted(failed),
            "cli_args": {
                "dataset_path": out_ds_path,
                "fetch_dataset_path": ds_path,
                "fetch_group": spec.fetch_group,
                "model_size": model_size,
            },
        },
    )
    logger.info(f"Done: {written} written, {skipped} skipped, {len(failed)} failed")
    if failed:
        logger.warning(
            f"{len(failed)} windows failed; re-run to retry them "
            f"(existing layers are skipped): {failed[:10]}..."
        )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the dataset-selection flags shared by create_windows and infer."""
    parser.add_argument("--ds_path", required=True)
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASETS),
        help="Preset fetch group / year for a known eval dataset.",
    )
    parser.add_argument("--fetch_group", default=None, help="Override the fetch group.")
    parser.add_argument(
        "--eval_groups",
        default=None,
        help="Comma-separated eval window groups (default: all but the fetch group).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Pin the fetch year instead of reading each window's own "
        "calendar-year range.",
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser(
        "write_fetch_config",
        help="Write the standalone --config the fetch runs against",
    )
    p_config.add_argument("--ds_path", required=True)
    p_config.add_argument(
        "--out_path",
        default=None,
        help=f"Default: <ds_path>/{FETCH_CONFIG_NAME}",
    )

    p_windows = sub.add_parser("create_windows", help="Create the fetch windows")
    add_common_args(p_windows)

    p_infer = sub.add_parser("infer", help="Run inference and write the layer")
    add_common_args(p_infer)
    p_infer.add_argument("--checkpoint_path", required=True)
    p_infer.add_argument(
        "--eval_ds_path",
        default=None,
        help="Dataset to write the layer and manifest into (default: --ds_path).",
    )
    p_infer.add_argument(
        "--model_size",
        default="medium",
        choices=["nano", "small", "medium", "large"],
        help="Student size, recorded in the provenance manifest (the "
        "architecture itself is read from the checkpoint).",
    )
    p_infer.add_argument("--batch_pixels", type=int, default=4096)
    p_infer.add_argument("--device", default=None)
    p_infer.add_argument("--overwrite", action="store_true")
    p_infer.add_argument("--read_workers", type=int, default=8)

    args = parser.parse_args()
    if args.command == "write_fetch_config":
        write_fetch_config(args.ds_path, out_path=args.out_path)
        return

    spec = resolve_spec(args.dataset, args.fetch_group, args.year)
    if args.eval_groups:
        spec.eval_groups = args.eval_groups.split(",")
    if args.command == "create_windows":
        create_windows(args.ds_path, spec)
    else:
        infer(
            args.ds_path,
            checkpoint_path=args.checkpoint_path,
            spec=spec,
            model_size=args.model_size,
            eval_ds_path=args.eval_ds_path,
            batch_pixels=args.batch_pixels,
            device=args.device,
            overwrite=args.overwrite,
            read_workers=args.read_workers,
        )


if __name__ == "__main__":
    main()
