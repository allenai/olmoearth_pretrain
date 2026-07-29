r"""Produce the ``tessera_v2`` layer of pastis_rslearn with our own v2 inference.

No precomputed Tessera v2 product is published (the geotessera bucket only
serves v1/v1.1), but the v2 student weights + inference code are public — see
docs/TesseraV2Inference.md. This script reproduces their pipeline on top of
rslearn instead of their bash + Rust preprocessing:

1. ``create_windows``: mirror every eval window (group ``pastis``) into a
   fetch-only group ``pastis_tessera_v2`` whose time range is the calendar
   year Tessera products are defined over (their published v1/v1.1 layers for
   PASTIS are year-2019 products, so the default is 2019 rather than the eval
   windows' Sept-Sept range).
2. rslearn fetches every acquisition in the year via the ``*_all`` layers of
   ``config_pastis_rslearn.json`` (``sentinel2_l2a_all`` incl. the SCL band,
   ``sentinel1_ascending_all``/``sentinel1_descending_all``), one item group
   per scene, chronological. Run prepare/materialize restricted to those
   layers::

    export DS_PATH=/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn
    python -m olmoearth_pretrain.evals.datasets.pastis_tessera_v2 \
        create_windows --ds_path $DS_PATH
    rslearn dataset prepare --root $DS_PATH --group pastis_tessera_v2 \
        --workers 64 --no-use-initial-job --retry-max-attempts 8 \
        --retry-backoff-seconds 60 \
        --enabled-layers sentinel2_l2a_all,sentinel1_ascending_all,sentinel1_descending_all
    rslearn dataset materialize ... (same flags)

   For many concurrent CPU jobs use the rslearn_projects launcher
   (https://github.com/allenai/rslearn_projects/tree/master/rslp/common):
   ``python -m rslp.main common launch_data_materialization_jobs ...`` with
   the materialize command above.
3. ``infer``: per eval window, assemble the per-pixel year of observations in
   exactly the d-pixel form their pipeline produces (S2 uint16 DN in their
   band order + SCL-derived cloud mask; S1 converted with their
   ``(20*log10(raw) + 50) * 200 -> int16`` transform, asc/desc separate;
   per-scene DOYs), run the vendored v2 student
   (olmoearth_pretrain/evals/models/tessera/tessera_v2_{model,infer}.py), and
   write the (128, H, W) embedding as the ``tessera_v2`` raster layer of the
   EVAL window — after which the ``tessera_v2_precomputed`` baseline evals it
   like any other precomputed product.

Weights: download from the Hugging Face ``geotessera`` org
(TESSERA-V-2.0-2B-{N,S,M,L} -> ckpt/student_{nano,small,medium,large}.pt) and
pass ``--checkpoint_path``.
"""

import argparse
import itertools
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
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

EVAL_GROUP = "pastis"
FETCH_GROUP = "pastis_tessera_v2"
LAYER_NAME = Modality.TESSERA_V2.name
PRODUCT_YEAR = 2019

S2_LAYER = "sentinel2_l2a_all"
S1_LAYERS = ("sentinel1_ascending_all", "sentinel1_descending_all")
# Band-set dirs as configured in config_pastis_rslearn.json.
S2_BAND_SETS = (
    ["B02", "B03", "B04", "B08"],
    ["B05", "B06", "B07", "B8A", "B11", "B12"],
)
SCL_BAND_SET = ["SCL"]
S1_BAND_SET = ["vv", "vh"]

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
        # a PASTIS window can only be a fetch-pipeline failure.
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


def create_windows(ds_path: str, year: int = PRODUCT_YEAR) -> None:
    """Mirror the eval windows into the calendar-year fetch group."""
    provider = RslearnWindowProvider(UPath(ds_path), groups=[EVAL_GROUP])
    windows = provider.load_windows()
    time_range = (
        datetime(year, 1, 1, tzinfo=UTC),
        datetime(year + 1, 1, 1, tzinfo=UTC),
    )
    for eval_window in windows:
        Window(
            storage=eval_window.storage,
            group=FETCH_GROUP,
            name=eval_window.name,
            projection=eval_window.projection,
            bounds=eval_window.bounds,
            time_range=time_range,
        ).save()
    logger.info(f"Created {len(windows)} {FETCH_GROUP} windows for year {year}")


def infer(
    ds_path: str,
    checkpoint_path: str,
    model_size: str,
    batch_pixels: int = 4096,
    device: str | None = None,
    overwrite: bool = False,
    read_workers: int = 4,
) -> None:
    """Run v2 student inference per window and write the tessera_v2 layer."""
    from olmoearth_pretrain.evals.models.tessera.tessera_v2_infer import encode_tile
    from olmoearth_pretrain.evals.models.tessera.tessera_v2_model import load_model

    torch_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(checkpoint_path, device=torch_device)

    provider = RslearnWindowProvider(UPath(ds_path), groups=[EVAL_GROUP])
    eval_windows = provider.load_windows()
    fetch_provider = RslearnWindowProvider(UPath(ds_path), groups=[FETCH_GROUP])
    fetch_windows = {w.name: w for w in fetch_provider.load_windows()}

    def read_one(
        eval_window: Window,
    ) -> tuple[Window, dict[str, Any] | None, bool]:
        """Load one window's inputs -> (window, inputs or None, failed)."""
        if not overwrite and provider.is_layer_written(eval_window, LAYER_NAME):
            return eval_window, None, False
        fetch_window = fetch_windows.get(eval_window.name)
        if fetch_window is None:
            raise ValueError(
                f"No {FETCH_GROUP} window named {eval_window.name}; "
                "run create_windows first"
            )
        try:
            return eval_window, build_dpixel_inputs(fetch_window), False
        except Exception:
            logger.exception(f"window {eval_window.name}: reading inputs failed")
            return eval_window, None, True

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

    write_manifest(
        UPath(ds_path),
        LAYER_NAME,
        {
            "product": LAYER_NAME,
            "product_version": f"v2-{model_size}",
            "checkpoint_path": checkpoint_path,
            "year": PRODUCT_YEAR,
            "windows_written": written,
            "windows_skipped_existing": skipped,
            "windows_failed": failed,
        },
    )
    logger.info(f"Done: {written} written, {skipped} skipped, {len(failed)} failed")
    if failed:
        logger.warning(
            f"{len(failed)} windows failed; re-run to retry them "
            f"(existing layers are skipped): {failed[:10]}..."
        )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_windows = sub.add_parser("create_windows", help="Create the fetch windows")
    p_windows.add_argument("--ds_path", required=True)
    p_windows.add_argument("--year", type=int, default=PRODUCT_YEAR)

    p_infer = sub.add_parser("infer", help="Run inference and write the layer")
    p_infer.add_argument("--ds_path", required=True)
    p_infer.add_argument("--checkpoint_path", required=True)
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
    p_infer.add_argument("--read_workers", type=int, default=4)

    args = parser.parse_args()
    if args.command == "create_windows":
        create_windows(args.ds_path, year=args.year)
    else:
        infer(
            args.ds_path,
            checkpoint_path=args.checkpoint_path,
            model_size=args.model_size,
            batch_pixels=args.batch_pixels,
            device=args.device,
            overwrite=args.overwrite,
            read_workers=args.read_workers,
        )


if __name__ == "__main__":
    main()
