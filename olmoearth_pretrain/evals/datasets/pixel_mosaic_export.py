"""Per-pixel cloud-aware monthly composites for the AEF-supplemental evals.

Our year-aligned eval reads twelve 30-day mosaics, but the archive holds a median
~72 S2 acquisitions per Ethiopia window per year (docs/TesseraV2Inference.md), and
the year-aligned re-export dropped the ``eo:cloud_cover < 50`` scene filter
entirely (scripts/tools/setup_extra_layers.py). So each stored month is whatever
rslearn's mosaic picked, cloud included, and the eval-time SCL mask can only
*blank* a cloudy pixel -- never *substitute* a clear one from another date,
because that date was never fetched into the eval dataset. Tessera, reading the
same archive, decides per pixel per acquisition.

This script closes that gap for one dataset at a time. For every 30-day period it
picks, **per pixel**, the acquisition with the least-contaminated SCL class, and
writes the result as that period's imagery layer. Every output pixel is still one
real observation -- this is a selection, not an average.

It reads the year-of-scenes fetch group that ``tessera_v2_export`` already
creates (so no new fetch is needed where that group survives) and writes into a
*separate* dataset, leaving the baseline untouchable.

Three paths are involved, and they are not interchangeable -- the fetch group
lives on the STAGING copy while the materialized gse/tessera_v2 layers, labels
and split tags live on the INGESTED copy (see setup_tessera_v2.DEFAULT_STAGE_ROOT
vs DEFAULT_EVAL_ROOT, and ``infer --ds_path <stage> --eval_ds_path <eval>``)::

    STAGE=/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/ethiopia_crops_year_aligned
    SRC=/weka/dfive-default/olmoearth/eval_datasets/ethiopia_crops_year_aligned
    DST=/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/ethiopia_crops_ccmos_year_aligned

    # 0. go/no-go: is there anything to choose between? Reads the fetch group.
    python -m olmoearth_pretrain.evals.datasets.pixel_mosaic_export probe \
        --ds_path $STAGE --dataset ethiopia_crops_year_aligned --sample 200

    # 1. clone from the INGESTED copy, dropping the imagery this script replaces.
    # Seeding from $SRC (not $STAGE) is what carries gse/tessera_v2 across, so
    # AEF and Tessera need no re-fetch -- the reanchor runbook's trick.
    rsync -a --exclude='sentinel2_l2a_mo*' --exclude='sentinel2_scl_mo*' \
        --exclude='*_tessera_v2' "$SRC/" "$DST/"

    # 2. composite: read the fetch group in $STAGE, write into $DST
    python -m olmoearth_pretrain.evals.datasets.pixel_mosaic_export composite \
        --ds_path $STAGE --dataset ethiopia_crops_year_aligned \
        --out_ds_path $DST --workers 8

STORED-RESOLUTION DEVIATION, read before comparing numbers: the parent stores S2
as three band sets at 10 m / 20 m / 40 m and lets the eval loader upsample to the
window grid. ``_read_scenes`` hands us every band already on the window grid, so
this script writes **one 12-band set at window resolution** instead. The model
therefore sees the same bilinear upsample, just applied at export rather than at
load; the deviation is the resampling *timing* and the uint16 rounding of
interpolated values (~0.03% of a typical DN), not the band values. Both arms
still go through the same loader and the same pretraining normalization.
"""

import argparse
import itertools
import json
import logging
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from rasterio.enums import Resampling
from rslearn.dataset import Window
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.evals.datasets.tessera_v2_export import (
    S2_BAND_SETS,
    S2_LAYER,
    SCL_BAND_SET,
    DatasetSpec,
    NoS2ScenesError,
    _read_scenes,
    check_names_unique,
    eval_windows_of,
    resolve_spec,
)
from olmoearth_pretrain.evals.embedding_materializer.materialize import write_manifest
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)

logger = logging.getLogger(__name__)

MONTHS = 12
PERIOD_DAYS = 30

# The 60 m band set, added to config_tessera_v2_fetch.json so the composite can
# be a full 12-band S2 input rather than the 10 the Tessera export needed.
B0109_BAND_SET = ["B01", "B09"]


# Band order of the written composite: S2_BAND_SETS concatenated, then (when
# requested) B01/B09. _read_scenes concatenates in the order passed, and the
# 12-band order is exactly the `bands:` list in the year-aligned model.yaml -- so
# nothing needs permuting (contrast build_dpixel_inputs, which reorders via
# TESSERA_S2_INDICES).
#
# include_60m is OPT-IN because rslearn marks materialization complete per LAYER,
# not per band set (dataset/manage.py: `if window.is_layer_completed(layer_name):
# return skipped=True`). So a fetch group materialized before B01/B09 was added to
# config_tessera_v2_fetch.json will NEVER backfill it -- the band set only lands on
# groups fetched from scratch afterwards. On an existing group, run at 10 bands and
# score the `mo` baseline at 10 bands too, so the band subset is a shared confound
# and compositing is the only difference.
def read_band_sets(include_60m: bool) -> tuple[list[str], ...]:
    """Band sets to read, in the order the composite concatenates them."""
    return (*S2_BAND_SETS, B0109_BAND_SET) if include_60m else S2_BAND_SETS


def composite_bands(include_60m: bool) -> list[str]:
    """Flat band list of the written composite, matching read_band_sets order."""
    return [band for band_set in read_band_sets(include_60m) for band in band_set]


S2_LAYERS = [f"sentinel2_l2a_mo{i:02d}" for i in range(1, MONTHS + 1)]
SCL_LAYERS = [f"sentinel2_scl_mo{i:02d}" for i in range(1, MONTHS + 1)]

# Per-class contamination ranking, best (0) to worst. Every other SCL consumer in
# the repo is a binary set test, and the three that exist disagree:
# INVALID_SCL_CLASSES = (0,1,2,3,8,9) masks dark and keeps cirrus, the loader's
# SCL_CLOUD_CLASSES = (0,1,3,8,9,10) keeps dark and masks cirrus, and
# SCL_CLOUDLESS_CLASSES = (8,9) masks neither. A *selection* needs a total order,
# so this is new. Rank 0 is INVALID_SCL_CLASSES' complement minus cirrus, i.e.
# "clear" means what it means to Tessera; cirrus sorts just behind it because
# their stacker accepts it. Class names per scripts/tools/check_scl_layers.py.
SCL_SEVERITY_TIERS: tuple[tuple[int, ...], ...] = (
    (4, 5, 6, 7, 11),  # vegetation, bare, water, unclassified, snow
    (10,),  # thin cirrus
    (2,),  # dark area
    (3,),  # cloud shadow
    (8,),  # cloud medium probability
    (9,),  # cloud high probability
    (1,),  # saturated / defective
    (0,),  # nodata
)
# uint8-indexable lookup. Unlisted codes (a future class 12, or garbage) sort
# last rather than silently ranking as clear.
WORST_SEVERITY = len(SCL_SEVERITY_TIERS)
SCL_SEVERITY: np.ndarray = np.full(256, WORST_SEVERITY, dtype=np.uint8)
for _tier, _classes in enumerate(SCL_SEVERITY_TIERS):
    for _cls in _classes:
        SCL_SEVERITY[_cls] = _tier
CLEAR_SEVERITY = 0


def period_index(doys: np.ndarray) -> np.ndarray:
    """Map day-of-year to its 30-day period, or -1 if no period covers it.

    The monthly layers are 30-day durations at offsets 0d..330d, so period p
    covers days-since-Jan-1 [30p, 30p+30), i.e. day-of-year 30p+1..30p+30. Days
    361+ (and 366 in a leap year) fall outside all twelve and are dropped, which
    is exactly what the ``mo`` layers do.

    Args:
        doys: (T,) day-of-year, 1-366.

    Returns:
        (T,) int array of period indices in [0, 12), or -1.
    """
    periods = (np.asarray(doys, dtype=np.int64) - 1) // PERIOD_DAYS
    return np.where(periods < MONTHS, periods, -1)


def period_midpoint_doy(period: int) -> float:
    """Day-of-year at the centre of a 30-day period."""
    return PERIOD_DAYS * period + (PERIOD_DAYS + 1) / 2


def select_best(severity: np.ndarray, doys: np.ndarray, period: int) -> np.ndarray:
    """Per-pixel index of the least-contaminated observation in one period.

    Ranks by severity tier, breaking ties toward the observation nearest the
    period midpoint so the composite's effective date stays near its nominal
    one. ``argmin`` takes the first minimum, so remaining ties resolve to the
    earliest acquisition -- deterministic across runs.

    Args:
        severity: (t, H, W) severity tiers for this period's observations.
        doys: (t,) day-of-year for the same observations.
        period: period index, for the midpoint tie-break.

    Returns:
        (H, W) int array indexing axis 0 of ``severity``.
    """
    # severity is small (<= 8) and the tie-break is < 366, so one scalar key
    # orders by (tier, distance) without a lexsort.
    distance = np.abs(np.asarray(doys, dtype=np.int64) - period_midpoint_doy(period))
    key = severity.astype(np.int64) * 512 + distance[:, None, None]
    return np.argmin(key, axis=0)


def composite_window(
    fetch_window: Window,
    include_60m: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Counter]:
    """Build every period's composite for one fetch window.

    Args:
        fetch_window: window in the fetch group holding the year of scenes.
        include_60m: read B01/B09 as well. Only possible on a fetch group
            materialized after they entered the fetch config (see
            read_band_sets).

    Returns:
        ``(imagery, scl, severity_hist)``. ``imagery`` maps layer name to an
        (H, W, C) uint16 array and ``scl`` maps layer name to an (H, W) uint8
        array of the *chosen* observation's SCL class, both keyed by
        ``S2_LAYERS`` / ``SCL_LAYERS``. Periods with no acquisition are absent
        from both, mirroring rslearn, which never materializes an empty month.
        ``severity_hist`` counts chosen tiers over all pixels and periods.

    Raises:
        NoS2ScenesError: if the window's S2 layer matched zero scenes.
        ValueError: if the imagery and SCL reads disagree on scene ordering.
    """
    imagery, doys = _read_scenes(fetch_window, S2_LAYER, read_band_sets(include_60m))
    if imagery.shape[0] == 0:
        raise NoS2ScenesError(
            f"window {fetch_window.group}/{fetch_window.name} has zero "
            f"{S2_LAYER} scenes"
        )
    # SCL is categorical: nearest resampling, no averaging across classes.
    scl, scl_doys = _read_scenes(
        fetch_window, S2_LAYER, (SCL_BAND_SET,), resampling=Resampling.nearest
    )
    if not np.array_equal(doys, scl_doys):
        raise ValueError("S2 band and SCL scene ordering diverged")

    scl_classes = scl[..., 0].astype(np.uint8)
    severity = SCL_SEVERITY[scl_classes]
    periods = period_index(doys)

    out_imagery: dict[str, np.ndarray] = {}
    out_scl: dict[str, np.ndarray] = {}
    hist: Counter = Counter()
    for period in range(MONTHS):
        in_period = periods == period
        if not in_period.any():
            continue
        chosen = select_best(severity[in_period], doys[in_period], period)
        take = chosen[None, :, :]
        picked = np.take_along_axis(imagery[in_period], take[..., None], axis=0)[0]
        out_imagery[S2_LAYERS[period]] = np.rint(picked).astype(np.uint16)
        out_scl[SCL_LAYERS[period]] = np.take_along_axis(
            scl_classes[in_period], take, axis=0
        )[0]
        hist.update(
            np.take_along_axis(severity[in_period], take, axis=0)[0].ravel().tolist()
        )
    return out_imagery, out_scl, hist


def period_time_range(window: Window, period: int) -> tuple[datetime, datetime]:
    """The 30-day range a period covers, from the window's calendar year.

    Mirrors the ``mo`` layers' 30d duration at offsets 0d..330d. Worth
    confirming against a parent ``sentinel2_l2a_moNN`` raster before trusting
    the stamped dates: the eval's time encodings are per-timestep.

    Args:
        window: the output window, whose time_range anchors the periods.
        period: period index in [0, 12).

    Returns:
        The period's (start, end).

    Raises:
        ValueError: if the window has no time range, which would otherwise
            stamp every composite with a silently wrong date.
    """
    if window.time_range is None:
        raise ValueError(
            f"window {window.group}/{window.name} has no time_range, so its "
            "period dates cannot be derived"
        )
    start = window.time_range[0] + timedelta(days=PERIOD_DAYS * period)
    return start, start + timedelta(days=PERIOD_DAYS)


def write_composite(
    window: Window,
    imagery: dict[str, np.ndarray],
    scl: dict[str, np.ndarray],
    bands: list[str],
) -> None:
    """Write one window's composite layers and mark them completed.

    Uses ``GeotiffRasterFormat`` directly rather than
    ``providers.write_embedding``, which derives band names from a ModalitySpec
    and hard-casts to float32 -- that would double a uint16 composite and lose
    its dtype.

    Args:
        window: the window in the output dataset.
        imagery: layer name to (H, W, C) uint16 array.
        scl: layer name to (H, W) uint8 array.
        bands: band names of the imagery, naming its raster dir.
    """
    raster_format = GeotiffRasterFormat()
    written: list[str] = []
    for period in range(MONTHS):
        s2_name = S2_LAYERS[period]
        if s2_name not in imagery:
            continue
        time_range = period_time_range(window, period)
        # Distinct loop names: binding `bands` here would shadow the parameter
        # and leak the previous period's band list into the next one.
        for name, array, layer_bands in (
            (s2_name, imagery[s2_name].transpose(2, 0, 1), bands),
            (SCL_LAYERS[period], scl[SCL_LAYERS[period]][None], SCL_BAND_SET),
        ):
            raster_format.encode_raster(
                window.get_raster_dir(name, layer_bands),
                window.projection,
                window.bounds,
                RasterArray(
                    chw_array=array,
                    time_range=time_range,
                    metadata=RasterMetadata(),
                ),
            )
            written.append(name)
    # Mark completed only once every raster is on disk. The resume guard tests
    # period 0, so marking as we went would let an interrupted window be skipped
    # forever with periods 1-11 missing. Interrupting before this point leaves
    # unmarked rasters, which a re-run simply overwrites.
    for name in written:
        window.mark_layer_completed(name)


def _bounded_map(
    pool: ThreadPoolExecutor, windows: list[Window], fn: Any, readahead: int
) -> Any:
    """``pool.map`` with bounded read-ahead.

    ``pool.map`` buffers every completed result, which for a year of scenes per
    window is hundreds of MB. Lifted from ``tessera_v2_export.infer``.
    """
    pending: deque = deque()
    window_iter = iter(windows)
    for window in itertools.islice(window_iter, 2 * readahead):
        pending.append(pool.submit(fn, window))
    while pending:
        yield pending.popleft().result()
        for window in itertools.islice(window_iter, 1):
            pending.append(pool.submit(fn, window))


CONFIG_BACKUP_NAME = "config.json.pre_pixel_mosaic"


def patch_config(out_ds_path: str, include_60m: bool = False) -> list[str]:
    """Rewrite the output dataset's S2/SCL band sets to match what we wrote.

    The clone inherits the parent's ``config.json``, whose ``sentinel2_l2a_moNN``
    layers declare THREE band sets at 10 m / 20 m / 40 m and whose SCL layers
    declare ``zoom_offset: -1``. ``_read_scenes`` returns every band already on
    the window grid, so the compositor writes ONE band set at window resolution
    instead -- and ``get_raster_dir`` names the directory after the band list, so
    without this rslearn would look for band-set dirs that do not exist.

    Idempotent, and keeps a one-time backup at ``CONFIG_BACKUP_NAME``.

    Args:
        out_ds_path: the composited dataset.
        include_60m: whether the composite carries B01/B09.

    Returns:
        The layer names that were rewritten.

    Raises:
        SystemExit: if the config has no monthly S2 layers to rewrite.
    """
    config_path = UPath(out_ds_path) / "config.json"
    config = json.loads(config_path.read_text())
    layers = config.get("layers", {})
    if not any(name in layers for name in S2_LAYERS):
        raise SystemExit(
            f"{config_path} declares no {S2_LAYERS[0]}-style layers; is this the "
            "composited dataset?"
        )

    wanted = {
        **{
            name: [{"bands": composite_bands(include_60m), "dtype": "uint16"}]
            for name in S2_LAYERS
        },
        **{name: [{"bands": SCL_BAND_SET, "dtype": "uint8"}] for name in SCL_LAYERS},
    }
    changed = [
        name
        for name, band_sets in wanted.items()
        if name in layers and layers[name].get("band_sets") != band_sets
    ]
    if not changed:
        logger.info("config.json already matches the composite; nothing to do")
        return []

    backup = UPath(out_ds_path) / CONFIG_BACKUP_NAME
    if not backup.exists():
        backup.write_text(config_path.read_text())
        logger.info(f"backed up the inherited config to {backup}")
    for name in changed:
        layers[name]["band_sets"] = wanted[name]
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"rewrote band_sets on {len(changed)} layers in {config_path}")
    return changed


def _providers(
    ds_path: str, out_ds_path: str, spec: DatasetSpec
) -> tuple[list[Window], dict[str, Window], RslearnWindowProvider]:
    """Resolve output windows and their fetch-group counterparts, matched by name."""
    out_provider = RslearnWindowProvider(UPath(out_ds_path), groups=spec.eval_groups)
    out_windows = eval_windows_of(out_provider, spec)
    check_names_unique(out_windows)
    fetch_provider = RslearnWindowProvider(UPath(ds_path), groups=[spec.fetch_group])
    fetch_windows = {w.name: w for w in fetch_provider.load_windows()}
    missing = [w.name for w in out_windows if w.name not in fetch_windows]
    if missing:
        raise SystemExit(
            f"{len(missing)} output windows have no fetch window, e.g. "
            f"{missing[:3]}. Run tessera_v2_export create_windows + the fetch first."
        )
    return out_windows, fetch_windows, out_provider


def composite(
    ds_path: str,
    out_ds_path: str,
    spec: DatasetSpec,
    workers: int = 8,
    overwrite: bool = False,
    include_60m: bool = False,
) -> None:
    """Composite every eval window and write the layers into the output dataset.

    Args:
        ds_path: dataset holding the fetch group (the year of scenes).
        out_ds_path: dataset to write the composite layers into.
        spec: fetch-group / eval-group selection.
        workers: raster-read threads.
        overwrite: recompute windows whose first period is already written.
        include_60m: include B01/B09 (see read_band_sets -- only possible on
            a fetch group materialized after they entered the fetch config).
    """
    out_windows, fetch_windows, out_provider = _providers(ds_path, out_ds_path, spec)
    bands = composite_bands(include_60m)
    logger.info(
        f"compositing {len(out_windows)} windows into {out_ds_path} "
        f"({len(bands)} bands)"
    )

    def read_one(
        out_window: Window,
    ) -> tuple[Window, tuple[dict, dict, Counter] | None, str]:
        if not overwrite and out_provider.is_layer_written(out_window, S2_LAYERS[0]):
            return out_window, None, "skipped"
        try:
            return (
                out_window,
                composite_window(
                    fetch_windows[out_window.name], include_60m=include_60m
                ),
                "ok",
            )
        except NoS2ScenesError as e:
            logger.warning(f"window {out_window.name}: coverage gap -- {e}")
            return out_window, None, "gap"
        except Exception:
            logger.exception(f"window {out_window.name}: compositing failed")
            return out_window, None, "failed"

    written = skipped = 0
    failed: list[str] = []
    gaps: list[str] = []
    severity_hist: Counter = Counter()
    empty_periods = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for out_window, result, status in _bounded_map(
            pool, out_windows, read_one, workers
        ):
            if status == "failed":
                failed.append(out_window.name)
                continue
            if status == "gap":
                gaps.append(out_window.name)
                continue
            if status == "skipped":
                skipped += 1
                continue
            assert result is not None
            imagery, scl, hist = result
            try:
                write_composite(out_window, imagery, scl, bands)
            except Exception:
                logger.exception(f"window {out_window.name}: writing failed")
                failed.append(out_window.name)
                continue
            severity_hist.update(hist)
            empty_periods += MONTHS - len(imagery)
            written += 1
            if written % 50 == 0:
                logger.info(f"{written}/{len(out_windows)} windows written")

    total_pixels = sum(severity_hist.values())
    write_manifest(
        UPath(out_ds_path),
        "pixel_mosaic",
        {
            "product": "pixel_mosaic",
            "product_version": "select-least-contaminated-v1",
            "bands": bands,
            "severity_tiers": [list(tier) for tier in SCL_SEVERITY_TIERS],
            # The experiment's own falsifier: if the chosen pixels are mostly
            # NOT tier 0, there was little to choose between and a flat eval
            # result needs no further explanation.
            "chosen_severity_histogram": {
                str(tier): count for tier, count in sorted(severity_hist.items())
            },
            "chosen_clear_fraction": (
                severity_hist[CLEAR_SEVERITY] / total_pixels if total_pixels else None
            ),
            "num_empty_periods": empty_periods,
            "num_windows_written": written,
            "num_windows_skipped_existing": skipped,
            "num_coverage_gaps": len(gaps),
            "coverage_gaps": sorted(gaps),
            "num_windows_without_year": 0,
            "windows_without_year": [],
            "num_windows_failed": len(failed),
            "windows_failed": sorted(failed),
            "cli_args": {
                "ds_path": ds_path,
                "out_ds_path": out_ds_path,
                "fetch_group": spec.fetch_group,
                "workers": workers,
            },
        },
    )
    logger.info(
        f"wrote {written}, skipped {skipped}, gaps {len(gaps)}, failed {len(failed)}"
    )
    if total_pixels:
        logger.info(
            f"chosen pixels clear: {severity_hist[CLEAR_SEVERITY] / total_pixels:.1%}"
        )


def probe(ds_path: str, spec: DatasetSpec, sample: int = 200) -> dict[str, Any]:
    """Report how many clear observations per pixel per period actually exist.

    The go/no-go gate: per-pixel cloud selection can only help where a period
    holds more than one usable observation. Reads the fetch group only, writes
    nothing.

    Args:
        ds_path: dataset holding the fetch group.
        spec: fetch-group selection.
        sample: number of fetch windows to read.

    Returns:
        A summary dict, also logged.
    """
    provider = RslearnWindowProvider(UPath(ds_path), groups=[spec.fetch_group])
    windows = sorted(provider.load_windows(), key=lambda w: w.name)[:sample]
    logger.info(f"probing {len(windows)} fetch windows")

    clear_counts: Counter = Counter()
    obs_counts: Counter = Counter()
    # Per period as well as pooled: whether the usable observations sit in the
    # growing season decides whether a composite can move a crop-type score at
    # all. Pooling hides that, and for ethiopia the Meher monsoon (roughly
    # periods 6-9) is exactly where clear looks are expected to be scarcest.
    per_period: dict[int, Counter] = {p: Counter() for p in range(MONTHS)}
    for i, window in enumerate(windows):
        try:
            scl, doys = _read_scenes(
                window, S2_LAYER, (SCL_BAND_SET,), resampling=Resampling.nearest
            )
        except Exception:
            logger.exception(f"window {window.name}: probe read failed")
            continue
        if scl.shape[0] == 0:
            continue
        clear = SCL_SEVERITY[scl[..., 0].astype(np.uint8)] == CLEAR_SEVERITY
        periods = period_index(doys)
        for period in range(MONTHS):
            in_period = periods == period
            if not in_period.any():
                continue
            obs_counts[int(in_period.sum())] += 1
            counts = clear[in_period].sum(axis=0).ravel().tolist()
            clear_counts.update(counts)
            per_period[period].update(counts)
        if (i + 1) % 25 == 0:
            logger.info(f"probed {i + 1}/{len(windows)}")

    def _fractions(counts: Counter) -> dict[str, Any]:
        """Zero-clear / has-a-choice / mean-clear for one bucket of counts."""
        total = sum(counts.values())
        if not total:
            return {"pixel_periods": 0}
        return {
            "pixel_periods": total,
            "zero_clear": round(counts[0] / total, 4),
            "has_choice": round(sum(c for n, c in counts.items() if n >= 2) / total, 4),
            "mean_clear": round(sum(n * c for n, c in counts.items()) / total, 2),
        }

    summary = {
        "windows_probed": len(windows),
        "acquisitions_per_period": dict(sorted(obs_counts.items())),
        "clear_obs_per_pixel_period": dict(sorted(clear_counts.items())),
        "pooled": _fractions(clear_counts),
        "by_period": {str(p): _fractions(per_period[p]) for p in range(MONTHS)},
    }
    logger.info(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # patch_config touches only the output dataset, so it takes none of the
    # fetch-group selection flags.
    p_patch = sub.add_parser(
        "patch_config",
        help="Point the clone's config.json at the composite band sets",
    )
    p_patch.add_argument("--out_ds_path", required=True, help="The composited dataset.")
    p_patch.add_argument("--include_60m", action="store_true")

    for name, help_text in (
        ("probe", "Report clear-observation counts; writes nothing"),
        ("composite", "Build and write the composite layers"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--ds_path", required=True, help="Dataset holding the fetch group."
        )
        p.add_argument("--dataset", default=None, help="Preset fetch group / year.")
        p.add_argument("--fetch_group", default=None, help="Override the fetch group.")
        p.add_argument(
            "--eval_groups",
            default=None,
            help="Comma-separated eval window groups (default: all but the fetch group).",
        )
        p.add_argument("--year", type=int, default=None, help="Pin the fetch year.")
        if name == "composite":
            p.add_argument(
                "--out_ds_path",
                required=True,
                help="Dataset to write the composite layers into.",
            )
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--overwrite", action="store_true")
            p.add_argument(
                "--include_60m",
                action="store_true",
                help="Also composite B01/B09. Only works on a fetch group "
                "materialized AFTER they entered the fetch config -- rslearn "
                "marks completion per layer, so existing groups never backfill.",
            )
        else:
            p.add_argument("--sample", type=int, default=200)

    args = parser.parse_args()
    if args.command == "patch_config":
        patch_config(args.out_ds_path, include_60m=args.include_60m)
        return
    spec = resolve_spec(args.dataset, args.fetch_group, args.year)
    if args.eval_groups:
        spec.eval_groups = args.eval_groups.split(",")
    if args.command == "probe":
        probe(args.ds_path, spec, sample=args.sample)
    else:
        composite(
            args.ds_path,
            args.out_ds_path,
            spec,
            workers=args.workers,
            overwrite=args.overwrite,
            include_60m=args.include_60m,
        )


if __name__ == "__main__":
    main()
