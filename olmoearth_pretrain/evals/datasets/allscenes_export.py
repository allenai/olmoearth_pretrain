"""Feed the eval a year of real S2 acquisitions instead of twelve monthly mosaics.

The year-aligned eval reads twelve 30-day Sentinel-2 mosaics, one per month. The
archive holds a median ~72 acquisitions per window per year
(docs/TesseraV2Inference.md), so eleven of every twelve looks are discarded before
the model sees anything. This script builds the other arm of that comparison: the
same windows, the same labels, the same splits, but S2 as a chronological stack of
individual acquisitions.

It reads the ``sentinel2_l2a_all`` fetch group that ``tessera_v2_export`` already
materialized -- every acquisition in the window's calendar year, one item group per
scene, sorted datetime-ascending -- and attaches a subset of those item groups to
the eval windows of a *separate* dataset, leaving the baseline untouched. Nothing
is re-fetched and nothing is resampled: the scene rasters are copied verbatim, so
they keep the fetch's native 10 m / 20 m band-set layout and the eval loader
upsamples at load time exactly as it does for the monthly layers. That matters --
it is the one deviation ``pixel_mosaic_export`` had to accept, and here there is no
reason to.

WHY A CAP. Token count at the eval's ws16/patch_size=1 convention is
16 x 16 x T, so the full ~72 scenes is six times the monthly arm's sequence and
far more than six times its attention cost. ``--cap`` thins the year to a fixed
budget of evenly spaced acquisitions (default 36, three times the monthly arm).
Selection is by TIME, not by list position: MGRS overlap zones return two scenes
for the same date, and taking every k-th list entry would spend the budget on
duplicate dates while skipping whole weeks elsewhere. Windows with fewer scenes
than the cap keep all of them.

VARIABLE LENGTH IS FINE, and is the reason the item-group layout is used rather
than a fixed list of ``_scNN`` layers. ``load_all_item_groups`` reads however many
groups a window has (rslearn only drops windows with zero), the eval collate pads
the batch to its longest sample and marks the padding MISSING
(``evals/datasets/utils.eval_collate_fn_variable_time``), and each timestep carries
its own acquisition date from the raster metadata. A fixed layer list would instead
drop every window that came up short of the cap, silently changing the window set.

B01/B09 ARE ABSENT, not zero-filled here. The fetch group carries only the 10 m and
20 m band sets -- B01/B09 were added to ``config_tessera_v2_fetch.json`` after these
groups were materialized, and rslearn cannot backfill a band set onto a materialized
layer. The generated model.yaml therefore declares the ten bands that exist and lets
the loader scatter them to their canonical channel positions, zeroing the other two
after normalization (``rslearn_dataset._init_band_scatter``). This is the same
compromise ``pixel_mosaic_export`` documents, and it is in-distribution: pretraining
uses band dropout. It does mean an all-scenes-vs-monthly delta bundles the temporal
change with the loss of two 60 m atmospheric bands.

Runbook, per dataset. Unlike the ccmos pilot there is no fetch or materialize step,
so the output is built directly in eval_datasets rather than in rslearn staging::

    NAME=ethiopia_crops
    # the fetch group lives on the STAGING copy...
    STAGE=/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/${NAME}_year_aligned
    # ...while labels, split tags and the gse/tessera_v2 layers live on the INGESTED one
    SRC=/weka/dfive-default/olmoearth/eval_datasets/${NAME}_year_aligned
    DST=/weka/dfive-default/olmoearth/eval_datasets/${NAME}_s2all36_year_aligned

    # 0. how many scenes are actually there, and what does the cap cost?
    python -m olmoearth_pretrain.evals.datasets.allscenes_export probe \
        --ds_path $STAGE --dataset ${NAME}_year_aligned --sample 200

    # 1. clone, dropping the S2 imagery this script replaces. Seeding from $SRC is
    # what carries gse/tessera_v2 across, so the baselines need no re-fetch and the
    # window set stays identical to the monthly arm's. model.yaml is excluded
    # because the clone gets its own (see step 3 of the ccmos runbook for why
    # inheriting the parent's is a trap).
    rsync -a --info=progress2 \
        --exclude='sentinel2_l2a_mo*' --exclude='sentinel2_scl_mo*' \
        --exclude='model.yaml' "$SRC/" "$DST/"

    # 2. attach the scenes and patch config.json
    python -m olmoearth_pretrain.evals.datasets.allscenes_export build \
        --ds_path $STAGE --dataset ${NAME}_year_aligned \
        --out_ds_path $DST --cap 36 --workers 16

    # 3. register the entry (clones the parent's, like the ccmos pilot)
    python scripts/tools/register_allscenes_entries.py --go
"""

import argparse
import itertools
import json
import logging
import shutil
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from rslearn.dataset import Window
from rslearn.train.dataset import read_layer_time_range
from upath import UPath

from olmoearth_pretrain.evals.datasets.tessera_v2_export import (
    S2_BAND_SETS,
    S2_LAYER,
    SCL_BAND_SET,
    DatasetSpec,
    check_names_unique,
    eval_windows_of,
    resolve_spec,
)
from olmoearth_pretrain.evals.embedding_materializer.materialize import write_manifest
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)

logger = logging.getLogger(__name__)

DEFAULT_CAP = 36

# The layer definition written into the output dataset's config.json. It mirrors
# what the fetch group MATERIALIZED, which is not the same as what
# config_tessera_v2_fetch.json currently DECLARES: the B01/B09 band set was added
# to that file after these groups were built and never landed on disk, so
# declaring it here would point rslearn at directories that do not exist. There is
# no data_source -- these rasters are placed by this script, never fetched, and a
# layer with a source is one stray `rslearn dataset materialize` away from pulling
# a year of imagery for every window (the warning at the top of
# config_tessera_v2_fetch.json).
SCENE_BAND_SETS: list[dict[str, Any]] = [
    {"bands": list(S2_BAND_SETS[0]), "dtype": "uint16"},
    {"bands": list(S2_BAND_SETS[1]), "dtype": "uint16", "zoom_offset": -1},
    {"bands": list(SCL_BAND_SET), "dtype": "uint8", "zoom_offset": -1},
]
SCENE_LAYER_CONFIG: dict[str, Any] = {"type": "raster", "band_sets": SCENE_BAND_SETS}

# Imagery bands the model.yaml reads, in the order the band sets concatenate.
# SCL rides along in the layer but is not a model input here (these tasks run
# unmasked); it is copied so a masked sibling stays possible without a rebuild.
SCENE_BANDS = [band for band_set in S2_BAND_SETS for band in band_set]

# Layer directories the S2 monthly scheme owns, dropped from the output config so
# it describes only what the clone actually holds. The rsync in the runbook drops
# the matching rasters.
DROP_LAYER_PREFIXES = ("sentinel2_l2a_mo", "sentinel2_scl_mo")


def scene_times(fetch_window: Window) -> list[tuple[int, datetime]]:
    """Acquisition instant of every materialized S2 item group in a fetch window.

    Read from ``items.json`` (one file per window) rather than from each group's
    raster metadata (one per scene): the two agree, and at ~72 scenes across tens
    of thousands of windows the difference is a weka round trip per scene.
    Restricted to groups marked completed, so a partial materialize shows up as
    missing scenes rather than as a copy of a directory that is not there.

    Args:
        fetch_window: window in the fetch group holding the year of scenes.

    Returns:
        ``(group_idx, acquisition instant)`` pairs, chronological.
    """
    completed = {
        group_idx
        for layer_name, group_idx in fetch_window.list_completed_layers()
        if layer_name == S2_LAYER
    }
    if not completed:
        return []
    layer_data = fetch_window.load_layer_datas().get(S2_LAYER)
    if layer_data is None:
        return []

    scenes: list[tuple[int, datetime]] = []
    for group_idx in sorted(completed):
        if group_idx >= len(layer_data.serialized_item_groups):
            logger.warning(
                f"window {fetch_window.name}: {S2_LAYER}.{group_idx} is materialized "
                "but absent from items.json; skipping it"
            )
            continue
        time_range = read_layer_time_range(layer_data, group_idx)
        if time_range is None:
            logger.warning(
                f"window {fetch_window.name}: {S2_LAYER}.{group_idx} has no item "
                "time range; skipping it"
            )
            continue
        scenes.append((group_idx, time_range[0]))
    scenes.sort(key=lambda pair: pair[1])
    return scenes


def thin_to_cap(
    scenes: list[tuple[int, datetime]], cap: int
) -> list[tuple[int, datetime]]:
    """Thin a year of acquisitions to ``cap`` evenly spaced in TIME.

    Places ``cap`` targets at the centres of equal slices of the window's observed
    span and takes the nearest unclaimed acquisition to each, which keeps the
    chosen dates spread over the year even where the archive is not: an MGRS
    overlap zone returns two scenes for one date, and index-based thinning would
    spend two slots on that date and none on the following week.

    Ties resolve to the earlier acquisition (``min`` keeps the first minimum over a
    chronologically sorted list), so the selection is deterministic across runs.

    Args:
        scenes: ``(group_idx, instant)`` pairs, chronological.
        cap: maximum number to keep. Non-positive means no cap.

    Returns:
        The kept subset, chronological.
    """
    if cap <= 0 or len(scenes) <= cap:
        return scenes

    span_start = scenes[0][1]
    span = (scenes[-1][1] - span_start).total_seconds()
    if span <= 0:
        return scenes[:cap]

    remaining = list(range(len(scenes)))
    chosen: list[int] = []
    for slot in range(cap):
        target = span * (slot + 0.5) / cap
        best = min(
            remaining,
            key=lambda i: abs((scenes[i][1] - span_start).total_seconds() - target),
        )
        remaining.remove(best)
        chosen.append(best)
    return [scenes[i] for i in sorted(chosen)]


def layer_dir(window: Window, group_idx: int) -> UPath:
    """The window's directory for one item group of the scene layer."""
    folder = S2_LAYER if group_idx == 0 else f"{S2_LAYER}.{group_idx}"
    return window.window_root / "layers" / folder


def attach_scenes(
    fetch_window: Window,
    out_window: Window,
    scenes: list[tuple[int, datetime]],
) -> int:
    """Copy the selected scene rasters onto an eval window, renumbered from zero.

    Item groups are renumbered to a contiguous 0..n-1 rather than keeping their
    fetch indices for two reasons: ``is_data_input_available`` probes group_idx=0
    specifically to decide whether a window has the layer at all, and a gapped
    numbering would make the on-disk indices meaningless as an ordering.

    The ``completed`` marker is written by ``mark_layer_completed`` after the
    rasters are in place, not copied along with them, so an interrupted run leaves
    unmarked directories that a re-run overwrites instead of half-copied groups
    that later look materialized.

    Args:
        fetch_window: source window in the fetch group.
        out_window: destination window in the output dataset.
        scenes: the selected ``(group_idx, instant)`` pairs, chronological.

    Returns:
        the number of item groups written.
    """
    for out_idx, (src_idx, _instant) in enumerate(scenes):
        src = layer_dir(fetch_window, src_idx)
        dst = layer_dir(out_window, out_idx)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("completed"))
    for out_idx in range(len(scenes)):
        out_window.mark_layer_completed(S2_LAYER, group_idx=out_idx)
    return len(scenes)


def stale_groups(out_window: Window) -> list[int]:
    """Scene item groups already on an output window, for resume/overwrite checks."""
    return sorted(
        group_idx
        for layer_name, group_idx in out_window.list_completed_layers()
        if layer_name == S2_LAYER
    )


CONFIG_BACKUP_NAME = "config.json.pre_allscenes"


def patch_config(out_ds_path: str) -> None:
    """Declare the scene layer in the output config and drop the monthly S2 ones.

    The clone inherits the parent's ``config.json``, which knows nothing about
    ``sentinel2_l2a_all`` -- and ``read_data_input`` looks the layer config up by
    name, so without this the dataset fails to build. The monthly S2 and SCL layers
    go the other way: the rsync left their rasters behind, so keeping their
    declarations would describe data the clone does not have.

    Idempotent, and keeps a one-time backup at ``CONFIG_BACKUP_NAME``.

    Args:
        out_ds_path: the output dataset root.
    """
    config_path = UPath(out_ds_path) / "config.json"
    config = json.loads(config_path.read_text())
    layers = config.get("layers", {})

    dropped = [name for name in layers if name.startswith(DROP_LAYER_PREFIXES)]
    if layers.get(S2_LAYER) == SCENE_LAYER_CONFIG and not dropped:
        logger.info("config.json already describes the scene layer; nothing to do")
        return

    backup = UPath(out_ds_path) / CONFIG_BACKUP_NAME
    if not backup.exists():
        backup.write_text(config_path.read_text())
        logger.info(f"backed up the inherited config to {backup}")

    for name in dropped:
        del layers[name]
    layers[S2_LAYER] = SCENE_LAYER_CONFIG
    config["layers"] = layers
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logger.info(
        f"declared {S2_LAYER} and dropped {len(dropped)} monthly S2 layers "
        f"in {config_path}"
    )


def _bounded_map(
    pool: ThreadPoolExecutor, windows: list[Window], fn: Any, readahead: int
) -> Iterator[Any]:
    """``pool.map`` with bounded read-ahead, lifted from pixel_mosaic_export."""
    pending: list[Any] = []
    window_iter = iter(windows)
    for window in itertools.islice(window_iter, 2 * readahead):
        pending.append(pool.submit(fn, window))
    while pending:
        yield pending.pop(0).result()
        for window in itertools.islice(window_iter, 1):
            pending.append(pool.submit(fn, window))


def _providers(
    ds_path: str, out_ds_path: str, spec: DatasetSpec
) -> tuple[list[Window], dict[str, Window]]:
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
    return out_windows, fetch_windows


def build(
    ds_path: str,
    out_ds_path: str,
    spec: DatasetSpec,
    cap: int = DEFAULT_CAP,
    workers: int = 8,
    overwrite: bool = False,
) -> None:
    """Attach the thinned year of scenes to every eval window of the output dataset.

    Args:
        ds_path: dataset holding the fetch group (the year of scenes).
        out_ds_path: dataset to write the scene layers into.
        spec: fetch-group / eval-group selection.
        cap: maximum acquisitions per window (see thin_to_cap).
        workers: copy threads.
        overwrite: rebuild windows that already carry scene groups.
    """
    out_windows, fetch_windows = _providers(ds_path, out_ds_path, spec)
    logger.info(
        f"attaching up to {cap} scenes per window to {len(out_windows)} windows "
        f"in {out_ds_path}"
    )
    patch_config(out_ds_path)

    def do_one(out_window: Window) -> tuple[Window, int, str]:
        """Attach one window's scenes -> (window, groups written, status)."""
        existing = stale_groups(out_window)
        if existing and not overwrite:
            return out_window, len(existing), "skipped"
        try:
            scenes = scene_times(fetch_windows[out_window.name])
        except Exception:
            logger.exception(f"window {out_window.name}: reading the fetch failed")
            return out_window, 0, "failed"
        if not scenes:
            logger.warning(f"window {out_window.name}: coverage gap -- zero S2 scenes")
            return out_window, 0, "gap"
        kept = thin_to_cap(scenes, cap)
        try:
            # Groups beyond the new selection would otherwise survive a rebuild at
            # a smaller cap and be read as extra timesteps.
            for group_idx in existing[len(kept) :]:
                shutil.rmtree(layer_dir(out_window, group_idx), ignore_errors=True)
            return (
                out_window,
                attach_scenes(fetch_windows[out_window.name], out_window, kept),
                "ok",
            )
        except Exception:
            logger.exception(f"window {out_window.name}: copying scenes failed")
            return out_window, 0, "failed"

    written = skipped = 0
    kept_hist: Counter = Counter()
    failed: list[str] = []
    gaps: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for out_window, count, status in _bounded_map(
            pool, out_windows, do_one, workers
        ):
            if status == "failed":
                failed.append(out_window.name)
            elif status == "gap":
                gaps.append(out_window.name)
            elif status == "skipped":
                skipped += 1
            else:
                written += 1
                kept_hist[count] += 1
                if written % 250 == 0:
                    logger.info(f"{written}/{len(out_windows)} windows written")

    write_manifest(
        UPath(out_ds_path),
        "allscenes",
        {
            "product": "allscenes",
            "product_version": f"s2-scene-stack-cap{cap}-v1",
            "layer": S2_LAYER,
            "bands": SCENE_BANDS,
            "cap": cap,
            "scenes_per_window_histogram": {
                str(count): n for count, n in sorted(kept_hist.items())
            },
            "num_windows_written": written,
            "num_windows_skipped_existing": skipped,
            "num_coverage_gaps": len(gaps),
            "coverage_gaps": sorted(gaps),
            "num_windows_failed": len(failed),
            "windows_failed": sorted(failed),
            "cli_args": {
                "ds_path": ds_path,
                "out_ds_path": out_ds_path,
                "fetch_group": spec.fetch_group,
                "cap": cap,
                "workers": workers,
            },
        },
    )
    total = sum(count * n for count, n in kept_hist.items())
    logger.info(
        f"wrote {written}, skipped {skipped}, gaps {len(gaps)}, failed {len(failed)}"
        + (f"; mean {total / written:.1f} scenes/window" if written else "")
    )
    if failed:
        logger.warning(f"{len(failed)} windows failed; re-run to retry them")


def probe(ds_path: str, spec: DatasetSpec, cap: int, sample: int = 200) -> None:
    """Report the scene-count distribution and what the cap would discard.

    Reads the fetch group only, writes nothing. Answers the two questions worth
    asking before building: is there materially more than twelve looks per window,
    and how much of the year does the cap throw away.

    Args:
        ds_path: dataset holding the fetch group.
        spec: fetch-group selection.
        cap: the cap to report against.
        sample: number of fetch windows to read.
    """
    provider = RslearnWindowProvider(UPath(ds_path), groups=[spec.fetch_group])
    windows = sorted(provider.load_windows(), key=lambda w: w.name)[:sample]
    logger.info(f"probing {len(windows)} fetch windows")

    counts: list[int] = []
    for i, window in enumerate(windows):
        try:
            counts.append(len(scene_times(window)))
        except Exception:
            logger.exception(f"window {window.name}: probe read failed")
        if (i + 1) % 50 == 0:
            logger.info(f"probed {i + 1}/{len(windows)}")

    if not counts:
        raise SystemExit("no fetch windows could be read")
    counts.sort()
    n = len(counts)
    logger.info(
        f"scenes/window over {n} windows: min {counts[0]}, p25 {counts[n // 4]}, "
        f"median {counts[n // 2]}, p75 {counts[3 * n // 4]}, max {counts[-1]}"
    )
    over = sum(1 for c in counts if c > cap)
    kept = sum(min(c, cap) for c in counts)
    logger.info(
        f"at cap {cap}: {over}/{n} windows thinned "
        f"({100 * over / n:.1f}%), keeping {kept / sum(counts):.1%} of all scenes, "
        f"{kept / n:.1f} scenes/window on average (monthly arm: 12)"
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["probe", "build"])
    parser.add_argument(
        "--ds_path", required=True, help="Dataset with the fetch group."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Preset fetch group / year for a known eval dataset "
        "(tessera_v2_export.DATASETS).",
    )
    parser.add_argument("--fetch_group", default=None, help="Override the fetch group.")
    parser.add_argument(
        "--eval_groups",
        default=None,
        help="Comma-separated eval window groups (default: all but the fetch group).",
    )
    parser.add_argument("--out_ds_path", default=None, help="Output dataset (build).")
    parser.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help="Max acquisitions kept per window; <=0 keeps every scene.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild windows that already carry scene groups.",
    )
    parser.add_argument("--sample", type=int, default=200, help="Windows to probe.")
    args = parser.parse_args()

    spec = resolve_spec(args.dataset, args.fetch_group)
    if args.eval_groups:
        spec.eval_groups = args.eval_groups.split(",")

    if args.command == "probe":
        probe(args.ds_path, spec, cap=args.cap, sample=args.sample)
        return
    if not args.out_ds_path:
        raise SystemExit("build needs --out_ds_path")
    build(
        args.ds_path,
        args.out_ds_path,
        spec,
        cap=args.cap,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
