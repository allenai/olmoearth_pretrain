"""Set up extra imagery layers (SCL, Landsat) on the *_year_aligned datasets.

Two layer sets, selected with ``--layer_sets`` (default: both):

``scl`` -- the pixel-level cloud-mask fix. The year-aligned re-export dropped
the original AEF-supplemental fetch's ``eo:cloud_cover < 50`` scene filter,
so in persistently cloudy regions (descals oil palm above all) the monthly
mosaics now carry cloud instead of being empty. Twelve ``sentinel2_scl_moNN``
layers are derived from the dataset's OWN ``sentinel2_l2a_moNN`` layers (same
data source, duration, time_offset; SCL band set at uint8, 20 m ->
zoom_offset -1, nearest resampling -- SCL is categorical), and every window's
S2 items.json entries are CLONED under the SCL names: the serialized items
already carry every STAC asset URL, SCL included, so the metadata copy IS the
prepare step -- identical scenes by construction, zero STAC queries. The eval
loader consumes them via ``RslearnToOlmoEarthDataset(scl_cloud_mask=True)``.

``landsat`` -- input parity with AEF, which ingests Landsat. Twelve
``landsat_moNN`` layers are taken from the pretraining config
(config_landsat.json, the exact config the pretraining dataset materialized
Landsat with), offsets rewritten to tile the calendar year (0d..330d, the
same rewrite reanchor_year_aligned_dataset.py applied to S2/S1). No items
exist to copy, so Landsat needs a real Beaker PREPARE pass before
materialize; ``launch`` picks the right stage per dataset automatically by
sampling window readiness, so re-running ``run --go`` advances the pipeline:
first invocation launches SCL materialize + Landsat prepare, a later one
(once prepare reads complete) launches Landsat materialize.

Commands:

- ``plan``   -- report per dataset x layer set: config state, window
  readiness, materialize progress, and the next launch stage. Read-only;
  re-run any time to watch progress.
- ``apply``  -- weka-side config.json edits + the SCL items copy.
- ``launch`` -- submit one Beaker job per dataset x layer set via rslp's
  ``common launch_data_materialization_jobs`` (the launcher used by
  launch_year_aligned_prepare.sh), hosts round-robin. ``--enabled-layers``
  restricts each job to its layer set, so the existing imagery layers are
  never rescanned (and pastis's *_all fetch layers cannot trigger); their
  rasters stay byte-identical to what the published year-aligned numbers
  were measured on. Prepare jobs get the rate-limited regime defaults
  (fewer workers, short backoff); materialize the worker-bound ones.
  Dry-run unless ``--go``.
- ``run``    -- apply then launch.

The PRETRAINING rslearn dataset is opt-in and off by default. It uses the
same monthly S2 layers (the eval re-export copied its configs), so the SCL
set applies there too when the time comes: pass its path via
``--pretrain_ds_path`` and it is handled alongside the eval datasets,
restricted to the ``res_10`` window group (``--pretrain_group``). Landsat is
skipped for it (the pretraining dataset already has Landsat). Note this only
lands SCL rasters on weka; using them in training additionally needs the
rslearn->olmoearth conversion step and dataloader/masking support, which are
separate work.

Usage, from the helios repo root on a weka-mounted machine::

    python scripts/tools/setup_extra_layers.py plan
    python scripts/tools/setup_extra_layers.py run \
        --hosts jupiter-cs-aus-134.reviz.ai2.in,jupiter-cs-aus-137.reviz.ai2.in --go
    # later: watch progress / advance landsat prepare -> materialize
    python scripts/tools/setup_extra_layers.py plan --only descals
    python scripts/tools/setup_extra_layers.py run --only descals --hosts ... --go
    # one layer set only
    python scripts/tools/setup_extra_layers.py run --layer_sets landsat --hosts ... --go
    # include the pretraining dataset (SCL only)
    python scripts/tools/setup_extra_layers.py run --pretrain_ds_path /weka/... --hosts ... --go

rslp is invoked from its own checkout/venv (``--rslp_dir``, default the
``rslearn_projects`` sibling of this repo). Beaker jobs query data sources
anonymously (the launcher cannot forward PC_SDK_SUBSCRIPTION_KEY); for
prepare that caps useful parallelism, so fan out across datasets, not by
stacking jobs on one dataset (the launcher does not shard).

After materialize completes (re-run ``plan`` until progress is ~100%):

1. ``python scripts/tools/check_scl_layers.py --ds_path ...`` per dataset to
   verify SCL landed wherever imagery exists.
2. ``python scripts/tools/backfill_eval_registry_provenance.py`` and commit
   registry.json: the registry pins config.json by sha256
   (``verify_config_json_hash``), so every eval on an edited dataset FAILS
   LOUDLY until the new hash is recorded. Run it once, after BOTH layer
   sets' config edits have landed, so the datasets go through one hash
   change.
3. Regenerate + re-sync model.yaml (build_year_aligned_eval_configs.py adds
   the optional ``scl`` and ``landsat`` inputs; the weka copy must match the
   repo copy for config_repo_dir provenance to hold).

Landsat eval tasks are deliberately NOT registered yet: the landsat input is
``required: false`` (a required input would shrink the evaluated window set
for every model and detach the published numbers), which means windows with
Landsat coverage gaps simply lack the input -- measure per-dataset
completeness first and register tasks with eyes open (the
docs/PrecomputedEmbeddingCoverage.md lesson).

The config.json edits change its hash, which invalidates
``.rslearn_dataset_index`` on their own -- no manual index deletion needed.
"""

import argparse
import json
import logging
import random
import shlex
import subprocess  # nosec B404
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rslearn.dataset import Dataset
from rslearn.dataset.window import Window, WindowLayerData
from upath import UPath

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "data" / "rslearn_dataset_configs"

MONTHS = 12
S2_LAYERS = [f"sentinel2_l2a_mo{i:02d}" for i in range(1, MONTHS + 1)]
SCL_LAYERS = [f"sentinel2_scl_mo{i:02d}" for i in range(1, MONTHS + 1)]
LANDSAT_LAYERS = [f"landsat_mo{i:02d}" for i in range(1, MONTHS + 1)]
S2_TO_SCL = dict(zip(S2_LAYERS, SCL_LAYERS))
LAYER_SET_NAMES = ("scl", "landsat")

# SCL ships at 20 m (zoom_offset -1, matching the B05/B06/... band set) and is
# categorical, so it must be resampled nearest -- bilinear would invent
# classes on upsample.
SCL_BAND_SET = {"bands": ["SCL"], "dtype": "uint8", "zoom_offset": -1}

DATASETS = (
    "africa_crop_mask_year_aligned",
    "canada_crops_coarse_year_aligned",
    "canada_crops_fine_year_aligned",
    "descals_year_aligned",
    "ethiopia_crops_year_aligned",
    "glance_year_aligned",
    "lcmap_lu_year_aligned",
    "us_trees_year_aligned",
    "pastis_year_aligned",
)

DEFAULT_STAGE_ROOT = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals"
DEFAULT_IMAGE = "favyen/rslpomp20260727a"
DEFAULT_RSLP_DIR = REPO_ROOT.parent / "rslearn_projects"

# tessera_v2_export.py adds a `<dataset>_tessera_v2` fetch group beside the
# eval windows (africa_crop_mask and ethiopia_crops carry one). Those windows
# have no monthly layers and are not part of any eval, so they are excluded
# from window listings and every Beaker job is scoped to the surviving groups
# -- otherwise a Landsat prepare/materialize would fetch a year of Landsat
# for thousands of fetch-only windows.
FETCH_GROUP_SUFFIX = "_tessera_v2"


def layer_names(layer_set: str) -> list[str]:
    """The twelve monthly layer names of a layer set."""
    return SCL_LAYERS if layer_set == "scl" else LANDSAT_LAYERS


# ---------------------------------------------------------------------------
# apply: config.json layers (+ items.json copies for scl)
# ---------------------------------------------------------------------------


def build_scl_layer(s2_layer: dict) -> dict:
    """Derive one SCL layer config from its sentinel2_l2a_moNN sibling.

    Copies the data source verbatim (class, duration, time_offset, harmonize
    -- harmless for SCL, which the Sentinel2 data source excludes from
    harmonization), so the layer targets exactly the same 30-day period as
    the imagery it masks.
    """
    layer = json.loads(json.dumps(s2_layer))
    layer["alias"] = "sentinel2_scl"
    layer["band_sets"] = [dict(SCL_BAND_SET)]
    layer["resampling_method"] = "nearest"
    return layer


def build_landsat_layers() -> dict[str, dict]:
    """Load the pretraining Landsat monthlies, re-offset to tile Jan 1 -> Dec 27.

    Same rewrite reanchor_year_aligned_dataset.build_monthly_layers applies to
    S2/S1: the pretraining offsets are centred on the window start
    (-180d..+150d); the year-aligned windows are (Jan 1 Y, Jan 1 Y+1), so the
    layers are moved to 0d..330d. Everything else (aws_landsat data source,
    band sets, sort_by cloud_cover) is the config pretraining materialized
    Landsat with, verbatim.
    """
    config = json.loads((CONFIG_ROOT / "config_landsat.json").read_text())
    layers: dict[str, dict] = {}
    for name in LANDSAT_LAYERS:
        layer = json.loads(json.dumps(config["layers"][name]))
        month = int(name[-2:])
        layer["data_source"]["time_offset"] = f"{(month - 1) * 30}d"
        layer["data_source"]["duration"] = "30d"
        layers[name] = layer
    return layers


def expected_layers(config: dict, layer_set: str) -> dict[str, dict]:
    """The layer configs a layer set should add to this dataset's config."""
    if layer_set == "scl":
        return {
            S2_TO_SCL[s2_name]: build_scl_layer(config["layers"][s2_name])
            for s2_name in S2_LAYERS
        }
    return build_landsat_layers()


def config_with_layers(
    config: dict, layer_set: str
) -> tuple[dict, list[str], list[str]]:
    """Return (updated config, layers to add, layers already present).

    Raises:
        ValueError: if the dataset lacks the monthly S2 layers (i.e. it is
            not a re-anchored year-aligned export) or an existing layer
            differs from what would be generated (manual edit -- resolve
            before re-running).
    """
    layers = config.get("layers", {})
    missing = [name for name in S2_LAYERS if name not in layers]
    if missing:
        raise ValueError(
            f"dataset lacks monthly S2 layers ({missing[:3]}...); "
            "run reanchor_year_aligned_dataset.py apply first"
        )

    updated = json.loads(json.dumps(config))
    added: list[str] = []
    present: list[str] = []
    for name, generated in expected_layers(config, layer_set).items():
        if name in layers:
            if layers[name] != generated:
                raise ValueError(
                    f"existing layer {name} differs from the generated "
                    "config; resolve the difference before re-running"
                )
            present.append(name)
            continue
        updated["layers"][name] = generated
        added.append(name)
    return updated, added, present


def copy_window_scl_items(window: Window) -> str:
    """Clone a window's S2 monthly layer datas under the SCL names.

    Returns:
        one of "copied", "partial" (some S2 months unprepared -- their SCL
        siblings mirror the gap), "already", or "unprepared" (no S2 layer
        datas at all; the window never finished prepare and is not in the
        eval set anyway).
    """
    layer_datas = window.load_layer_datas()
    s2_present = [name for name in S2_LAYERS if name in layer_datas]
    if not s2_present:
        return "unprepared"

    changed = False
    for s2_name in s2_present:
        scl_name = S2_TO_SCL[s2_name]
        if scl_name in layer_datas:
            continue
        serialized = json.loads(json.dumps(layer_datas[s2_name].serialize()))
        serialized["layer_name"] = scl_name
        serialized["materialized"] = False
        layer_datas[scl_name] = WindowLayerData.deserialize(serialized)
        changed = True
    if changed:
        window.save_layer_datas(layer_datas)
    if not changed:
        return "already"
    return "partial" if len(s2_present) < MONTHS else "copied"


def window_state(window: Window, layer_set: str) -> str:
    """Classify one window's readiness for a layer set (read-only).

    For scl: whether the items copy has run. For landsat: whether Beaker
    prepare has written its twelve layer-data entries.
    """
    layer_datas = window.load_layer_datas()
    if layer_set == "landsat":
        have = sum(1 for name in LANDSAT_LAYERS if name in layer_datas)
        if have == MONTHS:
            return "prepared"
        return "partially prepared" if have else "needs prepare"
    s2_present = [name for name in S2_LAYERS if name in layer_datas]
    if not s2_present:
        return "unprepared"
    if all(S2_TO_SCL[name] in layer_datas for name in s2_present):
        return "already"
    return "partial s2" if len(s2_present) < MONTHS else "needs copy"


def list_windows(ds_path: UPath, group: str | None, workers: int = 8) -> list[Window]:
    """List a dataset's windows, restricted to one group when set.

    Without an explicit group, tessera_v2 fetch groups are excluded (see
    FETCH_GROUP_SUFFIX).
    """
    windows = Dataset(ds_path).storage.get_windows(
        groups=[group] if group else None, workers=workers
    )
    if group is None:
        windows = [w for w in windows if not w.group.endswith(FETCH_GROUP_SUFFIX)]
    return windows


def apply_dataset(
    ds_path: UPath, group: str | None, layer_sets: list[str], workers: int
) -> None:
    """Write the config layers and (for scl) clone every window's items."""
    config_path = ds_path / "config.json"
    config = json.loads(config_path.read_text())
    added_any = False
    for layer_set in layer_sets:
        config, added, present = config_with_layers(config, layer_set)
        added_any = added_any or bool(added)
        logger.info(
            f"  config.json [{layer_set}]: +{len(added)} layers "
            f"({len(present)} present)"
        )
    if added_any:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

    if "scl" not in layer_sets:
        return
    windows = list_windows(ds_path, group)
    logger.info(f"  copying SCL layer datas for {len(windows)} windows...")
    counts: Counter = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, state in enumerate(pool.map(copy_window_scl_items, windows)):
            counts[state] += 1
            if (i + 1) % 10000 == 0:
                logger.info(f"    {i + 1}/{len(windows)}")
    logger.info(f"  scl items: {dict(counts)}")


# ---------------------------------------------------------------------------
# plan / progress
# ---------------------------------------------------------------------------


def materialize_progress(sampled: list[Window], layer_set: str) -> str:
    """Completed-marker counts over a window sample.

    Months whose period matched no scene never get a marker, so mature
    datasets legitimately top out below 100% -- markedly so for Landsat,
    whose 16-day revisit leaves genuinely empty months. The authoritative
    completion signal is a re-run reporting nothing left to do.
    """
    done = sum(
        1
        for w in sampled
        for name in layer_names(layer_set)
        if w.is_layer_completed(name)
    )
    return f"{done}/{len(sampled) * MONTHS} sampled layer-months"


def plan_dataset(
    ds_path: UPath, group: str | None, layer_sets: list[str], sample: int
) -> None:
    """Report config state, window readiness and materialize progress."""
    config = json.loads((ds_path / "config.json").read_text())
    windows = list_windows(ds_path, group)
    random.seed(0)
    sampled = random.sample(windows, min(sample, len(windows)))
    print(f"  windows: {len(windows)} ({len(sampled)} sampled)")

    for layer_set in layer_sets:
        try:
            _, added, present = config_with_layers(config, layer_set)
        except ValueError as e:
            print(f"  [{layer_set}] BLOCKED -- {e}")
            continue
        states = Counter(window_state(w, layer_set) for w in sampled)
        print(
            f"  [{layer_set}] config: {len(present)} present, {len(added)} to add"
            f" | windows: {dict(states)}"
            f" | materialized: {materialize_progress(sampled, layer_set)}"
            f" | next launch: {launch_stage(states, layer_set) or 'apply first'}"
        )


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


def launch_stage(states: Counter, layer_set: str) -> str | None:
    """Which rslearn command a launch would run now, or None if not ready.

    scl is ready to materialize once the items copy has run. landsat needs a
    prepare pass first; once every sampled window that exists is prepared,
    the next stage is materialize.
    """
    if layer_set == "scl":
        if states["needs copy"] + states["partial s2"]:
            return None
        return "materialize"
    return (
        "materialize"
        if states["needs prepare"] + states["partially prepared"] == 0
        else "prepare"
    )


def job_command(
    stage: str,
    layer_set: str,
    groups: list[str],
    args: argparse.Namespace,
) -> list[str]:
    """The in-job rslearn command; {ds_path} is substituted by the launcher.

    prepare and materialize are bound by different resources, so they get
    different worker/backoff defaults (see launch_year_aligned_prepare.sh
    for the measured reasoning: prepare is rate-limited -- a long backoff
    parks workers on their first 403 and collapses parallelism; materialize
    is worker-bound and a long backoff just covers real mid-download
    failures).
    """
    if stage == "prepare":
        workers, backoff = args.prepare_workers, args.prepare_retry_backoff
    else:
        workers, backoff = args.workers, args.retry_backoff
    command = [
        "rslearn",
        "dataset",
        stage,
        "--root",
        "{ds_path}",
        "--workers",
        str(workers),
        "--no-use-initial-job",
        "--retry-max-attempts",
        str(args.retry_attempts),
        "--retry-backoff-seconds",
        str(backoff),
        "--ignore-errors",
        # Only this layer set does work; without this the job would also scan
        # the imagery layers (harmless but slow) and, on pastis, try to fetch
        # the *_all scene layers (max_matches 150 -- NOT harmless).
        "--enabled-layers",
        ",".join(layer_names(layer_set)),
    ]
    if groups:
        # Scope to the eval window groups (rslearn's --group takes several),
        # keeping the job off any tessera_v2 fetch group.
        command += ["--group", *groups]
    return command


def launch_job(
    ds_path: UPath,
    stage: str,
    layer_set: str,
    groups: list[str],
    host: str | None,
    clusters: list[str],
    args: argparse.Namespace,
) -> bool:
    """Submit (or print) one Beaker job for one dataset x layer set."""
    rslp_python = Path(args.rslp_dir) / ".venv" / "bin" / "python"
    if not rslp_python.exists():
        rslp_python = Path("python")

    argv = [
        str(rslp_python),
        "-m",
        "rslp.main",
        "common",
        "launch_data_materialization_jobs",
        "--image",
        args.image,
        "--ds_path",
        str(ds_path),
        f"--priority={args.priority}",
    ]
    if host is not None:
        argv.append(f"--hosts+={host}")
    else:
        argv.extend(f"--clusters+={c}" for c in clusters)
        argv.append("--num_jobs=1")
    argv += ["--command", json.dumps(job_command(stage, layer_set, groups, args))]

    if not args.go:
        print("\n" + " ".join(shlex.quote(a) for a in argv))
        return True
    logger.info(
        f"  launching {layer_set} {stage} for {ds_path.name} on {host or clusters}"
    )
    # argv is a fixed arg list (no shell); the only variable parts are local
    # paths and the operator's own CLI values.
    result = subprocess.run(argv, cwd=args.rslp_dir)  # nosec B603
    if result.returncode != 0:
        logger.error(f"  FAILED to launch {ds_path.name}")
        return False
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def resolve_targets(
    args: argparse.Namespace, layer_sets: list[str]
) -> list[tuple[str, UPath, str | None, list[str]]]:
    """Resolve CLI args to (name, ds_path, window group, layer sets) targets.

    The nine eval datasets live under --stage_root with no group restriction
    and take every requested layer set; the pretraining dataset (when
    --pretrain_ds_path is given) is restricted to --pretrain_group and takes
    only scl -- it already has Landsat. --only selects one by base name,
    full name, or "pretrain".
    """
    stage_root = UPath(args.stage_root)
    targets: list[tuple[str, UPath, str | None, list[str]]] = [
        (name, stage_root / name, None, layer_sets) for name in DATASETS
    ]
    if args.pretrain_ds_path:
        pretrain_sets = [s for s in layer_sets if s == "scl"]
        if pretrain_sets:
            targets.append(
                (
                    "pretrain",
                    UPath(args.pretrain_ds_path),
                    args.pretrain_group or None,
                    pretrain_sets,
                )
            )
    if args.only is None:
        return targets
    matches = [t for t in targets if t[0] in (args.only, f"{args.only}_year_aligned")]
    if not matches:
        names = [t[0] for t in targets]
        raise ValueError(f"--only {args.only!r} matches none of {names}")
    return matches


def main() -> int:
    """CLI entry point."""
    # Log to stdout so logger and print() lines interleave in order.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
    )
    # rslearn logs one INFO line per items.json save -- 162k lines across the
    # eval datasets; this script's own progress lines cover it.
    logging.getLogger("rslearn.dataset.storage.file").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["plan", "apply", "launch", "run"])
    parser.add_argument(
        "--layer_sets",
        default=",".join(LAYER_SET_NAMES),
        help=f"Comma-separated subset of {LAYER_SET_NAMES}.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="One dataset (base or full name, or 'pretrain').",
    )
    parser.add_argument("--stage_root", default=DEFAULT_STAGE_ROOT)
    parser.add_argument(
        "--pretrain_ds_path",
        default=None,
        help="Path to the pretraining rslearn dataset; when set it is handled "
        "alongside the eval datasets (scl only -- it already has Landsat), "
        "restricted to --pretrain_group.",
    )
    parser.add_argument(
        "--pretrain_group",
        default="res_10",
        help="Window group of the pretraining dataset to process (the group "
        "internal_docs.md's pretraining materialize targets).",
    )
    parser.add_argument(
        "--sample", type=int, default=200, help="Readiness sample size."
    )
    parser.add_argument(
        "--copy_workers", type=int, default=32, help="Items-copy threads."
    )
    # Launch targeting: hosts round-robin (one job per dataset x layer set),
    # or clusters.
    parser.add_argument("--hosts", default=None, help="Comma-separated Beaker hosts.")
    parser.add_argument(
        "--clusters",
        default=None,
        help="Comma-separated Beaker clusters. NB: cluster launches reserve a "
        "GPU per job (rslp sets gpuCount=1 when no host is pinned) and "
        "neither prepare nor materialize touches one -- prefer --hosts.",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--priority",
        default="high",
        choices=["low", "normal", "high", "immediate", "urgent"],
    )
    parser.add_argument("--workers", type=int, default=64, help="Materialize workers.")
    parser.add_argument("--retry_backoff", type=int, default=60)
    parser.add_argument("--prepare_workers", type=int, default=16)
    parser.add_argument("--prepare_retry_backoff", type=int, default=2)
    parser.add_argument("--retry_attempts", type=int, default=12)
    parser.add_argument("--rslp_dir", default=str(DEFAULT_RSLP_DIR))
    parser.add_argument(
        "--go",
        action="store_true",
        help="Actually submit Beaker jobs; without it launch/run print the commands.",
    )
    args = parser.parse_args()

    layer_sets = [s for s in args.layer_sets.replace(",", " ").split() if s]
    unknown = [s for s in layer_sets if s not in LAYER_SET_NAMES]
    if unknown:
        parser.error(f"unknown layer set(s) {unknown}; choose from {LAYER_SET_NAMES}")

    targets = resolve_targets(args, layer_sets)

    do_apply = args.command in ("apply", "run")
    do_launch = args.command in ("launch", "run")

    if do_launch:
        if bool(args.hosts) == bool(args.clusters):
            parser.error("launch needs exactly one of --hosts or --clusters")
        if not args.go:
            logger.info(
                "DRY RUN -- pass --go to actually write changes and submit "
                "Beaker jobs. Nothing is modified or launched without it."
            )
    hosts = [h for h in (args.hosts or "").replace(",", " ").split() if h]
    clusters = [c for c in (args.clusters or "").replace(",", " ").split() if c]

    failures = 0
    launched = 0
    for name, ds_path, group, target_sets in targets:
        print(f"\n{name}:" + (f" (group {group})" if group else ""))
        if not ds_path.exists():
            print("  SKIP -- not staged")
            continue

        if args.command == "plan":
            plan_dataset(ds_path, group, target_sets, args.sample)
            continue

        if do_apply:
            # In `run` mode --go gates the weka writes too, so a dry run is
            # fully read-only; the explicit `apply` command always executes.
            if args.command == "apply" or args.go:
                apply_dataset(ds_path, group, target_sets, args.copy_workers)
            else:
                config = json.loads((ds_path / "config.json").read_text())
                for layer_set in target_sets:
                    try:
                        _, added, present = config_with_layers(config, layer_set)
                    except ValueError as e:
                        print(f"  [{layer_set}] BLOCKED -- {e}")
                        continue
                    print(
                        f"  DRY RUN [{layer_set}]: would add {len(added)} config "
                        f"layers ({len(present)} present)"
                        + (" and copy window items" if layer_set == "scl" else "")
                    )

        if do_launch:
            windows = list_windows(ds_path, group)
            if not windows:
                print("  SKIP -- no eval windows")
                continue
            groups = sorted({w.group for w in windows})
            random.seed(0)
            sampled = random.sample(windows, min(25, len(windows)))
            for layer_set in target_sets:
                states = Counter(window_state(w, layer_set) for w in sampled)
                stage = launch_stage(states, layer_set)
                if stage is None:
                    print(f"  SKIP {layer_set} -- not ready: {dict(states)}; run apply")
                    failures += 1
                    continue
                host = hosts[launched % len(hosts)] if hosts else None
                if not launch_job(
                    ds_path, stage, layer_set, groups, host, clusters, args
                ):
                    failures += 1
                launched += 1

    if args.command == "plan":
        print(
            "\nplan is read-only; `apply` edits configs/items, `launch --go` "
            "submits Beaker jobs (prepare or materialize, per readiness), "
            "`run --go` does both."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
