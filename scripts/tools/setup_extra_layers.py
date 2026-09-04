"""Set up the Landsat imagery layers on the *_year_aligned datasets.

Input parity with AEF, which ingests Landsat. Twelve ``landsat_moNN`` layers
are taken from the pretraining config (config_landsat.json, the exact config
the pretraining dataset materialized Landsat with), offsets rewritten to tile
the calendar year (0d..330d, the same rewrite reanchor_year_aligned_dataset.py
applied to S2/S1). No items exist to copy, so Landsat needs a real Beaker
PREPARE pass before materialize; ``launch`` picks the right stage per dataset
automatically by sampling window readiness, so re-running ``run --go``
advances the pipeline: first invocation launches prepare, a later one (once
prepare reads complete) launches materialize.

Commands:

- ``plan``   -- report per dataset: config state, window readiness,
  materialize progress, and the next launch stage. Read-only;
  re-run any time to watch progress.
- ``apply``  -- weka-side config.json edits.
- ``launch`` -- submit one Beaker job per dataset directly via beaker-py (no
  rslp checkout needed), hosts round-robin, host-pinned so no GPU is
  reserved. ``--enabled-layers`` restricts each job to the Landsat layers,
  so the existing imagery layers are never rescanned (and pastis's
  *_all fetch layers cannot trigger); their rasters stay byte-identical to
  what the published year-aligned numbers were measured on. Prepare jobs
  get the rate-limited regime defaults (fewer workers, short backoff);
  materialize the worker-bound ones. Dry-run unless ``--go``.
- ``run``    -- apply then launch.

Usage, from the helios repo root on a weka-mounted machine::

    python scripts/tools/setup_extra_layers.py plan
    python scripts/tools/setup_extra_layers.py run \
        --hosts jupiter-cs-aus-134.reviz.ai2.in,jupiter-cs-aus-137.reviz.ai2.in --go
    # later: watch progress / advance landsat prepare -> materialize
    python scripts/tools/setup_extra_layers.py plan --only descals
    python scripts/tools/setup_extra_layers.py run --only descals --hosts ... --go

Jobs are submitted with beaker-py from this repo's own environment, so the
launch machine only needs Beaker auth (BEAKER_TOKEN or ~/.beaker/config.yml).
Experiments get readable names (``landsat-<stage>-<dataset>-<suffix>``).
Jobs get AWS credentials injected from the workspace secrets named
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY -- the usgs-landsat bucket is requester-pays, so
prepare/materialize fail with NoCredentialsError without them. Rate limits
are per-account either way, so fan out across datasets, not by stacking
jobs on one dataset (identical jobs do not shard).

After materialize completes (re-run ``plan`` until progress is ~100%):

1. ``python scripts/tools/backfill_eval_registry_provenance.py`` and commit
   registry.json: the registry pins config.json by sha256
   (``verify_config_json_hash``), so every eval on an edited dataset FAILS
   LOUDLY until the new hash is recorded.
2. Regenerate + re-sync model.yaml (build_year_aligned_eval_configs.py adds
   the optional ``landsat`` input; the weka copy must match the repo copy for
   config_repo_dir provenance to hold).

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
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from upath import UPath

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "data" / "rslearn_dataset_configs"

MONTHS = 12
S2_LAYERS = [f"sentinel2_l2a_mo{i:02d}" for i in range(1, MONTHS + 1)]
LANDSAT_LAYERS = [f"landsat_mo{i:02d}" for i in range(1, MONTHS + 1)]
LAYER_SET = "landsat"

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

# Beaker submission settings (mirroring rslp's data-materialization launcher).
BEAKER_WORKSPACE = "ai2/earth-systems"
BEAKER_BUDGET = "ai2/atec-olmoearth"
WEKA_BUCKET = "dfive-default"
# Workspace secrets holding AWS credentials, injected into every job (the
# usgs-landsat bucket is requester-pays).
AWS_SECRET_NAMES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")

# tessera_v2_export.py adds a `<dataset>_tessera_v2` fetch group beside the
# eval windows (africa_crop_mask and ethiopia_crops carry one). Those windows
# have no monthly layers and are not part of any eval, so they are excluded
# from window listings and every Beaker job is scoped to the surviving groups
# -- otherwise a Landsat prepare/materialize would fetch a year of Landsat
# for thousands of fetch-only windows.
FETCH_GROUP_SUFFIX = "_tessera_v2"


# ---------------------------------------------------------------------------
# apply: config.json layers
# ---------------------------------------------------------------------------


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


def config_with_layers(config: dict) -> tuple[dict, list[str], list[str]]:
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
    for name, generated in build_landsat_layers().items():
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


def window_state(window: Window) -> str:
    """Classify one window's readiness (read-only).

    Has Beaker prepare written its twelve layer-data entries?
    """
    layer_datas = window.load_layer_datas()
    have = sum(1 for name in LANDSAT_LAYERS if name in layer_datas)
    if have == MONTHS:
        return "prepared"
    return "partially prepared" if have else "needs prepare"


def list_windows(ds_path: UPath, workers: int = 8) -> list[Window]:
    """List a dataset's eval windows.

    tessera_v2 fetch groups are excluded (see FETCH_GROUP_SUFFIX).
    """
    windows = Dataset(ds_path).storage.get_windows(workers=workers)
    return [w for w in windows if not w.group.endswith(FETCH_GROUP_SUFFIX)]


def apply_dataset(ds_path: UPath) -> None:
    """Write the Landsat layers into the dataset's config.json."""
    config_path = ds_path / "config.json"
    config = json.loads(config_path.read_text())
    config, added, present = config_with_layers(config)
    logger.info(f"  config.json: +{len(added)} layers ({len(present)} present)")
    if added:
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# plan / progress
# ---------------------------------------------------------------------------


def materialize_progress(sampled: list[Window]) -> str:
    """Completed-marker counts over a window sample.

    Months whose period matched no scene never get a marker, so mature
    datasets legitimately top out below 100% -- markedly so for Landsat,
    whose 16-day revisit leaves genuinely empty months. The authoritative
    completion signal is a re-run reporting nothing left to do.
    """
    done = sum(
        1 for w in sampled for name in LANDSAT_LAYERS if w.is_layer_completed(name)
    )
    return f"{done}/{len(sampled) * MONTHS} sampled layer-months"


def plan_dataset(ds_path: UPath, sample: int) -> None:
    """Report config state, window readiness and materialize progress."""
    config = json.loads((ds_path / "config.json").read_text())
    windows = list_windows(ds_path)
    random.seed(0)
    sampled = random.sample(windows, min(sample, len(windows)))
    print(f"  windows: {len(windows)} ({len(sampled)} sampled)")

    try:
        _, added, present = config_with_layers(config)
    except ValueError as e:
        print(f"  BLOCKED -- {e}")
        return
    states = Counter(window_state(w) for w in sampled)
    print(
        f"  config: {len(present)} present, {len(added)} to add"
        f" | windows: {dict(states)}"
        f" | materialized: {materialize_progress(sampled)}"
        f" | next launch: {launch_stage(states)}"
    )


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


def launch_stage(states: Counter) -> str:
    """Which rslearn command a launch would run now.

    Landsat needs a prepare pass first; once every sampled window that exists
    is prepared, the next stage is materialize.
    """
    return (
        "materialize"
        if states["needs prepare"] + states["partially prepared"] == 0
        else "prepare"
    )


def job_command(
    ds_path: UPath,
    stage: str,
    groups: list[str],
    args: argparse.Namespace,
) -> list[str]:
    """The rslearn command the Beaker job runs.

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
        str(ds_path),
        "--workers",
        str(workers),
        "--no-use-initial-job",
        "--retry-max-attempts",
        str(args.retry_attempts),
        "--retry-backoff-seconds",
        str(backoff),
        "--ignore-errors",
        # Only the Landsat layers do work; without this the job would also scan
        # the imagery layers (harmless but slow) and, on pastis, try to fetch
        # the *_all scene layers (max_matches 150 -- NOT harmless).
        "--enabled-layers",
        ",".join(LANDSAT_LAYERS),
    ]
    if groups:
        # Scope to the eval window groups (rslearn's --group takes several),
        # keeping the job off any tessera_v2 fetch group.
        command += ["--group", *groups]
    return command


def get_beaker_client() -> Any:
    """Create the Beaker client (lazy import so plan/apply need no beaker)."""
    from beaker import Beaker

    return Beaker.from_env(default_workspace=BEAKER_WORKSPACE)


def launch_job(
    beaker_client: Any,
    ds_path: UPath,
    stage: str,
    groups: list[str],
    host: str,
    args: argparse.Namespace,
) -> bool:
    """Submit (or print) one host-pinned Beaker job for one dataset.

    Host-pinned jobs reserve no GPU. Jobs carry AWS credentials from the
    workspace secrets in AWS_SECRET_NAMES (requester-pays bucket).
    """
    command = job_command(ds_path, stage, groups, args)
    name = f"{LAYER_SET}-{stage}-{ds_path.name}-{uuid.uuid4().hex[:8]}"

    if not args.go:
        print(f"\n  would submit {name} on {host}:\n    {' '.join(command)}")
        return True

    from beaker import Constraints, DataMount, DataSource, EnvVar, ExperimentSpec

    env_vars = [EnvVar(name=secret, secret=secret) for secret in AWS_SECRET_NAMES]
    spec = ExperimentSpec.new(
        budget=BEAKER_BUDGET,
        task_name=name,
        beaker_image=args.image,
        priority=args.priority,
        command=command,
        env_vars=env_vars,
        datasets=[
            DataMount(
                source=DataSource(weka=WEKA_BUCKET),
                mount_path=f"/weka/{WEKA_BUCKET}",
            )
        ],
        constraints=Constraints(hostname=[host]),
        preemptible=True,
    )
    logger.info(f"  submitting {name} on {host}")
    try:
        beaker_client.experiment.create(name, spec)
    except Exception:
        logger.exception(f"  FAILED to submit {name}")
        return False
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def resolve_targets(args: argparse.Namespace) -> list[tuple[str, UPath]]:
    """Resolve CLI args to (name, ds_path) targets.

    The nine eval datasets live under --stage_root; --only selects one by base
    name or full name.
    """
    stage_root = UPath(args.stage_root)
    targets: list[tuple[str, UPath]] = [(name, stage_root / name) for name in DATASETS]
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
    parser.add_argument("--only", default=None, help="One dataset (base or full name).")
    parser.add_argument("--stage_root", default=DEFAULT_STAGE_ROOT)
    parser.add_argument(
        "--sample", type=int, default=200, help="Readiness sample size."
    )
    # Launch targeting: hosts round-robin, one job per dataset (times
    # --jobs_per_dataset). Host-pinned jobs reserve no GPU.
    parser.add_argument("--hosts", default=None, help="Comma-separated Beaker hosts.")
    parser.add_argument(
        "--jobs_per_dataset",
        type=int,
        default=1,
        help="Identical jobs per dataset, rotated across hosts. "
        "Effective for MATERIALIZE: rslearn shuffles each job's window order, "
        "so concurrent jobs work mostly-disjoint windows and skip completed "
        "ones cheaply (some duplicated work near the tail). Near-useless for "
        "prepare, which is bound by per-account rate limits, not workers.",
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
    parser.add_argument(
        "--go",
        action="store_true",
        help="Actually submit Beaker jobs; without it launch/run print the commands.",
    )
    args = parser.parse_args()

    targets = resolve_targets(args)

    do_apply = args.command in ("apply", "run")
    do_launch = args.command in ("launch", "run")

    hosts = [h for h in (args.hosts or "").replace(",", " ").split() if h]
    beaker_client = None
    if do_launch:
        if not hosts:
            parser.error("launch needs --hosts")
        if not args.go:
            logger.info(
                "DRY RUN -- pass --go to actually write changes and submit "
                "Beaker jobs. Nothing is modified or launched without it."
            )
        else:
            beaker_client = get_beaker_client()

    failures = 0
    launched = 0
    for name, ds_path in targets:
        print(f"\n{name}:")
        if not ds_path.exists():
            print("  SKIP -- not staged")
            continue

        if args.command == "plan":
            plan_dataset(ds_path, args.sample)
            continue

        if do_apply:
            # In `run` mode --go gates the weka writes too, so a dry run is
            # fully read-only; the explicit `apply` command always executes.
            if args.command == "apply" or args.go:
                apply_dataset(ds_path)
            else:
                config = json.loads((ds_path / "config.json").read_text())
                try:
                    _, added, present = config_with_layers(config)
                except ValueError as e:
                    print(f"  BLOCKED -- {e}")
                    continue
                print(
                    f"  DRY RUN: would add {len(added)} config layers "
                    f"({len(present)} present)"
                )

        if do_launch:
            windows = list_windows(ds_path)
            if not windows:
                print("  SKIP -- no eval windows")
                continue
            groups = sorted({w.group for w in windows})
            random.seed(0)
            sampled = random.sample(windows, min(25, len(windows)))
            states = Counter(window_state(w) for w in sampled)
            stage = launch_stage(states)
            for _ in range(max(1, args.jobs_per_dataset)):
                host = hosts[launched % len(hosts)]
                if not launch_job(beaker_client, ds_path, stage, groups, host, args):
                    failures += 1
                launched += 1

    if args.command == "plan":
        print(
            "\nplan is read-only; `apply` edits configs, `launch --go` "
            "submits Beaker jobs (prepare or materialize, per readiness), "
            "`run --go` does both."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
