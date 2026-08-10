"""Stage the ``tessera_v2`` layer onto the remaining *_year_aligned datasets.

pastis (on pastis_rslearn), africa_crop_mask and ethiopia_crops already carry
a ``tessera_v2`` layer produced by running the released v2 student ourselves
(docs/TesseraV2Inference.md;
olmoearth_pretrain/evals/datasets/tessera_v2_export.py). This script drives
that same pipeline across all eight supplemental ``*_year_aligned`` datasets,
so the remaining six -- canada_crops_coarse, canada_crops_fine, descals,
glance, lcmap_lu, us_trees -- get the identical treatment from one command
instead of six hand-run runbooks. The two finished datasets stay in the list
as living fixtures: ``plan`` should read them as ``done``, and if it does not,
the stage detection is wrong, not the datasets.

Stages, detected per dataset from what is already on weka, so re-running
``run --go`` (or ``plan`` to watch) advances the pipeline exactly like
setup_extra_layers.py does for SCL/Landsat:

  apply        write ``<ds>/config_tessera_v2_fetch.json`` and mirror every
               eval window into the ``<base>_tessera_v2`` fetch group on that
               window's own calendar year. All eight datasets are re-anchored
               to (Jan 1 Y, Jan 1 Y+1), so the year is read per window --
               check the year histogram create_windows logs against the label
               years before launching the fetch.
  prepare      one Beaker CPU job per dataset: ``rslearn dataset prepare
               --config <fetch config> --group <fetch group>`` in the
               rate-limited regime (16 workers, 2s backoff; a long backoff
               parks a worker for minutes on its first routine 403 and
               collapses parallelism -- see launch_year_aligned_prepare.sh).
               Planetary Computer's rate limit is per-account, so fan out
               across datasets, never by stacking jobs on one dataset.
  materialize  same job shape, worker-bound regime (64 workers, 60s backoff);
               ``--jobs_per_dataset`` stacks cooperating jobs (rslearn
               shuffles each job's window order, so they mostly avoid each
               other and skip completed windows cheaply).
  infer        NOT submitted -- the exact command is printed. Inference needs
               this repo's environment plus one GPU (it is I/O bound: ~50 min
               per 2.5k windows on an L40, so budget ~15 h for us_trees) and
               reads scenes from the staging tree while writing the layer +
               manifest into the registered eval tree (``--eval_ds_path``).
               The student is pinned to **large** -- every dataset must match
               pastis/africa/ethiopia or the numbers are not one Tessera
               column.
  wire         NOT run -- the wire_embedding_modalities.py and
               backfill_eval_registry_provenance.py commands are printed once
               the manifest exists. ``--required`` is included only when the
               manifest's ``num_windows_failed`` is 0 (the wiring script
               hard-blocks on any failure anyway); a nonzero count usually
               means scenes that 404 permanently at the source, which is what
               ``infer --allow_unmaterialized_s1`` is for (S1 only, recorded
               in the manifest).

Scale, before launching all six at once: they total ~120k+ eval windows
against the ~5.1k of africa+ethiopia combined (canada_coarse 16.1k,
canada_fine 14.6k, descals 16.7k, lcmap_lu 26.5k, us_trees 45.4k, glance
TBD -- ``plan`` prints the counts). The africa+ethiopia fetch was ~15 GB in
~3M small files and took ~12 h of materialize wall clock on 4 jobs, so expect
~25x that: ~350 GB, tens of millions of inodes, and days of materialize
unless it is fanned out hard. The fetch groups are disposable -- delete
``<stage>/<ds>/windows/<base>_tessera_v2`` once inference has written the
manifest, or the inode bill stays forever.

Usage, from the helios repo root on a weka-mounted machine::

    python scripts/tools/setup_tessera_v2.py plan
    python scripts/tools/setup_tessera_v2.py run --hosts <h1>,<h2> --go
    # later, as stages complete (re-run to advance prepare -> materialize):
    python scripts/tools/setup_tessera_v2.py plan --only us_trees
    python scripts/tools/setup_tessera_v2.py run --only us_trees \
        --jobs_per_dataset 4 --hosts ... --go

Jobs are submitted natively with beaker-py from this repo's environment
(Beaker auth via BEAKER_TOKEN or ~/.beaker/config.yml); experiments are named
``tesserav2-<stage>-<dataset>-<suffix>`` and host-pinned, so no GPU is
reserved. The fetch layers live in the standalone ``--config`` file and never
enter the dataset's own config.json, so its hash -- and every registered
eval -- is untouched until the wiring step deliberately changes model.yaml
and the registry.
"""

import argparse
import json
import logging
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rslearn.dataset import Dataset
from upath import UPath

from olmoearth_pretrain.evals.datasets.tessera_v2_export import (
    FETCH_CONFIG_NAME,
    FETCH_LAYERS,
    YEAR_ALIGNED_DATASETS,
    create_windows,
    resolve_spec,
    write_fetch_config,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "data" / "rslearn_dataset_configs"

DATASETS = tuple(f"{name}_year_aligned" for name in YEAR_ALIGNED_DATASETS)

DEFAULT_STAGE_ROOT = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals"
DEFAULT_EVAL_ROOT = "/weka/dfive-default/olmoearth/eval_datasets"
DEFAULT_CHECKPOINT = (
    "/weka/dfive-default/helios/models/tessera_v2/ckpt/student_large.pt"
)
# pastis/africa/ethiopia were run with the large student; every further
# dataset must match or the eight are not one Tessera column.
MODEL_SIZE = "large"
MANIFEST_NAME = "embedding_materializer_manifest_tessera_v2.json"

DEFAULT_IMAGE = "favyen/rslpomp20260727a"
BEAKER_WORKSPACE = "ai2/earth-systems"
BEAKER_BUDGET = "ai2/atec-olmoearth"
WEKA_BUCKET = "dfive-default"

STAGES = ("apply", "prepare", "materialize", "infer", "wire", "done")


@dataclass
class DatasetState:
    """Everything ``plan``/``launch`` need to know about one dataset."""

    has_fetch_config: bool
    num_eval: int
    num_fetch: int
    # Over the sampled fetch windows:
    num_sampled: int
    num_unprepared: int  # windows missing any of the three fetch layers
    num_item_groups: int  # prepared scenes (item groups) across the sample
    num_completed: int  # materialized scenes across the sample
    manifest: dict[str, Any] | None  # from the registered eval tree
    wired: bool  # repo model.yaml declares a tessera_v2 input

    @property
    def materialized_fraction(self) -> float:
        """Materialized share of the sampled prepared scenes."""
        return (
            self.num_completed / self.num_item_groups if self.num_item_groups else 0.0
        )


def read_dataset_state(
    stage_path: UPath, eval_path: UPath, name: str, sample: int, workers: int
) -> DatasetState:
    """Inspect one dataset's staging + eval trees (read-only)."""
    spec = resolve_spec(name)
    windows = Dataset(stage_path).storage.get_windows(workers=workers)
    eval_windows = [w for w in windows if w.group != spec.fetch_group]
    fetch_windows = [w for w in windows if w.group == spec.fetch_group]

    random.seed(0)
    sampled = random.sample(fetch_windows, min(sample, len(fetch_windows)))
    unprepared = 0
    item_groups = 0
    completed = 0
    for window in sampled:
        layer_datas = window.load_layer_datas()
        if any(layer not in layer_datas for layer in FETCH_LAYERS):
            unprepared += 1
            continue
        item_groups += sum(
            len(layer_datas[layer].serialized_item_groups) for layer in FETCH_LAYERS
        )
        completed += sum(
            1 for layer, _ in window.list_completed_layers() if layer in FETCH_LAYERS
        )

    manifest_path = eval_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    model_yaml = CONFIG_ROOT / name / "model.yaml"
    wired = model_yaml.exists() and "tessera_v2" in model_yaml.read_text()

    return DatasetState(
        has_fetch_config=(stage_path / FETCH_CONFIG_NAME).exists(),
        num_eval=len(eval_windows),
        num_fetch=len(fetch_windows),
        num_sampled=len(sampled),
        num_unprepared=unprepared,
        num_item_groups=item_groups,
        num_completed=completed,
        manifest=manifest,
        wired=wired,
    )


def next_stage(state: DatasetState, materialize_done_fraction: float) -> str:
    """Which stage the dataset needs next.

    The materialize threshold is a sampled fraction, not equality: scenes that
    404 permanently at the source never materialize (africa lost 3 windows'
    ascending S1 that way), and months of progress should not hide behind
    them. ``infer`` is strict about unmaterialized-but-prepared layers anyway,
    so declaring materialize done a touch early fails loudly there, not
    silently.
    """
    if not state.has_fetch_config or state.num_fetch < state.num_eval:
        return "apply"
    if state.num_unprepared:
        return "prepare"
    if state.materialized_fraction < materialize_done_fraction:
        return "materialize"
    manifest = state.manifest
    if manifest is None:
        return "infer"
    have = manifest["num_windows_written"] + manifest["num_windows_skipped_existing"]
    if have < state.num_eval or manifest["num_windows_failed"]:
        return "infer"
    return "done" if state.wired else "wire"


# ---------------------------------------------------------------------------
# printed next-step commands (infer needs a GPU + this repo; wire edits the
# repo and registry -- neither belongs in a fire-and-forget Beaker job)
# ---------------------------------------------------------------------------


def infer_command(
    name: str, stage_path: UPath, eval_path: UPath, checkpoint: str
) -> str:
    """The GPU-session command that writes the layer + manifest."""
    return (
        "python -m olmoearth_pretrain.evals.datasets.tessera_v2_export infer "
        f"--ds_path {stage_path} --eval_ds_path {eval_path} --dataset {name} "
        f"--checkpoint_path {checkpoint} --model_size {MODEL_SIZE}"
    )


def wire_commands(name: str, state: DatasetState) -> list[str]:
    """The repo-side wiring commands, with --required only on a clean bake."""
    manifest = state.manifest
    required = " --required" if manifest and not manifest["num_windows_failed"] else ""
    commands = [
        "python scripts/tools/wire_embedding_modalities.py "
        f"--datasets {name} --products tessera_v2{required}",
        "python scripts/tools/backfill_eval_registry_provenance.py",
    ]
    if not required:
        commands.insert(
            0,
            f"# {manifest['num_windows_failed'] if manifest else '?'} windows failed "
            "-- retry `infer` first (existing layers are skipped); --required "
            "left off deliberately",
        )
    return commands


def print_next_steps(
    name: str,
    stage: str,
    state: DatasetState,
    stage_path: UPath,
    eval_path: UPath,
    checkpoint: str,
) -> None:
    """Print the manual command(s) for the infer/wire/done stages."""
    if stage == "infer":
        failed = state.manifest["num_windows_failed"] if state.manifest else None
        note = f" (retrying {failed} failed windows)" if failed else ""
        print(
            f"  next (GPU session){note}:\n    {infer_command(name, stage_path, eval_path, checkpoint)}"
        )
    elif stage == "wire":
        print("  next (repo side, then commit registry.json + model.yaml and")
        print("  re-sync the weka model.yaml copy):")
        for command in wire_commands(name, state):
            print(f"    {command}")
    elif stage == "done":
        print(
            f"  fetch group is disposable now: {stage_path}/windows/"
            f"{resolve_spec(name).fetch_group} (millions of small files)"
        )


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def apply_dataset(name: str, stage_path: UPath, state: DatasetState) -> None:
    """Write the fetch config and mirror the eval windows into the fetch group.

    Both halves are idempotent: the config is regenerated deterministically
    from the dataset's own config.json + the shared layer file, and
    create_windows rewrites each fetch window's metadata from its eval
    sibling (same grid, same per-window year) without touching items.json.
    """
    if not state.has_fetch_config:
        write_fetch_config(str(stage_path))
    else:
        logger.info(f"  {FETCH_CONFIG_NAME} already present")
    if state.num_fetch < state.num_eval:
        # Logs the year histogram -- eyeball it against the label years.
        create_windows(str(stage_path), resolve_spec(name))
    else:
        logger.info(f"  fetch group complete ({state.num_fetch} windows)")


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


def job_command(
    stage_path: UPath, stage: str, fetch_group: str, args: argparse.Namespace
) -> list[str]:
    """The rslearn command the Beaker job runs.

    prepare and materialize are bound by different resources (rate limit vs
    workers), hence the split worker/backoff defaults -- measured reasoning in
    launch_year_aligned_prepare.sh. The standalone --config carries ONLY the
    three *_all layers, so no --enabled-layers is needed and the dataset's own
    monthly layers cannot be rescanned; --group keeps the job off the ~10-45k
    eval windows, for which a year-of-scenes fetch would be ruinous.
    """
    if stage == "prepare":
        workers, backoff = args.prepare_workers, args.prepare_retry_backoff
    else:
        workers, backoff = args.workers, args.retry_backoff
    return [
        "rslearn",
        "dataset",
        stage,
        "--root",
        str(stage_path),
        "--config",
        str(stage_path / FETCH_CONFIG_NAME),
        "--group",
        fetch_group,
        "--workers",
        str(workers),
        "--no-use-initial-job",
        "--retry-max-attempts",
        str(args.retry_attempts),
        "--retry-backoff-seconds",
        str(backoff),
        "--ignore-errors",
    ]


def get_beaker_client() -> Any:
    """Create the Beaker client (lazy import so plan/apply need no beaker)."""
    from beaker import Beaker

    return Beaker.from_env(default_workspace=BEAKER_WORKSPACE)


def launch_job(
    beaker_client: Any,
    stage_path: UPath,
    stage: str,
    fetch_group: str,
    host: str,
    args: argparse.Namespace,
) -> bool:
    """Submit (or print) one host-pinned Beaker job. Pinned jobs hold no GPU."""
    command = job_command(stage_path, stage, fetch_group, args)
    name = f"tesserav2-{stage}-{stage_path.name}-{uuid.uuid4().hex[:8]}"

    if not args.go:
        print(f"\n  would submit {name} on {host}:\n    {' '.join(command)}")
        return True

    from beaker import Constraints, DataMount, DataSource, ExperimentSpec

    spec = ExperimentSpec.new(
        budget=BEAKER_BUDGET,
        task_name=name,
        beaker_image=args.image,
        priority=args.priority,
        command=command,
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


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout
    )
    logging.getLogger("rslearn.dataset.storage.file").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["plan", "apply", "launch", "run"])
    parser.add_argument("--only", default=None, help="One dataset (base or full name).")
    parser.add_argument("--stage_root", default=DEFAULT_STAGE_ROOT)
    parser.add_argument(
        "--eval_root",
        default=DEFAULT_EVAL_ROOT,
        help="Registered (ingested) tree the layer + manifest are written to.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--sample", type=int, default=200, help="Fetch-readiness sample size."
    )
    parser.add_argument(
        "--materialize_done_fraction",
        type=float,
        default=0.999,
        help="Sampled scene fraction above which materialize counts as done "
        "(exact completion is unreachable when source scenes 404 permanently).",
    )
    parser.add_argument("--list_workers", type=int, default=16)
    parser.add_argument("--hosts", default=None, help="Comma-separated Beaker hosts.")
    parser.add_argument(
        "--jobs_per_dataset",
        type=int,
        default=1,
        help="Identical jobs per dataset, rotated across hosts. Effective for "
        "MATERIALIZE (rslearn shuffles window order, so jobs cooperate); "
        "useless for prepare, which is rate-limit-bound.",
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
        help="Actually write windows/configs and submit Beaker jobs; without "
        "it launch/run are dry runs.",
    )
    args = parser.parse_args()

    names = list(DATASETS)
    if args.only is not None:
        matches = [n for n in names if n in (args.only, f"{args.only}_year_aligned")]
        if not matches:
            parser.error(f"--only {args.only!r} matches none of {names}")
        names = matches

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
    for name in names:
        stage_path = UPath(args.stage_root) / name
        eval_path = UPath(args.eval_root) / name
        print(f"\n{name}:")
        if not stage_path.exists():
            print("  SKIP -- not staged")
            continue

        state = read_dataset_state(
            stage_path, eval_path, name, args.sample, args.list_workers
        )
        stage = next_stage(state, args.materialize_done_fraction)
        print(
            f"  eval windows: {state.num_eval} | fetch windows: {state.num_fetch}"
            f" | unprepared: {state.num_unprepared}/{state.num_sampled} sampled"
            f" | materialized: {state.num_completed}/{state.num_item_groups}"
            " sampled scenes"
            f" | manifest: "
            + (
                "none"
                if state.manifest is None
                else (
                    f"{state.manifest['num_windows_written']} written / "
                    f"{state.manifest['num_windows_skipped_existing']} skipped / "
                    f"{state.manifest['num_windows_failed']} failed"
                )
            )
            + f" | wired: {state.wired} | next: {stage}"
        )

        if args.command == "plan":
            print_next_steps(name, stage, state, stage_path, eval_path, args.checkpoint)
            continue

        if do_apply and stage == "apply":
            # In `run` mode --go gates the weka writes too, so a dry run is
            # fully read-only; the explicit `apply` command always executes.
            if args.command == "apply" or args.go:
                apply_dataset(name, stage_path, state)
                state = read_dataset_state(
                    stage_path, eval_path, name, args.sample, args.list_workers
                )
                stage = next_stage(state, args.materialize_done_fraction)
            else:
                print(
                    "  DRY RUN: would write the fetch config and create "
                    f"{state.num_eval - state.num_fetch} fetch windows"
                )
                stage = "prepare"

        if not do_launch:
            continue
        if stage not in ("prepare", "materialize"):
            print_next_steps(name, stage, state, stage_path, eval_path, args.checkpoint)
            continue
        jobs = args.jobs_per_dataset if stage == "materialize" else 1
        for _ in range(max(1, jobs)):
            host = hosts[launched % len(hosts)]
            if not launch_job(
                beaker_client,
                stage_path,
                stage,
                resolve_spec(name).fetch_group,
                host,
                args,
            ):
                failures += 1
            launched += 1

    if args.command == "plan":
        print(
            "\nplan is read-only; `apply` writes fetch configs/windows, "
            "`launch --go` submits Beaker prepare/materialize jobs per "
            "readiness, `run --go` does both. infer (GPU) and wire (repo) "
            "stay manual -- their commands are printed when a dataset "
            "reaches them."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
