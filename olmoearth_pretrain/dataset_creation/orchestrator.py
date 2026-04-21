"""End-to-end corpus-dataset orchestrator (Beaker-native).

Pipeline (left-to-right, each modality runs sequentially to avoid config
contention on the dataset root):

    create_windows      (inline, local; metadata-only, cheap)
    ├── modality_1
    │   ├── prepare     (single Beaker task)
    │   ├── ingest      (single Beaker task, only if needed)
    │   ├── materialize (1..N sharded Beaker tasks)
    │   └── export      (single Beaker task -> rslearn_to_olmoearth.<mod>)
    ├── modality_2 ...
    └── osm_rasterize   (single Beaker task; depends on openstreetmap_export)

Why each modality gets its own `modality_root`
----------------------------------------------
rslearn reads configuration from ``{root}/config.json``. If we swapped a
single shared config.json in the dataset root between stages, a Beaker task
that is preempted and re-launched much later could execute against the
wrong config. Instead, for each modality we build a tiny sub-root
(`{ds_path}/modality_roots/{modality}/`) with its own `config.json` copied in
and a `windows -> ../../windows` symlink. Per-modality roots are stable for
the life of the dataset, so preemption/resume is safe.

Progress
--------
State is persisted to `{ds_path}/build_progress.json`. Each entry tracks
launched experiment ids and terminal state so restarts skip completed stages
and only relaunch missing ones. Live, in-flight progress flows the other
direction via heartbeat files under `{ds_path}/.heartbeats/` (written by
`HeartbeatWriter` inside each shard task) that the orchestrator aggregates
into each Beaker experiment's description.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from beaker import Beaker
from upath import UPath

from olmoearth_pretrain.dataset_creation.beaker_launcher import (
    BeakerJobConfig,
    classify_terminal_states,
    launch_beaker_task,
    set_experiment_description,
    wait_for_experiments,
)
from olmoearth_pretrain.dataset_creation.progress import (
    format_rollup,
    read_all_heartbeats,
)
from olmoearth_pretrain.dataset_creation.shard_windows import (
    enumerate_window_names,
    partition_windows,
    shard_file_path,
    write_shard_files,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalitySpec:
    """Static per-modality metadata driving pipeline execution."""

    name: str
    config_filename: str
    export_module: str  # dotted module under rslearn_to_olmoearth.*
    needs_ingest: bool
    layer_names: list[str]  # used for heartbeat "done" counting
    materialize_shards: int = 1

    @property
    def prepare_stage(self) -> str:
        return f"{self.name}__prepare"

    @property
    def ingest_stage(self) -> str:
        return f"{self.name}__ingest"

    @property
    def materialize_stage(self) -> str:
        return f"{self.name}__materialize"

    @property
    def export_stage(self) -> str:
        return f"{self.name}__export"


# Default modality table. Layer lists mirror the `data/rslearn_dataset_configs/`
# config JSONs. Sharding defaults to 1 (conservative, matches previous behavior);
# the user can bump individual modalities via the orchestrator CLI.
_S2_LAYERS = ["sentinel2_l2a_freq"] + [f"sentinel2_l2a_mo{m:02d}" for m in range(1, 13)]
_S1_LAYERS = ["sentinel1_freq"] + [f"sentinel1_mo{m:02d}" for m in range(1, 13)]
_LANDSAT_LAYERS = ["landsat_freq"] + [f"landsat_mo{m:02d}" for m in range(1, 13)]
_WC_LAYERS = [
    "tc-annual-temporarycrops-classification",
    "tc-maize-main-irrigation-classification",
    "tc-maize-main-maize-classification",
    "tc-maize-second-irrigation-classification",
    "tc-maize-second-maize-classification",
    "tc-springcereals-springcereals-classification",
    "tc-wintercereals-irrigation-classification",
    "tc-wintercereals-wintercereals-classification",
]

DEFAULT_MODALITIES: list[ModalitySpec] = [
    ModalitySpec("sentinel2_l2a", "config_sentinel2_l2a.json", "sentinel2_l2a", False, _S2_LAYERS),
    ModalitySpec("sentinel1", "config_sentinel1.json", "sentinel1", False, _S1_LAYERS),
    ModalitySpec("landsat", "config_landsat.json", "landsat", False, _LANDSAT_LAYERS),
    ModalitySpec("worldcover", "config_worldcover.json", "worldcover", True, ["worldcover"]),
    ModalitySpec("openstreetmap", "config_openstreetmap.json", "openstreetmap", True, ["openstreetmap"]),
    ModalitySpec("worldcereal", "config_worldcereal.json", "worldcereal", True, _WC_LAYERS),
    ModalitySpec("srtm", "config_srtm.json", "srtm", True, ["srtm"]),
    ModalitySpec("cdl", "config_cdl.json", "cdl", True, ["cdl"]),
    ModalitySpec("wri_canopy_height_map", "config_wri_canopy_height_map.json", "wri_canopy_height_map", True, ["wri_canopy_height_map"]),
]


@dataclass
class StageRecord:
    """Progress-file entry for a single stage."""

    experiment_ids: list[str] = field(default_factory=list)
    state: str = "pending"  # pending | running | succeeded | failed
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class OrchestratorConfig:
    """All the knobs for a pipeline run."""

    ds_path: UPath
    tiles_path: UPath
    configs_dir: Path
    job: BeakerJobConfig
    workers_per_task: int = 32
    corpus_csv: str | None = None  # studio corpus CSV (preferred for new runs)
    lonlats_json: str | None = None  # legacy lon/lat JSON list
    group: str = "res_10"
    init_config_filename: str = "config_corpus_init.json"
    poll_interval_s: float = 30.0
    stages_only: set[str] | None = None
    stages_skip: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not (bool(self.corpus_csv) ^ bool(self.lonlats_json)):
            raise ValueError("Exactly one of corpus_csv / lonlats_json must be set.")

    @property
    def run_prefix(self) -> str:
        """Short hash of ds_path used to namespace Beaker experiment names per run."""
        return hashlib.blake2b(str(self.ds_path).encode(), digest_size=4).hexdigest()


# ─── progress file helpers ────────────────────────────────────────────────


def _progress_path(ds_path: UPath) -> UPath:
    return ds_path / "build_progress.json"


def load_progress(ds_path: UPath) -> dict[str, StageRecord]:
    p = _progress_path(ds_path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: StageRecord(**v) for k, v in raw.items()}


def save_progress(ds_path: UPath, progress: dict[str, StageRecord]) -> None:
    p = _progress_path(ds_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.__dict__ for k, v in progress.items()}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ─── dataset root / shard scaffolding ─────────────────────────────────────


def _setup_modality_root(
    cfg: OrchestratorConfig, modality: ModalitySpec
) -> UPath:
    """Build ``{ds_path}/modality_roots/{modality}/`` with config.json + windows/ symlink.

    Idempotent: re-running overwrites `config.json` (cheap, identical content)
    and leaves the `windows` symlink alone.
    """
    root = cfg.ds_path / "modality_roots" / modality.name
    root.mkdir(parents=True, exist_ok=True)

    src_cfg = cfg.configs_dir / modality.config_filename
    dst_cfg = root / "config.json"
    shutil.copy2(src_cfg, dst_cfg)

    # rslearn scans `{root}/windows/`; point it at the shared windows dir.
    windows_link = root / "windows"
    if not windows_link.exists() and not windows_link.is_symlink():
        # Absolute target — relative symlinks via UPath are fragile across
        # mount points and UPath.symlink_to may not handle Path objects.
        windows_link.symlink_to(str(cfg.ds_path / "windows"))
    return root


def _setup_init_root(cfg: OrchestratorConfig) -> UPath:
    """Dataset root used during `create_windows` (uses the corpus_init config)."""
    root = cfg.ds_path
    src_cfg = cfg.configs_dir / cfg.init_config_filename
    dst_cfg = root / "config.json"
    root.mkdir(parents=True, exist_ok=True)
    if dst_cfg.is_symlink() or dst_cfg.exists():
        dst_cfg.unlink()
    shutil.copy2(src_cfg, dst_cfg)
    return root


# ─── stage: create_windows (inline, local) ────────────────────────────────


def _run_create_windows(cfg: OrchestratorConfig) -> None:
    """Invoke the unified create_windows CLI in the orchestrator process.

    Metadata-only operation — no heavy I/O — so there's no benefit to farming
    it out to Beaker. Runs against the top-level dataset with the init config.
    """
    _setup_init_root(cfg)
    cmd = [
        sys.executable,
        "-m",
        "olmoearth_pretrain.dataset_creation.create_windows",
        "--ds_path",
        str(cfg.ds_path),
        "--workers",
        str(cfg.workers_per_task),
    ]
    if cfg.corpus_csv:
        cmd += ["--corpus_csv", cfg.corpus_csv]
    else:
        cmd += ["--lonlats_json", str(cfg.lonlats_json)]
    logger.info("create_windows: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ─── stage: launch Beaker rslearn shards ──────────────────────────────────


_RSLEARN_RETRY_ARGS = [
    "--retry-max-attempts",
    "4",
    "--retry-backoff-seconds",
    "30",
    "--ignore-errors",
]


def _build_shard_command(
    cfg: OrchestratorConfig,
    modality: ModalitySpec,
    shard_idx: int,
    subcommand: str,
    shard_file: UPath | None,
) -> list[str]:
    """Build the container command for a single rslearn-shard Beaker task."""
    root = cfg.ds_path / "modality_roots" / modality.name
    cmd = [
        "python",
        "-u",
        "-m",
        "olmoearth_pretrain.dataset_creation.run_rslearn_shard",
        "--ds_path",
        str(cfg.ds_path),
        "--root",
        str(root),
        "--shard_id",
        f"{modality.name}-{subcommand}-{shard_idx:04d}",
        "--modality",
        modality.name,
        "--layer_names",
        *modality.layer_names,
        "--group",
        cfg.group,
        "--heartbeat_dir",
        str(cfg.ds_path / ".heartbeats"),
        "--rslearn_subcommand",
        subcommand,
    ]
    if shard_file is not None:
        cmd += ["--shard_file", str(shard_file)]
    cmd.append("--")
    cmd += [
        "--workers",
        str(cfg.workers_per_task),
        "--no-use-initial-job",
        *_RSLEARN_RETRY_ARGS,
    ]
    return cmd


def _build_export_command(cfg: OrchestratorConfig, modality: ModalitySpec) -> list[str]:
    """Command for the export stage (rslearn_to_olmoearth.<modality>)."""
    return [
        "python",
        "-u",
        "-m",
        f"olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.{modality.export_module}",
        "--ds_path",
        str(cfg.ds_path / "modality_roots" / modality.name),
        "--olmoearth_path",
        str(cfg.tiles_path),
        "--workers",
        str(cfg.workers_per_task),
    ]


def _launch_single_task_stage(
    beaker: Beaker,
    cfg: OrchestratorConfig,
    stage_name: str,
    task_name: str,
    description: str,
    command: list[str],
) -> str:
    """Launch a single-task Beaker experiment and return its id."""
    return launch_beaker_task(
        beaker,
        name=task_name,
        description=description,
        command=command,
        job=cfg.job,
    )


def _launch_sharded_materialize(
    beaker: Beaker,
    cfg: OrchestratorConfig,
    modality: ModalitySpec,
) -> list[str]:
    """Partition windows, persist shard files, launch one Beaker task per shard."""
    num_shards = max(1, modality.materialize_shards)
    window_names = enumerate_window_names(cfg.ds_path, cfg.group)
    if not window_names:
        raise RuntimeError(
            f"No windows found under {cfg.ds_path / 'windows' / cfg.group}. "
            "Did create_windows run?"
        )
    shards = partition_windows(window_names, num_shards)
    write_shard_files(cfg.ds_path, modality.name, shards)

    exp_ids: list[str] = []
    for i, names in enumerate(shards):
        if not names:
            continue
        sf = shard_file_path(cfg.ds_path, modality.name, i)
        cmd = _build_shard_command(
            cfg, modality, shard_idx=i, subcommand="materialize", shard_file=sf
        )
        exp_id = launch_beaker_task(
            beaker,
            name=f"{cfg.run_prefix}-{modality.name}-materialize-{i:04d}",
            description=(
                f"{modality.name} materialize shard {i}/{num_shards} "
                f"({len(names)} windows)"
            ),
            command=cmd,
            job=cfg.job,
        )
        exp_ids.append(exp_id)
    return exp_ids


# ─── stage runner ─────────────────────────────────────────────────────────


def _should_run(cfg: OrchestratorConfig, stage_name: str) -> bool:
    if stage_name in cfg.stages_skip:
        return False
    if cfg.stages_only is not None and stage_name not in cfg.stages_only:
        return False
    return True


def _run_beaker_stage(
    beaker: Beaker,
    cfg: OrchestratorConfig,
    stage_name: str,
    progress: dict[str, StageRecord],
    launch_fn,
) -> bool:
    """Launch (or resume) a Beaker stage and wait for all its experiments.

    `launch_fn(beaker, cfg) -> list[str]` must be idempotent w.r.t. experiment
    names so a retry after a partial launch reuses the existing experiments
    (beaker_launcher handles name collisions as resume).

    Returns True on success.
    """
    record = progress.get(stage_name, StageRecord())
    if record.state == "succeeded":
        logger.info("[%s] already succeeded, skipping", stage_name)
        return True

    if not record.experiment_ids:
        logger.info("[%s] launching", stage_name)
        record.started_at = record.started_at or _now_iso()
        record.experiment_ids = launch_fn(beaker, cfg)
        record.state = "running"
        progress[stage_name] = record
        save_progress(cfg.ds_path, progress)
    else:
        logger.info(
            "[%s] resuming; watching %d existing experiments",
            stage_name,
            len(record.experiment_ids),
        )

    # Each shard writes heartbeats independently; we aggregate them into the
    # PARENT (orchestrator) experiment description if we can find it. Each
    # child task experiment description is left as-is (static text set on
    # launch). Updating per-task descriptions in a tight loop is not worth the
    # API churn; the rollup is what a human reads.
    heartbeat_dir = cfg.ds_path / ".heartbeats"
    orchestrator_exp_id = _self_experiment_id()

    def _on_poll(states: dict[str, str]) -> None:
        summary = _stage_summary_line(stage_name, states)
        hbs = read_all_heartbeats(heartbeat_dir)
        body = format_rollup(hbs)
        description = f"{summary}\n\n{body}"
        if orchestrator_exp_id is not None:
            set_experiment_description(beaker, orchestrator_exp_id, description)
        logger.info("[%s] %s", stage_name, summary)

    terminal = wait_for_experiments(
        beaker,
        record.experiment_ids,
        poll_interval_s=cfg.poll_interval_s,
        on_poll=_on_poll,
    )
    succeeded, failed = classify_terminal_states(terminal)
    record.finished_at = _now_iso()
    if failed:
        record.state = "failed"
        progress[stage_name] = record
        save_progress(cfg.ds_path, progress)
        logger.error(
            "[%s] %d/%d experiments failed: %s",
            stage_name,
            len(failed),
            len(record.experiment_ids),
            failed,
        )
        return False
    record.state = "succeeded"
    progress[stage_name] = record
    save_progress(cfg.ds_path, progress)
    logger.info("[%s] all %d experiments succeeded", stage_name, len(succeeded))
    return True


def _stage_summary_line(stage_name: str, states: dict[str, str]) -> str:
    from collections import Counter

    c = Counter(states.values())
    pieces = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
    return f"{stage_name}: {len(states)} tasks ({pieces})"


def _self_experiment_id() -> str | None:
    """Beaker injects BEAKER_EXPERIMENT_ID when running as an experiment."""
    import os

    return os.environ.get("BEAKER_EXPERIMENT_ID")


# ─── top-level pipeline ───────────────────────────────────────────────────


def run(cfg: OrchestratorConfig, modalities: list[ModalitySpec]) -> bool:
    """Execute the full pipeline. Returns True if every requested stage succeeded."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    beaker = Beaker.from_env(default_workspace=cfg.job.workspace)
    progress = load_progress(cfg.ds_path)

    # Stage: create_windows.
    if _should_run(cfg, "create_windows"):
        if progress.get("create_windows", StageRecord()).state != "succeeded":
            _run_create_windows(cfg)
            progress["create_windows"] = StageRecord(
                state="succeeded",
                started_at=_now_iso(),
                finished_at=_now_iso(),
            )
            save_progress(cfg.ds_path, progress)

    all_ok = True
    for modality in modalities:
        _setup_modality_root(cfg, modality)
        # prepare (always single-task)
        if _should_run(cfg, modality.prepare_stage):
            all_ok &= _run_beaker_stage(
                beaker,
                cfg,
                modality.prepare_stage,
                progress,
                lambda b, c, m=modality: [
                    _launch_single_task_stage(
                        b,
                        c,
                        m.prepare_stage,
                        task_name=f"{c.run_prefix}-{m.name}-prepare",
                        description=f"{m.name} prepare",
                        command=_build_shard_command(
                            c, m, 0, "prepare", shard_file=None
                        ),
                    )
                ],
            )

        if modality.needs_ingest and _should_run(cfg, modality.ingest_stage):
            all_ok &= _run_beaker_stage(
                beaker,
                cfg,
                modality.ingest_stage,
                progress,
                lambda b, c, m=modality: [
                    _launch_single_task_stage(
                        b,
                        c,
                        m.ingest_stage,
                        task_name=f"{c.run_prefix}-{m.name}-ingest",
                        description=f"{m.name} ingest",
                        command=_build_shard_command(
                            c, m, 0, "ingest", shard_file=None
                        ),
                    )
                ],
            )

        if _should_run(cfg, modality.materialize_stage):
            all_ok &= _run_beaker_stage(
                beaker,
                cfg,
                modality.materialize_stage,
                progress,
                lambda b, c, m=modality: _launch_sharded_materialize(b, c, m),
            )

        if _should_run(cfg, modality.export_stage):
            all_ok &= _run_beaker_stage(
                beaker,
                cfg,
                modality.export_stage,
                progress,
                lambda b, c, m=modality: [
                    _launch_single_task_stage(
                        b,
                        c,
                        m.export_stage,
                        task_name=f"{c.run_prefix}-{m.name}-export",
                        description=f"{m.name} export to olmoearth tiles",
                        command=_build_export_command(c, m),
                    )
                ],
            )

    # OSM rasterization (post-export).
    if _should_run(cfg, "osm_rasterize"):
        all_ok &= _run_beaker_stage(
            beaker,
            cfg,
            "osm_rasterize",
            progress,
            lambda b, c: [
                _launch_single_task_stage(
                    b,
                    c,
                    "osm_rasterize",
                    task_name=f"{c.run_prefix}-osm-rasterize",
                    description="rasterize openstreetmap",
                    command=[
                        "python",
                        "-u",
                        "-m",
                        "olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.rasterize_openstreetmap",
                        "--olmoearth_path",
                        str(c.tiles_path),
                        "--workers",
                        str(c.workers_per_task),
                    ],
                )
            ],
        )

    _print_summary(progress)
    return all_ok


def _print_summary(progress: dict[str, StageRecord]) -> None:
    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    for name, rec in sorted(progress.items()):
        print(f"  [{rec.state:>9s}] {name}  ({len(rec.experiment_ids)} exps)")


# ─── dry-run helper ───────────────────────────────────────────────────────


def dry_run_plan(
    cfg: OrchestratorConfig, modalities: list[ModalitySpec]
) -> list[tuple[str, list[str]]]:
    """Return [(stage_name, command_argv)] for every task that `run(cfg)` would launch.

    Useful for a CLI `--dry_run`. Does not touch Beaker.
    """
    plan: list[tuple[str, list[str]]] = []
    plan.append(("create_windows", ["(runs inline in orchestrator)"]))
    for m in modalities:
        plan.append(
            (m.prepare_stage, _build_shard_command(cfg, m, 0, "prepare", None))
        )
        if m.needs_ingest:
            plan.append(
                (m.ingest_stage, _build_shard_command(cfg, m, 0, "ingest", None))
            )
        for i in range(max(1, m.materialize_shards)):
            sf = shard_file_path(cfg.ds_path, m.name, i)
            plan.append(
                (
                    f"{m.materialize_stage}[{i}]",
                    _build_shard_command(cfg, m, i, "materialize", sf),
                )
            )
        plan.append((m.export_stage, _build_export_command(cfg, m)))
    plan.append(
        (
            "osm_rasterize",
            [
                "python",
                "-m",
                "olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.rasterize_openstreetmap",
                "--olmoearth_path",
                str(cfg.tiles_path),
            ],
        )
    )
    return plan
