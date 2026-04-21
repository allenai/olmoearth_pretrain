"""Build a corpus-driven rslearn → olmoearth dataset end-to-end on Beaker.

Three subcommands:

    dry_run     Print the full set of planned Beaker tasks.
    run         Execute the pipeline (all Beaker tasks, wait, update progress).
                Typically invoked inside the orchestrator-as-experiment.
    launch      Submit the orchestrator itself as a Beaker experiment.

Example::

    # 1. Build + push the Docker image first (one-shot):
    scripts/tools/build_dataset_creation_image.sh

    # 2. Inspect the plan (local, no Beaker calls):
    python scripts/tools/build_corpus_dataset.py dry_run \
        --corpus_csv /weka/.../pretraining_corpus.csv \
        --ds_path   /weka/.../rslearn_ds \
        --tiles_path /weka/.../tiles

    # 3. Launch the orchestrator on Beaker:
    python scripts/tools/build_corpus_dataset.py launch \
        --corpus_csv /weka/.../pretraining_corpus.csv \
        --ds_path    /weka/.../rslearn_ds \
        --tiles_path /weka/.../tiles \
        --beaker_image hankh/oep-dataset-creation-<sha> \
        --clusters ai2/jupiter-cirrascale-2

    # 4. Inside the Beaker task, the image re-invokes `run` automatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

from upath import UPath

from olmoearth_pretrain.dataset_creation.beaker_launcher import (
    DEFAULT_BEAKER_IMAGE,
    BeakerJobConfig,
)
from olmoearth_pretrain.dataset_creation.orchestrator import (
    DEFAULT_MODALITIES,
    ModalitySpec,
    OrchestratorConfig,
    dry_run_plan,
    run,
)

DEFAULT_CLUSTERS = (
    "ai2/jupiter-cirrascale-2,ai2/saturn-cirrascale,ai2/ceres-cirrascale"
)


def _common_parser_opts(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus_csv", help="Studio corpus CSV with lon/lat/start_time")
    src.add_argument("--lonlats_json", help="Legacy [lon,lat] JSON list")

    p.add_argument("--ds_path", required=True, help="rslearn dataset root path")
    p.add_argument("--tiles_path", required=True, help="olmoearth tiles output path")
    p.add_argument(
        "--configs_dir",
        default="data/rslearn_dataset_configs",
        help="directory containing config_*.json modality configs",
    )
    p.add_argument("--workers_per_task", type=int, default=32)
    p.add_argument(
        "--materialize_shards",
        type=int,
        default=1,
        help="Number of shards per modality's materialize stage (default 1).",
    )
    p.add_argument("--group", default="res_10")
    p.add_argument(
        "--init_config",
        default="config_corpus_init.json",
        help="Config used by create_windows (defines the window grid).",
    )
    p.add_argument("--only", nargs="*", default=None, help="Only run these stages.")
    p.add_argument("--skip", nargs="*", default=[], help="Skip these stages.")


def _beaker_parser_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--beaker_image", default=DEFAULT_BEAKER_IMAGE,
                    help=f"Beaker image (default: {DEFAULT_BEAKER_IMAGE})")
    p.add_argument("--clusters", default=DEFAULT_CLUSTERS)
    p.add_argument("--workspace", default="ai2/earth-systems")
    p.add_argument("--budget", default="ai2/d5")
    p.add_argument("--priority", default="normal")
    p.add_argument("--no_preemptible", action="store_true")
    p.add_argument(
        "--beaker_token_secret",
        default=None,
        help=(
            "Override the Beaker secret name injected as BEAKER_TOKEN. "
            "Auto-detected as '<whoami>_BEAKER_TOKEN' if not provided."
        ),
    )
    p.add_argument(
        "--pc_subscription_secret",
        default=None,
        help=(
            "Optional Beaker secret name exposed as PC_SDK_SUBSCRIPTION_KEY "
            "for higher Planetary Computer rate limits (not required)."
        ),
    )


def _resolve_beaker_secrets() -> tuple[str, str, str]:
    """Auto-detect Beaker username and derive secret names.

    Returns (beaker_token_secret, github_token_secret, username).
    Follows the convention in olmoearth_pretrain/internal/common.py:
    ``<username>_BEAKER_TOKEN``, ``<username>_GITHUB_TOKEN``.
    """
    from beaker import Beaker

    beaker = Beaker.from_env()
    username = beaker.account.whoami().name
    print(f"Auto-detected beaker user: {username}")
    return f"{username}_BEAKER_TOKEN", f"{username}_GITHUB_TOKEN", username


def _get_git_info() -> tuple[str, str, str]:
    """Get the current git ref, branch, and repo URL."""
    ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    print(f"Git: {url} @ {branch} ({ref[:8]})")
    return ref, branch, url


def _build_cfg(args: argparse.Namespace) -> tuple[OrchestratorConfig, list[ModalitySpec]]:
    env_secrets: dict[str, str] = {}
    git_ref, git_branch, git_repo_url = "", "", ""

    if getattr(args, "command", None) in ("run", "launch"):
        beaker_token_secret, github_token_secret, _ = _resolve_beaker_secrets()
        env_secrets["BEAKER_TOKEN"] = beaker_token_secret
        env_secrets["GITHUB_TOKEN"] = github_token_secret
        git_ref, git_branch, git_repo_url = _get_git_info()

    if getattr(args, "pc_subscription_secret", None):
        env_secrets["PC_SDK_SUBSCRIPTION_KEY"] = args.pc_subscription_secret

    job = BeakerJobConfig(
        beaker_image=getattr(args, "beaker_image", DEFAULT_BEAKER_IMAGE),
        clusters=tuple(getattr(args, "clusters", DEFAULT_CLUSTERS).split(",")),
        workspace=getattr(args, "workspace", "ai2/earth-systems"),
        budget=getattr(args, "budget", "ai2/d5"),
        priority=getattr(args, "priority", "normal"),
        preemptible=not getattr(args, "no_preemptible", False),
        git_ref=git_ref,
        git_branch=git_branch,
        git_repo_url=git_repo_url,
        env_secrets=env_secrets,
    )
    modalities = [
        ModalitySpec(
            name=m.name,
            config_filename=m.config_filename,
            export_module=m.export_module,
            needs_ingest=m.needs_ingest,
            layer_names=m.layer_names,
            materialize_shards=args.materialize_shards,
        )
        for m in DEFAULT_MODALITIES
    ]
    cfg = OrchestratorConfig(
        ds_path=UPath(args.ds_path),
        tiles_path=UPath(args.tiles_path),
        configs_dir=Path(args.configs_dir),
        job=job,
        workers_per_task=args.workers_per_task,
        corpus_csv=args.corpus_csv,
        lonlats_json=args.lonlats_json,
        group=args.group,
        init_config_filename=args.init_config,
        stages_only=set(args.only) if args.only else None,
        stages_skip=set(args.skip),
    )
    return cfg, modalities


# ─── subcommand: dry_run ──────────────────────────────────────────────────


def _cmd_dry_run(args: argparse.Namespace) -> int:
    cfg, modalities = _build_cfg(args)
    plan = dry_run_plan(cfg, modalities)
    print(f"Planned stages: {len(plan)}")
    for stage, cmd in plan:
        print(f"\n[{stage}]")
        print("  " + " ".join(cmd))
    return 0


# ─── subcommand: run ──────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace) -> int:
    cfg, modalities = _build_cfg(args)
    ok = run(cfg, modalities)
    return 0 if ok else 1


# ─── subcommand: launch ───────────────────────────────────────────────────


def _cmd_launch(args: argparse.Namespace) -> int:
    """Submit the orchestrator itself as a single Beaker experiment.

    The submitted task re-invokes this same script with `run` and the same
    arguments (minus the launch-specific flags). That task is the one that
    launches child experiments and monitors them.
    """
    from beaker import Beaker

    from olmoearth_pretrain.dataset_creation.beaker_launcher import (
        BeakerJobConfig,
        launch_beaker_task,
    )

    cfg, _ = _build_cfg(args)  # also validates args
    beaker = Beaker.from_env(default_workspace=cfg.job.workspace)

    # Re-invoke ourselves inside the container with `run` + same args.
    run_cmd = [
        "python",
        "-u",
        "scripts/tools/build_corpus_dataset.py",
        "run",
    ]
    passthrough = _forward_args(args, drop={"command", "no_preemptible"})
    run_cmd.extend(passthrough)

    suffix = uuid.uuid4().hex[:6]
    exp_id = launch_beaker_task(
        beaker,
        name=f"oep-build-corpus-{suffix}",
        description=(
            f"Orchestrator: build corpus dataset @ {args.ds_path} "
            f"({args.materialize_shards} mat-shards/mod)"
        ),
        command=run_cmd,
        job=cfg.job,
        cpu_count=2,
    )
    print(f"Launched orchestrator experiment {exp_id}")
    return 0


def _forward_args(args: argparse.Namespace, drop: set[str]) -> list[str]:
    """Re-serialize argparse Namespace back to a flag list.

    Used so `launch` can re-invoke `run` inside the Beaker container with the
    exact same user-supplied arguments. Not every Namespace field is a CLI
    flag (e.g. `command`), so `drop` names those to omit.
    """
    out: list[str] = []
    for k, v in vars(args).items():
        if k in drop or v is None:
            continue
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
            continue
        if isinstance(v, list):
            if v:
                out.append(f"--{k}")
                out.extend(str(x) for x in v)
            continue
        out.append(f"--{k}")
        out.append(str(v))
    return out


# ─── entrypoint ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Execute the pipeline (launches Beaker tasks).")
    _common_parser_opts(run_p)
    _beaker_parser_opts(run_p)

    dry_p = sub.add_parser("dry_run", help="Print the planned tasks and exit.")
    _common_parser_opts(dry_p)
    dry_p.add_argument("--beaker_image", default="<dry-run-image>")
    dry_p.add_argument("--clusters", default=DEFAULT_CLUSTERS)
    dry_p.add_argument("--workspace", default="ai2/earth-systems")
    dry_p.add_argument("--budget", default="ai2/d5")
    dry_p.add_argument("--priority", default="normal")
    dry_p.add_argument("--no_preemptible", action="store_true")
    dry_p.add_argument("--beaker_token_secret", default=None)
    dry_p.add_argument("--pc_subscription_secret", default=None)

    launch_p = sub.add_parser("launch", help="Submit the orchestrator as a Beaker task.")
    _common_parser_opts(launch_p)
    _beaker_parser_opts(launch_p)

    return p


def main() -> None:
    """CLI entrypoint."""
    args = _build_parser().parse_args()
    if args.command == "dry_run":
        sys.exit(_cmd_dry_run(args))
    if args.command == "run":
        sys.exit(_cmd_run(args))
    if args.command == "launch":
        sys.exit(_cmd_launch(args))
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
