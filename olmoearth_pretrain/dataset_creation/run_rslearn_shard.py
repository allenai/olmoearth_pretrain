"""Worker-side entrypoint for a single rslearn shard task on Beaker.

Responsibilities (deliberately minimal):

1. Spawn a `HeartbeatWriter` background thread so the parent orchestrator can
   observe live progress via weka (no Beaker API calls from the worker).
2. Invoke the `rslearn` CLI with the provided subcommand + args.
3. Translate the subprocess exit code into our own exit code + capture error
   string into the final heartbeat.

Deliberately NOT responsible for:

* Config management. The orchestrator prepares per-modality dataset roots
  (with the right `config.json`) before launching shards. The `--root` passed
  to rslearn points at that pre-built root. This makes shards preemption-safe:
  if Beaker re-runs a task much later, the root still has the correct config.
* Sharding itself. Shards come in pre-computed as a newline-delimited file of
  window names.

Invocation (invoked by the orchestrator as a Beaker task command)::

    python -m olmoearth_pretrain.dataset_creation.run_rslearn_shard \
        --ds_path /weka/.../rslearn_ds \
        --root /weka/.../rslearn_ds/modality_roots/sentinel2_l2a \
        --shard_id s2-shard-0 \
        --modality sentinel2_l2a \
        --layer_name sentinel2 \
        --group res_10 \
        --shard_file /weka/.../rslearn_ds/.beaker_shards/sentinel2_l2a/shard_0000.txt \
        --heartbeat_dir /weka/.../rslearn_ds/.heartbeats \
        --rslearn_subcommand materialize \
        -- \
        --workers 32 --ignore-errors --retry-max-attempts 4 --retry-backoff-seconds 30

Everything after the lone ``--`` is passed to rslearn verbatim (after we
inject `--root`, `--group`, and `--window ...`).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from upath import UPath

from olmoearth_pretrain.dataset_creation.progress import HeartbeatWriter
from olmoearth_pretrain.dataset_creation.shard_windows import load_shard_file

logger = logging.getLogger(__name__)


def _build_rslearn_cmd(
    subcommand: str,
    root: str,
    group: str,
    window_names: list[str],
    passthrough: list[str],
) -> list[str]:
    cmd = [
        "rslearn",
        "dataset",
        subcommand,
        "--root",
        root,
        "--group",
        group,
    ]
    if window_names:
        cmd.append("--window")
        cmd.extend(window_names)
    cmd.extend(passthrough)
    return cmd


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split our args from rslearn passthrough args at the first lone ``--``."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_rslearn_shard",
        description="Run a single sharded rslearn stage with heartbeat reporting.",
    )
    p.add_argument("--ds_path", required=True, help="top-level rslearn dataset path")
    p.add_argument("--root", required=True, help="rslearn --root (has config.json)")
    p.add_argument("--shard_id", required=True, help="unique id for this shard task")
    p.add_argument("--modality", required=True, help="e.g. sentinel2_l2a")
    p.add_argument(
        "--layer_names",
        required=True,
        nargs="+",
        help="layer(s) whose completion markers we count (window done <=> all layers done)",
    )
    p.add_argument("--group", required=True, help="window group, e.g. res_10")
    p.add_argument(
        "--shard_file",
        default=None,
        help="Newline-delimited window names. Omit to run on all windows in --group.",
    )
    p.add_argument("--heartbeat_dir", required=True, help="directory for heartbeat files")
    p.add_argument(
        "--rslearn_subcommand",
        required=True,
        choices=("prepare", "ingest", "materialize"),
    )
    p.add_argument(
        "--heartbeat_interval_s",
        type=float,
        default=30.0,
        help="How often to refresh the heartbeat file.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Returns the exit code of the underlying rslearn process."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raw = sys.argv[1:] if argv is None else argv
    our_args, passthrough = _split_argv(raw)
    args = _build_parser().parse_args(our_args)

    window_names: list[str] = (
        load_shard_file(UPath(args.shard_file)) if args.shard_file else []
    )
    logger.info(
        "Shard %s (%s, layers=%s) running rslearn %s on %d windows",
        args.shard_id,
        args.modality,
        ",".join(args.layer_names),
        args.rslearn_subcommand,
        len(window_names),
    )

    cmd = _build_rslearn_cmd(
        subcommand=args.rslearn_subcommand,
        root=args.root,
        group=args.group,
        window_names=window_names,
        passthrough=passthrough,
    )
    logger.info("Invoking: %s", " ".join(cmd))

    rc = 1
    try:
        with HeartbeatWriter(
            heartbeat_dir=UPath(args.heartbeat_dir),
            shard_id=args.shard_id,
            modality=args.modality,
            layer_names=args.layer_names,
            ds_path=UPath(args.ds_path),
            group=args.group,
            window_names=window_names,
            interval_s=args.heartbeat_interval_s,
        ):
            result = subprocess.run(cmd, check=False)
            rc = result.returncode
    except Exception:
        logger.exception("shard runner failed")
        return 2

    logger.info("rslearn exited with rc=%d", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
