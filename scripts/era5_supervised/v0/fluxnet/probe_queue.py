"""Run a queue of eval-only checkpoint sweeps inside one Beaker job.

Each queue entry re-probes one source run's saved checkpoints with
``base.py evaluate`` in checkpoint-sweep mode (the same path as the probe-LR
sweep launchers), so many runs share one GPU allocation. The source run's
encoder / decoder flags are read from its ``step<N>/config.json`` instead of
being restated by hand: the script defaults changed between batches (pooled
LayerNorm, conv-stem widths, patch day of year), and a checkpoint loaded with
the wrong parameter-free flags loads fine but probes a different model.

An entry's ``tasks`` (e.g. ``"[fluxnet_gpp_eval,lfmc_woody_eval]"``)
overrides the queue's ``--tasks``. An entry with ``"random_init": true`` probes an untrained encoder instead (the
step-0 random-projection floor): flags come from ``source``'s config, nothing
is loaded, and ``init_seed`` picks the initialization.

Every eval also lands as a JSON line in ``<results_dir>/<name>.jsonl``. A
finished entry leaves ``<done_dir>/<name>.done``; a requeued job skips it.

Queue file (JSON list)::

    [{"name": "era5enc_1306_halo75_nogate_flux", "source": "era5enc_1306_halo75_nogate",
      "interval": 2000, "load_step": 50000, "extra": []},
     {"name": "era5enc_1601_doyc_flux_init", "source": "era5enc_1601_doyc",
      "random_init": true, "init_seed": 12536}, ...]

Usage (inside the job, repo root as cwd)::

    python scripts/era5_supervised/v0/fluxnet/probe_queue.py --queue <file> --cluster ai2/saturn
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CKPT_ROOT = Path("/weka/dfive-default/olmoearth_pretrain/checkpoints/hadriens")
DEFAULT_DONE_DIR = Path("/weka/dfive-default/rslearn-eai/datasets/fluxnet/probe_runs")
DEFAULT_RESULTS_DIR = Path(
    "/weka/dfive-default/rslearn-eai/datasets/fluxnet/probe_results"
)
BASE = "scripts/era5_supervised/v0/base.py"


def _value(value: Any) -> str:
    """Render a config value as a ``common.*`` override value.

    Args:
        value: Scalar, list or None.

    Returns:
        The override string (``null`` for None, ``[a,b]`` for lists).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_value(v) for v in value) + "]"
    return str(value)


def source_flags(config_path: Path) -> list[str]:
    """``common.*`` overrides that rebuild a saved run's encoder and decoder.

    Fields absent from older configs map to the behaviour of the code that
    trained them: no position embedding, patch-mean day of year, no pooled
    LayerNorm, no decoder key position embedding.

    Args:
        config_path: The run's ``step<N>/config.json``.

    Returns:
        Overrides for ``script.py``'s common components.
    """
    model = json.loads(config_path.read_text())["model"]
    enc = model["encoder_config"]
    flags = {
        "encoder_embedding_size": enc["embedding_size"],
        "encoder_depth": enc["depth"],
        "encoder_num_heads": enc["num_heads"],
        "encoder_pooling": enc["pooling"],
        "encoder_use_conv_stem": enc["use_conv_stem"],
        "encoder_stem_hidden_dims": enc.get("stem_hidden_dims"),
        "encoder_patch_kernel_size": enc.get("patch_kernel_size") or 14,
        "encoder_patch_stride": enc.get("patch_stride") or 7,
        "encoder_position_embedding": enc.get("position_embedding") or "none",
        "encoder_patch_day_of_year": enc.get("patch_day_of_year") or "mean",
        "encoder_pooled_norm": enc.get("pooled_norm") or "none",
        "encoder_swt_input": enc["is_swt_input"],
        "encoder_swt_input_levels": enc["swt_input_levels"],
        "encoder_swt_input_include_approx": enc["swt_input_include_approx"],
    }
    recon = model.get("reconstruction_objective")
    if recon is not None:
        dec = recon["decoder"]
        flags.update(
            enable_reconstruction=True,
            recon_decoder_depth=dec["depth"],
            recon_decoder_num_heads=dec["num_heads"],
            recon_decoder_key_position_embedding=dec.get("key_position_embedding")
            or "none",
            recon_contrastive_lambda=recon.get("contrastive_lambda") or 0.0,
        )
    else:
        flags.update(enable_reconstruction=False)
    return [f"common.{key}={_value(value)}" for key, value in flags.items()]


def entry_command(entry: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """The ``base.py evaluate`` command of one queue entry.

    Args:
        entry: Queue entry (``name, source, interval, load_step, extra``, or
            ``random_init`` + ``init_seed``).
        args: Queue-level options (cluster, project, tasks, LR grid).

    Returns:
        The command line.
    """
    run_dir = CKPT_ROOT / entry["source"]
    load_step = int(entry.get("load_step", 50000))
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", BASE, "evaluate",
        entry["name"], args.cluster,
        *source_flags(run_dir / f"step{load_step}" / "config.json"),
        f"common.eval_tasks={entry.get('tasks', args.tasks)}",
        f"common.eval_probe_lr_grid={args.lr_grid}",
        "common.eval_probe_seed=1202",
        "common.eval_run_on_test=True",
        f"common.eval_results_jsonl={Path(args.results_dir) / (entry['name'] + '.jsonl')}",
        "common.max_steps=0",
        "common.max_epochs=1",
        "--trainer.callbacks.beaker.enabled=False",
        f"--trainer.callbacks.wandb.project={args.project}",
        f"--trainer.callbacks.wandb.name={entry['name']}",
    ]  # fmt: skip
    if entry.get("random_init", False):
        # Untrained encoder: the startup eval (eval_on_startup) is the floor.
        cmd += [f"--init_seed={int(entry['init_seed'])}"]
    else:
        cmd += [
            f"common.eval_checkpoint_sweep_dir={run_dir}",
            f"common.eval_checkpoint_sweep_interval={int(entry.get('interval', 50000))}",
            f"--trainer.load_path={run_dir}/step{load_step}",
            "--trainer.load_trainer_state=False",
            "--trainer.load_optim_state=False",
        ]
    return cmd + list(entry.get("extra", []))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queue", required=True)
    ap.add_argument("--cluster", default="ai2/saturn")
    ap.add_argument("--project", default="era5_fluxnet_eval")
    ap.add_argument(
        "--tasks", required=True, help="e.g. [fluxnet_gpp_eval,lfmc_woody_eval]"
    )
    ap.add_argument("--lr-grid", default="[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]")
    ap.add_argument("--done-dir", default=str(DEFAULT_DONE_DIR))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    queue = json.loads(Path(args.queue).read_text())
    done_dir = Path(args.done_dir)
    done_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for k, entry in enumerate(queue, start=1):
        marker = done_dir / f"{entry['name']}.done"
        if marker.exists():
            logger.info("[%d/%d] %s: done already", k, len(queue), entry["name"])
            continue
        cmd = entry_command(entry, args)
        logger.info("[%d/%d] %s\n%s", k, len(queue), entry["name"], " ".join(cmd))
        if args.dry_run:
            continue
        start = time.monotonic()
        result = subprocess.run(cmd, check=False)
        minutes = (time.monotonic() - start) / 60
        if result.returncode == 0:
            marker.write_text(f"{minutes:.1f} min\n")
            logger.info(
                "[%d/%d] %s finished in %.1f min", k, len(queue), entry["name"], minutes
            )
        else:
            failed.append(entry["name"])
            logger.error(
                "[%d/%d] %s failed (exit %d)",
                k,
                len(queue),
                entry["name"],
                result.returncode,
            )
    if failed:
        logger.error("failed entries: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
