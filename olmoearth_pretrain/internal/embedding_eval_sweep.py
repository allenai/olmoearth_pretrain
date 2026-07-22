"""Sweep the embedding-product evals (EMBEDDING_EVAL_TASKS).

The third sweep type next to the KNN/LP sweep (full_eval_sweep.py) and the
finetuning sweep (full_eval_sweep_finetune.py). It runs only the per-pixel
ws16/ps1 embedding-convention tasks (EMBEDDING_EVALS=1 selects them in
all_evals.py) and differs from the full sweep in that:

- normalization is held fixed: pretraining stats for OlmoEarth, and the
  precomputed embedding products are consumed exactly as stored (NO_NORM);
- only the probe LR is swept, and only for the linear-probe tasks — the KNN
  twins have no hyperparameters and run once in their own job;
- only OlmoEarth checkpoints and the precomputed embedding products (aef,
  tessera_precomputed) are supported.

e.g.
  # OlmoEarth checkpoint
  python -m olmoearth_pretrain.internal.embedding_eval_sweep \
      --cluster=ai2/saturn-cirrascale \
      --checkpoint_path=/weka/.../step370000 \
      --module_path=scripts/.../nano.py

  # Precomputed baselines
  python -m olmoearth_pretrain.internal.embedding_eval_sweep \
      --cluster=ai2/saturn-cirrascale --model=aef
"""

import argparse
import json
import os
import subprocess  # nosec
import uuid
from logging import getLogger

from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.models import BaselineModelName, get_launch_script_path
from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS
from olmoearth_pretrain.internal.constants import EVAL_LAUNCH_PATH, EVAL_WANDB_PROJECT
from olmoearth_pretrain.internal.experiment import SubCmd
from olmoearth_pretrain.internal.full_eval_sweep import (
    LAUNCH_OVERRIDES,
    MAX_DURATION_OVERRIDE,
    PRECOMPUTED_MODEL_TO_MODALITY,
    LP_LRs,
    _get_checkpoint_args,
    _get_sub_command,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import EvalMode

logger = getLogger(__name__)

SUPPORTED_BASELINES = tuple(PRECOMPUTED_MODEL_TO_MODALITY)

LP_TASK_NAMES = [
    name
    for name, task in EMBEDDING_EVAL_TASKS.items()
    if task.eval_mode == EvalMode.LINEAR_PROBE
]
KNN_TASK_NAMES = [
    name
    for name, task in EMBEDDING_EVAL_TASKS.items()
    if task.eval_mode == EvalMode.KNN
]


def _task_arg(task_name: str, field_name: str, value: object) -> str:
    """Build one per-task downstream-evaluator override."""
    return (
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}"
        f".{field_name}={value}"
    )


def _capable_tasks(task_names: list[str], modality: str) -> list[str]:
    """Task names whose dataset carries the given precomputed modality."""
    return [
        name
        for name in task_names
        if modality
        in dataset_to_config(EMBEDDING_EVAL_TASKS[name].dataset).supported_modalities
    ]


def _model_task_names(
    model: BaselineModelName | None,
) -> tuple[list[str], list[str]]:
    """(LP, KNN) task names the given model can run; fail fast on zero tasks."""
    if model is None:
        return LP_TASK_NAMES, KNN_TASK_NAMES
    modality, product = PRECOMPUTED_MODEL_TO_MODALITY[model]
    lp = _capable_tasks(LP_TASK_NAMES, modality)
    knn = _capable_tasks(KNN_TASK_NAMES, modality)
    skipped = sorted(set(LP_TASK_NAMES + KNN_TASK_NAMES) - set(lp + knn))
    if skipped:
        logger.warning(
            f"--model={model}: skipping tasks whose dataset does not carry the "
            f"'{modality}' modality: {', '.join(skipped)}"
        )
    if not lp and not knn:
        raise SystemExit(
            f"No embedding eval task's dataset supports the precomputed "
            f"'{modality}' modality, so --model={model} would run zero tasks. "
            f"Bake the embeddings into the eval datasets first (embedding "
            f"materializer / pastis_processor --embedding_products {product}) "
            f"and list '{modality}' in the dataset's supported_modalities."
        )
    return lp, knn


def _model_args(model: BaselineModelName | None, task_names: list[str]) -> str:
    """Per-task and trainer args pinning each model's normalization/quantization.

    The task configs already carry these values, but the sweep pins them
    explicitly so the convention cannot drift: OlmoEarth's forward-pass
    embeddings are always int8 round-tripped (scored as an embedding product,
    pretraining-stats normalization), while the precomputed products are
    consumed exactly as stored — no re-normalization and no second int8
    round-trip (they are already int8 at source) — reading the embedding
    modality instead of imagery.
    """
    if model is None:
        args = [" --trainer.no_checkpoints=False"]
        for task_name in task_names:
            args.append(_task_arg(task_name, "norm_stats_from_pretrained", "True"))
            args.append(_task_arg(task_name, "quantize_embeddings", "True"))
        return " ".join(args)
    modality, _ = PRECOMPUTED_MODEL_TO_MODALITY[model]
    args = [" --trainer.no_checkpoints=True"]
    for task_name in task_names:
        args.append(_task_arg(task_name, "norm_stats_from_pretrained", "False"))
        args.append(_task_arg(task_name, "norm_method", "NormMethod.NO_NORM"))
        args.append(_task_arg(task_name, "input_modalities", f"[{modality}]"))
        args.append(_task_arg(task_name, "quantize_embeddings", "False"))
    return " ".join(args)


def _tasks_to_run_arg(task_names: list[str]) -> str:
    """Restrict the evaluator to the given tasks (compact JSON; see full sweep)."""
    return (
        " --trainer.callbacks.downstream_evaluator.tasks_to_run="
        f"'{json.dumps(task_names, separators=(',', ':'))}'"
    )


def _select_best_val_args(task_names: list[str]) -> str:
    """Per-LP-task early-stopping args (best epoch by primary val metric)."""
    return " " + " ".join(
        f"{_task_arg(name, 'select_best_by_primary_metric', 'True')} "
        f"{_task_arg(name, 'linear_probe_eval_interval', '5')}"
        for name in task_names
    )


def _base_run_name(args: argparse.Namespace) -> str:
    """Base run name from --model_name, checkpoint path, or model."""
    if args.model_name is not None:
        return args.model_name
    if args.checkpoint_path is not None:
        parent_dir = os.path.basename(os.path.dirname(args.checkpoint_path))[:100]
        step_num = os.path.basename(args.checkpoint_path)
        return f"{parent_dir}_{step_num}"
    return f"{args.model}_{str(uuid.uuid4())[:4]}"


def build_commands(args: argparse.Namespace, extra_cli: list[str]) -> list[str]:
    """Build one command per LP learning rate plus one KNN command."""
    model: BaselineModelName | None = args.model
    if model is None and args.module_path is None:
        raise ValueError("Provide --module_path (and --checkpoint_path) or --model")

    lp_tasks, knn_tasks = _model_task_names(model)

    module_path = args.module_path if model is None else get_launch_script_path(model)
    sub_command = _get_sub_command(args)
    launch_command = "torchrun" if sub_command == SubCmd.evaluate else "python3"
    launch_overrides = LAUNCH_OVERRIDES if sub_command == SubCmd.launch_evaluate else ""
    checkpoint_args = _get_checkpoint_args(args.checkpoint_path)
    project_name = args.project_name or EVAL_WANDB_PROJECT
    extra = " " + " ".join(extra_cli) if extra_cli else ""
    base_run_name = _base_run_name(args) + "_emb"

    env_prefix = f"TRAIN_SCRIPT_PATH={module_path} EMBEDDING_EVALS=1"
    if args.skip_mismatched_weights:
        env_prefix += " OE_LOAD_SKIP_MISMATCHED_KEYS=1"
    common = (
        f"{env_prefix} {launch_command} {EVAL_LAUNCH_PATH} "
        f"{sub_command} {{run_name}} {args.cluster} {launch_overrides} "
        f"{checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra}"
        f" {MAX_DURATION_OVERRIDE}"
    )

    commands = []
    if lp_tasks:
        lp_model_args = _model_args(model, lp_tasks)
        for lr in LP_LRs:
            cmd = common.format(run_name=f"{base_run_name}_lr{lr}")
            cmd += lp_model_args
            cmd += " " + " ".join(_task_arg(name, "probe_lr", lr) for name in lp_tasks)
            if args.select_best_val:
                cmd += _select_best_val_args(lp_tasks)
            cmd += _tasks_to_run_arg(lp_tasks)
            commands.append(cmd)
    if knn_tasks:
        cmd = common.format(run_name=f"{base_run_name}_knn")
        cmd += _model_args(model, knn_tasks)
        cmd += _tasks_to_run_arg(knn_tasks)
        commands.append(cmd)
    return commands


def _parse_model_arg(value: str) -> BaselineModelName:
    """Parse --model, restricted to the precomputed embedding products."""
    try:
        model = BaselineModelName(value)
    except ValueError:
        model = None
    if model not in SUPPORTED_BASELINES:
        raise argparse.ArgumentTypeError(
            f"Invalid model: {value}. The embedding sweep supports "
            f"{[m.value for m in SUPPORTED_BASELINES]} (or omit --model and pass "
            "--checkpoint_path/--module_path for an OlmoEarth checkpoint)."
        )
    return model


def main() -> None:
    """Run the embedding-product eval sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="OlmoEarth checkpoint path (omit for precomputed baselines)",
    )
    parser.add_argument(
        "--module_path",
        type=str,
        default=None,
        help="Path to the OlmoEarth model-config module .py",
    )
    parser.add_argument(
        "--model",
        type=_parse_model_arg,
        default=None,
        help=f"Precomputed baseline: {[m.value for m in SUPPORTED_BASELINES]}. "
        "Omit to evaluate an OlmoEarth checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="If set, use this as the base run name",
    )
    parser.add_argument(
        "--project_name", type=str, required=False, help="Wandb project name"
    )
    parser.add_argument(
        "--select_best_val",
        action="store_true",
        help="Select the best test epoch by the primary validation metric",
    )
    parser.add_argument(
        "--skip_mismatched_weights",
        action="store_true",
        help="Skip checkpoint weights whose shape mismatches the current model "
        "(they keep their fresh init). For checkpoints saved before benign "
        "architecture drift, e.g. the srtm terrain-band change; the skipped "
        "keys are logged loudly in the job output.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print the configs that would be run",
    )
    args, extra_cli = parser.parse_known_args()

    commands_to_run = build_commands(args, extra_cli)
    logger.info(f"Running {len(commands_to_run)} commands")
    for cmd in commands_to_run:
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)  # nosec
    logger.info(f"Finished running {len(commands_to_run)} commands")


if __name__ == "__main__":
    main()
