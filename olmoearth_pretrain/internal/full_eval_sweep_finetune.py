"""Launch fine-tune evaluation sweeps for OlmoEarth Pretrain checkpoints.

Example run:
python olmoearth_pretrain/internal/full_eval_sweep_finetune.py --project_name 2025_10_06_phase1_finetune --module_path olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py --cluster ai2/jupiter --model_name dino_v3 --defaults_only

python olmoearth_pretrain/internal/full_eval_sweep_finetune.py --checkpoint_path /weka/dfive-default/helios/checkpoints/yawenzzzz/base_v6_default/step400000 --project_name 2025_10_06_phase1_finetune --module_path scripts/2025_09_10_phase1/script.py --cluster ai2/jupiter --defaults_only
"""

import argparse
import os
import subprocess  # nosec
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from olmoearth_pretrain.evals.models import get_launch_script_path
from olmoearth_pretrain.internal.all_evals import FT_EVAL_TASKS
from olmoearth_pretrain.internal.constants import EVAL_LAUNCH_PATH, EVAL_WANDB_PROJECT
from olmoearth_pretrain.internal.experiment import SubCmd

logger = getLogger(__name__)

# Fine-tune learning rates to sweep over.
FT_LRS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

TASK_ARG_PREFIX = "--trainer.callbacks.downstream_evaluator.tasks"
FT_TASK_NAMES = list(FT_EVAL_TASKS.keys())


def _format_per_task_args(overrides: dict[str, Any]) -> list[str]:
    """Repeat overrides for each downstream task."""
    if not overrides:
        return []
    args: list[str] = []
    for task in FT_TASK_NAMES:
        for f, v in overrides.items():  # type: ignore
            value_str = v if isinstance(v, str) else str(v)
            args.append(f"{TASK_ARG_PREFIX}.{task}.{f}={value_str}")
    return args


FT_MODE_ARGS = _format_per_task_args({"eval_mode": "finetune"})
DATASET_STATS_ARGS = _format_per_task_args({"norm_stats_from_pretrained": "False"})


def _format_ft_lr_args(lr: float) -> list[str]:
    return _format_per_task_args({"ft_lr": lr})


@dataclass(frozen=True)
class ModelPreset:
    """Model-specific overrides used for evaluation normalisation."""

    per_task_overrides: dict[str, Any] = field(default_factory=dict)
    global_args: tuple[str, ...] = ()
    # By default, we include the dataset stats.
    include_dataset_stats: bool = True
    launch_script_key: str | None = None


# TODO: We need to add the final norm settings for each model.
MODEL_PRESETS: dict[str, ModelPreset] = {
    "dino_v3": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_YES_CLIP_MIN_MAX_INT"},
        launch_script_key="dino_v3",
    ),
    "panopticon": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        launch_script_key="panopticon",
    ),
    "terramind": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NO_NORM"},
        global_args=("--model.use_pretrained_normalizer=True",),
        launch_script_key="terramind",
    ),
    "galileo": ModelPreset(
        per_task_overrides={
            "norm_method": "NormMethod.NO_NORM",
            "embedding_batch_size": "8",
        },
        global_args=("--model.use_pretrained_normalizer=True",),
        launch_script_key="galileo",
    ),
    "satlas": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NO_NORM"},
        global_args=("--model.use_pretrained_normalizer=True",),
        launch_script_key="satlas",
    ),
    "croma": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_YES_CLIP_2_STD"},
        launch_script_key="croma",
    ),
    "clay": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NO_NORM"},
        global_args=("--model.use_pretrained_normalizer=True",),
        launch_script_key="clay",
    ),
    "copernicusfm": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_YES_CLIP_2_STD"},
    ),
    "presto": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NO_NORM"},
        global_args=("--model.use_pretrained_normalizer=True",),
        launch_script_key="presto",
    ),
    "anysat": ModelPreset(
        per_task_overrides={
            "norm_method": "NormMethod.STANDARDIZE",
            "embedding_batch_size": "4",
        },
    ),
    "tessera": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NO_NORM"},
        global_args=("--model.use_pretrained_normalizer=True",),
        launch_script_key="tessera",
    ),
    "prithvi_v2": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NO_NORM"},
        global_args=("--model.use_pretrained_normalizer=True",),
        launch_script_key="prithvi_v2",
    ),
}


def _selected_model_flag(args: argparse.Namespace) -> str | None:
    """Get the selected model flag."""
    flags = [name for name in MODEL_PRESETS if getattr(args, name)]
    if len(flags) > 1:
        raise ValueError(
            f"Specify at most one model preset flag (got: {', '.join(sorted(flags))})."
        )
    return flags[0] if flags else None


def _build_model_args(selected_flag: str | None) -> list[str]:
    """Build the model arguments."""
    if selected_flag is None:
        return []
    preset = MODEL_PRESETS[selected_flag]
    args = list(DATASET_STATS_ARGS) if preset.include_dataset_stats else []
    args.extend(_format_per_task_args(preset.per_task_overrides))
    args.extend(preset.global_args)
    return args


def _resolve_module_path(args: argparse.Namespace, selected_flag: str | None) -> str:
    """Get the module path."""
    if args.module_path:
        logger.info(f"Using module path {args.module_path}")
        return args.module_path

    if selected_flag is None:
        raise ValueError(
            "Provide --module_path or specify a model preset flag that implies one."
        )

    preset = MODEL_PRESETS[selected_flag]
    if preset.launch_script_key is None:
        raise ValueError(
            f"--{selected_flag} has no default launch script. Pass --module_path explicitly."
        )

    module_path = get_launch_script_path(preset.launch_script_key)
    logger.info(f"Using module path {module_path}")
    return module_path


def _get_sub_command(args: argparse.Namespace) -> str:
    """Get the sub command."""
    if args.dry_run:
        return SubCmd.dry_run
    if args.cluster == "local":
        return SubCmd.train
    return SubCmd.launch


def _get_base_run_name(args: argparse.Namespace) -> str:
    """Get the base run name."""
    if args.model_name is not None:
        logger.info("Overriding checkpoint name with %s", args.model_name)
        return args.model_name
    if args.checkpoint_path is not None:
        parent_dir = os.path.basename(os.path.dirname(args.checkpoint_path))[:100]
        step_num = os.path.basename(args.checkpoint_path)
        return f"{parent_dir}_{step_num}"
    logger.warning("No model name or checkpoint path provided; using random run name")
    return str(uuid.uuid4())[:8]


def _get_checkpoint_args(checkpoint_path: str | None) -> list[str]:
    """Get the checkpoint arguments."""
    if checkpoint_path:
        return [f"--trainer.load_path={checkpoint_path}"]
    return []


def _format_launch_command(
    *,
    module_path: str,
    launch_command: str,
    sub_command: str,
    run_name: str,
    cluster: str,
    project_name: str,
    checkpoint_args: list[str],
    extra_cli: Iterable[str],
    model_args: list[str],
    lr: float,
) -> str:
    """Format the launch command."""
    parts = [
        f"TRAIN_SCRIPT_PATH={module_path}",
        launch_command,
        EVAL_LAUNCH_PATH,
        sub_command,
        run_name,
        cluster,
        "--launch.priority=high",
        "--launch.task_name=eval",
    ]
    parts.extend(checkpoint_args)
    parts.append(f"--trainer.callbacks.wandb.project={project_name}")
    parts.extend(extra_cli)
    parts.extend(model_args)
    parts.extend(FT_MODE_ARGS)
    parts.extend(_format_ft_lr_args(lr))
    parts.append("--train_module.dp_config=null")
    return " ".join(parts)


def build_commands(args: argparse.Namespace, extra_cli: list[str]) -> list[str]:
    """Build the commands for the sweep."""
    project_name = args.project_name or EVAL_WANDB_PROJECT
    sub_command = _get_sub_command(args)
    base_run_name = _get_base_run_name(args)
    launch_command = "python3" if sub_command != SubCmd.train else "torchrun"

    selected_flag = _selected_model_flag(args)
    model_args = _build_model_args(selected_flag)
    module_path = _resolve_module_path(args, selected_flag)
    checkpoint_args = _get_checkpoint_args(args.checkpoint_path)

    lrs = [FT_LRS[0]] if args.defaults_only else FT_LRS
    commands: list[str] = []
    for lr in lrs:
        if args.defaults_only:
            logger.info("Running FT defaults with lr=%s", lr)
            run_suffix = "FT_defaults"
        else:
            logger.info("Running FT with lr=%s", lr)
            run_suffix = f"FT_lr{lr}"
        run_name = f"{base_run_name}_{run_suffix}"
        commands.append(
            _format_launch_command(
                module_path=module_path,
                launch_command=launch_command,
                sub_command=sub_command,
                run_name=run_name,
                cluster=args.cluster,
                project_name=project_name,
                checkpoint_args=checkpoint_args,
                extra_cli=extra_cli,
                model_args=model_args,
                lr=lr,
            )
        )
    return commands


def main() -> None:
    """Run the fine-tune evaluation sweep."""
    parser = argparse.ArgumentParser(description="Run finetune evaluation sweeps.")
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--module_path",
        type=str,
        required=False,
        default=None,
        help="Path to module .py (overrides model preset defaults)",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=False,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--defaults_only",
        action="store_true",
        help="Only run with the default learning rate",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands without launching them",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Use this as the base run name",
    )

    for flag, preset in MODEL_PRESETS.items():
        parser.add_argument(f"--{flag}", action="store_true")

    args, extra_cli = parser.parse_known_args()

    env = os.environ.copy()
    env["FINETUNE"] = "1"
    commands_to_run = build_commands(args, extra_cli)
    for cmd in commands_to_run:
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True, env=env)  # nosec


if __name__ == "__main__":
    main()
