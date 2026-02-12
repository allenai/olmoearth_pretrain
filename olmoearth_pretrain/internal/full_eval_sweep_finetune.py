"""Launch fine-tune evaluation sweeps for OlmoEarth and other models.

Usage examples:

1. Finetune all eval tasks using TerraMind model (default lr only):
   python olmoearth_pretrain/internal/full_eval_sweep_finetune.py \
       --project_name test_finetune \
       --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py \
       --cluster ai2/jupiter \
       --model terramind \
       --defaults_only

2. Finetune all eval tasks using OlmoEarth model (default lr only):
   python olmoearth_pretrain/internal/full_eval_sweep_finetune.py \
       --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
       --project_name test_finetune \
       --module_path scripts/official/base.py \
       --cluster ai2/jupiter \
       --defaults_only

3. To run a subset of tasks, add:
     --trainer.callbacks.downstream_evaluator.tasks_to_run='["m_eurosat","m_so2sat","mados"]'
   You can also launch multiple jobs with different tasks_to_run values to speed up the finetuning.

Flags:
  --defaults_only  Runs just one job: lr = 1e-4
  (omit)           Sweeps lrs: [1e-4, 5e-4, 1e-3]

OlmoEarth: normalization method (pretrained vs dataset) is *not* swept.
Each FT eval task's normalization is defined in all_evals.py.
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

# Learning rates to sweep over.
FT_LRS = [1e-4, 5e-4, 1e-3]

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


def _format_task_specific_args(task_overrides: dict[str, dict[str, Any]]) -> list[str]:
    """Generate overrides for specific downstream tasks."""
    if not task_overrides:
        return []
    args: list[str] = []
    for task, overrides in task_overrides.items():
        if task not in FT_TASK_NAMES:
            raise ValueError(f"Unknown fine-tune task override: {task}")
        for field_name, value in overrides.items():
            value_str = value if isinstance(value, str) else str(value)
            args.append(f"{TASK_ARG_PREFIX}.{task}.{field_name}={value_str}")
    return args


FT_MODE_ARGS = _format_per_task_args({"eval_mode": "FINETUNE"})
DATASET_STATS_ARGS = _format_per_task_args({"norm_stats_from_pretrained": "False"})


def _format_ft_lr_args(lr: float) -> list[str]:
    return _format_per_task_args({"ft_lr": lr})


@dataclass(frozen=True)
class ModelPreset:
    """Model-specific overrides used for evaluation normalisation."""

    per_task_overrides: dict[str, Any] = field(default_factory=dict)
    task_specific_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_args: tuple[str, ...] = ()
    include_dataset_stats: bool = True
    launch_script_key: str | None = None
    # Whether this model supports --model.use_pretrained_normalizer
    supports_pretrained_normalizer: bool = False


MODEL_PRESETS: dict[str, ModelPreset] = {
    "dino_v3": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_YES_CLIP_MIN_MAX_INT"},
        launch_script_key="dino_v3",
    ),
    "panopticon": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        launch_script_key="panopticon",
    ),
    "croma": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_YES_CLIP_2_STD"},
        launch_script_key="croma",
    ),
    "croma_large": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_YES_CLIP_2_STD"},
        global_args=("--model.size=large",),
        launch_script_key="croma",
    ),
    # by default, AnySat uses patch size of 4
    "anysat": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        task_specific_overrides={
            "m_sa_crop_type": {"ft_batch_size": 4},
            "pastis_sentinel2": {"ft_batch_size": 4},
            "m_cashew_plant": {"ft_batch_size": 4},
            "m_forestnet": {"ft_batch_size": 4},
        },
    ),
    "anysat_ps8": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        global_args=("--model.patch_size=8",),
        task_specific_overrides={
            "m_cashew_plant": {"ft_batch_size": 4, "patch_size": 8},
        },
    ),
    "anysat_ps16": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        global_args=("--model.patch_size=16",),
        task_specific_overrides={
            "m_cashew_plant": {"ft_batch_size": 4, "patch_size": 16},
            "m_forestnet": {"ft_batch_size": 4, "patch_size": 16},
        },
    ),
    # Models with pretrained normalizer
    "terramind": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        launch_script_key="terramind",
        supports_pretrained_normalizer=True,
    ),
    "terramind_large": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        global_args=("--model.size=large",),
        launch_script_key="terramind",
        supports_pretrained_normalizer=True,
    ),
    # by default, Galileo uses patch size of 4
    "galileo": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_NO_CLIP_2_STD"},
        task_specific_overrides={
            "m_sa_crop_type": {"ft_batch_size": 1},
            "pastis_sentinel2": {"ft_batch_size": 2},
            "m_cashew_plant": {"ft_batch_size": 4},
        },
        launch_script_key="galileo",
        supports_pretrained_normalizer=True,
    ),
    "galileo_ps8": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_NO_CLIP_2_STD"},
        global_args=("--model.patch_size=8",),
        task_specific_overrides={
            "m_cashew_plant": {"ft_batch_size": 4, "patch_size": 8},
        },
        launch_script_key="galileo",
        supports_pretrained_normalizer=True,
    ),
    "galileo_ps16": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_NO_CLIP_2_STD"},
        global_args=("--model.patch_size=16",),
        task_specific_overrides={
            "m_cashew_plant": {"ft_batch_size": 4, "patch_size": 16},
        },
        launch_script_key="galileo",
        supports_pretrained_normalizer=True,
    ),
    "satlas": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.NORM_YES_CLIP"},
        task_specific_overrides={
            "pastis_sentinel2": {"ft_batch_size": 4},
        },
        launch_script_key="satlas",
        supports_pretrained_normalizer=True,
    ),
    "clay": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        launch_script_key="clay",
        supports_pretrained_normalizer=True,
    ),
    "prithvi_v2": ModelPreset(
        per_task_overrides={"norm_method": "NormMethod.STANDARDIZE"},
        launch_script_key="prithvi_v2",
        supports_pretrained_normalizer=True,
    ),
}


def _build_model_args(
    selected_preset: str | None, normalizer: bool | None = None
) -> list[str]:
    """Build the model arguments.

    Always return the preset overrides. When normalizer is True, force
    --model.use_pretrained_normalizer=True and set per-task norm to NO_NORM.
    When normalizer is False, explicitly disable the pretrained normalizer.
    """
    if selected_preset is None:
        return []
    preset = MODEL_PRESETS[selected_preset]

    args = list(DATASET_STATS_ARGS) if preset.include_dataset_stats else []
    args.extend(_format_per_task_args(preset.per_task_overrides))
    args.extend(_format_task_specific_args(preset.task_specific_overrides))
    args.extend(preset.global_args)

    if normalizer is True and preset.supports_pretrained_normalizer:
        args.append("--model.use_pretrained_normalizer=True")
        args.extend(_format_per_task_args({"norm_method": "NormMethod.NO_NORM"}))
    elif normalizer is False and preset.supports_pretrained_normalizer:
        args.append("--model.use_pretrained_normalizer=False")

    return args


def _resolve_module_path(args: argparse.Namespace, selected_preset: str | None) -> str:
    """Get the module path."""
    if args.module_path:
        logger.info(f"Using module path {args.module_path}")
        return args.module_path

    if selected_preset is None:
        raise ValueError(
            "Provide --module_path or select a model preset key that implies one."
        )

    preset = MODEL_PRESETS[selected_preset]
    if preset.launch_script_key is None:
        raise ValueError(
            f"Model preset '{selected_preset}' has no default launch script. Pass --module_path explicitly."
        )

    module_path = get_launch_script_path(preset.launch_script_key)
    logger.info(f"Using module path {module_path}")
    return module_path


def _get_sub_command(args: argparse.Namespace) -> str:
    """Get the sub command."""
    if args.dry_run:
        return SubCmd.dry_run_evaluate
    # If cluster is local, we run eval locally, if not, we launch evaluation on beaker
    if args.cluster == "local":
        return SubCmd.evaluate
    else:
        return SubCmd.launch_evaluate


def _get_base_run_name(args: argparse.Namespace, selected_preset: str | None) -> str:
    """Get the base run name."""
    if args.model is not None:
        logger.info("Overriding checkpoint name with %s", args.model)
        return args.model
    if args.checkpoint_path is not None:
        parent_dir = os.path.basename(os.path.dirname(args.checkpoint_path))[:100]
        step_num = os.path.basename(args.checkpoint_path)
        return f"{parent_dir}_{step_num}"
    if selected_preset is not None:
        logger.info("Using model preset key %s as base run name", selected_preset)
        return selected_preset
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
    seed_args: Iterable[str],
) -> str:
    """Format the launch command."""
    parts = [
        f"TRAIN_SCRIPT_PATH={module_path}",
        launch_command,
        EVAL_LAUNCH_PATH,
        sub_command,
        run_name,
        cluster,
        # Overwrite the max duration to enable eval of the last step of the checkpoint
        "--trainer.max_duration.value=10000000",
        "--trainer.max_duration.unit=steps",
    ]
    if cluster != "local":
        parts.extend(
            [
                "--launch.priority=high",
                "--launch.num_gpus=1",
                "--launch.preemptible=True",
                "--launch.task_name=eval",
            ]
        )
    parts.extend(checkpoint_args)
    parts.append(f"--trainer.callbacks.wandb.project={project_name}")
    parts.extend(extra_cli)
    parts.extend(model_args)
    parts.extend(FT_MODE_ARGS)
    parts.extend(_format_ft_lr_args(lr))
    parts.extend(seed_args)
    parts.append("--train_module.dp_config=null")
    return " ".join(parts)


def build_commands(
    args: argparse.Namespace,
    extra_cli: list[str],
) -> list[str]:
    """Build the commands for the sweep."""
    project_name = args.project_name or EVAL_WANDB_PROJECT
    sub_command = _get_sub_command(args)
    selected_preset = args.model
    base_run_name = _get_base_run_name(args, selected_preset)
    launch_command = "python3" if not sub_command == SubCmd.evaluate else "torchrun"

    module_path = _resolve_module_path(args, selected_preset)
    checkpoint_args = _get_checkpoint_args(args.checkpoint_path)

    # LR sweep
    lrs = [FT_LRS[0]] if args.defaults_only else FT_LRS

    # Pretrained normalizer: default True for supported presets.
    normalizer_value: bool | None = None
    if selected_preset is not None:
        preset = MODEL_PRESETS[selected_preset]
        if preset.supports_pretrained_normalizer:
            normalizer_value = not args.use_dataset_normalizer
        else:
            if not args.use_dataset_normalizer:
                logger.warning(
                    "Model preset %s does not support pretrained normalization; "
                    "falling back to dataset statistics.",
                    selected_preset,
                )

    # Seed sweep
    seed_args: list[str] = []
    if args.finetune_seed is not None:
        seed_args.extend(_format_per_task_args({"finetune_seed": args.finetune_seed}))
    commands: list[str] = []
    for lr in lrs:
        if args.defaults_only:
            run_suffix = "FT_defaults"
        elif args.checkpoint_path:
            run_suffix = f"FT_lr{lr}"
        else:
            norm_suffix = ""
            if normalizer_value is not None:
                norm_suffix = (
                    "_norm_pretrained_True"
                    if normalizer_value
                    else "_norm_pretrained_False"
                )
            run_suffix = f"FT_lr{lr}{norm_suffix}"

        seed_suffix = (
            f"_seed{args.finetune_seed}" if args.finetune_seed is not None else ""
        )
        run_name = f"{base_run_name}{seed_suffix}_{run_suffix}"
        model_args = _build_model_args(selected_preset, normalizer_value)

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
                seed_args=seed_args,
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
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset key to apply (defaults to none).",
    )
    parser.add_argument(
        "--use_dataset_normalizer",
        action="store_true",
        help=(
            "Use dataset statistics instead of the pretrained normalizer when supported."
        ),
    )
    parser.add_argument(
        "--finetune_seed",
        type=int,
        default=None,
        help="Base random seed applied to every finetune task (optional).",
    )

    args, extra_cli = parser.parse_known_args()

    env = os.environ.copy()
    env["FINETUNE"] = "1"
    commands_to_run = build_commands(args, extra_cli)
    for cmd in commands_to_run:
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True, env=env)  # nosec


if __name__ == "__main__":
    main()
