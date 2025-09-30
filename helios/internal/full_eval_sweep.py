"""Run an evaluation sweep for an arbitrary helios checkpoint.

e.g. python3 scripts/run_all_evals/full_eval_sweep.py --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0078125/step450000  --module_path=scripts/2025_06_26_dataset_percentage_experiments/latent_mim_all_data.py (extra args here e.g --model.decoder_config.depth=1)
"""

import argparse
import os
import subprocess  # nosec
import uuid
from collections.abc import Generator
from logging import getLogger
from typing import Any

from helios.evals.datasets.configs import dataset_to_config, get_eval_mode
from helios.evals.models import get_launch_script_path
from helios.internal.all_evals import EVAL_TASKS
from helios.internal.experiment import SubCmd
from helios.nn.flexihelios import PoolingType

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
Normalization_MODES = ["dataset", "pre_trained"]
pooling_types = [PoolingType.MEAN, PoolingType.MAX]

logger = getLogger(__name__)


def create_linear_probe_arg(task_name: str, field_name: str) -> str:
    """Create a linear probe argument for a given task name."""
    initial_str = (
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.{field_name}="
    )
    return initial_str + "{arg}"


lr_args = " ".join(
    [
        create_linear_probe_arg(task_name, "probe_lr")
        for task_name, task in EVAL_TASKS.items()
        if get_eval_mode(dataset_to_config(task.dataset).task_type) == "linear_probe"
    ]
)

pooling_args = " ".join(
    [" "]
    + [
        create_linear_probe_arg(task_name, "pooling_type")
        for task_name, task in EVAL_TASKS.items()
    ]
)

dataset_args = " ".join(
    [" "]
    + [
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_stats_from_pretrained=False"
        for task_name in EVAL_TASKS.keys()
    ]
)

helios_args = " ".join(
    [""]
    + [
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_stats_from_pretrained=True"
        for task_name in EVAL_TASKS.keys()
    ]
)


def loop_through_params() -> Generator[dict[str, Any], None, None]:
    """Yield a dict of the hps we are sweeping over."""
    for lr in LP_LRs:
        for norm_mode in Normalization_MODES:
            for pooling_type in pooling_types:
                yield {
                    "lr": lr,
                    "norm_mode": norm_mode,
                    "pooling_type": pooling_type,
                }


def no_norm_sweep() -> Generator[dict[str, Any], None, None]:
    """Yield a dict of the hps we are sweeping over."""
    for pooling_type in pooling_types:
        for lr in LP_LRs:
            yield {
                "lr": lr,
                "pooling_type": pooling_type,
            }


def get_dino_v3_args() -> str:
    """Get the dino v3 arguments."""
    # Normalization strategy is to scale with min max to 0 - 256 and then scale back to 0 - 1
    # Normalization is then applied by the eval wrapper by default
    dino_v3_args = dataset_args
    dino_v3_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NORM_YES_CLIP_MIN_MAX_INT"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    return dino_v3_args


def get_croma_args() -> str:
    """Get the croma arguments."""
    croma_args = dataset_args
    croma_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NORM_YES_CLIP_2_STD"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    return croma_args


def get_tessera_args(pretrained_normalizer: bool = True) -> str:
    """Get the tessera arguments."""
    tessera_args = dataset_args
    if pretrained_normalizer:
        # To use galileo pretrained normalizer we want to leave normalization to the galileo wrapper
        tessera_args = dataset_args
        tessera_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NO_NORM"
                for task_name in EVAL_TASKS.keys()
            ]
        )

        tessera_args += " " + "--model.use_pretrained_normalizer=True"
    else:
        tessera_args += " " + "--model.use_pretrained_normalizer=False"
        tessera_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.STANDARDIZE"
                for task_name in EVAL_TASKS.keys()
            ]
        )
    return tessera_args


def get_panopticon_args() -> str:
    """Get the panopticon arguments."""
    panopticon_args = dataset_args
    panopticon_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.STANDARDIZE"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    return panopticon_args


def get_copernicusfm_args() -> str:
    """Get the copernicusfm arguments."""
    copernicusfm_args = dataset_args
    copernicusfm_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NORM_YES_CLIP_2_STD"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    return copernicusfm_args


def get_anysat_args() -> str:
    """Get the anysat arguments."""
    anysat_args = dataset_args
    anysat_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.STANDARDIZE"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    anysat_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.embedding_batch_size=4"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    return anysat_args


def get_galileo_args(pretrained_normalizer: bool = True) -> str:
    """Get the galileo arguments."""
    galileo_args = dataset_args
    if pretrained_normalizer:
        # To use galileo pretrained normalizer we want to leave normalization to the galileo wrapper
        galileo_args = dataset_args
        galileo_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NO_NORM"
                for task_name in EVAL_TASKS.keys()
            ]
        )

        galileo_args += " " + "--model.use_pretrained_normalizer=True"
    else:
        # IF we use dataset stats we want to turn off the pretrained normalizer
        galileo_args += " " + "--model.use_pretrained_normalizer=False"
    galileo_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.embedding_batch_size=8"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    return galileo_args


def get_satlas_args(pretrained_normalizer: bool = True) -> str:
    """Get the satlas arguments."""
    satlas_args = dataset_args
    if pretrained_normalizer:
        # To use satlas pretrained normalizer we want to leave normalization to the satlas wrapper
        satlas_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NO_NORM"
                for task_name in EVAL_TASKS.keys()
            ]
        )

        satlas_args += " " + "--model.use_pretrained_normalizer=True"
    else:
        satlas_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NORM_YES_CLIP"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        # IF we use dataset stats we want to turn off the pretrained normalizer
        satlas_args += " " + "--model.use_pretrained_normalizer=False"
    return satlas_args


def get_presto_args(pretrained_normalizer: bool = True) -> str:
    """Get the presto arguments."""
    presto_args = dataset_args
    if pretrained_normalizer:
        # To use presto pretrained normalizer we want to leave normalization to the presto wrapper
        presto_args = dataset_args
        presto_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NO_NORM"
                for task_name in EVAL_TASKS.keys()
            ]
        )

        presto_args += " " + "--model.use_pretrained_normalizer=True"
    else:
        # IF we use dataset stats we want to turn off the pretrained normalizer
        presto_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.STANDARDIZE"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        presto_args += " " + "--model.use_pretrained_normalizer=False"
    return presto_args


def get_prithviv2_args(pretrained_normalizer: bool = True) -> str:
    """Get the Prithvi arguments."""
    prithvi_args = dataset_args
    if pretrained_normalizer:
        # To use Prithvi pretrained normalizer we want to leave normalization to the Prithvi wrapper
        prithvi_args = dataset_args
        prithvi_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NO_NORM"
                for task_name in EVAL_TASKS.keys()
            ]
        )

        prithvi_args += " " + "--model.use_pretrained_normalizer=True"
    else:
        prithvi_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.STANDARDIZE"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        # IF we use dataset stats we want to turn off the pretrained normalizer
        prithvi_args += " " + "--model.use_pretrained_normalizer=False"

    return prithvi_args


def _get_sub_command(args: argparse.Namespace) -> str:
    """Determine the sub command based on args and cluster."""
    if args.dry_run:
        return SubCmd.dry_run
    elif args.cluster == "local":
        return SubCmd.train
    else:
        return SubCmd.launch


def _get_base_run_name(args: argparse.Namespace) -> str:
    """Generate the base run name from checkpoint path or model name."""
    if args.model_name is not None:
        logger.info(f"Overiding checkpoint name with {args.model_name}")
        run_name = args.model_name
    elif args.checkpoint_path is not None:
        parent_dir = os.path.basename(os.path.dirname(args.checkpoint_path))[:100]
        step_num = os.path.basename(args.checkpoint_path)
        run_name = f"{parent_dir}_{step_num}"
    else:
        logger.warning(
            "No model name provided or checkpoint path, using random run name"
        )
        run_name = str(uuid.uuid4())[:8]
    return run_name


def _get_checkpoint_args(checkpoint_path: str) -> str:
    """Generate checkpoint arguments string."""
    if checkpoint_path is not None:
        return f"--trainer.load_path={checkpoint_path}"
    return ""


def _get_model_specific_args(args: argparse.Namespace) -> str:
    """Get model-specific command arguments."""
    if args.dino_v3:
        return get_dino_v3_args()
    elif args.panopticon:
        return get_panopticon_args()
    elif args.galileo:
        return get_galileo_args()
    elif args.satlas:
        return get_satlas_args()
    elif args.croma:
        return get_croma_args()
    elif args.copernicusfm:
        return get_copernicusfm_args()
    elif args.presto:
        return get_presto_args()
    elif args.anysat:
        return get_anysat_args()
    elif args.tessera:
        return get_tessera_args()
    elif args.prithvi_v2:
        return get_prithviv2_args()
    return ""


def _get_normalization_args(args: argparse.Namespace, norm_mode: str) -> str:
    """Get normalization-specific command arguments."""
    model_map = {
        "galileo": get_galileo_args,
        "tessera": get_tessera_args,
        "prithvi_v2": get_prithviv2_args,
        "satlas": get_satlas_args,
        "presto": get_presto_args,
    }
    for model, func in model_map.items():
        if getattr(args, model, False):
            return func(pretrained_normalizer=(norm_mode == "pre_trained"))
    if norm_mode == "dataset":
        return dataset_args
    if norm_mode == "pre_trained":
        return helios_args
    return ""


def _build_default_command(
    args: argparse.Namespace,
    base_run_name: str,
    sub_command: str,
    launch_command: str,
    checkpoint_args: str,
    project_name: str,
    extra: str,
) -> str:
    """Build command for running with default hyperparameters."""
    lr = LP_LRs[0]
    norm_mode = Normalization_MODES[0]
    pooling_type = pooling_types[0]
    logger.info(
        f"Running defaults: {norm_mode} normalization, lr={lr}, pooling={pooling_type}"
    )
    run_name = f"{base_run_name}_defaults"

    cmd_args = _get_model_specific_args(args)
    module_path = (
        args.module_path if args.module_path is not None else _get_module_path(args)
    )
    logger.info(f"Using module path {module_path}")

    return (
        f"TRAIN_SCRIPT_PATH={module_path} {launch_command} helios/internal/all_evals.py "
        f"{sub_command} {run_name} {args.cluster} --launch.priority=high "
        f"--launch.task_name=eval {checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra} {cmd_args}"
    )


def _build_hyperparameter_command(
    args: argparse.Namespace,
    params: dict,
    base_run_name: str,
    sub_command: str,
    launch_command: str,
    checkpoint_args: str,
    project_name: str,
    extra: str,
) -> str:
    """Build command for running with specific hyperparameters."""
    lr = params.get("lr", None)
    norm_mode = params.get("norm_mode", "fixed")
    pooling_type = params.get("pooling_type", "default")

    logger.info(f"Running with {norm_mode} normalization and {lr} learning rate")
    logger.info(
        f"Running with module path {args.module_path} on cluster {args.cluster}"
    )

    run_name = f"{base_run_name}_{norm_mode}_lr{lr}_pooling{pooling_type}"
    cmd_args = lr_args.format(arg=lr)

    if pooling_type != "default":
        cmd_args += pooling_args.format(arg=pooling_type)

    # Add model-specific args
    cmd_args += _get_model_specific_args(args)

    # Add normalization-specific args
    # These args will override the model-specific args
    cmd_args += _get_normalization_args(args, norm_mode)

    return (
        f"TRAIN_SCRIPT_PATH={args.module_path} {launch_command} helios/internal/all_evals.py "
        f"{sub_command} {run_name} {args.cluster} --launch.priority=high {cmd_args} "
        f"--launch.task_name=eval {checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra}"
    )


def _get_module_path(args: argparse.Namespace) -> str:
    """Get the module path for the launch script."""
    if args.dino_v3:
        return get_launch_script_path("dino_v3")
    elif args.panopticon:
        return get_launch_script_path("panopticon")
    elif args.croma:
        return get_launch_script_path("croma")
    elif args.galileo:
        return get_launch_script_path("galileo")
    elif args.presto:
        return get_launch_script_path("presto")
    elif args.satlas:
        return get_launch_script_path("satlas")
    elif args.tessera:
        return get_launch_script_path("tessera")
    elif args.prithvi_v2:
        return get_launch_script_path("prithvi_v2")
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")


def build_commands(args: argparse.Namespace, extra_cli: list[str]) -> list[str]:
    """Build the commands for the sweep."""
    project_name = args.project_name or "helios_in_loop_evals"
    extra = " " + " ".join(extra_cli) if extra_cli else ""

    sub_command = _get_sub_command(args)
    base_run_name = _get_base_run_name(args)
    launch_command = "python3" if not sub_command == SubCmd.train else "torchrun"
    checkpoint_args = _get_checkpoint_args(args.checkpoint_path)

    commands_to_run = []

    if args.defaults_only:
        # Just run with the first/default values
        cmd = _build_default_command(
            args,
            base_run_name,
            sub_command,
            launch_command,
            checkpoint_args,
            project_name,
            extra,
        )
        commands_to_run.append(cmd)
    else:
        hp_params = (
            loop_through_params()
            if not args.dino_v3
            and not args.panopticon
            and not args.copernicusfm  # Only use the dataset normalization stats for these models
            and not args.tessera  # Only use the dataset normalization stats for these models
            else no_norm_sweep()
        )

        for params in hp_params:
            cmd = _build_hyperparameter_command(
                args,
                params,
                base_run_name,
                sub_command,
                launch_command,
                checkpoint_args,
                project_name,
                extra,
            )
            commands_to_run.append(cmd)

    return commands_to_run


def main() -> None:
    """Run the full evaluation sweep or just the defaults."""
    parser = argparse.ArgumentParser()
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
        help="Path to module .py",
    )
    parser.add_argument(
        "--project_name", type=str, required=False, help="Wandb project name"
    )
    parser.add_argument(
        "--defaults_only",
        action="store_true",
        help="If set, only run with default values (no sweep)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print the commands that would be run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="If set, use this as the  base run name",
    )
    parser.add_argument(
        "--dino_v3",
        action="store_true",
        help="If set, use the dino v3 normalization settings",
    )
    parser.add_argument(
        "--panopticon",
        action="store_true",
        help="If set, use the panopticon normalization settings",
    )
    parser.add_argument(
        "--galileo",
        action="store_true",
        help="If set, use the galileo normalization settings",
    )
    parser.add_argument(
        "--satlas",
        action="store_true",
        help="If set, use the satlas normalization settings",
    )
    parser.add_argument(
        "--croma",
        action="store_true",
        help="If set, use the croma normalization settings",
    )
    parser.add_argument(
        "--copernicusfm",
        action="store_true",
        help="If set, use the copernicusfm normalization settings",
    )
    parser.add_argument(
        "--presto",
        action="store_true",
        help="If set, use the presto normalization settings",
    )
    parser.add_argument(
        "--anysat",
        action="store_true",
        help="If set, use the anysat normalization settings",
    )
    parser.add_argument(
        "--tessera",
        action="store_true",
        help="If set, use the tessera normalization settings",
    )
    parser.add_argument(
        "--prithvi_v2",
        action="store_true",
        help="If set, use the prithvi normalization settings",
    )
    args, extra_cli = parser.parse_known_args()

    commands_to_run = build_commands(args, extra_cli)

    for cmd in commands_to_run:
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
