"""Run an evaluation sweep for an arbitrary OlmoEarth Pretrain checkpoint.

e.g. python -m olmoearth_pretrain.internal.full_eval_sweep --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0078125/step450000  --module_path=scripts/2025_06_26_dataset_percentage_experiments/latent_mim_all_data.py (extra args here e.g --model.decoder_config.depth=1)
"""

import argparse
import json
import os
import subprocess  # nosec
import uuid
from collections.abc import Generator
from logging import getLogger
from typing import Any

from olmoearth_pretrain.evals.datasets.configs import dataset_to_config, get_eval_mode
from olmoearth_pretrain.evals.models import (
    MODELS_WITH_MULTIPLE_SIZES,
    BaselineModelName,
    get_launch_script_path,
)
from olmoearth_pretrain.internal.all_evals import EVAL_TASKS
from olmoearth_pretrain.internal.constants import EVAL_LAUNCH_PATH, EVAL_WANDB_PROJECT
from olmoearth_pretrain.internal.experiment import SubCmd
from olmoearth_pretrain.nn.flexi_vit import PoolingType

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
Normalization_MODES = ["pre_trained", "dataset"]
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

quantize_args = " ".join(
    [" "]
    + [
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.quantize_embeddings=True"
        for task_name in EVAL_TASKS.keys()
    ]
)


def get_embedding_dim_args(dim: int) -> str:
    """Get embedding dim args for all tasks."""
    return " ".join(
        [" "]
        + [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.embedding_dim={dim}"
            for task_name in EVAL_TASKS.keys()
        ]
    )


dataset_args = " ".join(
    [" "]
    + [
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_stats_from_pretrained=False"
        for task_name in EVAL_TASKS.keys()
    ]
)

olmoearth_args = " ".join(
    [""]
    + [
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_stats_from_pretrained=True"
        for task_name in EVAL_TASKS.keys()
    ]
)


def loop_through_params(no_norm: bool = False) -> Generator[dict[str, Any], None, None]:
    """Yield a dict of the hps we are sweeping over."""
    if no_norm:
        normalization_modes = ["dataset"]
    else:
        normalization_modes = Normalization_MODES
    for lr in LP_LRs:
        for norm_mode in normalization_modes:
            for pooling_type in pooling_types:
                yield {
                    "lr": lr,
                    "norm_mode": norm_mode,
                    "pooling_type": pooling_type,
                }


def lr_only_params() -> Generator[dict[str, Any], None, None]:
    """Yield a dict of the hps we are sweeping over."""
    for lr in LP_LRs:
        yield {
            "lr": lr,
        }


def select_best_val_args() -> str:
    """Get the early stopping arguments.

    This is used to select the final test miou based on the epoch of the max val miou.
    """
    return " ".join(
        [
            f" --trainer.callbacks.downstream_evaluator.tasks.{task_name}.select_final_test_miou_based_on_epoch_of_max_val_miou=True  --trainer.callbacks.downstream_evaluator.tasks.{task_name}.linear_probe_eval_interval=5"
            for task_name in EVAL_TASKS.keys()
        ]
    )


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


def get_terramind_args(pretrained_normalizer: bool = True) -> str:
    """Get the terramind arguments."""
    terramind_args = dataset_args
    if pretrained_normalizer:
        # To use terramind pretrained normalizer we want to leave normalization to the terramind wrapper
        terramind_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NO_NORM"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        terramind_args += " " + "--model.use_pretrained_normalizer=True"
    else:
        # IF we use dataset stats we want to turn off the pretrained normalizer
        terramind_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.STANDARDIZE"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        terramind_args += " " + "--model.use_pretrained_normalizer=False"
    return terramind_args


def get_clay_args(pretrained_normalizer: bool = True) -> str:
    """Get the clay arguments."""
    clay_args = dataset_args
    if pretrained_normalizer:
        # To use clay pretrained normalizer we want to leave normalization to the clay wrapper
        clay_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NO_NORM"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        clay_args += " " + "--model.use_pretrained_normalizer=True"
    else:
        # IF we use dataset stats we want to turn off the pretrained normalizer
        clay_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.STANDARDIZE"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        clay_args += " " + "--model.use_pretrained_normalizer=False"
    return clay_args


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
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.embedding_batch_size=2"
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
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NORM_NO_CLIP_2_STD"
                for task_name in EVAL_TASKS.keys()
            ]
        )
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
        return SubCmd.dry_run_evaluate
    # If cluster is local, we run eval locally, if not, we launch evaluation on beaker
    if args.cluster == "local":
        return SubCmd.evaluate
    else:
        return SubCmd.launch_evaluate


def _get_base_run_name(
    args: argparse.Namespace, size: str | None = None, use_uuid: bool = True
) -> str:
    """Generate the base run name from checkpoint path or model name."""
    if use_uuid:
        uuid_str = "_" + str(uuid.uuid4())[:4]
    else:
        uuid_str = ""
    if args.model_name is not None:
        logger.info(f"Overiding checkpoint name with {args.model_name}")
        run_name = args.model_name
    elif args.checkpoint_path is not None:
        parent_dir = os.path.basename(os.path.dirname(args.checkpoint_path))[:100]
        step_num = os.path.basename(args.checkpoint_path)
        run_name = f"{parent_dir}_{step_num}"
    elif args.model is not None:
        if size is not None:
            size_str = f"_{size}"
        else:
            size_str = ""
        run_name = args.model + size_str + uuid_str
    else:
        logger.warning(
            "No model name provided or checkpoint path, using random run name"
        )
        run_name = uuid_str
    return run_name


def _get_checkpoint_args(checkpoint_path: str) -> str:
    """Generate checkpoint arguments string."""
    if checkpoint_path is not None:
        return f"--trainer.load_path={checkpoint_path}"
    return ""


# TODO: Explain why some models are not in the map
def _get_model_specific_args(model: BaselineModelName | None) -> str:
    """Get model-specific command arguments."""
    model_args_map = {
        BaselineModelName.DINO_V3: get_dino_v3_args,
        BaselineModelName.PANOPTICON: get_panopticon_args,
        BaselineModelName.GALILEO: get_galileo_args,
        BaselineModelName.SATLAS: get_satlas_args,
        BaselineModelName.CROMA: get_croma_args,
        BaselineModelName.PRESTO: get_presto_args,
        BaselineModelName.ANYSAT: get_anysat_args,
        BaselineModelName.TESSERA: get_tessera_args,
        BaselineModelName.PRITHVI_V2: get_prithviv2_args,
        BaselineModelName.TERRAMIND: get_terramind_args,
        BaselineModelName.CLAY: get_clay_args,
    }
    if model is None or model not in model_args_map:
        return ""

    return model_args_map[model]()  # type: ignore


def get_olmoearth_args(pretrained_normalizer: bool = True) -> str:
    """Get the olmoearth arguments."""
    if pretrained_normalizer:
        return olmoearth_args
    else:
        olmoearth_dataset_args = dataset_args
        olmoearth_dataset_args += " " + " ".join(
            [
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method=NormMethod.NORM_NO_CLIP_2_STD"
                for task_name in EVAL_TASKS.keys()
            ]
        )
        return olmoearth_dataset_args


# TODO: Explain why some models are not in the map
def _get_normalization_args(model: BaselineModelName | None, norm_mode: str) -> str:
    """Get normalization-specific command arguments."""
    if model is None:
        # If model is None, we want to use the olmoearth arguments
        return get_olmoearth_args(pretrained_normalizer=(norm_mode == "pre_trained"))
    model_map = {
        BaselineModelName.GALILEO: get_galileo_args,
        BaselineModelName.TESSERA: get_tessera_args,
        BaselineModelName.PRITHVI_V2: get_prithviv2_args,
        BaselineModelName.SATLAS: get_satlas_args,
        BaselineModelName.PRESTO: get_presto_args,
        BaselineModelName.TERRAMIND: get_terramind_args,
        BaselineModelName.CLAY: get_clay_args,
    }

    if model in model_map:
        return model_map[model](pretrained_normalizer=(norm_mode == "pre_trained"))

    if norm_mode == "dataset":
        return dataset_args
    if norm_mode == "pre_trained":
        return olmoearth_args
    return ""


def _get_model_size_args(model: BaselineModelName | None, size: str | None) -> str:
    """Get the model size arguments."""
    if model in MODELS_WITH_MULTIPLE_SIZES:
        if size is not None:
            return f" --model.size={size}"
    return ""


def _get_load_checkpoints_args(model: BaselineModelName | None) -> str:
    """Get the no checkpoints arguments."""
    if model is None:
        # Allow load model for olmoearth checkpoints
        return " --trainer.no_checkpoints=False"
    return " --trainer.no_checkpoints=True"


def _get_norm_mode_str(norm_mode: str) -> str:
    """Get the normalization mode string."""
    if norm_mode == "default":
        norm_mode_str = "df"
    else:
        norm_mode_str = norm_mode
    return norm_mode_str


def _get_pooling_type_str(pooling_type: str) -> str:
    """Get the pooling type string."""
    if pooling_type == "default":
        pooling_type_str = "df"
    else:
        pooling_type_str = pooling_type
    return pooling_type_str


LAUNCH_OVERRIDES = "--launch.priority=high --launch.num_gpus=1 --launch.task_name=eval"


def _build_default_command(
    args: argparse.Namespace,
    base_run_name: str,
    sub_command: str,
    launch_command: str,
    checkpoint_args: str,
    project_name: str,
    extra: str,
    size: str | None = None,
) -> str:
    """Build command for running with default hyperparameters."""
    lr = LP_LRs[0]
    norm_mode = Normalization_MODES[0]
    pooling_type = pooling_types[0]
    logger.info(
        f"Running defaults: {norm_mode} normalization, lr={lr}, pooling={pooling_type}"
    )
    run_name = f"{base_run_name}_df"
    cmd_args = _get_model_specific_args(args.model)

    # Add normalization-specific args
    cmd_args += _get_normalization_args(args.model, norm_mode)

    module_path = (
        args.module_path
        if args.module_path is not None
        else _get_module_path(args.model)
    )
    logger.info(f"Using module path {module_path}")
    cmd_args += _get_model_size_args(args.model, size)
    cmd_args += _get_load_checkpoints_args(args.model)

    # Add quantization args if enabled
    if getattr(args, "quantize_embeddings", False):
        cmd_args += quantize_args
        run_name += "_qt"

    # Add embedding dim args if enabled
    embedding_dim = getattr(args, "embedding_dim", None)
    if embedding_dim is not None:
        cmd_args += get_embedding_dim_args(embedding_dim)
        run_name += f"_dim{embedding_dim}"

    launch_overrides = LAUNCH_OVERRIDES if sub_command == SubCmd.launch_evaluate else ""
    return (
        f"TRAIN_SCRIPT_PATH={module_path} {launch_command} {EVAL_LAUNCH_PATH} "
        f"{sub_command} {run_name} {args.cluster} {launch_overrides} "
        f"{checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra} {cmd_args}"
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
    size: str | None = None,
) -> str:
    """Build command for running with specific hyperparameters."""
    lr = params.get("lr", None)
    norm_mode = params.get("norm_mode", "fixed")
    pooling_type = params.get("pooling_type", "default")

    logger.info(f"Running with {norm_mode} normalization and {lr} learning rate")
    logger.info(
        f"Running with module path {args.module_path} on cluster {args.cluster}"
    )
    # map default to df
    norm_mode_str = _get_norm_mode_str(norm_mode)
    pooling_type_str = _get_pooling_type_str(pooling_type)
    run_name = f"{base_run_name}_{norm_mode_str}_lr{lr}_pt{pooling_type_str}"
    cmd_args = lr_args.format(arg=lr)

    if pooling_type != "default":
        cmd_args += pooling_args.format(arg=pooling_type)

    # Add model-specific args
    cmd_args += _get_model_specific_args(args.model)

    # Add normalization-specific args
    # These args will override the model-specific args
    cmd_args += _get_normalization_args(args.model, norm_mode)
    module_path = (
        args.module_path
        if args.module_path is not None
        else _get_module_path(args.model)
    )
    cmd_args += _get_load_checkpoints_args(args.model)
    cmd_args += _get_model_size_args(args.model, size)

    # Add quantization args if enabled
    if getattr(args, "quantize_embeddings", False):
        cmd_args += quantize_args
        run_name += "_qt"

    # Add embedding dim args if enabled
    embedding_dim = getattr(args, "embedding_dim", None)
    if embedding_dim is not None:
        cmd_args += get_embedding_dim_args(embedding_dim)
        run_name += f"_dim{embedding_dim}"

    launch_overrides = LAUNCH_OVERRIDES if sub_command == SubCmd.launch_evaluate else ""
    # if init_seed is set add to base run name
    if "init_seed" in extra:
        run_name += f"_seed{extra.split('init_seed=')[1].split(' ')[0]}"
    return (
        f"TRAIN_SCRIPT_PATH={module_path} {launch_command} {EVAL_LAUNCH_PATH} "
        f"{sub_command} {run_name} {args.cluster} {launch_overrides} {cmd_args} "
        f"{checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra}"
    )


def _build_command_from_eval_settings(
    args: argparse.Namespace,
    eval_settings_dict: dict,
    base_run_name: str,
    sub_command: str,
    launch_command: str,
    checkpoint_args: str,
    project_name: str,
    extra: str,
    size: str | None = None,
) -> str:
    """Build a command from eval settings with per-task lr, pooling, and normalization."""
    logger.info("Building command with per-task eval settings from loaded JSON")
    logger.info(
        f"Running with module path {args.module_path} on cluster {args.cluster}"
    )

    # Build per-task command arguments
    cmd_args_parts = []

    # Track unique settings for the run name
    pooling_types_used = set()
    lrs_used = set()
    norm_modes_used = set()

    for task_name, task_data in eval_settings_dict.items():
        settings = task_data["settings"]

        # Extract settings for this task
        pooling_type = settings.get("pooling_type", "mean")
        probe_lr = settings.get("probe_lr", None)
        norm_from_pretrained = settings.get("norm_stats_from_pretrained", False)
        quantize_embeddings = settings.get("quantize_embeddings", False)

        # Track for run name
        pooling_types_used.add(pooling_type)
        if probe_lr is not None:
            lrs_used.add(probe_lr)
        norm_modes_used.add("pre_trained" if norm_from_pretrained else "dataset")

        # Build per-task args
        task_args = []

        # Add probe_lr if this is a linear probe task
        if probe_lr is not None:
            task_args.append(
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.probe_lr={probe_lr}"
            )

        # Add pooling type
        task_args.append(
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.pooling_type={pooling_type}"
        )

        # Add normalization setting
        task_args.append(
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_stats_from_pretrained={norm_from_pretrained}"
        )

        # Add quantization setting if specified in JSON
        if quantize_embeddings:
            task_args.append(
                f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.quantize_embeddings=True"
            )

        cmd_args_parts.extend(task_args)

    # Create a descriptive run name
    # Use "mixed" if multiple different settings are used
    pooling_str = (
        "mixed"
        if len(pooling_types_used) > 1
        else _get_pooling_type_str(list(pooling_types_used)[0])
    )
    lr_str = (
        "mixed"
        if len(lrs_used) > 1
        else (f"lr{list(lrs_used)[0]}" if lrs_used else "knn")
    )
    norm_str = (
        "mixed"
        if len(norm_modes_used) > 1
        else _get_norm_mode_str(list(norm_modes_used)[0])
    )

    run_name = f"{base_run_name}_{norm_str}_{lr_str}_pt{pooling_str}"

    # Check if quantization is enabled (either from args or from JSON settings)
    quantize_enabled = getattr(args, "quantize_embeddings", False) or any(
        task_data.get("settings", {}).get("quantize_embeddings", False)
        for task_data in eval_settings_dict.values()
    )
    if quantize_enabled:
        run_name += "_qt"

    cmd_args = " " + " ".join(cmd_args_parts)

    # Add quantization args for all tasks if enabled via command line (not already in JSON)
    if getattr(args, "quantize_embeddings", False):
        # Only add if not already specified per-task in JSON
        for task_name in EVAL_TASKS.keys():
            if task_name not in eval_settings_dict or not eval_settings_dict[
                task_name
            ].get("settings", {}).get("quantize_embeddings", False):
                cmd_args += f" --trainer.callbacks.downstream_evaluator.tasks.{task_name}.quantize_embeddings=True"

    # Add model-specific args
    cmd_args += _get_model_specific_args(args.model)

    # Add model-specific normalization args if needed (this may get overridden by task-specific args)
    # Use the first norm mode found, or default to dataset
    first_norm_mode = list(norm_modes_used)[0] if norm_modes_used else "dataset"
    cmd_args += _get_normalization_args(args.model, first_norm_mode)

    module_path = (
        args.module_path
        if args.module_path is not None
        else _get_module_path(args.model)
    )
    cmd_args += _get_load_checkpoints_args(args.model)
    cmd_args += _get_model_size_args(args.model, size)

    # Add quantization args if enabled
    if getattr(args, "quantize_embeddings", False):
        cmd_args += quantize_args
        run_name += "_qt"

    # Add embedding dim args if enabled
    embedding_dim = getattr(args, "embedding_dim", None)
    if embedding_dim is not None:
        cmd_args += get_embedding_dim_args(embedding_dim)
        run_name += f"_dim{embedding_dim}"

    launch_overrides = LAUNCH_OVERRIDES if sub_command == SubCmd.launch_evaluate else ""
    # if init_seed is set add to base run name
    if "init_seed" in extra:
        run_name += f"_seed{extra.split('init_seed=')[1].split(' ')[0]}"
    return (
        f"TRAIN_SCRIPT_PATH={module_path} {launch_command} {EVAL_LAUNCH_PATH} "
        f"{sub_command} {run_name} {args.cluster} {launch_overrides} {cmd_args} "
        f"{checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra}"
    )


def _get_module_path(model: BaselineModelName | None) -> str:
    """Get the module path for the launch script."""
    if model is None:
        raise ValueError("Model must be specified when module_path is not provided")
    return get_launch_script_path(model)


def build_commands(args: argparse.Namespace, extra_cli: list[str]) -> list[str]:
    """Build the commands for the sweep."""
    project_name = args.project_name or EVAL_WANDB_PROJECT
    extra = " " + " ".join(extra_cli) if extra_cli else ""

    sub_command = _get_sub_command(args)
    launch_command = "python3" if not sub_command == SubCmd.evaluate else "torchrun"
    checkpoint_args = _get_checkpoint_args(args.checkpoint_path)

    commands_to_run = []

    if args.defaults_only:
        if args.model == "all":
            raise ValueError("Cannot run defaults with all models")
        # Just run with the first/default values
        base_run_name = _get_base_run_name(args, args.size)
        cmd = _build_default_command(
            args,
            base_run_name,
            sub_command,
            launch_command,
            checkpoint_args,
            project_name,
            extra,
            args.size,
        )
        commands_to_run.append(cmd)
    elif args.lr_only:
        # only sweep the learning rates use mean pooling  and whatever normalization works best
        base_run_name = _get_base_run_name(args, args.size)
        lr_params = lr_only_params()

        for params in lr_params:
            cmd = _build_hyperparameter_command(
                args,
                params,
                base_run_name,
                sub_command,
                launch_command,
                checkpoint_args,
                project_name,
                extra,
                args.size,
            )
            commands_to_run.append(cmd)
    else:
        if args.model == "all":
            models = list(BaselineModelName)
            # Filter out skipped models if model-skip-names is provided
            if args.model_skip_names:
                skip_names = [name.strip() for name in args.model_skip_names.split(",")]
                models = [model for model in models if model not in skip_names]
        else:
            models = [args.model]
        for model in models:
            args.model = model
            # Models that only use dataset normalization or need dataset normalization to scale to 0 - 1 then always use pretrained
            dataset_norm_only_models = {
                BaselineModelName.DINO_V3,
                BaselineModelName.PANOPTICON,
                BaselineModelName.TESSERA,
            }
            if args.size is not None:
                model_sizes = [args.size]
            else:
                model_sizes = (
                    MODELS_WITH_MULTIPLE_SIZES.get(
                        args.model,
                        [None],  # type: ignore # TODO: Fix this
                    )
                    if args.all_sizes
                    else [None]
                )

            for size in model_sizes:
                base_run_name = _get_base_run_name(args, size)

                # Optionally load imported settings from json file
                if args.load_eval_settings_from_json:
                    with open(args.load_eval_settings_from_json) as f:
                        eval_settings = json.load(f)
                    # Get all tasks for this group/run
                    base_run_name_og = _get_base_run_name(args, size, use_uuid=False)
                    if "step" in base_run_name_og:
                        base_run_name_og = base_run_name_og.split("_step")[0]
                    suffixes_to_try = ["", "_base", "_large"]
                    eval_settings_dict = None

                    for suffix in suffixes_to_try:
                        try:
                            lookup_name = base_run_name_og + suffix
                            eval_settings_dict = eval_settings[lookup_name]
                            base_run_name = lookup_name  # Update base_run_name to the successful lookup
                            break
                        except KeyError:
                            continue

                    if eval_settings_dict is None:
                        raise KeyError(
                            f"Could not find eval settings for {base_run_name} with any of the suffixes: {suffixes_to_try}"
                        )

                    base_run_name += "_from_json_settings"

                    cmd = _build_command_from_eval_settings(
                        args,
                        eval_settings_dict,  # This now contains all tasks with their settings
                        base_run_name,
                        sub_command,
                        launch_command,
                        checkpoint_args,
                        project_name,
                        extra,
                        size,
                    )
                    commands_to_run.append(cmd)
                    continue

                hp_params = loop_through_params(
                    no_norm=(args.model in dataset_norm_only_models)
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
                        size,
                    )
                    commands_to_run.append(cmd)

    if args.select_best_val:
        commands_to_run_new = []
        for cmd in commands_to_run:
            logger.info(f"Adding select best val args to {cmd}")
            cmd += select_best_val_args()
            commands_to_run_new.append(cmd)
        commands_to_run = commands_to_run_new

    # Filter out skipped tasks if task-skip-names is provided
    if args.task_skip_names:
        skip_names = [name.strip() for name in args.task_skip_names.split(",")]
        tasks_to_run = [task for task in EVAL_TASKS.keys() if task not in skip_names]
        tasks_to_run_arg = f" --trainer.callbacks.downstream_evaluator.tasks_to_run='{json.dumps(tasks_to_run)}'"
        commands_to_run_new = []
        for cmd in commands_to_run:
            logger.info(f"Adding tasks_to_run filter to {cmd}")
            cmd += tasks_to_run_arg
            commands_to_run_new.append(cmd)
        commands_to_run = commands_to_run_new

    return commands_to_run


def _parse_model_arg(value: str) -> BaselineModelName | str:
    """Parse the model argument, returning either a BaselineModelName or 'all'."""
    if value == "all":
        return value
    try:
        return BaselineModelName(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid model: {value}. Must be one of {list(BaselineModelName)} or 'all'"
        )


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
        "--model",
        type=_parse_model_arg,
        required=False,
        default=None,
        help="Baseline model to use (e.g., dino_v3, galileo, satlas) or all",
    )
    parser.add_argument(
        "--all_sizes",
        action="store_true",
        help="If set, run all sizes for each model",
    )
    parser.add_argument(
        "--lr_only",
        action="store_true",
        help="If set, only run with default values (no sweep)",
    )
    parser.add_argument(
        "--select_best_val",
        action="store_true",
        help="If set, use select best val on the linear probe evals",
    )
    parser.add_argument(
        "--model-skip-names",
        type=str,
        required=False,
        help="Comma-separated list of model names to skip when --model=all is set",
    )
    parser.add_argument(
        "--task-skip-names",
        type=str,
        required=False,
        help="Comma-separated list of task names to skip (e.g., pastis128_sentinel2,pastis128_sentinel1)",
    )
    parser.add_argument(
        "--size",
        type=str,
        required=False,
        help="Model size to use",
    )
    parser.add_argument(
        "--load_eval_settings_from_json",
        type=str,
        required=False,
        help="Path to the eval settings json file",
    )
    parser.add_argument(
        "--quantize_embeddings",
        action="store_true",
        help="If set, quantize embeddings to int8 for all tasks",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="If set, reduce embeddings to this dimensionality via PCA (e.g., 128, 64)",
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
