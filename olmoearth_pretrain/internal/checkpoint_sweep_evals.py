"""Evaluate multiple checkpoints from a training run, logging to a single wandb run.

Each checkpoint's evaluation metrics are logged at its training step,
so you can visualize eval performance over the course of training.

Usage via full_eval_sweep.py (recommended):
    python -m olmoearth_pretrain.internal.full_eval_sweep \
        --checkpoint_dir=/weka/.../checkpoints/henryh/my_run \
        --cluster=ai2/saturn-cirrascale \
        --module_path=scripts/my_train.py

Direct local usage:
    TRAIN_SCRIPT_PATH=scripts/my_train.py \
    CHECKPOINT_DIR=/weka/.../checkpoints/henryh/my_run \
    torchrun olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
        evaluate my_run_sweep local

Beaker launch:
    TRAIN_SCRIPT_PATH=scripts/my_train.py \
    CHECKPOINT_DIR=/weka/.../checkpoints/henryh/my_run \
    python3 olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
        launch_evaluate my_run_sweep ai2/saturn-cirrascale
"""

import gc
import logging
import os
import re
import sys
import time
from typing import cast

import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.distributed.utils import get_rank
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    BeakerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

from olmoearth_pretrain.internal.all_evals import EVAL_TASKS, load_user_module
from olmoearth_pretrain.internal.constants import EVAL_WANDB_PROJECT, WANDB_ENTITY
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthEvaluateConfig,
    SubCmd,
    build_evaluate_config,
    launch,
)
from olmoearth_pretrain.internal.utils import (
    MockLatentMIMTrainModule,
    MockOlmoEarthDataLoader,
)
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamEvaluatorCallback,
)

logger = logging.getLogger(__name__)


def discover_checkpoints(
    checkpoint_dir: str, steps: list[int] | None = None
) -> list[tuple[int, str]]:
    """Find step{N}/ directories in checkpoint_dir, sorted by step number."""
    step_dirs = []
    for entry in os.listdir(checkpoint_dir):
        match = re.match(r"^step(\d+)$", entry)
        if match:
            step_num = int(match.group(1))
            full_path = os.path.join(checkpoint_dir, entry)
            if os.path.isdir(full_path):
                if steps is None or step_num in steps:
                    step_dirs.append((step_num, full_path))
    step_dirs.sort()
    return step_dirs


def evaluate_checkpoints(
    config: OlmoEarthEvaluateConfig,
    checkpoint_dir: str,
    steps: list[int] | None = None,
) -> None:
    """Evaluate all checkpoints in checkpoint_dir, logging to one wandb run."""
    seed_all(config.init_seed)

    checkpoints = discover_checkpoints(checkpoint_dir, steps=steps)
    if not checkpoints:
        raise ValueError(f"No step directories found in {checkpoint_dir}")
    logger.info(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    # Build model
    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    data_loader = MockOlmoEarthDataLoader()

    # Build train module if available (needed for proper model architecture init)
    if config.train_module is not None:
        train_module = config.train_module.build(model)
        data_loader.min_patch_size = model.encoder.min_patch_size
        data_loader.max_patch_size = model.encoder.max_patch_size
    else:
        train_module = MockLatentMIMTrainModule()
    train_module.model = model

    # Build trainer (wires up callbacks including evaluators and wandb)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    wandb_callback = cast(OlmoEarthWandBCallback, trainer.callbacks["wandb"])
    wandb_callback.config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Init wandb (without running evals or starting the training loop)
    wandb_callback.pre_train()

    # Tell wandb to use checkpoint_step as the x-axis for eval metrics
    if wandb_callback.enabled and get_rank() == 0:
        wandb_callback.wandb.define_metric("checkpoint_step")
        wandb_callback.wandb.define_metric("eval/*", step_metric="checkpoint_step")
        wandb_callback.wandb.define_metric("eval/test/*", step_metric="checkpoint_step")
        wandb_callback.wandb.define_metric("eval_time/*", step_metric="checkpoint_step")

    # Get the evaluator callback (contains the built evaluator objects)
    eval_callback = trainer.callbacks.get("downstream_evaluator")
    if not isinstance(eval_callback, DownstreamEvaluatorCallback):
        raise ValueError("downstream_evaluator callback not found or disabled")

    for step_num, step_path in checkpoints:
        logger.info(f"=== Evaluating checkpoint step {step_num}: {step_path} ===")

        # Load model weights from the distributed checkpoint
        train_module_dir = os.path.join(step_path, "model_and_optim")
        load_model_and_optim_state(train_module_dir, model)
        model.to(device)

        # Run all evaluators and collect metrics for this checkpoint
        metrics: dict[str, float | int] = {"checkpoint_step": step_num}

        for evaluator in eval_callback.evaluators:
            if not eval_callback._check_supported_modalities(evaluator):
                logger.info(
                    f"  Skipping {evaluator.evaluation_name} (unsupported modalities)"
                )
                continue
            if not eval_callback._check_input_requirements(evaluator):
                logger.info(
                    f"  Skipping {evaluator.evaluation_name} (input requirements)"
                )
                continue

            start_time = time.monotonic()
            result = evaluator.val()
            eval_time = time.monotonic() - start_time

            val_result = result.val_result
            test_result = result.test_result

            if val_result is not None:
                metrics[f"eval/{evaluator.evaluation_name}"] = val_result.primary
                for k, v in val_result.metrics.items():
                    metrics[f"eval/{evaluator.evaluation_name}/{k}"] = v

            if eval_callback.run_on_test and test_result is not None:
                metrics[f"eval/test/{evaluator.evaluation_name}"] = test_result.primary
                for k, v in test_result.metrics.items():
                    metrics[f"eval/test/{evaluator.evaluation_name}/{k}"] = v

            metrics[f"eval_time/{evaluator.evaluation_name}"] = eval_time

            logger.info(
                f"  {evaluator.evaluation_name}: "
                f"val={val_result.primary if val_result else 'N/A'}, "
                f"test={test_result.primary if test_result else 'N/A'} "
                f"({eval_time:.1f}s)"
            )

        # Log all metrics for this checkpoint in one call
        if wandb_callback.enabled and get_rank() == 0:
            wandb_callback.wandb.log(metrics)
            logger.info(f"Logged {len(metrics)} metrics for step {step_num}")

        gc.collect()
        torch.cuda.empty_cache()

    if wandb_callback.enabled and get_rank() == 0:
        wandb_callback.wandb.finish()
    logger.info("Checkpoint sweep evaluation complete.")


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build trainer config for checkpoint sweep (no training, no auto-eval)."""
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=EVAL_WANDB_PROJECT,
        entity=WANDB_ENTITY,
        enabled=True,
        upload_dataset_distribution_pre_train=False,
        upload_modality_data_band_distribution_pre_train=False,
    )
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            save_folder=common.save_folder,
            cancel_check_interval=1,
            metrics_collect_interval=10,
            max_duration=Duration.epochs(300),
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=EVAL_TASKS,
                eval_on_startup=False,
                cancel_after_first_eval=False,
                run_on_test=True,
            ),
        )
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
        .with_callback("beaker", BeakerCallback())
    )
    return trainer_config


def parse_steps(steps_str: str | None) -> list[int] | None:
    """Parse a comma-separated string of step numbers (e.g. '5000,10000,15000')."""
    if steps_str is None:
        return None
    return [int(s.strip()) for s in steps_str.split(",") if s.strip()]


if __name__ == "__main__":
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir is None:
        raise ValueError("CHECKPOINT_DIR environment variable must be set")

    module_path = os.environ.get("TRAIN_SCRIPT_PATH")
    if module_path is None:
        raise ValueError("TRAIN_SCRIPT_PATH environment variable must be set")

    # Optional: only evaluate specific steps (comma-separated)
    steps = parse_steps(os.environ.get("CHECKPOINT_STEPS"))

    user_mod = load_user_module(module_path)

    try:
        build_common_components = user_mod.build_common_components
    except AttributeError:
        from olmoearth_pretrain.internal.common import build_common_components

    try:
        build_train_module_config = user_mod.build_train_module_config
    except AttributeError:
        build_train_module_config = None

    try:
        build_model_config = user_mod.build_model_config
    except AttributeError:
        raise AttributeError(
            f"Module at {module_path} has no 'build_model_config'. "
            f"Point --module_path to the size-specific script "
            f"(e.g. scripts/official/base.py instead of scripts/official/script.py)."
        )

    usage = (
        f"Usage: CHECKPOINT_DIR=... TRAIN_SCRIPT_PATH=... "
        f"[CHECKPOINT_STEPS=5000,10000,...] "
        f"python3/torchrun {sys.argv[0]} "
        f"evaluate|launch_evaluate|dry_run_evaluate RUN_NAME CLUSTER [OVERRIDES...]"
    )
    if len(sys.argv) < 4:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv
    common = build_common_components(script, cmd, run_name, cluster, overrides)

    cmd = SubCmd(cmd)

    config = build_evaluate_config(
        common=common,
        model_config_builder=build_model_config,
        trainer_config_builder=build_trainer_config,
        overrides=overrides,
        train_module_config_builder=build_train_module_config,
    )

    if cmd == SubCmd.launch_evaluate:
        prepare_cli_environment()
        launch(config)
    elif cmd == SubCmd.evaluate:
        prepare_training_environment()
        try:
            evaluate_checkpoints(config, checkpoint_dir, steps=steps)
        finally:
            teardown_training_environment()
    elif cmd == SubCmd.dry_run_evaluate:
        prepare_cli_environment()
        logger.info(config)
        checkpoints = discover_checkpoints(checkpoint_dir, steps=steps)
        logger.info(
            f"Would evaluate {len(checkpoints)} checkpoints: "
            f"{[s for s, _ in checkpoints]}"
        )
    else:
        raise ValueError(
            f"Unsupported command: {cmd}. "
            f"Use evaluate, launch_evaluate, or dry_run_evaluate."
        )
