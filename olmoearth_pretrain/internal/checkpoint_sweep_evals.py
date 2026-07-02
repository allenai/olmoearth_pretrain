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
from functools import partial
from types import ModuleType
from typing import cast

import torch
from beaker import Experiment
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

from olmoearth_pretrain.internal.all_evals import (
    EMBED_DIAG_TASKS,
    EVAL_TASKS,
    load_user_module,
)
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
    CHECKPOINT_STEP_METRIC,
    DownstreamEvaluatorCallback,
    define_checkpoint_step_metrics,
    eval_result_log_dict,
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
    """Evaluate selected checkpoints and stream metrics to one wandb run.

    Each evaluator result is logged as soon as it finishes, with
    ``checkpoint_step`` included so W&B plots use the training checkpoint step as
    the x-axis instead of wall-clock eval order.
    """
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
        define_checkpoint_step_metrics(wandb_callback.wandb)

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
            metrics: dict[str, float | int] = {CHECKPOINT_STEP_METRIC: step_num}

            if val_result is not None:
                metrics.update(
                    eval_result_log_dict("eval", evaluator.evaluation_name, val_result)
                )

            if eval_callback.run_on_test and test_result is not None:
                metrics.update(
                    eval_result_log_dict(
                        "eval/test", evaluator.evaluation_name, test_result
                    )
                )

            if result.embedding_diagnostics:
                for k, v in result.embedding_diagnostics.items():
                    metrics[
                        f"eval_embed_diagnostics/{evaluator.evaluation_name}/{k}"
                    ] = v

            metrics[f"eval_time/{evaluator.evaluation_name}"] = eval_time

            logger.info(
                f"  {evaluator.evaluation_name}: "
                f"val={val_result.primary if val_result else 'N/A'}, "
                f"test={test_result.primary if test_result else 'N/A'} "
                f"({eval_time:.1f}s)"
            )

            if wandb_callback.enabled and get_rank() == 0:
                wandb_callback.wandb.log(metrics)
                logger.info(
                    f"Logged {len(metrics)} metrics for "
                    f"{evaluator.evaluation_name} at step {step_num}"
                )

        gc.collect()
        torch.cuda.empty_cache()

    if wandb_callback.enabled and get_rank() == 0:
        wandb_callback.wandb.finish()
    logger.info("Checkpoint sweep evaluation complete.")


def _get_eval_tasks() -> dict:
    """Select task set based on EMBEDDING_DIAGNOSTICS_ONLY env var."""
    if os.environ.get("EMBEDDING_DIAGNOSTICS_ONLY"):
        return EMBED_DIAG_TASKS
    return EVAL_TASKS


def build_trainer_config(
    common: CommonComponents, eval_tasks: dict | None = None
) -> TrainerConfig:
    """Build trainer config for checkpoint sweep (no training, no auto-eval).

    ``eval_tasks`` overrides the task set the downstream evaluator runs. In-loop
    eval jobs pass the *training run's own* tasks here (see ``__main__``) so task
    names and hyperparameters match the run that launched the job -- including
    experiment-specific tasks that are not in the shared ``all_evals`` catalog.
    When ``None`` (e.g. standalone checkpoint sweeps) we fall back to the catalog.
    """
    tasks = eval_tasks if eval_tasks is not None else _get_eval_tasks()
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
                tasks=tasks,
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


def get_train_run_eval_tasks(
    user_mod: ModuleType, common: CommonComponents
) -> dict | None:
    """Read the downstream_evaluator tasks from the training run's trainer config.

    Used by in-loop eval jobs so the job evaluates exactly the tasks the *training
    run* configured (matching names + hyperparameters, including experiment-specific
    tasks absent from the shared catalog). Returns ``None`` if the module has no
    ``build_trainer_config``, it fails to build, or it has no evaluator tasks -- in
    which case the caller falls back to the shared catalog.
    """
    build = getattr(user_mod, "build_trainer_config", None)
    if build is None:
        return None
    try:
        train_trainer_config = build(common)
    except Exception as e:  # noqa: BLE001 - best-effort; fall back to catalog
        logger.warning(
            "Could not build the training run's trainer config to read eval tasks "
            "(%s); falling back to the shared eval catalog.",
            e,
        )
        return None
    evaluator_config = train_trainer_config.callbacks.get("downstream_evaluator")
    tasks = getattr(evaluator_config, "tasks", None)
    return tasks or None


def add_experiment_to_loop_eval_group(
    config: OlmoEarthEvaluateConfig, experiment: Experiment
) -> None:
    """Add a just-launched in-loop eval experiment to its per-run Beaker group.

    All eval jobs for one training run share a Beaker group (named via the
    ``OE_LOOP_EVAL_BEAKER_GROUP`` env var set by the in-loop launcher) so they
    appear as a single entry in the Beaker console instead of many loose
    experiments. Best-effort: any failure here only logs a warning and never fails
    the eval launch. No-op when the env var is unset (e.g. standalone sweeps).
    """
    group_name = os.environ.get("OE_LOOP_EVAL_BEAKER_GROUP")
    if not group_name:
        return
    try:
        from beaker import Beaker, GroupConflict, GroupNotFound

        beaker = Beaker.from_env()
        workspace = config.launch.workspace if config.launch is not None else None
        try:
            group = beaker.group.get(group_name)
        except GroupNotFound:
            # First eval job for this run creates the group; concurrent launches
            # may race, so treat a conflict as "already created" and re-fetch.
            try:
                group = beaker.group.create(group_name, workspace=workspace)
            except GroupConflict:
                group = beaker.group.get(group_name)
        beaker.group.add_experiments(group, experiment)
        logger.info(
            "Added eval experiment %s to Beaker group %r", experiment.id, group_name
        )
    except Exception as e:  # noqa: BLE001 - best-effort, never block the eval launch
        logger.warning(
            "Could not add eval experiment to Beaker group %r (%s)", group_name, e
        )


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

    # For in-loop eval jobs, evaluate the training run's *own* configured tasks
    # instead of the shared catalog, so task names + hyperparameters match the run
    # that launched the job (and experiment-specific tasks not in the catalog run
    # too). Gated on an env var set by the in-loop launcher so standalone checkpoint
    # sweeps keep using the full catalog.
    trainer_config_builder = build_trainer_config
    if os.environ.get("OE_LOOP_EVAL_FROM_TRAIN_CONFIG"):
        train_eval_tasks = get_train_run_eval_tasks(user_mod, common)
        if train_eval_tasks:
            logger.info(
                "In-loop eval: using %d tasks from the training run's config: %s",
                len(train_eval_tasks),
                sorted(train_eval_tasks),
            )
            trainer_config_builder = partial(
                build_trainer_config, eval_tasks=train_eval_tasks
            )

    config = build_evaluate_config(
        common=common,
        model_config_builder=build_model_config,
        trainer_config_builder=trainer_config_builder,
        overrides=overrides,
        train_module_config_builder=build_train_module_config,
    )

    if cmd == SubCmd.launch_evaluate:
        prepare_cli_environment()
        experiment = launch(config)
        add_experiment_to_loop_eval_group(config, experiment)
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
