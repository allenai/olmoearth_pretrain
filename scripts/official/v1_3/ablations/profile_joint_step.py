"""Short single-GPU PROFILING run for a joint-latent arm (not a training run).

Picks the arm from the run name (``PROFILE_TARGETS``), keeps its model, sampler, train
module and microbatch, and swaps the trainer for a profiling one: one GPU at the real
per-rank batch (global batch 64), no in-loop evals, no W&B, no checkpoints beyond the
ones olmo-core needs, and olmo-core's ``ProfilerCallback`` over ``PROFILE_ACTIVE`` steps
after ``PROFILE_SKIP`` steps of warm-up (so FlexAttention recompiles have settled). The
profiler logs the top kernels by GPU and by CPU time and saves a Chrome trace under
``<save_folder>/profiler``.

Launch with ``--launch.num_gpus=1``, e.g. ``python .../profile_joint_step.py launch
prof_tmlp1 ai2/jupiter --launch.num_gpus=1 --launch.priority=urgent``.
"""

import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import build_common_components, build_visualize_config  # noqa: E402
from olmo_core.train.callbacks import ProfilerCallback  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

# Run-name prefix -> arm module whose model/sampler/train module to profile.
PROFILE_TARGETS = {
    "prof_tmlp1": "pure_perceiver_joint_latentread_rstride_tmlp1",
    "prof_latentread": "pure_perceiver_joint_latentread",
}
# One GPU at the real per-rank batch: v1.3's 512 global batch over 8 GPUs = 64.
GLOBAL_BATCH_SIZE = 64
PROFILE_SKIP = 300
PROFILE_WARMUP = 3
PROFILE_ACTIVE = 10
MAX_STEPS = PROFILE_SKIP + 1 + PROFILE_WARMUP + PROFILE_ACTIVE + 5


def _target(common: CommonComponents):
    for prefix, module in PROFILE_TARGETS.items():
        if common.run_name.startswith(prefix):
            return importlib.import_module(module)
    raise ValueError(
        f"run name {common.run_name!r} must start with one of {list(PROFILE_TARGETS)}"
    )


def build_model_config(common: CommonComponents):
    """The target arm's model."""
    return _target(common).build_model_config(common)


def build_train_module_config(common: CommonComponents):
    """The target arm's train module (keeps its rank microbatch)."""
    return _target(common).build_train_module_config(common)


def build_dataset_config(common: CommonComponents):
    """The target arm's dataset."""
    return _target(common).build_dataset_config(common)


def build_dataloader_config(common: CommonComponents):
    """The target arm's sampler at the real per-rank batch on one GPU."""
    config = _target(common).build_dataloader_config(common)
    config.global_batch_size = GLOBAL_BATCH_SIZE
    return config


def build_trainer_config(common: CommonComponents):
    """The arm's trainer, stripped to a short profiled run."""
    trainer_config = _target(common).build_trainer_config(common)
    trainer_config.max_duration = Duration.steps(MAX_STEPS)
    trainer_config.callbacks.pop("downstream_evaluator", None)
    trainer_config.callbacks["wandb"].enabled = False
    checkpointer = trainer_config.callbacks["checkpointer"]
    checkpointer.save_interval = 10**9
    checkpointer.ephemeral_save_interval = None
    trainer_config.callbacks["profiler"] = ProfilerCallback(
        skip_first=PROFILE_SKIP,
        wait=1,
        warmup=PROFILE_WARMUP,
        active=PROFILE_ACTIVE,
        repeat=1,
        with_stack=False,
    )
    return trainer_config


def run() -> None:
    """Run the profiling job."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
