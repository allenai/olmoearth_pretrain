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
    # Token-mixing window size: the same arm at 3x3 (as trained) and 5x5.
    "prof_mix6r1": "pure_perceiver_mix6_read6",
    "prof_mix6r2": "pure_perceiver_mix6_read6",
    "prof_mix6d128r1": "pure_perceiver_mix6_read6_d128",
    "prof_mix6d128r2": "pure_perceiver_mix6_read6_d128",
    # Latent budget memory check for the random-stride point arm: 512 (as trained) vs 1024.
    "prof_rstride512": "pure_perceiver_joint_latentread_rstride",
    "prof_rstride1024": "pure_perceiver_joint_latentread_rstride",
    # Local-radius mask cost (2026-10-01): the trained rad2lat3 arm ran 2.3x slower per
    # step than its global control. Control vs local (precomputed cell coordinates) vs
    # local + cell-major latents.
    "prof_rsfast2global": "pure_perceiver_joint_latentread_rstride_fast",
    "prof_rad2lat3plain": "pure_perceiver_joint_latentread_rstride_local",
    "prof_rad2lat3cellsort": "pure_perceiver_joint_latentread_rstride_local",
}
# Perceiver-config overrides per prefix, applied on top of the arm's model.
PROFILE_PERCEIVER_OVERRIDES: dict[str, dict] = {
    "prof_mix6r1": {"token_mix_radius": 1},
    "prof_mix6r2": {"token_mix_radius": 2},
    "prof_mix6d128r1": {"token_mix_radius": 1},
    "prof_mix6d128r2": {"token_mix_radius": 2},
    "prof_rstride512": {"max_latents": 512},
    "prof_rstride1024": {"max_latents": 1024},
    "prof_rad2lat3cellsort": {"sort_latents_by_cell": True},
}
# Longer runs for memory checks: more batches drawn at the per-sample latent ceiling.
PROFILE_MAX_STEPS: dict[str, int] = {"prof_rstride512": 1000, "prof_rstride1024": 1000}
# One GPU at the real per-rank batch: v1.3's 512 global batch over 8 GPUs = 64.
GLOBAL_BATCH_SIZE = 64
PROFILE_SKIP = 300
PROFILE_WARMUP = 3
PROFILE_ACTIVE = 10
MAX_STEPS = PROFILE_SKIP + 1 + PROFILE_WARMUP + PROFILE_ACTIVE + 5


def _prefix(common: CommonComponents) -> str:
    # Longest match first, so no prefix can shadow a longer one.
    for prefix in sorted(PROFILE_TARGETS, key=len, reverse=True):
        if common.run_name.startswith(prefix):
            return prefix
    raise ValueError(
        f"run name {common.run_name!r} must start with one of {list(PROFILE_TARGETS)}"
    )


def _target(common: CommonComponents):
    return importlib.import_module(PROFILE_TARGETS[_prefix(common)])


def build_model_config(common: CommonComponents):
    """The target arm's model (plus any per-prefix Perceiver overrides)."""
    config = _target(common).build_model_config(common)
    for field, value in PROFILE_PERCEIVER_OVERRIDES.get(_prefix(common), {}).items():
        setattr(config.encoder_config.perceiver_config, field, value)
    return config


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
    trainer_config.max_duration = Duration.steps(
        PROFILE_MAX_STEPS.get(_prefix(common), MAX_STEPS)
    )
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
