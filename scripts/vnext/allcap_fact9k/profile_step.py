"""Profile a few allcap_fact9k training steps (torch.profiler op tables + chrome trace).

Same model, data and microbatch as ``base.py``; the global batch is cut to 64 (8
microbatches of 8) so a profiled step takes seconds, in-loop evals are off, and
olmo-core's ProfilerCallback logs the per-op GPU/CPU tables and saves a chrome
trace under ``<save_folder>/profiler``.

Launch (1 GPU, debug W&B project):
    PYTHONPATH=. python scripts/vnext/allcap_fact9k/profile_step.py launch <run_name> ai2/ceres \
        --launch.num_gpus=1 --trainer.callbacks.wandb.project=helios-debug
"""

import importlib.util
from pathlib import Path

from olmo_core.train.callbacks import ProfilerCallback
from olmo_core.train.common import Duration

from olmoearth_pretrain.internal.experiment import CommonComponents, main

# base.py imports the v1.2 script as module "base", so load it under another name.
_spec = importlib.util.spec_from_file_location(
    "allcap_fact9k_base", Path(__file__).with_name("base.py")
)
assert _spec is not None and _spec.loader is not None
allcap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(allcap)

PROFILE_GLOBAL_BATCH_SIZE = 64


def build_dataloader_config(common: CommonComponents):
    """base.py's dataloader with a small global batch."""
    config = allcap.build_dataloader_config(common)
    config.global_batch_size = PROFILE_GLOBAL_BATCH_SIZE
    return config


def build_trainer_config(common: CommonComponents):
    """base.py's trainer without evals, plus the profiler; a few steps only."""
    config = allcap.build_trainer_config(common)
    config.callbacks.pop("downstream_evaluator")
    config.max_duration = Duration.steps(8)
    config.callbacks["profiler"] = ProfilerCallback(
        wait=1, warmup=2, active=2, with_stack=True
    )
    return config


if __name__ == "__main__":
    main(
        common_components_builder=allcap.build_common_components,
        model_config_builder=allcap.build_model_config,
        train_module_config_builder=allcap.build_train_module_config,
        dataset_config_builder=allcap.build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=allcap.v1_2_base.build_visualize_config,
    )
