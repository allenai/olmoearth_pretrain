"""Continue-training / annealing ablation.

Loads a pretrained checkpoint and continues training on a filtered subset
with either a short cosine anneal or a constant LR. Much cheaper than full
retraining; intended as a fast way to rank data splits.

Why one scheduler class covers both cosine and constant:
    CosWithWarmup(alpha_f=0.0) -> cosine decay from peak LR to 0
    CosWithWarmup(alpha_f=1.0) -> eta_min == initial_lr, LR stays at peak

Warmup trick:
    Setting warmup = RESUME_STEP makes the cosine start from the peak LR
    at the resume step. The saved optimizer param group still has
    initial_lr = 0.0001 from the original run, so the new schedule
    recovers the full peak LR on resume.

Launch example (8 GPU, 50k cosine anneal on the infra-rich split):

    python scripts/official/ablations/continue_anneal.py launch \
        anneal_infra_rich_cos50k ai2/jupiter \
        --launch.num_gpus=8 \
        --launch.clusters=[ai2/jupiter,ai2/titan,ai2/ceres] \
        --trainer.load_path=/weka/.../data_ablation_v2_full_baseline_1/step200000 \
        --dataset.filter_idx_file=/weka/.../ablation_filters/infrastructure_content_infrastructure_rich.npy \
        --data_loader.num_dataset_repeats_per_epoch=10 \
        --trainer.max_duration.value=250000 \
        --trainer.max_duration.unit=steps \
        --train_module.scheduler.warmup=200000 \
        --train_module.scheduler.t_max=250000 \
        --train_module.scheduler.alpha_f=0.0 \
        --trainer.callbacks.wandb.project=2026_04_continue_anneal
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vnext" / "speedups"))

from base_speedup import build_model_config  # noqa: E402
from base_speedup import (  # noqa: E402
    build_train_module_config as _build_train_module_config_base,
)
from olmo_core.optim.scheduler import CosWithWarmup  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from script import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (  # noqa: E402
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Same as base_speedup, but with a scheduler whose warmup/t_max/alpha_f
    are meant to be overridden per-experiment via CLI dotlist."""
    cfg = _build_train_module_config_base(common)
    cfg.scheduler = CosWithWarmup(
        warmup=200000,
        t_max=250000,
        alpha_f=0.0,
    )
    return cfg


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
