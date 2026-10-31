"""v1.3 control (``claude_findings/control_pixtgt.py``) with flexible masking + count band dropout.

Only masking and band dropout differ from the control ``v1_3_rc_ld1_pixtgt_pix512``:

* Masking ``flex_modality_with_decode`` (see ``FlexModalityMaskingStrategy``): per rank
  batch a visible band-set count ``e ~ U{1..3}`` (an instance with fewer present band
  sets draws its own ``e ~ U{1..n}``); per instance one mask type (random / temporal /
  spatial, 1/3 each) and mask ratio ``r ~ U(0.3, 0.7)``; visible band sets encode
  ``1 - r`` and decode ``r`` of their tokens, hidden band sets decode ``r`` and keep
  the rest target-encoder-only; maps stay decode-only.
  At 8 GPUs the rank batch (512 / 8 = 64) is one microbatch, so ``e`` is per microbatch.
* Band dropout ``count`` on S2, Landsat and S1: per sample, with probability 0.75 keep
  ``k ~ U{1..N}`` random bands (the control: per-band rate ~ U(0, 0.2), S2 + Landsat).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))

from common import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from common import build_dataloader_config as _build_dataloader_config  # noqa: E402
from common import build_model_config as _build_model_config  # noqa: E402
from common import build_train_module_config as _build_train_module_config  # noqa: E402
from common import build_trainer_config as _build_trainer_config  # noqa: E402
from v1_2.base import ONLY_DECODE_MODALITIES  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.masking import MaskingConfig  # noqa: E402
from olmoearth_pretrain.train.olmo_core_patches import (  # noqa: E402
    drop_async_checkpoint_duration_metric,
)
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

# The async-save duration metric hangs the gloo bookkeeping all-reduce; see
# olmo_core_patches (same patch as the NISAR arms).
drop_async_checkpoint_duration_metric()

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/flex_masking/flex_masking.py"
WANDB_PROJECT = "2026_10_30_flex_masking"

MIN_MASK_RATIO = 0.3
MAX_MASK_RATIO = 0.7
BAND_DROPOUT_APPLY_PROB = 0.75
BAND_DROPOUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.LANDSAT.name,
    Modality.SENTINEL1.name,
]


def _masking_config(common: CommonComponents) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "flex_modality_with_decode",
            "min_mask_ratio": MIN_MASK_RATIO,
            "max_mask_ratio": MAX_MASK_RATIO,
            "mode_weights": [1.0, 1.0, 1.0],
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
            "num_visible_per_call": True,
        },
        tokenization_config=common.tokenization_config,
    )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The control model with count band dropout on S2, Landsat and S1."""
    config = _build_model_config(common)
    encoder_config = config.encoder_config
    encoder_config.band_dropout_mode = "count"
    encoder_config.band_dropout_rate = BAND_DROPOUT_APPLY_PROB
    encoder_config.random_band_dropout = False
    encoder_config.band_dropout_modalities = list(BAND_DROPOUT_MODALITIES)
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The control dataloader with the flex masking strategy."""
    config = _build_dataloader_config(common)
    config.masking_config = _masking_config(common)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The control train module (pixel MIM targets) with the flex masking strategy."""
    config = _build_train_module_config(common, pixel_targets=True)
    config.masking_config = _masking_config(common)
    return config


def build_trainer_config(common: CommonComponents):
    """The control's in-loop evals (re-importing THIS module), own W&B project.

    Duration stays the control's 300 epochs = 667,200 steps.
    """
    trainer_config = _build_trainer_config(common, MODULE_PATH)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


def run() -> None:
    """Run the experiment."""
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
