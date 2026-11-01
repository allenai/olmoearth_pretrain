"""v1.3 control (``claude_findings/control_pixtgt.py``) without the per-pixel hidden layer and band dropout.

Ablation for ``drop_embed.py`` / ``nohid.py``: ``patch_embed_hidden_sizes=None`` (each token
is one linear projection of its raw patch pixels x bands) and ``band_dropout_rate=0``
(no band dropout; the control drops each S2/Landsat band at rate ~ U(0, 0.2)), and no
dropped-band embedding. Masking is the control's.
"""
import logging
import os
import sys
from pathlib import Path

# Numerics unchanged; avoids allocator fragmentation retries (see flex_masking.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))

from common import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from common import build_model_config as _build_model_config  # noqa: E402
from common import build_train_module_config as _build_train_module_config  # noqa: E402
from common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
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

MODULE_PATH = "scripts/official/v1_3/drop_embed/nohid_nodrop.py"
WANDB_PROJECT = "2026_10_30_flex_masking"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The control model with a linear patch projection and no band dropout."""
    config = _build_model_config(common)
    config.encoder_config.patch_embed_hidden_sizes = None
    config.encoder_config.band_dropout_rate = 0.0
    config.encoder_config.random_band_dropout = False
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The control train module (pixel MIM targets)."""
    return _build_train_module_config(common, pixel_targets=True)


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
