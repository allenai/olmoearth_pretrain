"""NAIP arm N4: register-grid regression of frozen DINOv3-sat features of NAIP.

The v1.3 control (``ablations/rc_ld1_pixtgt_pix512.py``) trained from scratch on the
naip_10 sibling H5, with ``naip_dino`` loaded from a side directory: DINOv3 ViT-L/16
(SAT-493M) run on the 0.625 m NAIP RGB of each sample (2048 x 2048 px -> a 128 x 128
patch grid, one feature per 10 m S2 pixel), PCA-whitened to 64 dims. naip_dino is a
target only: one more register-grid supervision head regresses it with a per-pixel
cosine loss. The teacher turns NAIP's sub-10 m detail into semantics a 10 m latent
can carry. Raw NAIP pixels are not the target (texture S2 cannot see). Samples
without features (~85%) contribute nothing to this head.

W&B project ``2026_10_23_naip_v13``; trained as ``v1_3_naip_n4_dino``. See
``naip_common.py``.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import naip_common  # noqa: E402
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import main  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionModalityConfig,
    SupervisionTaskType,
)

MODULE_PATH = "scripts/official/v1_3/naip/n4_naip_dino.py"

TARGET_ONLY = [Modality.NAIP_DINO.name]
# 1 - cos lies in [0, 2] and starts near 1, the scale of the L1 regression heads.
NAIP_DINO_SUPERVISION_WEIGHT = 1.0

build_common_components = partial(naip_common.build_common_components, TARGET_ONLY)


def build_model_config(common):
    """The control's model plus a naip_dino cosine-regression supervision head."""
    config = naip_common.build_model_config(common, naip_common.CONTROL_MODALITIES)
    assert config.supervision_head_config is not None
    config.supervision_head_config.modality_configs[Modality.NAIP_DINO.name] = (
        SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=Modality.NAIP_DINO.num_bands,
            weight=NAIP_DINO_SUPERVISION_WEIGHT,
            regression_loss_type="cosine",
        )
    )
    return config


def build_train_module_config(common):
    """The control's train module; naip_dino decode-only (never encoded)."""
    return naip_common.build_train_module_config(common, TARGET_ONLY)


def build_dataloader_config(common):
    """The control's dataloader; naip_dino out of the token budget."""
    return naip_common.build_dataloader_config(common, TARGET_ONLY)


def build_trainer_config(common):
    """The control's evals, re-importing THIS module."""
    return naip_common.build_trainer_config(common, MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=naip_common.build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
