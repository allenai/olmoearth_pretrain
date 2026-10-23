"""NAIP arm N3: register-grid supervision of NAIP sub-pixel statistics.

The v1.3 control (``ablations/rc_ld1_pixtgt_pix512.py``) trained from scratch on the
naip_10 sibling H5. naip_10 is NOT an encoder input or MIM target. It is a target only:
one more register-grid supervision head (next to worldcover/srtm/osm/...) predicts,
for every 10 m pixel, its 4 x 4 NAIP block in low-passed form: the per-band means at
5 m (2 x 2 per S2 pixel) and the per-band std over the 10 m pixel (8 channels, L1).
Raw 2.5 m pixels are not the target: past NAIP-reconstruction attempts spent their
capacity on texture that S2 cannot see. These statistics are what a 10 m latent can
plausibly carry: sub-pixel layout and heterogeneity. Samples without NAIP (~82.5%)
contribute nothing to this head.

W&B project ``2026_10_23_naip_v13``; trained as ``v1_3_naip_n3_subpix``. See
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

MODULE_PATH = "scripts/official/v1_3/naip/n3_naip_subpixel.py"

TARGET_ONLY = [Modality.NAIP_10.name]
# Same weight as the other regression heads (base.py: base_weight 1.0 x 1.0).
NAIP_SUPERVISION_WEIGHT = 1.0

build_common_components = partial(naip_common.build_common_components, TARGET_ONLY)


def build_model_config(common):
    """The control's model plus a naip_10 sub-pixel-stats supervision head."""
    config = naip_common.build_model_config(common, naip_common.CONTROL_MODALITIES)
    assert config.supervision_head_config is not None
    config.supervision_head_config.modality_configs[Modality.NAIP_10.name] = (
        SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=2 * Modality.NAIP_10.num_bands,
            weight=NAIP_SUPERVISION_WEIGHT,
            regression_loss_type="l1",
            target_transform="subpixel_stats",
        )
    )
    return config


def build_train_module_config(common):
    """The control's train module; naip_10 decode-only (never encoded)."""
    return naip_common.build_train_module_config(common, TARGET_ONLY)


def build_dataloader_config(common):
    """The control's dataloader; naip_10 out of the token budget."""
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
