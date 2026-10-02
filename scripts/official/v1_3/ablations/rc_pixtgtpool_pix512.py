"""``rc_pixtgt_pix512`` with POOLED pixel targets: drawn from all masked pixels at once.

The shared and independent draws give every masked token exactly one target pixel.
Here, per (sample, modality), as many targets are drawn as there are masked tokens,
uniformly and without replacement from every masked pixel (each pixel of each masked
token's footprint, at that token's timestep). A token's footprint can get zero, one or
several target pixels; the total stays one per masked token, so the decoder cost is
unchanged. The decoder reads these as a flat list of queries
(``Predictor.forward_pooled``), each built exactly as the token's own query would be,
moved to its pixel. See ``olmoearth_pretrain/nn/pixel_targets.py``.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_pixtgtpool_pix512``.
Partners: ``v1_3_rc_pixtgtind_pix512`` (independent), ``v1_3_rc_pixtgt_pix512``
(shared), ``v1_3_rc_pix512`` (patch targets).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_pix512 import build_model_config  # noqa: E402
from rc_pixtgt_pix512 import (
    build_train_module_config as _pixtgt_build_train_module_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_pixtgtpool_pix512.py"


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """rc_pixtgt_pix512's train module with targets pooled over all masked pixels."""
    config = _pixtgt_build_train_module_config(common)
    config.pixel_target_draw = "pooled"
    return config


def build_trainer_config(common: CommonComponents):
    """rc_pix512's evals (incl. ps4 per pixel), re-importing THIS module."""
    return build_mix_trainer_config(common, MODULE_PATH, ps4_student_evals=True)


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
