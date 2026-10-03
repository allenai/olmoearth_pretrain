"""``rc_pixtgtind_pix512`` with 1 Perceiver ``[read -> self-attend]`` layer instead of 4.

Everything else is ``rc_pixtgtind_pix512.py``: the RC with per-pixel random-stride
latents (512 budget) and pixel-resolution MIM targets with an independent pixel per
token. Partners: ``rc_ld1_pixtgt_pix512`` (shared draw) and ``rc_ld1_pix512`` (patch
targets), all at depth 1.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_ld1_pixtgtind_pix512``.
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
from rc_ld1_pix512 import build_model_config  # noqa: E402
from rc_pixtgtind_pix512 import build_train_module_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_pixtgtind_pix512.py"


def build_trainer_config(common: CommonComponents):
    """rc_pix512's evals plus ps2 on PASTIS + 2 AEF tasks, re-importing THIS module."""
    return build_mix_trainer_config(
        common, MODULE_PATH, ps4_student_evals=True, ps2_student_evals=True
    )


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
