"""``rc_ld1_pixtgt_pix512`` (released v1.3) with a per-cell slot merge after 4 ViT blocks.

After block 4, each patch cell's visible tokens (all timesteps and modalities, ``n`` of
them) are merged into ``max(ceil(n / 3), min(n, 4))`` slots by slot attention
(``nn/slot_merge.py``: softmax over slots, chunk-mean init in (modality, time) order,
3 GRU rounds, point RoPE at the attention-weighted mean time); cells with 4 or fewer
tokens bypass it. Blocks 5-12 and the Perceiver read run over the merged sequence.
Everything else, including the in-loop evals, is ``rc_ld1_pixtgt_pix512.py``, so the
pair is a one-flag A/B.

Speed (1 H100, random-init merge, ps2, 16 px windows, S1+S2+Landsat x 12 months):
~1.76x faster per window than v1.3 (36 tokens per cell -> 12 slots). Training saves
much less (~4% per step), because masking leaves ~5 visible tokens per cell.

W&B project ``20260921_perceiver_shapes``; trained as ``v1_3_rc_ld1_pixtgt_slot4r3_pix512``.
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
from rc_ld1_pix512 import build_model_config as _rc_ld1_model_config  # noqa: E402
from rc_pixtgt_pix512 import build_train_module_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.slot_merge import SlotMergeConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/ablations/rc_ld1_pixtgt_slot4r3_pix512.py"

SLOT_MERGE = SlotMergeConfig(merge_after=4, ratio=3, min_slots=4, iters=3)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """rc_ld1_pixtgt_pix512's model with the slot merge."""
    config = _rc_ld1_model_config(common)
    config.encoder_config.slot_merge_config = SLOT_MERGE
    return config


def build_trainer_config(common: CommonComponents):
    """rc_ld1_pixtgt_pix512's evals, re-importing THIS module."""
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
