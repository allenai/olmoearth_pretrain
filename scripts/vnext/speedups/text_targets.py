"""Text-embedding targets for map modalities, based on base_speedup.

Replaces target-encoder outputs for WorldCover, OSM raster, and WorldCereal with
pre-computed semantic text embeddings from bge-base-en-v1.5.

Uses masked-negatives patch discrimination loss since many patches will share
identical text embedding targets for the same class.

Launch example:
    python scripts/vnext/speedups/text_targets.py dry_run test_text_targets local \
        --trainer.max_duration=10steps --dataloader.num_workers=2 \
        --dataloader.global_batch_size=8 --train_module.rank_microbatch_size=8
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from base_speedup import build_model_config  # noqa: E402
from olmo_core.optim import AdamWConfig  # noqa: E402
from script import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from script import (
    build_train_module_config as _build_train_module_config_base,
)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.text_targets import TextEmbeddingTargetConfig  # noqa: E402
from olmoearth_pretrain.train.loss import LossConfig  # noqa: E402

logger = logging.getLogger(__name__)

TEXT_EMBEDDINGS_PATH = "/weka/dfive-default/henryh/helios/olmoearth_pretrain/olmoearth_pretrain/data/text_embeddings.pt"
TEXT_TARGET_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WORLDCEREAL.name,
]


def build_train_module_config(common: CommonComponents):
    """Base speedup train module + text embedding targets for map modalities."""
    cfg = _build_train_module_config_base(common)
    cfg.optim_config = AdamWConfig(lr=0.0001, weight_decay=0.02, fused=True)
    cfg.loss_config = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_masked_negatives",
            "tau": 0.1,
            "mask_negatives_for_modalities": TEXT_TARGET_MODALITIES,
        }
    )
    cfg.compile_model = True
    cfg.text_target_config = TextEmbeddingTargetConfig(
        embeddings_path=TEXT_EMBEDDINGS_PATH,
        modalities=TEXT_TARGET_MODALITIES,
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
