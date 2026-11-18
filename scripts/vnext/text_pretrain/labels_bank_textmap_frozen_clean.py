r"""labels_bank_frozen_clean.py with the projection on the text side.

v1.3's 768-d location tokens are used as they are; an MLP (hidden 1,024) maps the
EmbeddingGemma label embeddings into their space. Everything else (bigger bank,
labels only, clean split, frozen published v1.3, 20k steps) is as labels_bank_frozen_clean.py.

python scripts/vnext/text_pretrain/labels_bank_textmap_frozen_clean.py launch labels_bank_textmap_frozen_v1_3pub_gemma_clean ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import labels_bank_frozen_clean as bank  # noqa: E402
import labels_frozen_clean as base  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """labels_frozen_clean.py's model, its head mapping text into the token space."""
    config = base.build_model_config(common)
    config.text_head_config.project_text = True
    return config


if __name__ == "__main__":
    main(
        common_components_builder=base.base.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=base.base.build_train_module_config,
        dataset_config_builder=bank.build_dataset_config,
        dataloader_config_builder=base.base.build_dataloader_config,
        trainer_config_builder=base.base.build_trainer_config,
        visualize_config_builder=base.base.build_visualize_config,
    )
