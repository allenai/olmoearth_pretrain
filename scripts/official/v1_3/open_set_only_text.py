r"""Launch script: ``open_set_only`` with CLIP-style text-embedding probe targets.

Identical to ``open_set_only.py`` except the classification head: instead of one
learned weight row per class, a shared ``Linear(768, 768) -> GELU -> Linear(768, 768)``
trunk maps each register cell into the space of frozen class *text embeddings*
(``all-mpnet-base-v2`` on "class name; dataset name", see
``open_set_segmentation_data.embed_class_names``) and the logits are scaled cosine
similarities. Classes that name the same concept across datasets therefore share a
target; the masked softmax within the source dataset is unchanged. Regression keeps a
linear head on the trunk output.

Requires the embeddings on weka (``open_set_base.CLASS_TEXT_EMBEDDINGS_PATH`` +
``.json`` sidecar), so the model can only be built where weka is mounted; the probe
refuses embeddings generated for a different class mapping.

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_only_text.py launch open_set_only_text \\
        ai2/jupiter --launch.num_gpus=8
"""

import logging

from base import build_visualize_config
from open_set_base import (
    CLASS_TEXT_EMBEDDINGS_PATH,
    build_common_components,
    build_dataloader_config,
    build_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import build_model_config as _build_model_config
from open_set_base import build_trainer_config as _build_open_set_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.open_set_latent_mim import OpenSetLatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/open_set_only_text.py"


def build_model_config(common: CommonComponents) -> OpenSetLatentMIMConfig:
    """v1.3 + open-set probe scored against frozen class text embeddings."""
    return _build_model_config(
        common, head_type="text", text_embeddings_path=CLASS_TEXT_EMBEDDINGS_PATH
    )


def build_dataset_config(common: CommonComponents):
    """Open-set supervised dataset only."""
    return build_open_set_dataset_config(common)


def build_trainer_config(common: CommonComponents):
    """Trainer with the in-loop evals pointed at this module."""
    return _build_open_set_trainer_config(common, MODULE_PATH)


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
