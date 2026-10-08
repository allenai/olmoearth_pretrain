r"""Launch script: osm_sampling + open-set with the text probe, AEF-clean.

The combination of ``open_set_osm_aef_clean.py`` (osm_sampling + open-set
concatenation, open-set half filtered to ``open_set_base.AEF_CLEAN_FILTER_IDX_FILE``)
and ``../open_set_only_text.py`` (probe logits are scaled cosine similarities against
frozen class text embeddings; needs ``open_set_base.CLASS_TEXT_EMBEDDINGS_PATH`` on
weka). The osm_sampling samples carry no labels, so they only see the SSL objectives,
as in ``open_set_osm``. See ``select_aef_clean_indices.py``.

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_aef_clean/open_set_osm_text_aef_clean.py \\
        launch open_set_osm_text_aef_clean ai2/jupiter --launch.num_gpus=8
"""

import logging
import sys
from pathlib import Path

# The shared open-set builders (and the v1.3 base) live one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import build_visualize_config  # noqa: E402
from open_set_base import (  # noqa: E402
    AEF_CLEAN_FILTER_IDX_FILE,
    CLASS_TEXT_EMBEDDINGS_PATH,
    build_common_components,
    build_dataloader_config,
    build_osm_plus_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import build_model_config as _build_model_config  # noqa: E402
from open_set_base import (
    build_trainer_config as _build_open_set_trainer_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.open_set_latent_mim import (  # noqa: E402
    OpenSetLatentMIMConfig,
)

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_3/open_set_aef_clean/open_set_osm_text_aef_clean.py"


def build_model_config(common: CommonComponents) -> OpenSetLatentMIMConfig:
    """v1.3 + open-set probe scored against frozen class text embeddings."""
    return _build_model_config(
        common, head_type="text", text_embeddings_path=CLASS_TEXT_EMBEDDINGS_PATH
    )


def build_dataset_config(common: CommonComponents):
    """osm_sampling + AEF-clean open-set concatenation."""
    return build_osm_plus_open_set_dataset_config(
        common, open_set_filter_idx_file=AEF_CLEAN_FILTER_IDX_FILE
    )


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
