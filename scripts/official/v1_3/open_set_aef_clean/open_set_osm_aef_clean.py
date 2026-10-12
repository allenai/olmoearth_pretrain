r"""Launch script: ``open_set_osm`` trained without the AEF-eval-overlapping samples.

Identical to ``../open_set_osm.py`` (v1.3 recipe + open-set probe, osm_sampling +
open-set concatenation) except that the open-set half is filtered to
``open_set_base.AEF_CLEAN_FILTER_IDX_FILE``: every open-set H5 sample whose footprint
overlaps an AlphaEarth supplemental eval window (any split) is dropped, so the AEF
evals stay valid. The class mapping is unchanged; probe rows of the AEF eval datasets
in the bank simply never receive gradient. See ``select_aef_clean_indices.py``.

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_aef_clean/open_set_osm_aef_clean.py launch \\
        open_set_osm_aef_clean ai2/jupiter --launch.num_gpus=8
"""

import logging
import sys
from pathlib import Path

# The shared open-set builders (and the v1.3 base) live one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import build_visualize_config  # noqa: E402
from open_set_base import (  # noqa: E402
    AEF_CLEAN_FILTER_IDX_FILE,
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_osm_plus_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import (
    build_trainer_config as _build_open_set_trainer_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_3/open_set_aef_clean/open_set_osm_aef_clean.py"


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
