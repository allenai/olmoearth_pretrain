r"""Launch script: ``open_set_only`` restricted to the high-quality open-set datasets.

Identical to ``../open_set_only.py`` (v1.3 recipe + open-set probe on the register
grid, open-set dataset only) except that the dataset is filtered to the H5 samples of
the ``HIGH_QUALITY_SLUGS`` in ``select_h5_indices.py`` through
``OlmoEarthDatasetConfig.filter_idx_file``. The SAME H5 directory and the same frozen
class mapping are used; probe rows of the datasets that are filtered out simply never
receive gradient, so results stay comparable with the full-bank runs.

Produce the index file once per H5 build (weka mounted)::

    python scripts/official/v1_3/open_set_hq/select_h5_indices.py

The file is named after the build's sample count (``select_h5_indices.DEFAULT_OUTPUT``
follows ``open_set_base.OPEN_SET_H5_DIR``), so it must be regenerated whenever the H5
directory changes; the dataset fails to load if it is missing.

The subset is roughly a tenth of the full bank, so the inherited
``Duration.epochs(300)`` is far fewer optimizer steps than the full run; override with
``--trainer.max_duration`` if you want to match step counts.

Usage (from the repo root)::

    python scripts/official/v1_3/open_set_hq/open_set_only_hq.py launch \\
        open_set_only_hq ai2/jupiter --launch.num_gpus=8
"""

import dataclasses
import logging
import sys
from pathlib import Path

# The shared open-set builders (and the v1.3 base) live one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import build_visualize_config  # noqa: E402
from open_set_base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_open_set_dataset_config,
    build_train_module_config,
)
from open_set_base import (
    build_trainer_config as _build_open_set_trainer_config,  # noqa: E402
)
from select_h5_indices import DEFAULT_OUTPUT as HQ_FILTER_IDX_FILE  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_3/open_set_hq/open_set_only_hq.py"


def build_dataset_config(common: CommonComponents):
    """Open-set supervised dataset, filtered to the high-quality slugs."""
    return dataclasses.replace(
        build_open_set_dataset_config(common),
        filter_idx_file=HQ_FILTER_IDX_FILE,
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
