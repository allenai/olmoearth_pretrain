r"""labels_finetune_clean.py with the larger open-set bank (labels_bank_frozen_clean.py's).

python scripts/vnext/text_pretrain/labels_bank_finetune_clean.py launch labels_bank_finetune_v1_3pub_gemma_clean ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import labels_finetune as base  # noqa: E402
from clean import clean_dataset  # noqa: E402
from labels_bank_frozen_clean import BANK_H5PY_DIR  # noqa: E402
from labels_finetune_clean import build_model_config  # noqa: E402
from text_mix import OPEN_SET_H5PY_DIR  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """labels_finetune_clean.py's datasets, the open-set windows from the larger bank."""
    config = clean_dataset(base.build_dataset_config(common))
    (open_set,) = [d for d in config.dataset_configs if d.h5py_dir == OPEN_SET_H5PY_DIR]
    open_set.h5py_dir = BANK_H5PY_DIR
    return config


if __name__ == "__main__":
    main(
        common_components_builder=base.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=base.build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        trainer_config_builder=base.build_trainer_config,
        visualize_config_builder=base.build_visualize_config,
    )
