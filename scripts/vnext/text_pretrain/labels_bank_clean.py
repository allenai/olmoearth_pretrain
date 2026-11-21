r"""text_mix.py from scratch, labels only, larger open-set bank, clean split.

text_mix.py's model, schedule (the published v1.3 run's 667,200 steps) and pretraining windows,
without caption_cell (as labels_finetune.py), the open-set windows from
labels_bank_frozen_clean.py's bank, and held-out classes and locations withheld (clean.py).
The from-scratch counterpart of labels_bank_frozen_clean.py and labels_bank_finetune_clean.py.

python scripts/vnext/text_pretrain/labels_bank_clean.py launch labels_bank_v1_3_gemma_clean ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import text_mix as base  # noqa: E402
from clean import clean_dataset, clean_model  # noqa: E402
from labels_bank_frozen_clean import BANK_H5PY_DIR  # noqa: E402
from labels_finetune import build_dataset_config as labels_dataset_config  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """text_mix.py's model, with held-out labels kept out of the distractors."""
    return clean_model(base.build_model_config(common))


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """text_mix.py's datasets without captions, the larger bank, held-out targets masked."""
    config = clean_dataset(labels_dataset_config(common))
    (open_set,) = [
        d for d in config.dataset_configs if d.h5py_dir == base.OPEN_SET_H5PY_DIR
    ]
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
