r"""text_mix_finetune.py without the cell captions: pixel-level class names only.

Closest to ClipEarth v5p's data (per-pixel class names from labelled datasets and OSM,
no captions): the same published-v1.3 start, schedule, label term, vocabulary, open-set
windows and pretraining windows as text_mix_finetune.py, with caption_cell dropped
from the dataset so the caption InfoNCE never runs. text_mix_finetune minus this arm
is what the captions add on top of the labels.

python scripts/vnext/text_pretrain/labels_finetune.py launch <run_name> ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from text_mix import build_dataset_config as _mix_build_dataset_config  # noqa: E402
from text_mix_finetune import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """text_mix.py's datasets without caption_cell."""
    config = _mix_build_dataset_config(common)
    for dataset in config.dataset_configs:
        dataset.training_modalities = [
            m for m in dataset.training_modalities if m != Modality.CAPTION_CELL.name
        ]
        dataset.caption_dir = None
    return config


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
