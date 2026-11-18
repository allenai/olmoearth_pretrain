r"""labels_finetune_clean.py's data on the frozen published v1.3: only the head trains.

Pixel-level class names only (no captions), clean split, text_mix_frozen.py's model and
head-only schedule. The control for labels_bank_frozen_clean.py.

python scripts/vnext/text_pretrain/labels_frozen_clean.py launch labels_frozen_v1_3pub_gemma_clean ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import labels_finetune  # noqa: E402
import text_mix_frozen as base  # noqa: E402
from clean import clean_dataset, clean_model  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """text_mix_frozen.py's model, with held-out labels kept out of the distractors."""
    return clean_model(base.build_model_config(common))


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """labels_finetune.py's datasets (no captions), with held-out text targets masked."""
    return clean_dataset(labels_finetune.build_dataset_config(common))


def run(dataset_config_builder=build_dataset_config) -> None:
    """Run the experiment."""
    main(
        common_components_builder=base.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=base.build_train_module_config,
        dataset_config_builder=dataset_config_builder,
        dataloader_config_builder=base.build_dataloader_config,
        trainer_config_builder=base.build_trainer_config,
        visualize_config_builder=base.build_visualize_config,
    )


if __name__ == "__main__":
    run()
