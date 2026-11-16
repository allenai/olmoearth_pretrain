r"""labels_finetune.py on the clean split: held-out classes and locations withheld (see clean.py).

python scripts/vnext/text_pretrain/labels_finetune_clean.py launch labels_finetune_v1_3pub_gemma_clean ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import labels_finetune as base  # noqa: E402
from clean import clean_dataset, clean_model  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """labels_finetune.py's model, with held-out labels kept out of the distractors."""
    return clean_model(base.build_model_config(common))


def build_dataset_config(
    common: CommonComponents,
) -> OlmoEarthDatasetConfig | OlmoEarthConcatDatasetConfig:
    """labels_finetune.py's datasets, with held-out text targets masked."""
    return clean_dataset(base.build_dataset_config(common))


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=base.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=base.build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=base.build_dataloader_config,
        trainer_config_builder=base.build_trainer_config,
        visualize_config_builder=base.build_visualize_config,
    )


if __name__ == "__main__":
    run()
