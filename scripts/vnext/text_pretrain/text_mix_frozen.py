r"""text_mix.py's caption and label head, trained on the published v1.3 kept frozen (see frozen.py).

python scripts/vnext/text_pretrain/text_mix_frozen.py launch <run_name> ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from frozen import build_train_module_config, freeze, shorten  # noqa: E402
from olmo_core.train.config import TrainerConfig  # noqa: E402
from text_mix import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from text_mix import build_model_config as _text_build_model_config  # noqa: E402
from text_mix import build_trainer_config as _text_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """text_mix.py's model, frozen at the published weights but for the head."""
    return freeze(_text_build_model_config(common))


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """text_mix.py's trainer, run for the head-only schedule."""
    return shorten(_text_build_trainer_config(common))


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
