"""v1.3 on AllCaptures, narrow stage 2 x d256 within cell -> <=24 slots -> 12 x d768 (config E).

See ``allcap_narrow.py`` for the model, data, compositing and evals. Benchmarked
(random init, 1 H100, ps2 16 px windows) at ~1.33 / 1.51 ms per
window for 200 / 300 every-capture tokens per cell, vs ~1.73 for v1.3's shape on
monthly inputs.

W&B project ``20261016_allcap_slot_narrow``; trained as ``v1_3_allcap_slot_n2d256``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allcap_narrow import (  # noqa: E402
    build_allcap_trainer_config,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_narrow_model_config,
    build_train_module_config,
)
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/allcap_slot_n2d256.py"
NARROW_DEPTH = 2
NARROW_DIM = 256


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.3 with a 2 x d256 narrow within-cell stage and capped slot merge."""
    return build_narrow_model_config(common, NARROW_DEPTH, NARROW_DIM)


def build_trainer_config(common: CommonComponents):
    """The allcap eval set, re-importing THIS module."""
    return build_allcap_trainer_config(common, MODULE_PATH)


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
