"""Arm N12: N7 with a linear patch projection (no per-pixel hidden layer).

N7 (N6 + MLP supervision heads + VIIRS footprint loss) with
``patch_embed_hidden_sizes=None``: each token is one linear projection of its raw patch
pixels x bands, as in Favyen's ``drop_embed/nohid.py`` (``v1_3_nohid``), instead of
the control's per-pixel Linear(bands, 64) + ReLU first. This applies to every encoded
modality, naip_10 included. N12 vs N7 isolates the hidden layer.

See ``hires_targets_common.py``. Trained as ``v1_3_naip_n12_mlp_viirsfp_linembed``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import n7_mlp_viirs_footprint as n7  # noqa: E402
import naip_common  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n12_mlp_viirs_footprint_linear_embed.py"

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = n7.build_common_components
build_train_module_config = n7.build_train_module_config
build_dataloader_config = n7.build_dataloader_config
build_dataset_config = n7.build_dataset_config
build_visualize_config = n7.build_visualize_config


def build_model_config(common):
    """N7's model with a linear patch projection."""
    config = n7.build_model_config(common)
    config.encoder_config.patch_embed_hidden_sizes = None
    return config


def build_trainer_config(common):
    """The control's evals, re-importing THIS module."""
    return naip_common.build_trainer_config(common, MODULE_PATH)


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
