"""Arm N11: N10 with a linear patch projection (no per-pixel hidden layer).

N10 (the v1.3 control + MLP supervision heads) with ``patch_embed_hidden_sizes=None``:
each token is one linear projection of its raw patch pixels x bands, as in Favyen's
``drop_embed/nohid.py`` (``v1_3_nohid``), instead of the control's per-pixel
Linear(bands, 64) + ReLU first. N11 vs N10 isolates the hidden layer given MLP heads;
N11 vs ``v1_3_nohid`` isolates the MLP heads given a linear projection.

W&B project ``2026_10_23_naip_v13``; trained as ``v1_3_naip_n11_control_mlp_linembed``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import n10_control_mlp_heads as n10  # noqa: E402
import naip_common  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n11_control_mlp_heads_linear_embed.py"

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = n10.build_common_components
build_train_module_config = n10.build_train_module_config
build_dataloader_config = n10.build_dataloader_config
build_dataset_config = n10.build_dataset_config
build_visualize_config = n10.build_visualize_config


def build_model_config(common):
    """N10's model with a linear patch projection."""
    config = n10.build_model_config(common)
    config.encoder_config.patch_embed_hidden_sizes = None
    return config


def build_trainer_config(common):
    """The control's evals, re-importing THIS module, in our W&B project."""
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
