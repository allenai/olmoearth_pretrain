"""Arm N14: N13 with a per-modality MLP in the first four encoder blocks.

The first ``ROUTED_MLP_DEPTH`` encoder blocks hold three MLPs, one each for Sentinel-2,
Sentinel-1 and Landsat, and every token goes through its own modality's
(``EncoderConfig.routed_mlp_*``). Attention stays shared in every block. A token still
passes through one MLP per block, so the encoder gains parameters (two extra MLPs in
each of four blocks) at the same compute. N14 vs N13 asks whether dedicated early
capacity per modality helps; N15 is the control with the same parameters and compute
but no specialisation.

See ``n13_n12_no_naip.py``. Trained as ``v1_3_naip_n14_modality_mlp4``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import n13_n12_no_naip as n13  # noqa: E402
import naip_common  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n14_modality_mlp.py"

# The first third of the 12 encoder blocks, the fraction "Unbundling Multimodal Vision
# Encoding" (arXiv:2611.17334) found best for modality-specific capacity (8 of 28).
ROUTED_MLP_DEPTH = 4

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = n13.build_common_components
build_train_module_config = n13.build_train_module_config
build_dataloader_config = n13.build_dataloader_config
build_dataset_config = n13.build_dataset_config
build_visualize_config = n13.build_visualize_config


def build_model_config(common):
    """N13's model with per-modality MLPs in the early blocks."""
    config = n13.build_model_config(common)
    encoder_config = config.encoder_config
    encoder_config.routed_mlp_depth = ROUTED_MLP_DEPTH
    encoder_config.routed_mlp_routing = "modality"
    encoder_config.routed_mlp_groups = [[m] for m in n13.ENCODER_INPUT_MODALITIES]
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
