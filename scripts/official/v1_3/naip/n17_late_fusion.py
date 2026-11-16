"""Arm N17: N13 with separate modality streams in every encoder layer (late fusion).

Every encoder layer runs Sentinel-2, Sentinel-1 and Landsat as separate streams, each
with its own whole block and attention only within itself
(``EncoderConfig.modality_stream_depth`` = depth). The modalities first meet in the
Perceiver read, which is unchanged (one layer), so each modality's encoder tokens do not
depend on which other modalities are present. Three encoders' worth of block parameters
for the least compute of the arms. N17 vs N16 asks how late the streams can merge.

See ``n13_n12_no_naip.py``. Trained as ``v1_3_naip_n17_late_fusion``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import n13_n12_no_naip as n13  # noqa: E402
import naip_common  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n17_late_fusion.py"

# Every encoder layer: the streams first meet in the Perceiver.
MODALITY_STREAM_DEPTH = 12

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = n13.build_common_components
build_train_module_config = n13.build_train_module_config
build_dataloader_config = n13.build_dataloader_config
build_dataset_config = n13.build_dataset_config
build_visualize_config = n13.build_visualize_config


def build_model_config(common):
    """N13's model with modality streams in every layer."""
    config = n13.build_model_config(common)
    encoder_config = config.encoder_config
    assert MODALITY_STREAM_DEPTH <= encoder_config.depth
    encoder_config.modality_stream_depth = MODALITY_STREAM_DEPTH
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
