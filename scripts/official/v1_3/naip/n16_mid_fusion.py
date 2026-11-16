"""Arm N16: N13 with separate modality streams in the first four encoder layers (mid fusion).

In the first ``MODALITY_STREAM_DEPTH`` encoder layers Sentinel-2, Sentinel-1 and Landsat
each have their own whole block and attend only to their own tokens
(``EncoderConfig.modality_stream_depth``); the remaining layers are shared as in N13.
More parameters than N13 (two extra blocks in each of four layers) for less compute,
since attention in those layers runs over one modality's tokens at a time. N16 vs N14
isolates separating the attention (N14 has per-modality MLPs with shared attention);
N17 keeps the streams apart for the whole encoder.

See ``n13_n12_no_naip.py``. Trained as ``v1_3_naip_n16_mid_fusion4``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import n13_n12_no_naip as n13  # noqa: E402
import naip_common  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n16_mid_fusion.py"

# The same four layers N14 gives per-modality MLPs.
MODALITY_STREAM_DEPTH = 4

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = n13.build_common_components
build_train_module_config = n13.build_train_module_config
build_dataloader_config = n13.build_dataloader_config
build_dataset_config = n13.build_dataset_config
build_visualize_config = n13.build_visualize_config


def build_model_config(common):
    """N13's model with modality streams in the early layers."""
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
