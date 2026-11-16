"""Arm N18: N13 with linear supervision heads.

The linear-head base of the architecture arms: N18 vs N13 isolates the heads on this
recipe, and N19-N21 are N14, N16 and N17 on top of it.

Every per-cell supervision head that N13 makes a 2-layer MLP (WorldCover, CDL,
WorldCereal, OSM, SRTM, canopy height, VIIRS) is a single linear layer again, as in the
v1.3 control. Nothing else changes: Open Buildings keeps its upsample decoder and VIIRS
is still scored on footprint means. Why: every MLP-head arm (N7, N10, N11, N12) ends
with about twice the control's student distillation loss, every linear-head arm at or
below it, and the MLP-head arms lose crop information in the 64-d embedding.

See ``n13_n12_no_naip.py``. Trained as ``v1_3_naip_n18_linear_heads``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import n13_n12_no_naip as parent  # noqa: E402
import naip_common  # noqa: E402

from olmoearth_pretrain.internal.experiment import main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/naip/n18_linear_heads.py"

# Module-level builders: checkpoint_sweep_evals.py (the in-loop eval jobs) imports this
# module and looks these up by name, falling back to defaults that do not match.
build_common_components = parent.build_common_components
build_train_module_config = parent.build_train_module_config
build_dataloader_config = parent.build_dataloader_config
build_dataset_config = parent.build_dataset_config
build_visualize_config = parent.build_visualize_config


def build_model_config(common):
    """N13's model with every MLP supervision head linear."""
    config = parent.build_model_config(common)
    assert config.supervision_head_config is not None
    heads = config.supervision_head_config.modality_configs
    flipped = 0
    for cfg in heads.values():
        if cfg.decoder == "mlp":
            cfg.decoder = "linear"
            flipped += 1
    # The seven per-cell heads; Open Buildings ("upsample") is left alone.
    assert flipped == 7, {name: cfg.decoder for name, cfg in heads.items()}
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
