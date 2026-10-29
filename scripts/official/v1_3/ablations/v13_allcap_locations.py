"""Data diagnostic: v1.3 exactly as released, restricted to the AllCaptures locations.

v1.3's own corpus, sampler and model (``rc_ld1_pixtgt_pix512`` with the sub-patch offset
loss at 1.0, as every allcap arm trains), but only on the v1.3 samples whose tile is one
of the 395,660 AllCaptures tiles (~35% of v1.3's 1,138,828). The step count is pinned to
v1.3's 667,200 so the schedule matches and each tile is seen ~3x as often, as in allcap.

If this lags v1.3 like the allcap runs (PASTIS -2.2 / -3.2 / -3.5 at 80k / 160k / 240k),
the gap is the amount of data; if it tracks v1.3, it is the AllCaptures pipeline.

The index file comes from matching both corpora's ``latlon_distribution.npy``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allcap_narrow import MAX_STEPS  # noqa: E402
from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_visualize_config,
)
from base import build_dataset_config as _v1_3_dataset_config  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_ld1_pix512 import build_model_config  # noqa: E402
from rc_pixtgt_pix512 import (  # noqa: E402
    build_train_module_config as _pixtgt_train_module_config,
)

from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/v13_allcap_locations.py"
SUBSET_IDX_FILE = "/weka/dfive-default/olmoearth_pretrain/v13_allcap_subset_idx.npy"


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """v1.3's corpus, only the samples at AllCaptures locations."""
    config = _v1_3_dataset_config(common)
    config.filter_idx_file = SUBSET_IDX_FILE
    return config


def build_train_module_config(common: CommonComponents):
    """v1.3's train module with the sub-patch offset loss, as in the allcap arms."""
    config = _pixtgt_train_module_config(common)
    config.subpatch_offset_loss_weight = 1.0
    return config


def build_trainer_config(common: CommonComponents):
    """v1.3's evals, at v1.3's step count (an epoch here is ~35% of v1.3's)."""
    config = build_mix_trainer_config(
        common, MODULE_PATH, ps4_student_evals=True, ps2_student_evals=True
    )
    config.max_duration = Duration.steps(MAX_STEPS)
    return config


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
