"""v1.3 + per-modality DOY (favyen go-ahead 2026-10-26 ~18:4xZ): each modality token carries its own date.

Exactly the "v1.3 + slot DOY" run (``v1_4_areal_doy.py``: E5 A_real captures, DOY harmonic time
code in encoder and decoder, encoder time-code dropout 0.3, v1.3's 8k warmup) with one change:
the dataset packs each modality's OWN capture date into ``timestamps``
(``allcap_per_modality_timestamps``; see ``olmoearth_pretrain.data.per_modality_timestamps``),
so the DOY code of an S1 token is its S1 capture's day of year instead of the slot's S2 date.
The captures, the slot grid, the slot-index encoding and the shared stamp are unchanged. The
dropout draw is shared by all modalities of a sample, as before.

Pilot length: ``E5_MAX_STEPS`` sets the schedule (667200, the slot-DOY run's length, so LR and
EMA match it step for step) and ``PERMOD_HARD_STOP_STEPS`` stops training early (90000 for the
pilot), so the 90k checkpoint is directly comparable with the slot-DOY run's 90k checkpoint.
Unset = no hard stop.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e5_allcap_common as e5  # noqa: E402
import v1_4_areal_doy as slot_doy  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.data.allcap import ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

HARD_STOP_STEPS = os.environ.get("PERMOD_HARD_STOP_STEPS")


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """The slot-DOY dataset (A_real) with per-modality capture dates packed in."""
    config = e5.build_dataset_config_for(ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES)(common)
    config.allcap_per_modality_timestamps = True
    return config


def build_trainer_config(common: CommonComponents):  # noqa: ANN201
    """E5 trainer (schedule length ``E5_MAX_STEPS``), optional early hard stop."""
    config = e5.build_trainer_config(common)
    if HARD_STOP_STEPS:
        config.hard_stop = Duration.steps(int(HARD_STOP_STEPS))
    return config


if __name__ == "__main__":
    main(
        common_components_builder=e5.build_common_components,
        model_config_builder=slot_doy.build_model_config,
        train_module_config_builder=slot_doy.build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=e5.build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=e5.build_visualize_config,
    )
