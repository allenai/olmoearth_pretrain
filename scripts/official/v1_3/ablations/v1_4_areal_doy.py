"""v1.4 candidate (favyen go-ahead 2026-10-25 on Hollis's exploration doc): A_real + DOY harmonics, full length.

Combines the two pretraining results that survived the 10-19..10-24 tests:

* Gandalf's E5 A_real data rule (``e5_allcap_arm_areal.py``): per 30-day slot the least-cloudy
  S2 capture and the S1 capture nearest the slot midpoint, stamped with their REAL capture dates,
  read from ``osm_allcaptures`` (J.F.'s eval-clean keep-list, 387,084 samples). At 90k steps
  A_real - A on PASTIS was +.087 and still growing.
* Wildcat's sun-clock arm C time code (``sunclock_arm_c.py``): the frozen day-of-year code
  (sin/cos at 1..6 cycles/yr) in place of v1.3's month code, with per-sample time-code dropout
  p=0.3 (encoder only), the exact configuration in which C tied A (DOY "costs nothing").

Everything else is E5's recipe (``e5_allcap_common.py`` = ``rc_ld1_pixtgt_pix512``: S1 + S2 + maps,
no Landsat, batch 512, seed 3622, lr 1e-4 cosine) with two full-length changes:

* the run is as long as v1.3 in steps: ``E5_MAX_STEPS`` (set to 667200 by the launcher);
* the warmup is v1.3's 8k steps (E5's 4k was scaled for 30k-step runs).

Unlike the sun-clock arms, the dataset is the whole keep-list (not the 51k subset) and lat/lon is
not passed through (the DOY code does not use it). Checkpoints: permanent every 10k (comparable
with the E5 30k/60k/90k reads), latest every 1k.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e5_allcap_common as e5  # noqa: E402

from olmoearth_pretrain.data.allcap import ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES  # noqa: E402
from olmoearth_pretrain.internal.experiment import main  # noqa: E402

TIME_CODE = "doy"
TIME_CODE_DROPOUT = float(os.environ.get("V14_TIME_CODE_DROPOUT", "0.3"))
WARMUP_STEPS = 8_000


def build_model_config(common):  # noqa: ANN001, ANN201
    """E5 model with the DOY time code in encoder and predictor (dropout: encoder only)."""
    config = e5.build_model_config(common)
    config.encoder_config.time_code = TIME_CODE
    config.encoder_config.time_code_dropout = TIME_CODE_DROPOUT
    config.decoder_config.time_code = TIME_CODE
    return config


def build_train_module_config(common):  # noqa: ANN001, ANN201
    """E5 train module with v1.3's 8k warmup."""
    config = e5.build_train_module_config(common)
    config.scheduler.warmup = WARMUP_STEPS
    return config


if __name__ == "__main__":
    main(
        common_components_builder=e5.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=e5.build_dataset_config_for(
            ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES
        ),
        dataloader_config_builder=e5.build_dataloader_config,
        trainer_config_builder=e5.build_trainer_config,
        visualize_config_builder=e5.build_visualize_config,
    )
