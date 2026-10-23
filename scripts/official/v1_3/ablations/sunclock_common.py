"""Wildcat / Q10 sun-clock test (Hollis 2026-10-23 04:54Z + 05:55Z): does the calendar code limit v1.3?

Three arms, identical except for the code added to dims [2n, 3n) of multitemporal tokens:

* A (``sunclock_arm_a.py``): v1.3's frozen month code (1 cycle/yr, 12 values);
* C (``sunclock_arm_c.py``): frozen day-of-year code, sin/cos at 1..6 cycles/yr;
* B (``sunclock_arm_b.py``): C plus a zero-init learned projection of a latitude-aware
  sun clock (TOA daily insolation, day length, d/dt).

C - A measures calendar resolution; B - C measures hemisphere/latitude awareness.
All arms: E5 A_real data (least-cloudy S2 + midpoint S1 per 30-day slot, REAL capture
dates, osm_allcaptures), the same 51,000 indices drawn uniformly from J.F.'s eval-clean
keep-list (seed 20261023), time-code dropout p=0.3 (per sample, encoder only; zeroes the
whole code), and otherwise E5's recipe (rc_ld1_pixtgt_pix512, S1 + S2 + maps, batch 512,
4k warmup, cosine to E5_MAX_STEPS). Latitude reaches the model only through B's sun clock
(``include_latlon`` passes it as a non-encoded field in every arm).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e5_allcap_common as e5  # noqa: E402

from olmoearth_pretrain.data.allcap import ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import main  # noqa: E402

INDEX_FILE = (
    "/weka/dfive-default/agent/favyen/2026-10-15-favyen-wildcat-persona/sunclock/"
    "sunclock_51k_allcap_indices.npy"
)
TIME_CODE_DROPOUT = float(os.environ.get("SUNCLOCK_DROPOUT", "0.3"))
WANDB_PROJECT = "wildcat_sunclock"


def run(time_code: str) -> None:
    """Run one arm with ``time_code`` in the encoder and predictor."""

    def build_model_config(common):  # noqa: ANN001, ANN202
        config = e5.build_model_config(common)
        config.encoder_config.time_code = time_code
        config.encoder_config.time_code_dropout = TIME_CODE_DROPOUT
        config.decoder_config.time_code = time_code
        return config

    def build_dataset_config(common):  # noqa: ANN001, ANN202
        return OlmoEarthDatasetConfig(
            h5py_dir=e5.ALLCAP_H5PY_DIR,
            training_modalities=common.training_modalities,
            filter_idx_file=INDEX_FILE,
            allcap_selection=ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES,
            include_latlon=True,
        )

    def build_trainer_config(common):  # noqa: ANN001, ANN202
        config = e5.build_trainer_config(common)
        config.callbacks["wandb"].project = WANDB_PROJECT
        if os.environ.get("SUNCLOCK_CPU") == "1":  # local CPU smoke test only
            config.callbacks.pop("gpu_memory_monitor", None)
        return config

    main(
        common_components_builder=e5.build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=e5.build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=e5.build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=e5.build_visualize_config,
    )
