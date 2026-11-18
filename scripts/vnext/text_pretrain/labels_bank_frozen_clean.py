r"""labels_frozen_clean.py with a larger open-set bank: up to 5,000 windows per dataset.

The bank (697,317 windows from 236 datasets, the mixed run's 103,112 first) is built in
the session dir 2026-11-17-piperw-text-labels-bank; ClipEarth's old unseen datasets are in
it, as the clean split's held-out classes and places replace them.

python scripts/vnext/text_pretrain/labels_bank_frozen_clean.py launch labels_bank_frozen_v1_3pub_gemma_clean ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import labels_frozen_clean as base  # noqa: E402
from text_mix import OPEN_SET_H5PY_DIR  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402

BANK_H5PY_DIR = (
    "/weka/dfive-default/agent/piperbots/2026-11-17-piperw-text-labels-bank/h5/697317"
)


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """labels_frozen_clean.py's datasets, the open-set windows from the larger bank."""
    config = base.build_dataset_config(common)
    (open_set,) = [d for d in config.dataset_configs if d.h5py_dir == OPEN_SET_H5PY_DIR]
    open_set.h5py_dir = BANK_H5PY_DIR
    return config


if __name__ == "__main__":
    base.run(build_dataset_config)
