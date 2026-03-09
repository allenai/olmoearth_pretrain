"""Wrapper script to make single_bandset_masked_neg experiments compatible with the full eval sweep.

Usage:
    EXPERIMENT=single_bandset_all12_random_band_dropout_random_decode_masked_neg \
    python -m olmoearth_pretrain.internal.full_eval_sweep \
        --module_path=scripts/vnext/single_bandset_band_dropout/eval_wrapper.py \
        --checkpoint_path=... --cluster=...
"""

import os
import sys

from single_bandset_masked_neg import EXPERIMENTS

exp_key = os.environ.get("EXPERIMENT")
if exp_key is None:
    print("EXPERIMENT environment variable must be set")
    print(f"Available: {list(EXPERIMENTS.keys())}")
    sys.exit(1)

if exp_key not in EXPERIMENTS:
    print(f"Unknown experiment: {exp_key}")
    print(f"Available: {list(EXPERIMENTS.keys())}")
    sys.exit(1)

common_builder, model_builder, train_module_builder, _dataloader_builder = EXPERIMENTS[
    exp_key
]

build_common_components = common_builder
build_model_config = model_builder
build_train_module_config = train_module_builder
