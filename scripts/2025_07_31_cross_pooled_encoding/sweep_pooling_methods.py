"""Run sweep over masking ratios."""

import argparse
import subprocess  # nosec

from helios.nn.pooled_modality_predictor import DimsToPool

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cluster", type=str, default="ai2/titan-cirrascale", help="Cluster to use"
)
parser.add_argument("--priority", type=str, help="Priority for the launch")
args = parser.parse_args()


POOLING_METHODS = [
    DimsToPool.MODALITY,
    # DimsToPool.TEMPORAL,
    # DimsToPool.MODALITY_TEMPORAL,
]

# Arguments to override masking config
masking_args = [
    "--train_module.masking_config.strategy_config.encode_ratio={encode_ratio}",
    "--train_module.masking_config.strategy_config.decode_ratio={decode_ratio}",
]

for pooling_method in POOLING_METHODS:
    priority_args = []
    if args.priority:
        priority_args = [f"--launch.priority={args.priority}"]

    subprocess.call(
        [
            "python",
            "scripts/2025_07_31_cross_pooled_encoding/train_pooled_encoder.py",
            "launch",
            f"lmim_cross_random0.5_pooledfix_{pooling_method}_mlp_pooling",
            args.cluster,
            "--launch.num_gpus=8",
            f"--model.encoder_config.dims_to_pool={pooling_method.upper()}",
        ]
        + priority_args
    )  # nosec
