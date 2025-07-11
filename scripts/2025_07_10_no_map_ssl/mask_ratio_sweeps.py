"""Run sweep over masking ratios."""

import argparse
import subprocess  # nosec

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cluster", type=str, default="ai2/titan-cirrascale", help="Cluster to use"
)
parser.add_argument("--priority", type=str, help="Priority for the launch")
args = parser.parse_args()

# Define masking ratios to sweep over
ENCODE_RATIOS = [0.25, 0.5, 0.75]

# Arguments to override masking config
masking_args = [
    "--train_module.masking_config.strategy_config.encode_ratio={encode_ratio}",
    "--train_module.masking_config.strategy_config.decode_ratio={decode_ratio}",
]

for encode_ratio in ENCODE_RATIOS:
    decode_ratio = 1 - encode_ratio
    priority_args = []
    if args.priority:
        priority_args = [f"--launch.priority={args.priority}"]

    subprocess.call(
        [
            "python",
            "scripts/2025_07_10_no_map_ssl/train_cross_random.py",
            "launch",
            f"debug_lmim_cross_random_no_map_mask_ratio_sweep_enc_{encode_ratio}_dec_{decode_ratio}",
            args.cluster,
            "--launch.num_gpus=8",
        ]
        + priority_args
        + [
            arg.format(encode_ratio=encode_ratio, decode_ratio=decode_ratio)
            for arg in masking_args
        ],
    )  # nosec
