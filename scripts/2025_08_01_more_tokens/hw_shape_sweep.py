"""Sweeping some min and max hw shapes."""

import argparse
import subprocess  # nosec

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
args = parser.parse_args()
cluster = args.cluster

# HW from 8 to 12
# HW always 12
# normal with maxt 6
# need flash attn to work for this
# HW max 16
# HW max 16 min 12
# max size 128 however you can crack it


def format_hw_list(hw_list):
    """Format the hw list for the shell."""
    # Format as \[1,2,3,…\] so shell sees escaped brackets
    return "\\[" + ",".join(str(x) for x in hw_list) + "\\]"


print(format_hw_list(list(range(8, 13))))
HW_LIST = [list(range(8, 13)), [12]]


for hw_list in HW_LIST:
    run_name = f"new_hw_shape_sweep_min_{hw_list[0]}_max_{hw_list[-1]}"
    run_cmd = f"python scripts/2025_08_01_more_tokens/train_cross_random_shape.py launch {run_name} {cluster} --launch.priority=high --launch.num_gpus=8 --data_loader.sampled_hw_p_list={format_hw_list(hw_list)} --train_module.rank_microbatch_size=32"
    print(run_cmd)
    subprocess.run(run_cmd, shell=True)  # nosec
