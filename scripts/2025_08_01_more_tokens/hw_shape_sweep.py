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

HW_LIST = [list(range(8, 13)), [12]]

# Run 1
for hw_list in HW_LIST:
    run_name = f"1_hw_shape_sweep_min_{hw_list[0]}_max_{hw_list[-1]}"
    run_cmd = f"python scripts/2025_08_01_more_tokens/train_cross_random_shape.py {run_name} {cluster} --launch.priority=high --launch.num_gpus=8 --data_loader.sampled_hw_p_list={hw_list}"
    print(run_cmd)
    subprocess.run(run_cmd, shell=True)  # nosec

# launch just the first run with nccl_debug
for hw_list in HW_LIST:
    run_name = f"1_hw_shape_sweep_min_{hw_list[0]}_max_{hw_list[-1]}_debug"
    run_cmd = f"python scripts/2025_08_01_more_tokens/train_cross_random_shape.py {run_name} {cluster} --launch.priority=high --launch.num_gpus=8 --data_loader.sampled_hw_p_list={hw_list} --common.nccl_debug=True"
    print(run_cmd)
    subprocess.run(run_cmd, shell=True)  # nosec
    break
