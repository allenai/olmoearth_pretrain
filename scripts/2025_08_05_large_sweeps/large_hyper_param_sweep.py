"""Large hyper param sweep."""

# Want to sweep over the following:
# learning rate, mask ratio, weight decay, and decoder depth
import argparse
import subprocess  # nosec

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cluster", type=str, default="ai2/titan-cirrascale", help="Cluster to run on"
)
args = parser.parse_args()
cluster = args.cluster

LRs = [0.0001, 0.00005]
MASK_RATIOS = [0.5]  # Left out 0.1 # 0.05 OOMed so turned off fused adamw
WEIGHT_DECAYS = [0.02, 0.05]
# fixing decoder depth for now
DECODER_DEPTHS = [4]  # [4, 8]

number_of_runs = len(LRs) * len(MASK_RATIOS) * len(WEIGHT_DECAYS) * len(DECODER_DEPTHS)
print(f"Number of runs: {number_of_runs}")
for lr in LRs:
    for mask_ratio in MASK_RATIOS:
        encode_ratio = mask_ratio
        decode_ratio = 1 - encode_ratio
        for weight_decay in WEIGHT_DECAYS:
            for decoder_depth in DECODER_DEPTHS:
                run_name = f"1_large_hyper_param_sweep_lr_{lr}_mask_ratio_{mask_ratio}_weight_decay_{weight_decay}_decoder_depth_{decoder_depth}"
                run_cmd = "launch"
                start_command = "python3" if run_cmd == "launch" else "torchrun"
                cmd = (
                    f"{start_command} scripts/2025_08_05_large_sweeps/train_cross_large_no_maps.py "
                    f"{run_cmd} {run_name} {cluster} --launch.priority=high --launch.num_gpus=8 --model.decoder_config.depth={decoder_depth} --train_module.optim_config.lr={lr} --train_module.masking_config.strategy_config.encode_ratio={encode_ratio} --train_module.masking_config.strategy_config.decode_ratio={decode_ratio} --train_module.optim_config.weight_decay={weight_decay}"
                )
                print(cmd)
                subprocess.run(cmd, shell=True)  # nosec
