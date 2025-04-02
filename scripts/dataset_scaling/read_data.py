#!/usr/bin/env python3
"""Script to launch multiple runs with different dataloader initialization seeds."""

import argparse
import subprocess  # nosec

# Base command template
BASE_COMMAND = (
    "python3 scripts/base_debug_scripts/latent_mim.py launch "
    "read_all_data_once_{seed} ai2/jupiter-cirrascale-2 "
    "--data_loader.seed={seed}"
)

def main():
    """Launch multiple runs with different dataloader initialization seeds."""
    parser = argparse.ArgumentParser(description="Launch runs with different dataloader seeds")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of runs to launch")
    parser.add_argument("--start_seed", type=int, default=1000, help="Starting seed value")
    args = parser.parse_args()

    # Launch runs with different seeds
    for i in range(args.num_runs):
        seed = args.start_seed + i
        run_name = f"seed_sweep_{seed}"

        # Construct full command
        command = BASE_COMMAND.format(seed=seed)

        print(f"Launching run {i+1}/{args.num_runs} with seed {seed}")
        print(f"Command: {command}")

        # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec

        print(f"Successfully launched run with seed {seed}")
        print("-" * 50)

if __name__ == "__main__":
    main()