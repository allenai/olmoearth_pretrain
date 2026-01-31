#!/usr/bin/env python3
"""Python launcher for data size sweep experiments.

Runs nano and tiny models with 500, 1000, 5000, and all data.

Usage:
    python launch_data_size_sweep.py [CLUSTER] [PRIORITY] [NUM_GPUS]

Arguments:
    CLUSTER: Cluster name (neptune, jupiter, ceres, or saturn)
             Default: saturn
    PRIORITY: Priority level (low, normal, high, urgent)
              Default: high
    NUM_GPUS: Number of GPUs per experiment
              Default: 1

Examples:
    python launch_data_size_sweep.py
    python launch_data_size_sweep.py ceres
    python launch_data_size_sweep.py neptune urgent
    python launch_data_size_sweep.py jupiter high 4
"""

import subprocess
import sys
from pathlib import Path

# Total dataset size (from h5py_dir name in script.py)
TOTAL_DATASET_SIZE = 1138828

# Model sizes and data sizes to run
MODELS = ["nano", "tiny"]
DATA_SIZES = [500, 1000, 5000, "all"]

# Script paths (relative to project root)
SCRIPT_DIR = Path(__file__).parent.parent / "official"
NANO_SCRIPT = SCRIPT_DIR / "nano.py"
TINY_SCRIPT = SCRIPT_DIR / "tiny.py"


def launch_experiment(
    model: str,
    data_size: int | str,
    cluster: str,
    priority: str,
    num_gpus: int,
) -> None:
    """Launch a single experiment.
    
    Args:
        model: Model size ("nano" or "tiny")
        data_size: Number of samples (int) or "all"
        cluster: Cluster name
        priority: Priority level
        num_gpus: Number of GPUs
    """
    # Select the appropriate script
    if model == "nano":
        script = NANO_SCRIPT
    elif model == "tiny":
        script = TINY_SCRIPT
    else:
        raise ValueError(f"Unknown model: {model}")

    # Create run name
    if data_size == "all":
        run_name = f"{model}_data_all"
    else:
        run_name = f"{model}_data_{data_size}"

    print(f"Launching: {run_name}")
    print(f"  Model: {model}")
    print(f"  Data size: {data_size}")

    # Build command
    cmd = [
        sys.executable,
        str(script),
        "launch",
        run_name,
        cluster,
        f"--launch.clusters=[{cluster}]",
        f"--launch.priority={priority}",
        f"--launch.num_gpus={num_gpus}",
    ]
    
    # Add dataset size parameter
    if data_size == "all":
        # For "all", don't set max_training_samples (use full dataset)
        pass
    else:
        # Use max_training_samples instead of dataset_percentage
        cmd.append(f"--dataset.max_training_samples={data_size}")
    
    # Increase batch size (default is 512, we'll increase to 1024)
    cmd.append("--data_loader.global_batch_size=1024")
    
    # Set wandb entity and project
    cmd.append("--trainer.callbacks.wandb.entity=prior-ai2")
    cmd.append("--trainer.callbacks.wandb.project=olmoearth")

    # Run the command
    subprocess.run(cmd, check=True)
    print()


def main() -> None:
    """Main entry point."""
    # Valid cluster options
    VALID_CLUSTERS = ["neptune", "jupiter", "ceres", "saturn"]
    
    # Parse command line arguments
    cluster_name = sys.argv[1] if len(sys.argv) > 1 else "saturn"
    priority = sys.argv[2] if len(sys.argv) > 2 else "high"
    num_gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    # Validate cluster name
    if cluster_name not in VALID_CLUSTERS:
        print(f"Error: Invalid cluster '{cluster_name}'")
        print(f"Valid clusters: {', '.join(VALID_CLUSTERS)}")
        sys.exit(1)
    
    # Format cluster name for beaker (ai2/{cluster}-cirrascale)
    cluster = f"ai2/{cluster_name}-cirrascale"

    print("=" * 50)
    print("Launching data size sweep experiments")
    print("=" * 50)
    print(f"Cluster: {cluster}")
    print(f"Priority: {priority}")
    print(f"Num GPUs: {num_gpus}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Data sizes: {', '.join(map(str, DATA_SIZES))})")
    print("=" * 50)
    print()

    # Launch all combinations
    for model in MODELS:
        for data_size in DATA_SIZES:
            try:
                launch_experiment(model, data_size, cluster, priority, num_gpus)
            except subprocess.CalledProcessError as e:
                print(f"Error launching {model} with {data_size} samples: {e}")
                sys.exit(1)

    print("=" * 50)
    print("All experiments launched!")
    print("=" * 50)


if __name__ == "__main__":
    main()

