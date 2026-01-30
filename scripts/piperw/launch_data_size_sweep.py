#!/usr/bin/env python3
"""Python launcher for data size sweep experiments.

Runs nano and tiny models with 500, 1000, 5000, and all data.

Usage:
    python launch_data_size_sweep.py [CLUSTER] [PRIORITY] [NUM_GPUS]

Example:
    python launch_data_size_sweep.py ai2/ceres-cirrascale urgent 4
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


def calculate_dataset_percentage(num_samples: int | str) -> float:
    """Calculate dataset_percentage from number of samples.
    
    Args:
        num_samples: Number of samples (int) or "all" for full dataset
        
    Returns:
        Dataset percentage (0.0 to 1.0)
    """
    if num_samples == "all":
        return 1.0
    elif isinstance(num_samples, int):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_samples > TOTAL_DATASET_SIZE:
            print(
                f"Warning: Requested {num_samples} samples but dataset only has "
                f"{TOTAL_DATASET_SIZE}. Using full dataset (1.0)"
            )
            return 1.0
        return num_samples / TOTAL_DATASET_SIZE
    else:
        raise ValueError(f"num_samples must be int or 'all', got {type(num_samples)}")


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

    # Calculate dataset percentage
    dataset_percentage = calculate_dataset_percentage(data_size)

    print(f"Launching: {run_name}")
    print(f"  Model: {model}")
    print(f"  Data size: {data_size}")
    print(f"  Dataset percentage: {dataset_percentage:.10f}")

    # Build command
    cmd = [
        sys.executable,
        str(script),
        "launch",
        run_name,
        cluster,
        f"--dataset.dataset_percentage={dataset_percentage}",
        f"--launch.clusters=[{cluster}]",
        f"--launch.priority={priority}",
        f"--launch.num_gpus={num_gpus}",
    ]

    # Run the command
    subprocess.run(cmd, check=True)
    print()


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    cluster = sys.argv[1] if len(sys.argv) > 1 else "ai2/saturn-cirrascale"
    priority = sys.argv[2] if len(sys.argv) > 2 else "high"
    num_gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 1

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

