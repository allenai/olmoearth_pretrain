#!/usr/bin/env python3
"""Python launcher for data size sweep experiments.

Runs nano and tiny models with 500, 1000, 5000, and all data.

Usage:
    python launch_data_size_sweep.py [CLUSTER] [PRIORITY] [NUM_GPUS] [--models MODEL1,MODEL2] [--data-sizes SIZE1 SIZE2 ...]

Arguments:
    CLUSTER: Cluster name(s) - one of:
             - Single cluster: neptune, jupiter, ceres, or saturn
             - All clusters: "all"
             - Multiple clusters: "neptune,jupiter,ceres" (comma-separated)
             Default: saturn
    PRIORITY: Priority level (low, normal, high, urgent)
              Default: high
    NUM_GPUS: Number of GPUs per experiment
              Default: 1
    --models: Optional filter for models (comma-separated: nano,tiny)
              Default: all models
    --data-sizes: Optional filter for data sizes (space-separated: 500 1000 all)
                  Default: all data sizes

Examples:
    python launch_data_size_sweep.py
    python launch_data_size_sweep.py ceres
    python launch_data_size_sweep.py all high 1
    python launch_data_size_sweep.py neptune urgent
    python launch_data_size_sweep.py jupiter high 4
    python launch_data_size_sweep.py all high 1 --models tiny --data-sizes 500 1000
    python launch_data_size_sweep.py neptune,jupiter urgent 4 --models nano --data-sizes 5000
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
    
<<<<<<< HEAD
    # Note: Evaluation frequency overrides are commented out due to config system limitations.
    # Overriding nested task config fields creates incomplete configs. The defaults will be used:
    # - m-eurosat/mados: 4000 steps
    # - m_so2sat/pastis: 20000 steps
    # To change eval frequency, modify script.py directly or use a different override approach.
    # cmd.append("--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.eval_interval.value=1000")
    # cmd.append("--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.eval_interval.unit=steps")
    
=======
>>>>>>> 913ee47416ff1aad7547575c1a2f5634b4263538
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
    cluster_arg = sys.argv[1] if len(sys.argv) > 1 else "saturn"
    priority = sys.argv[2] if len(sys.argv) > 2 else "high"
    num_gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    # Optional filters: --models and --data-sizes
    # Example: python launch_data_size_sweep.py saturn high 1 --models tiny --data-sizes 500 1000
    models_to_run = MODELS
    data_sizes_to_run = DATA_SIZES
    
    if "--models" in sys.argv:
        idx = sys.argv.index("--models")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            models_to_run = sys.argv[idx + 1].split(",")
            # Validate models
            for model in models_to_run:
                if model not in MODELS:
                    print(f"Error: Invalid model '{model}'. Valid models: {', '.join(MODELS)}")
                    sys.exit(1)
    
    if "--data-sizes" in sys.argv:
        idx = sys.argv.index("--data-sizes")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            data_sizes_to_run = []
            i = idx + 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                size = sys.argv[i]
                if size == "all":
                    data_sizes_to_run.append("all")
                else:
                    try:
                        data_sizes_to_run.append(int(size))
                    except ValueError:
                        print(f"Error: Invalid data size '{size}'. Must be an integer or 'all'")
                        sys.exit(1)
                i += 1
    
    # Determine which clusters to use
    if cluster_arg.lower() == "all":
        cluster_names = VALID_CLUSTERS
    elif "," in cluster_arg:
        # Comma-separated list of clusters
        cluster_names = [c.strip() for c in cluster_arg.split(",")]
    else:
        cluster_names = [cluster_arg]
    
    # Validate cluster names
    for cluster_name in cluster_names:
        if cluster_name not in VALID_CLUSTERS:
            print(f"Error: Invalid cluster '{cluster_name}'")
            print(f"Valid clusters: {', '.join(VALID_CLUSTERS)} (or 'all' for all clusters)")
            sys.exit(1)
    
    # Format cluster names for beaker (ai2/{cluster}-cirrascale)
    clusters = [f"ai2/{cluster_name}-cirrascale" for cluster_name in cluster_names]

    print("=" * 50)
    print("Launching data size sweep experiments")
    print("=" * 50)
    print(f"Clusters: {', '.join(clusters)}")
    print(f"Priority: {priority}")
    print(f"Num GPUs: {num_gpus}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Data sizes: {', '.join(map(str, data_sizes_to_run))})")
    print("=" * 50)
    print()

    # Launch filtered combinations across all clusters
    for cluster in clusters:
        for model in models_to_run:
            for data_size in data_sizes_to_run:
                try:
                    launch_experiment(model, data_size, cluster, priority, num_gpus)
                except subprocess.CalledProcessError as e:
                    print(f"Error launching {model} with {data_size} samples on {cluster}: {e}")
                    sys.exit(1)

    print("=" * 50)
    print("All experiments launched!")
    print("=" * 50)


if __name__ == "__main__":
    main()

