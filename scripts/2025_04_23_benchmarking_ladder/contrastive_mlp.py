"""doing contrastive with MLPs"""

"""Sweep over contrastive weights for Galileo model."""

import subprocess  # nosec

# Model configuration
MODEL_SIZE = "base_super_shallow_decoder"
LEARNING_RATE = 1e-4

# Sweep parameters
CLUSTER = "ai2/jupiter-cirrascale-2"
CONTRASTIVE_WEIGHTS = [0.1, 0.2, 0.5]

# Base command template
BASE_COMMAND = (
    "python3 scripts/2025_04_23_benchmarking_ladder/base_galileo_max.py launch "
    "{run_name} {cluster} "
    "--train_module.contrastive_config.loss_config.type=InfoNCE "
    "--train_module.contrastive_config.loss_config.weight={contrastive_weight} "
    "--launch.priority=urgent "
    "--launch.num_gpus=8 "
)

print(f"Running {len(CONTRASTIVE_WEIGHTS)} jobs")

for contrastive_weight in CONTRASTIVE_WEIGHTS:
    # Construct run name
    run_name = f"galileo_contrastive_mlp_{contrastive_weight}_s2_s1_wc"

    # Construct full command
    command = BASE_COMMAND.format(
        run_name=run_name,
        cluster=CLUSTER,
        contrastive_weight=contrastive_weight,
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
