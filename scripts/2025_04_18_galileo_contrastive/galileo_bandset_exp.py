"""Script for running a sweep of the Galileo model."""

import subprocess  # nosec

DECODER_DEPTHS = [2]
LEARNING_RATES = [0.0001]
CONTRASTIVE_WEIGHTS = [0.05]


BASE_COMMAND = (
    "python3 scripts/2025_04_18_galileo_contrastive/galileo_base.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--model.decoder_config.depth={decoder_depth} "
    "--train_module.optim_config.lr={lr} "
    "--train_module.contrastive_config.loss_config.type=InfoNCE "
    "--train_module.contrastive_config.loss_config.weight={contrastive_weight} "
    "--launch.num_gpus=8 "
    "--launch.priority=urgent"
)

# 12 experiments
for decoder_depth in DECODER_DEPTHS:
    for lr in LEARNING_RATES:
        for contrastive_weight in CONTRASTIVE_WEIGHTS:
            run_name = "20250501_galileo_contrastive_base_semantic_bandsets"
            command = BASE_COMMAND.format(
                run_name=run_name,
                decoder_depth=decoder_depth,
                lr=lr,
                contrastive_weight=contrastive_weight,
            )
            print(command)
            # Execute the command
            subprocess.run(command, shell=True, check=True)  # nosec
