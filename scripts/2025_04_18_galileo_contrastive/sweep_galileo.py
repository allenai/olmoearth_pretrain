"""Script for running a sweep of the Galileo model."""

# (1) model size: base
# (2) dataset size: presto + osm
# (3) decoder depth: 2, 4

import subprocess  # nosec

DECODER_DEPTHS = [2, 4]
LEARNING_RATES = [0.00001, 0.0001, 0.004]


BASE_COMMAND = (
    "python3 scripts/2025_04_18_galileo_contrastive/galileo_base.py launch {run_name} ai2/titan-cirrascale "  # Modify cluster name
    "--model.decoder_config.depth={decoder_depth} "
    "--train_module.optim_config.lr={lr} "
    "--launch.num_gpus=8"
)

# 4 experiments
for decoder_depth in DECODER_DEPTHS:
    for lr in LEARNING_RATES:
        run_name = f"3_galileo_base_decoder_{decoder_depth}_lr_{lr}"
        command = BASE_COMMAND.format(
            run_name=run_name,
            decoder_depth=decoder_depth,
            lr=lr,
        )
        print(command)
        # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec
