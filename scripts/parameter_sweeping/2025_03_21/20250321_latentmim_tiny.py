"""This script is used to sweep the hyperparameters for the latentmin tiny model."""

import itertools
import subprocess  # nosec

# Masking configurations
MASKING_TYPES = [
    "random",
    "time",
    "space",
    "modality",
    "space_time",
    "modality_space_time",
]

# Token exit configurations
VARIED_TOKEN_EXIT_CFG = {
    "sentinel2_l2a": 12,
    "sentinel1": 12,
    "latlon": 12,
    "worldcover": 0,
}
VARIED_TOKEN_EXIT_ARGS = " ".join(
    f"--train_module.token_exit_cfg.{key}={value}"
    for key, value in VARIED_TOKEN_EXIT_CFG.items()
)
FULL_ENCODER_TOKEN_EXIT_CFG = {
    "sentinel2_l2a": 12,
    "sentinel1": 12,
    "latlon": 12,
    "worldcover": 12,
}
FULL_ENCODER_TOKEN_EXIT_ARGS = " ".join(
    f"--train_module.token_exit_cfg.{key}={value}"
    for key, value in FULL_ENCODER_TOKEN_EXIT_CFG.items()
)
HALF_ENCODER_TOKEN_EXIT_CFG = {
    "sentinel2_l2a": 6,
    "sentinel1": 6,
    "latlon": 6,
    "worldcover": 6,
}
HALF_ENCODER_TOKEN_EXIT_ARGS = " ".join(
    f"--train_module.token_exit_cfg.{key}={value}"
    for key, value in HALF_ENCODER_TOKEN_EXIT_CFG.items()
)
ZERO_ENCODER_TOKEN_EXIT_CFG = {
    "sentinel2_l2a": 0,
    "sentinel1": 0,
    "latlon": 0,
    "worldcover": 0,
}
ZERO_ENCODER_TOKEN_EXIT_ARGS = " ".join(
    f"--train_module.token_exit_cfg.{key}={value}"
    for key, value in ZERO_ENCODER_TOKEN_EXIT_CFG.items()
)
TOKEN_EXIT_ARGS = [
    (VARIED_TOKEN_EXIT_ARGS, "varied"),
    (FULL_ENCODER_TOKEN_EXIT_ARGS, "full"),
    (HALF_ENCODER_TOKEN_EXIT_ARGS, "half"),
    (ZERO_ENCODER_TOKEN_EXIT_ARGS, "zero"),
]

# Loss function new is the memory efficient loss function
LOSS_TYPES = ["patch_discrimination_new", "l2", "all_discrimination"]

# Base command template
BASE_COMMAND = (
    "python3 scripts/parameter_sweeping/2025_03_21/latent_mim_base_script.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--train_module.masking_config.strategy_config.type={masking_type} "
    "--train_module.loss_config.loss_config.type={loss_type} --train_module.rank_microbatch_size={rank_microbatch_size} "
    "{token_exit_args}"
)

# Iterate over all combinations of hyperparameters
for masking_type, loss_type, token_exit_args in itertools.product(
    MASKING_TYPES,
    LOSS_TYPES,
    TOKEN_EXIT_ARGS,
):
    # Construct run name indicating hyperparameters
    run_name = f"3latentmim_tiny_masking_{masking_type}_loss_{loss_type}_token_exit_{token_exit_args[1]}"

    if loss_type == "all_discrimination":
        # Need to reduce rank microbatch size for all discrimination to avoid OOM
        rank_microbatch_size = 32
    elif "modality" in masking_type:
        # Sometimes we pick both S1 and S2 and the number of tokens is a little too much for 128
        rank_microbatch_size = 64
    else:
        rank_microbatch_size = 128

    # Construct full command
    command = BASE_COMMAND.format(
        run_name=run_name,
        masking_type=masking_type,
        loss_type=loss_type,
        token_exit_args=token_exit_args[0],
        rank_microbatch_size=rank_microbatch_size,
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
