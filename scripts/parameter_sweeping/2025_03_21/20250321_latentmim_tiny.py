"""This script is used to sweep the hyperparameters for the latentmin tiny model."""

import itertools
import subprocess  # nosec

# Fixed training parameters
NUM_WORKERS = 8

# Fixed model parameters
ENCODER_EMBEDDING_SIZE = 192
DECODER_EMBEDDING_SIZE = 192
ENCODER_DEPTH = 12
DECODER_DEPTH = 12
ENCODER_NUM_HEADS = 3
DECODER_NUM_HEADS = 3
MLP_RATIO = 4.0

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

# Loss function
LOSS_TYPES = ["patch_discrimination", "l2"]


# Sweep parameters
LEARNING_RATES = [2e-3]
WEIGHT_DECAYS = [2e-2]
WARMUP_EPOCHS = [10]

# Base command template
BASE_COMMAND = (
    "python3 scripts/latent_mim.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--model.encoder_config.embedding_size={encoder_embedding_size} "
    "--model.encoder_config.depth={encoder_depth} "
    "--model.encoder_config.num_heads={encoder_num_heads} "
    "--model.encoder_config.mlp_ratio={mlp_ratio} "
    "--model.decoder_config.encoder_embedding_size={encoder_embedding_size} "
    "--model.decoder_config.decoder_embedding_size={decoder_embedding_size} "
    "--model.decoder_config.depth={decoder_depth} "
    "--model.decoder_config.num_heads={decoder_num_heads} "
    "--model.decoder_config.mlp_ratio={mlp_ratio} "
    "--data_loader.num_workers={num_workers} "
    "--train_module.masking_config.strategy_config.type={masking_type} "
    "--train_module.loss_config.loss_config.type={loss_type} "
    "--train_module.optim_config.lr={lr} "
    "--train_module.optim_config.weight_decay={wd} "
    "--train_module.warmup_duration.value={warmup} "
    "--train_module.warmup_duration.unit=epochs "
    "{token_exit_args}"
)

# Iterate over all combinations of hyperparameters
for lr, wd, warmup, masking_type, loss_type, token_exit_args in itertools.product(
    LEARNING_RATES,
    WEIGHT_DECAYS,
    WARMUP_EPOCHS,
    MASKING_TYPES,
    LOSS_TYPES,
    TOKEN_EXIT_ARGS,
):
    # Construct run name indicating hyperparameters
    run_name = f"latentmim_tiny_masking_{masking_type}_loss_{loss_type}_token_exit_{token_exit_args[1]}"

    # Construct full command
    command = BASE_COMMAND.format(
        run_name=run_name,
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        encoder_depth=ENCODER_DEPTH,
        decoder_depth=DECODER_DEPTH,
        encoder_num_heads=ENCODER_NUM_HEADS,
        decoder_num_heads=DECODER_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_workers=NUM_WORKERS,
        masking_type=masking_type,
        loss_type=loss_type,
        lr=lr,
        wd=wd,
        warmup=warmup,
        token_exit_args=token_exit_args[0],
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
