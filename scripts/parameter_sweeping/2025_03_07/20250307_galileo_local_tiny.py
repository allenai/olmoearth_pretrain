"""This script is used to sweep the hyperparameters for the Galileo local tiny model."""

import itertools
import subprocess  # nosec

# Fixed training parameters
NUM_WORKERS = 8

# Fixed model parameters
ENCODER_EMBEDDING_SIZE = 256
DECODER_EMBEDDING_SIZE = 256
ENCODER_DEPTH = 4
DECODER_DEPTH = 4
ENCODER_NUM_HEADS = 8
DECODER_NUM_HEADS = 8
MLP_RATIO = 4

# Fixed token exit
TOKEN_EXIT_CFG = {
    "sentinel2_l2a": 0,
    "sentinel1": 0,
    "latlon": 0,
    "worldcover": 0,
}
token_exit_args = " ".join(
    f"--train_module.token_exit_cfg.{key}={value}"
    for key, value in TOKEN_EXIT_CFG.items()
)

# Sweep parameters
LEARNING_RATES = [1e-3, 2e-3, 3e-3]
WEIGHT_DECAYS = [1e-2, 2e-2, 3e-2]
WARMUP_EPOCHS = [2, 10, 20, 30]

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
    "--train_module.optim_config.lr={lr} "
    "--train_module.optim_config.weight_decay={wd} "
    "--train_module.warmup_duration.value={warmup} "
    "--train_module.warmup_duration.unit=epochs " + token_exit_args
)

# Iterate over all combinations of hyperparameters
for lr, wd, warmup in itertools.product(LEARNING_RATES, WEIGHT_DECAYS, WARMUP_EPOCHS):
    # Construct run name indicating hyperparameters
    run_name = f"galileo_local_tiny_lr_{lr}_wd_{wd}_warmup_{warmup}"

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
        lr=lr,
        wd=wd,
        warmup=warmup,
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
