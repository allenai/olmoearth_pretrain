"""This script is used to sweep the hyperparameters for the latentmin tiny model."""

import itertools
import subprocess  # nosec

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

TINY_MODEL_CFG = {
    "encoder_config.embedding_size": 192,
    "decoder_config.encoder_embedding_size": 192,
    "decoder_config.decoder_embedding_size": 192,
    "encoder_config.depth": 12,
    "decoder_config.depth": 12,
    "encoder_config.num_heads": 3,
    "decoder_config.num_heads": 3,
}
TINY_MODEL_ARGS = " ".join(
    f"--model.{key}={value}" for key, value in TINY_MODEL_CFG.items()
)

TINYVAR_MODEL_CFG = {
    "encoder_config.embedding_size": 192,
    "decoder_config.encoder_embedding_size": 192,
    "decoder_config.decoder_embedding_size": 192,
    "encoder_config.depth": 12,
    "decoder_config.depth": 4,
    "encoder_config.num_heads": 3,
    "decoder_config.num_heads": 3,
}
TINYVAR_MODEL_ARGS = " ".join(
    f"--model.{key}={value}" for key, value in TINYVAR_MODEL_CFG.items()
)

BASE_MODEL_CFG = {
    "encoder_config.embedding_size": 768,
    "decoder_config.encoder_embedding_size": 768,
    "decoder_config.decoder_embedding_size": 768,
    "encoder_config.depth": 12,
    "decoder_config.depth": 12,
    "encoder_config.num_heads": 12,
    "decoder_config.num_heads": 12,
}
BASE_MODEL_ARGS = " ".join(
    f"--model.{key}={value}" for key, value in BASE_MODEL_CFG.items()
)

BASEVAR_MODEL_CFG = {
    "encoder_config.embedding_size": 768,
    "decoder_config.encoder_embedding_size": 768,
    "decoder_config.decoder_embedding_size": 768,
    "encoder_config.depth": 12,
    "decoder_config.depth": 4,
    "encoder_config.num_heads": 12,
    "decoder_config.num_heads": 12,
}
BASEVAR_MODEL_ARGS = " ".join(
    f"--model.{key}={value}" for key, value in BASEVAR_MODEL_CFG.items()
)

MODEL_ARGS = [
    (TINY_MODEL_ARGS, "tiny"),
    (TINYVAR_MODEL_ARGS, "tinyvar"),
    (BASE_MODEL_ARGS, "base"),
    (BASEVAR_MODEL_ARGS, "basevar"),
]

# Base command template
BASE_COMMAND = (
    "python3 scripts/2025_04_09_multidataset/train.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--train_module.rank_microbatch_size={rank_microbatch_size} "
    "{token_exit_args} "
    "{model_args} "
    "--launch.num_gpus=4 "
    "--data_loader.global_batch_size=512"
)

# Iterate over all combinations of hyperparameters
for token_exit_args, model_args in itertools.product(
    TOKEN_EXIT_ARGS,
    MODEL_ARGS,
):
    # Construct run name indicating hyperparameters
    run_name = f"20250409_02_exit_{token_exit_args[1]}_model_{model_args[1]}"

    if "base" in model_args[1]:
        # Lower batch size, otherwise it is too much memory usage.
        rank_microbatch_size = 32
    else:
        rank_microbatch_size = 128

    # Construct full command
    command = BASE_COMMAND.format(
        run_name=run_name,
        token_exit_args=token_exit_args[0],
        model_args=model_args[0],
        rank_microbatch_size=rank_microbatch_size,
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
