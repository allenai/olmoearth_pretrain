"""Looking at base model with momentum and different learning rates"""

import subprocess  # nosec

# MASKING_TYPES = [
#     "random",
#     "space_time",
# ]
MODEL_SIZE_ARGS = {
    "base": {
        "decoder_depth": 12,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
}

EMA_DECAYS = [0.841, 0.946, 0.974, 0.987, 0.992, 0.997, 0.9993]

LEARNING_RATES = [3e-4, 1e-3, 2e-3]

# Base command template
BASE_COMMAND = (
    "python3 scripts/model_ladder/latent_mim_base_script.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--model.encoder_config.embedding_size={encoder_embedding_size} "
    "--model.decoder_config.encoder_embedding_size={encoder_embedding_size} "
    "--model.decoder_config.decoder_embedding_size={decoder_embedding_size} "
    "--model.encoder_config.depth={encoder_depth} "
    "--model.decoder_config.depth={decoder_depth} "
    "--model.encoder_config.num_heads={encoder_num_heads} "
    "--model.decoder_config.num_heads={decoder_num_heads} "
    "--model.encoder_config.mlp_ratio={mlp_ratio} "
    "--model.decoder_config.mlp_ratio={mlp_ratio} "
    "--train_module.ema_decay=\\[{ema_decay}, {ema_decay}\\] "
    "--train_module.optim_config.lr={lr} "
    "--launch.num_gpus=4"
)
# Iterate over all combinations of hyperparameters
for size_str, args in MODEL_SIZE_ARGS.items():
    for lr in LEARNING_RATES:
        for ema_decay in EMA_DECAYS:
            # Construct run name indicating hyperparameters
            run_name = f"base_latent_mim_momentum_{ema_decay}_lr_{lr}_{size_str}"

            # Construct full command
            command = BASE_COMMAND.format(
                run_name=run_name,
                **args,
                ema_decay=ema_decay,
                lr=lr,
            )

            print(f"Launching: {command}")
            break
        break

            # Execute the command
            # subprocess.run(command, shell=True, check=True)  # nosec
