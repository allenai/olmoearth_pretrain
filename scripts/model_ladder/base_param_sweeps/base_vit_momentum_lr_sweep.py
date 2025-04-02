"""Looking at base and largemodel with momentum and different learning rates."""

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
    "large": {
        "decoder_depth": 24,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
}

EMA_DECAYS = [0.9993]
EMA_DECAYS = EMA_DECAYS[::-1]

LEARNING_RATES = [2e-4, 3e-4, 1e-3, 2e-3]

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
    '--train_module.ema_decay="[{ema_decay}, {ema_decay}]" '
    "--train_module.optim_config.lr={lr} "
    "--train_module.rank_microbatch_size={rank_microbatch_size} "
    "--launch.num_gpus={num_gpus}"
)
# Iterate over all combinations of hyperparameters
for size_str, args in MODEL_SIZE_ARGS.items():
    if size_str == "large":
        rank_microbatch_size = 16
        num_gpus = 8
    else:
        rank_microbatch_size = 32
        num_gpus = 4
    # we may need the rank microbatch to be 16 with 8 gpusfor large models without fsdp
    for lr in LEARNING_RATES:
        for ema_decay in EMA_DECAYS:
            # Construct run name indicating hyperparameters
            run_name = f"2base_latent_mim_momentum_{ema_decay}_lr_{lr}_{size_str}"

            # Construct full command
            command = BASE_COMMAND.format(
                run_name=run_name,
                **args,
                ema_decay=ema_decay,
                lr=lr,
                rank_microbatch_size=rank_microbatch_size,
                num_gpus=num_gpus,
            )

            print(f"Launching: {command}")

            # Execute the command
            subprocess.run(command, shell=True, check=True)  # nosec
