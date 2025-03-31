"""Try some of the best configurations for each dataset across a bunch of different Model Sizes.

I want to try these at Base and Large as well

I want to try these witha couple different decoder depths 2, 6, 12

random_masking_patch_disc_new_exit_zero
was the best overall run
Eurosat best:


    space time and space loss exit zero

    MADOS best:
    all disc exit zero random
    all disc exit hald
    random_masking_patch_disc_new_exit_zero


    space time patch disc exit zero
    all disc modality space time exit half
"""

import subprocess  # nosec

MASKING_TYPES = [
    "random",
    "space_time",
]
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
    "large_shallow_decoder": {
        "decoder_depth": 6,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "base_shallow_decoder": {
        "decoder_depth": 6,
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
    "--train_module.masking_config.strategy_config.type={masking_type} "
    "--train_module.ema_decay=\[{ema_decay}, {ema_decay}\] "
    "--train_module.optim_config.lr={lr} "
    "--launch.num_gpus=4"
)

# Iterate over all combinations of hyperparameters
for size_str, args in MODEL_SIZE_ARGS.items():
    # Construct run name indicating hyperparameters
    for masking_type in MASKING_TYPES:
        run_name = f"5latent_mim_{masking_type}_patch_disc_new_exit_zero_{size_str}"

        # Construct full command
        command = BASE_COMMAND.format(
            run_name=run_name,
            **args,
            masking_type=masking_type,
        )

        print(f"Launching: {command}")

        # # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec
