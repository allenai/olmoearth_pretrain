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

from helios.internal.utils import MODEL_SIZE_ARGS, build_token_exit_config

MASKING_TYPES = [
    "random",
    "space_time",
]


EXIT_CONFIG_TYPES = ["zero", "full"]

LEARNING_RATE_ARGS = [4e-5, 4e-4, 2e-3]


# Base command template
BASE_COMMAND = (
    "python3 scripts/model_ladder/latent_mim_base_model_ladder_script.py launch {run_name} ai2/jupiter-cirrascale-2 "
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
    "--train_module.optim_config.lr={lr} "
    "{token_exit_args} "
    "--launch.num_gpus={num_gpus}"
)


def main() -> None:
    """Run the model ladder sweep."""
    number_of_runs = len(MODEL_SIZE_ARGS) * len(MASKING_TYPES) * len(EXIT_CONFIG_TYPES)
    print(f"Number of runs: {number_of_runs}")
    # Iterate over all combinations of hyperparameters
    for size_str, args in MODEL_SIZE_ARGS.items():
        for masking_type in MASKING_TYPES:
            for exit_config in EXIT_CONFIG_TYPES:
                for lr in LEARNING_RATE_ARGS:
                    # Modality names for token exit configuration
                    modality_names = [
                        "sentinel2_l2a",
                        "sentinel1",
                        "latlon",
                        "worldcover",
                    ]

                    encoder_depth = int(args["encoder_depth"])
                    # Build token exit config arguments
                    token_exit_args = build_token_exit_config(
                        exit_config, modality_names, encoder_depth
                    )

                    # Construct run name indicating hyperparameters
                    run_name = f"8latent_mim_{masking_type}_patch_disc_new_exit_{exit_config}_lr_{lr}_{size_str}"

                    # Construct full command
                    command = BASE_COMMAND.format(
                        run_name=run_name,
                        **args,
                        masking_type=masking_type,
                        token_exit_args=token_exit_args,
                        lr=lr,
                        num_gpus=4,  # Added num_gpus param
                    )

                    print(f"Launching: {command}")

                    # Execute the command
                    subprocess.run(command, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
