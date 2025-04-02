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
    "tiny": {
        "decoder_depth": 12,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
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
    "large_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "base_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
}

EXIT_CONFIG_TYPES = ["zero", "half", "full", "varied"]


# TODO: THis should be added to the code so we don't have to configure directly anymore
def build_token_exit_config(
    config_type: str, modality_names: list[str], encoder_depth: int
) -> str:
    """Build the token exit config for an experiment."""
    if config_type == "zero":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}=0"
            for modality_name in modality_names
        )
    elif config_type == "half":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth // 2}"
            for modality_name in modality_names
        )
    elif config_type == "full":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
            for modality_name in modality_names
        )
    elif config_type == "varied":
        varied_args = []
        for modality_name in modality_names:
            if modality_name not in ["latlon", "worldcover"]:
                varied_args.append(
                    f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
                )
            else:
                varied_args.append(f"--train_module.token_exit_cfg.{modality_name}=0")
        return " ".join(varied_args)
    else:
        raise ValueError(f"Invalid config type: {config_type}")


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
                run_name = f"7latent_mim_{masking_type}_patch_disc_new_exit_{exit_config}_{size_str}"

                # Construct full command
                command = BASE_COMMAND.format(
                    run_name=run_name,
                    **args,
                    masking_type=masking_type,
                    token_exit_args=token_exit_args,
                    num_gpus=4,  # Added num_gpus param
                )

                print(f"Launching: {command}")

                # Execute the command
                subprocess.run(command, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
