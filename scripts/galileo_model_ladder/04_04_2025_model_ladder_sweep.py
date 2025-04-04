"""Galileo Model Ladder Sweep."""

import subprocess  # nosec

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
    "tiny_shallow_decoder": {
        "decoder_depth": 6,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "tiny_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
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


# Trying all combinations of masking types
MASKING_COMBINATIONS = [
    ("random", "random"),
    ("random", "space_time"),
    ("space_time", "random"),
    ("space_time", "space_time"),
]


LEARNING_RATE_ARGS = [4e-5, 4e-4, 4e-3]

DATASET_ARGS = {
    "presto": "/weka/dfive-default/helios/dataset/presto/h5py_data/latlon_sentinel1_sentinel2_l2a_worldcover/98856",
    "osm": "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data/latlon_sentinel1_sentinel2_l2a_worldcover/348102",
}

# datasets presto and OSM

# Base command template
BASE_COMMAND = (
    "python3 scripts/model_ladder/latent_mim_base_model_ladder_script.py dry_run {run_name} ai2/jupiter-cirrascale-2 "
    "--model.encoder_config.embedding_size={encoder_embedding_size} "
    "--model.decoder_config.encoder_embedding_size={encoder_embedding_size} "
    "--model.decoder_config.decoder_embedding_size={decoder_embedding_size} "
    "--model.encoder_config.depth={encoder_depth} "
    "--model.decoder_config.depth={decoder_depth} "
    "--model.encoder_config.num_heads={encoder_num_heads} "
    "--model.decoder_config.num_heads={decoder_num_heads} "
    "--model.encoder_config.mlp_ratio={mlp_ratio} "
    "--model.decoder_config.mlp_ratio={mlp_ratio} "
    "--train_module.masking_config_a.strategy_config.type={masking_type_a} "
    "--train_module.masking_config_b.strategy_config.type={masking_type_b} "
    "--train_module.optim_config.lr={lr} "
    "--launch.num_gpus=4"
)


def main() -> None:
    """Run the model ladder sweep."""
    number_of_runs = (
        len(MODEL_SIZE_ARGS)
        * len(MASKING_COMBINATIONS)
        * len(LEARNING_RATE_ARGS)
        * len(DATASET_ARGS)
    )
    print(f"Number of runs: {number_of_runs}")
    # Iterate over all combinations of hyperparameters
    for size_str, args in MODEL_SIZE_ARGS.items():
        for masking_combination in MASKING_COMBINATIONS:
            for dataset_name, dataset_path in DATASET_ARGS.items():
                for lr in LEARNING_RATE_ARGS:
                    # Construct run name indicating hyperparameters
                    run_name = f"galileo_dataset_{dataset_name}_{masking_combination[0]}_{masking_combination[1]}_{lr}_{size_str}"

                # Construct full command
                command = BASE_COMMAND.format(
                    run_name=run_name,
                    **args,
                    masking_type_a=masking_combination[0],
                    masking_type_b=masking_combination[1],
                    lr=lr,
                )

                print(f"Launching: {command}")

                # Execute the command
                subprocess.run(command, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
