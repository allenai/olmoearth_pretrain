"""Investigations of global batch size on larger datasets."""

import itertools
import subprocess  # nosec

from helios.internal.utils import MODEL_SIZE_ARGS

# Model size configurations
MODEL_SIZES = {
    "large": MODEL_SIZE_ARGS["large_shallow_decoder"],
    # "giga": MODEL_SIZE_ARGS["giga_shallow_decoder"],
}

# Sweep parameters
CLUSTERS = ["ai2/jupiter-cirrascale-2"]

DECODER_DEPTHS = [2, 4]
LEARNING_RATES = [2e-4, 1e-4, 4e-4, 5e-5]
CONTRASTIVE_WEIGHTS = [0.05]
DROP_RATIO = [0.0, 0.2, 0.3]

# Base command template
BASE_COMMAND = (
    "python3 scripts/2025_04_23_benchmarking_ladder/base_galileo_weka.py launch {run_name} "
    "{cluster} "
    "--model.encoder_config.embedding_size={encoder_embedding_size} "
    "--model.encoder_config.depth={encoder_depth} "
    "--model.encoder_config.num_heads={encoder_num_heads} "
    "--model.encoder_config.mlp_ratio={mlp_ratio} "
    "--model.decoder_config.encoder_embedding_size={encoder_embedding_size} "
    "--model.decoder_config.decoder_embedding_size={decoder_embedding_size} "
    "--model.decoder_config.depth={decoder_depth} "
    "--model.decoder_config.num_heads={decoder_num_heads} "
    "--model.decoder_config.mlp_ratio={mlp_ratio} "
    "--train_module.contrastive_config.loss_config.type=InfoNCE "
    "--train_module.contrastive_config.loss_config.weight={contrastive_weight} "
    "--train_module.optim_config.lr={lr} "
    "--model.encoder_config.drop_path={drop_path} "
    "--launch.priority=urgent "
    "--launch.num_gpus=8 "
)
# Generate all combinations
all_combinations = list(
    itertools.product(MODEL_SIZES.items(), DECODER_DEPTHS, LEARNING_RATES, DROP_RATIO)
)
print(f"Running {len(all_combinations)} jobs")
# First 6 jobs go to titan, the rest to jupiter
for i, ((size_name, size_config), decoder_depth, lr, drop_ratio) in enumerate(
    all_combinations
):
    cluster = CLUSTERS[0]
    # Construct run name
    run_name = f"2_galileo_contrastive_0.05_s2_s1_wc_{size_name}_dec{decoder_depth}_lr{lr}_drop{drop_ratio}_jupiter"

    # Construct full command
    command = BASE_COMMAND.format(
        run_name=run_name,
        cluster=cluster,
        encoder_embedding_size=size_config["encoder_embedding_size"],
        decoder_embedding_size=size_config["decoder_embedding_size"],
        encoder_depth=size_config["encoder_depth"],
        decoder_depth=decoder_depth,  # Use sweep parameter
        encoder_num_heads=size_config["encoder_num_heads"],
        decoder_num_heads=size_config["decoder_num_heads"],
        mlp_ratio=size_config["mlp_ratio"],
        contrastive_weight=CONTRASTIVE_WEIGHTS[0],
        lr=lr,  # Use sweep parameter
        drop_path=drop_ratio,
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
