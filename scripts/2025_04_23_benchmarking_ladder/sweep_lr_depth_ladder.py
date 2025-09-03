"""This script sweeps across model sizes, decoder depths, and learning rates for Galileo."""

import itertools
import subprocess  # nosec

from helios.internal.utils import MODEL_SIZE_ARGS

# Model size configurations
MODEL_SIZES = {
    "large": MODEL_SIZE_ARGS["large_shallow_decoder"],
    "giga": MODEL_SIZE_ARGS["giga_shallow_decoder"],
}

# Sweep parameters
CLUSTERS = ["ai2/jupiter-cirrascale-2", "ai2/titan-cirrascale"]

DECODER_DEPTHS = [2, 4]
LEARNING_RATES = [4e-3, 1e-4, 4e-4, 1e-5]
CONTRASTIVE_WEIGHTS = [0.05]

# Base command template
BASE_COMMAND = (
    "python3 scripts/2025_04_23_benchmarking_ladder/base_galileo_max.py launch {run_name} "
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
    "--launch.priority=urgent "
    "--launch.num_gpus=8 "
)
# Generate all combinations
all_combinations = list(
    itertools.product(MODEL_SIZES.items(), DECODER_DEPTHS, LEARNING_RATES)
)
print(f"Running {len(all_combinations)} jobs")
# First 6 jobs go to titan, the rest to jupiter
for i, ((size_name, size_config), decoder_depth, lr) in enumerate(all_combinations):
    # Determine which cluster to use
    cluster = "ai2/titan-cirrascale" if i < 8 else "ai2/jupiter-cirrascale-2"

    cluster_name = "titan" if "titan" in cluster else "jupiter"
    # Construct run name
    run_name = f"1_galileo_contrastive_0.05_s2_s1_wc_{size_name}_dec{decoder_depth}_lr{lr}_{cluster_name}"

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
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
