"""Script for running a sweep of the Galileo model."""

# (1) model size: base
# (2) dataset size: presto + osm
# (3) contrastive loss weight: 0.05, 0.1, 0.2
# (4) decoder depth: 2, 4

import subprocess  # nosec

DECODER_DEPTHS = [2]
CONTRASTIVE_WEIGHTS = [0.05, 0.0]
SEEDS = [3622, 42, 114514, 4289, 97]
CLUSTERS = ["ai2/titan-cirrascale", "ai2/jupiter-cirrascale-2"]

BASE_COMMAND = (
    "python3 scripts/2025_05_02_randomness/galileo_random_base.py launch {run_name} {cluster} "
    "--model.decoder_config.depth={decoder_depth} "
    "--train_module.contrastive_config.loss_config.type=InfoNCE "
    "--train_module.contrastive_config.loss_config.weight={contrastive_weight} "
    "--launch.num_gpus=8 "
    "--data_loader.seed={seed} "
    "--launch.priority=urgent"
)

total_experiments = len(DECODER_DEPTHS) * len(CONTRASTIVE_WEIGHTS) * len(SEEDS)
experiment_counter = 0
for decoder_depth in DECODER_DEPTHS:
    for contrastive_weight in CONTRASTIVE_WEIGHTS:
        for seed in SEEDS:
            run_name = f"random_variation_galileo_vs_contrastive_base_decoder_{decoder_depth}_seed_{seed}_weight_{contrastive_weight}"
            if experiment_counter < 7:
                cluster = CLUSTERS[0]
            else:
                cluster = CLUSTERS[1]
            command = BASE_COMMAND.format(
                run_name=run_name,
                cluster=cluster,
                decoder_depth=decoder_depth,
                contrastive_weight=contrastive_weight,
                seed=seed,
            )
            print(command)
            # Execute the command
            subprocess.run(command, shell=True, check=True)  # nosec
            experiment_counter += 1
            print(f"Experiment {experiment_counter} of {total_experiments} completed")
