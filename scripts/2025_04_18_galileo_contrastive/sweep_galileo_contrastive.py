"""Script for running a sweep of the Galileo model."""

# (1) model size: base
# (2) dataset size: presto + osm
# (3) contrastive loss weight: 0.05, 0.1, 0.2
# (4) decoder depth: 2, 4

import subprocess  # nosec

MODEL_CONFIGS = {
    # "tiny": {
    #     "encoder_embedding_size": 192,
    #     "decoder_embedding_size": 192,
    #     "encoder_depth": 12,
    #     "encoder_num_heads": 3,
    #     "decoder_num_heads": 3,
    #     "mlp_ratio": 4.0,
    # },
    "base": {
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    # "large": {
    #     "encoder_embedding_size": 1024,
    #     "decoder_embedding_size": 1024,
    #     "encoder_depth": 24,
    #     "encoder_num_heads": 16,
    #     "decoder_num_heads": 16,
    #     "mlp_ratio": 4.0,
    # },
}

DECODER_DEPTHS = [2, 4]
LEARNING_RATES = [0.0001, 0.004]
CONTRASTIVE_WEIGHTS = [0.05, 0.1, 0.2]


BASE_COMMAND = (
    "python3 scripts/2025_04_18_galileo_contrastive/galileo.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--model.encoder_config.embedding_size={encoder_embedding_size} "
    "--model.decoder_config.encoder_embedding_size={encoder_embedding_size} "
    "--model.decoder_config.decoder_embedding_size={decoder_embedding_size} "
    "--model.encoder_config.depth={encoder_depth} "
    "--model.decoder_config.depth={decoder_depth} "
    "--model.encoder_config.num_heads={encoder_num_heads} "
    "--model.decoder_config.num_heads={decoder_num_heads} "
    "--model.encoder_config.mlp_ratio={mlp_ratio} "
    "--model.decoder_config.mlp_ratio={mlp_ratio} "
    "--train_module.optim_config.lr={lr} "
    "--train_module.contrastive_config.loss_config.type=InfoNCE "
    "--train_module.contrastive_config.loss_config.weight={contrastive_weight} "
    "--launch.num_gpus=8"
)

# 12 experiments
for model_name, model_config in MODEL_CONFIGS.items():
    for decoder_depth in DECODER_DEPTHS:
        for lr in LEARNING_RATES:
            for contrastive_weight in CONTRASTIVE_WEIGHTS:
                run_name = f"galileo_contrastive_{model_name}_decoder_{decoder_depth}_lr_{lr}_weight_{contrastive_weight}"
                command = BASE_COMMAND.format(
                    run_name=run_name,
                    **model_config,
                    decoder_depth=decoder_depth,
                    lr=lr,
                    contrastive_weight=contrastive_weight,
                )
                print(command)
                # Execute the command
                subprocess.run(command, shell=True, check=True)  # nosec
