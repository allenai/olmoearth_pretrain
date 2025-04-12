"""2025_04_11 Model Ladder Sweep."""

# (1) model size: tiny / base / large
# (2) dataset size: presto / presto + osm
# (3) decoder ratio: 0.5 / 0.7 / 0.9

import subprocess  # nosec

MODEL_TRAINING_CONFIGS = {
    # "tiny": {
    #     "decoder_depth": 4,
    #     "encoder_embedding_size": 192,
    #     "decoder_embedding_size": 192,
    #     "encoder_depth": 12,
    #     "encoder_num_heads": 3,
    #     "decoder_num_heads": 3,
    #     "mlp_ratio": 4.0,
    #     "lr": 0.0001,
    #     "wd": 0.02,
    # },
    # "base": {
    #     "decoder_depth": 4,
    #     "encoder_embedding_size": 768,
    #     "decoder_embedding_size": 768,
    #     "encoder_depth": 12,
    #     "encoder_num_heads": 12,
    #     "decoder_num_heads": 12,
    #     "mlp_ratio": 4.0,
    #     "lr": 0.0001,
    #     "wd": 0.03,
    # },
    "large": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
        "lr": 0.0001,
        "wd": 0.02,
    },
}

BASE_SCRIPT_NAMES = [
    "latentmim_dataset_presto",
    "latentmim_dataset_osm_presto",
]

DECODER_RATIOS = [0.75, 0.9]

BASE_COMMAND = (
    "python3 scripts/model_ladder/2025_04_11/{script_name}_script.py launch {run_name} ai2/jupiter-cirrascale-2 "
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
    "--train_module.optim_config.weight_decay={wd} "
    "--train_module.masking_config.strategy_config.decode_ratio={decode_ratio} "
    "--launch.num_gpus=4"
)

for script_name in BASE_SCRIPT_NAMES:
    for model_name, model_config in MODEL_TRAINING_CONFIGS.items():
        for decoder_ratio in DECODER_RATIOS:
            run_name = f"{script_name}_{model_name}_decoder_{decoder_ratio}_lr_1e-4"
            command = BASE_COMMAND.format(
                script_name=script_name,
                run_name=run_name,
                **model_config,
                decode_ratio=decoder_ratio,
            )
            print(command)
            # Execute the command
            subprocess.run(command, shell=True, check=True)  # nosec
