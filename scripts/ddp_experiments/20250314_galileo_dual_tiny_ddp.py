"""This script is running ddp experiments for the Galileo tiny model."""

# Example run: https://wandb.ai/eai-ai2/helios-train/runs/in6gc4g7

import itertools
import subprocess  # nosec

# Fixed training parameters
PREFETCH_FACTOR = 1
NUM_WORKERS = 4
GLOBAL_BATCH_SIZE = 512
RANK_MICROBATCH_SIZE = 64

# Fixed model parameters
ENCODER_EMBEDDING_SIZE = 256
DECODER_EMBEDDING_SIZE = 256
ENCODER_DEPTH = 4
DECODER_DEPTH = 2
ENCODER_NUM_HEADS = 8
DECODER_NUM_HEADS = 8
MLP_RATIO = 4

# Sweep parameters
LEARNING_RATES = [2e-3]
WEIGHT_DECAYS = [3e-2]
WARMUP_EPOCHS = [30]

# Base command template
BASE_COMMAND = (
    "python3 scripts/galileo.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--model.encoder_config.embedding_size={encoder_embedding_size} "
    "--model.encoder_config.depth={encoder_depth} "
    "--model.encoder_config.num_heads={encoder_num_heads} "
    "--model.encoder_config.mlp_ratio={mlp_ratio} "
    "--model.decoder_config.encoder_embedding_size={encoder_embedding_size} "
    "--model.decoder_config.decoder_embedding_size={decoder_embedding_size} "
    "--model.decoder_config.depth={decoder_depth} "
    "--model.decoder_config.num_heads={decoder_num_heads} "
    "--model.decoder_config.mlp_ratio={mlp_ratio} "
    "--data_loader.num_workers={num_workers} "
    "--data_loader.prefetch_factor={prefetch_factor} "
    "--data_loader.global_batch_size={global_batch_size} "
    "--train_module.rank_microbatch_size={rank_microbatch_size} "
    "--train_module.optim_config.lr={lr} "
    "--train_module.optim_config.weight_decay={wd} "
    "--train_module.warmup_duration.value={warmup} "
    "--train_module.warmup_duration.unit=epochs "
    "--launch.num_gpus=8 --launch.shared_memory=512GiB "
)

# Iterate over all combinations of hyperparameters
for lr, wd, warmup in itertools.product(LEARNING_RATES, WEIGHT_DECAYS, WARMUP_EPOCHS):
    # Construct run name indicating hyperparameters
    run_name = f"galileo_tiny_modality_ddp_lr_{lr}_wd_{wd}_warmup_{warmup}_2"

    # Construct full command
    command = BASE_COMMAND.format(
        run_name=run_name,
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        encoder_depth=ENCODER_DEPTH,
        decoder_depth=DECODER_DEPTH,
        encoder_num_heads=ENCODER_NUM_HEADS,
        decoder_num_heads=DECODER_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        global_batch_size=GLOBAL_BATCH_SIZE,
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        lr=lr,
        wd=wd,
        warmup=warmup,
    )

    print(f"Launching: {command}")

    # Execute the command
    subprocess.run(command, shell=True, check=True)  # nosec
