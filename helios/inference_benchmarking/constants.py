"""Shared constants important for inference throughput benchmarking."""

# BEAKER-LAND
BEAKER_BUDGET = "ai2/d5"
BEAKER_WORKSPACE = "ai2/earth-systems"
BEAKER_IMAGE_NAME = "chrisw/helios-inf-throughput-flashattn-no-mask-2024-04-21_01"
WEKA_BUCKET = "dfive-default"
BEAKER_TASK_PRIORITY = "normal"
BEAKER_GPU_TO_CLUSTER_MAP = {
    "H100": [
        "ai2/jupiter-cirrascale-2",
    ],
    "A100": [
        "ai2/saturn-cirrascale",
    ],
    "L40S": [
        "ai2/neptune-cirrascale",
    ],
}

# GCP-land
# TODO: gcp project, bucket, gcr image, gpu to instance type

# wandb-land
PROJECT_NAME = "inference-throughput-no-mask"
ENTITY_NAME = "eai-ai2"

# representative model weights at each size, predictions not important for benchmark, only computation itself
NANO_PATH = "helios/checkpoints/henryh/missing_mosality_test_updated_2/step197250"
TINY_PATH = "helios/checkpoints/henryh/8latent_mim_random_patch_disc_new_exit_full_lr_0.0004_tiny/step154250"
BASE_PATH = "helios/checkpoints/henryh/8latent_mim_random_patch_disc_new_exit_full_lr_0.0004_base_shallow_decoder/step95000"
LARGE_PATH = "helios/checkpoints/henryh/8latent_mim_random_patch_disc_new_exit_full_lr_0.0004_large_shallow_decoder/step95000"

MODEL_SIZE_MAP = {
    "NANO": NANO_PATH,
    "TINY": TINY_PATH,
    "BASE": BASE_PATH,
    "LARGE": LARGE_PATH,
}

ARTIFACTS_DIR = "/artifacts"

# METRICS
PER_BATCH_TOKEN_RATE_METRIC = "per_batch_token_rate"  # nosec
MEAN_BATCH_TOKEN_RATE_METRIC = "mean_batch_token_rate"  # nosec
MEAN_BATCH_TIME_METRIC = "mean_batch_time"  # nosec
NUM_TOKENS_PER_BATCH_METRIC = "num_tokens_per_batch"  # nosec

PARAM_KEYS = dict(
    model_size="MODEL_SIZE",
    checkpoint_path="CHECKPOINT_PATH",
    use_s1="USE_S1",
    use_s2="USE_S2",
    use_landsat="USE_LANDSAT",
    image_size="IMAGE_SIZE",
    patch_size="PATCH_SIZE",
    num_timesteps="NUM_TIMESTEPS",
    batch_size="BATCH_SIZE",
    batch_sizes="BATCH_SIZES",
    gpu_type="GPU_TYPE",
    bf16="BF16",
    benchmark_interval_s="BENCHMARK_INTERVAL_S",
    min_batches_per_interval="MIN_BATCHES_PER_INTERVAL",
    project="PROJECT",
    owner="OWNER",
    name="NAME",
)
