"""Shared constants important for inference throughput benchmarking."""

# BEAKER-LAND
BEAKER_BUDGET = "ai2/es-platform"
BEAKER_WORKSPACE = "ai2/earth-systems"
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

ARTIFACTS_DIR = "/artifacts"

# METRICS
PER_BATCH_TOKEN_RATE_METRIC = "per_batch_token_rate"  # nosec
MEAN_BATCH_TOKEN_RATE_METRIC = "mean_batch_token_rate"  # nosec
MEAN_BATCH_TIME_METRIC = "mean_batch_time"  # nosec
NUM_TOKENS_PER_BATCH_METRIC = "num_tokens_per_batch"  # nosec
SQUARE_KM_PER_SECOND_METRIC = "square_km_per_second"  # nosec
PIXELS_PER_SECOND_METRIC = "pixels_per_second"  # nosec
OOM_OCCURRED_METRIC = "oom_occurred"  # nosec 1 if OOM occurred, 0 if not
GPU_NAME_METRIC = "gpu_name"  # nosec

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


# a bunch of different Sweep dicts that we could use
sweep_batch_sizes = {
    "batch_size": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
}
sweep_image_sizes = {
    "image_size": [1, 2, 4, 8, 16, 32, 64, 128],
}
sweep_patch_sizes = {"patch_size": [1, 2, 4, 8]}
sweep_num_timesteps = {"num_timesteps": [1, 2, 4, 6, 8, 12]}
sweep_use_s1 = {"use_s1": [True, False]}
sweep_use_s2 = {"use_s2": [True, False]}
sweep_use_landsat = {"use_landsat": [True, False]}
sweep_bf16 = {"bf16": [True, False]}
sweep_model_size = {"model_size": ["nano", "tiny", "base", "large"]}


SWEEPS = {
    "batch": sweep_batch_sizes,
    "image": sweep_image_sizes,
    "patch": sweep_patch_sizes,
    "time": sweep_num_timesteps,
    "use_s1": sweep_use_s1,
    "use_s2": sweep_use_s2,
    "use_landsat": sweep_use_landsat,
    "bf16": sweep_bf16,
    "model_size": sweep_model_size,
    "all": sweep_batch_sizes
    | sweep_image_sizes
    | sweep_patch_sizes
    | sweep_num_timesteps
    | sweep_use_s1
    | sweep_use_s2
    | sweep_use_landsat
    | sweep_bf16
    | sweep_model_size,
}
