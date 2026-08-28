"""Common utiities for laucnhing experiments on beaker."""

import logging
import os

from olmo_core.internal.common import get_beaker_username
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
    is_running_in_beaker,
)
from olmo_core.utils import generate_uuid
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthBeakerLaunchConfig,
    OlmoEarthVisualizeConfig,
    SubCmd,
)

logger = logging.getLogger(__name__)
BUDGET = "ai2/atec-olmoearth"
WORKSPACE = "ai2/earth-systems"

DEFAULT_OLMOEARTH_PRETRAIN_WEKA_BUCKET = BeakerWekaBucket(
    "dfive-default", "/weka/dfive-default"
)
PROJECT_NAME = "olmoearth_pretrain"

WEKA_CLUSTER_NAMES = [
    "jupiter",
    "saturn",
    "neptune",
    "ceres",
    "triton",
    "titan",
    "rhea",
]

LOCAL_CLUSTER_NAME = "local"
ANONYMOUS_USER = "anonymous"


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config for an experiment."""
    return OlmoEarthVisualizeConfig(
        num_samples=50,
        output_dir=str(UPath(common.save_folder) / "visualizations"),
        std_multiplier=2.0,
    )


def get_root_dir(cluster: str) -> str:
    """Get the root directory for the experiment.

    This is where the save_folder will be stored
    """
    if any(weka_cluster_name in cluster for weka_cluster_name in WEKA_CLUSTER_NAMES):
        root_dir = (
            f"/weka/{DEFAULT_OLMOEARTH_PRETRAIN_WEKA_BUCKET.bucket}/{PROJECT_NAME}"
        )
    elif "augusta" in cluster:
        # There does not seem to be any way to set the result directory in olmo-core.
        # Here we use /unused/ which is the default result directory in beaker-py, it
        # should work but it is not meant to be used like this, it is just meant to be
        # a placeholder.
        root_dir = f"/unused/{PROJECT_NAME}"
    elif LOCAL_CLUSTER_NAME in cluster:
        root_dir = "./local_output"
    else:
        raise ValueError(f"Cluster {cluster} is not supported")
    return root_dir


def extract_nccl_debug_from_overrides(overrides: list[str]) -> bool:
    """Extract the nccl_debug flag from the overrides."""
    for override in overrides:
        if override.startswith("--common.nccl_debug="):
            return override.split("=")[1].lower() in ("true", "1", "yes")
    return False


def set_nccl_debug_env_vars(
    nccl_debug: bool, local: bool = False
) -> list[BeakerEnvVar] | None:
    """Set the NCCL debug environment variables.

    If on_beaker is True, returns a list of BeakerEnvVar for use in a Beaker launch config.
    Otherwise, sets these variables in the local environment and returns None.
    """
    nccl_settings = {
        "NCCL_DEBUG": "DETAIL" if nccl_debug else "WARN",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "1000000000" if nccl_debug else "0",
        "TORCH_NCCL_BLOCKING_WAIT": "1" if nccl_debug else "0",
    }

    if not local:
        return [BeakerEnvVar(name=k, value=v) for k, v in nccl_settings.items()]
    else:
        for k, v in nccl_settings.items():
            os.environ[k] = v
        return None


def build_launch_config(
    *,
    name: str,
    cmd: list[str],
    clusters: list[str] | str,
    task_name: str = "train",
    workspace: str = WORKSPACE,
    budget: str = BUDGET,
    nccl_debug: bool = False,
) -> OlmoEarthBeakerLaunchConfig:
    """Build a launch config for an OlmoEarth Pretrain experiment.

    THis will be the default setup, any changes that are temporary should be overriden
    on the commandline
    """
    if isinstance(clusters, str):
        clusters = [clusters]
    weka_buckets: list[BeakerWekaBucket]
    # We cannot mount Weka on Augusta.
    # We just check if the first cluster is Augusta here since we assume users
    # targeting Augusta won't target any other cluster.
    weka_buckets = [DEFAULT_OLMOEARTH_PRETRAIN_WEKA_BUCKET]
    for c in clusters:
        if "augusta" in c:
            if len(clusters) > 1:
                raise ValueError(
                    "Jobs targeting Augusta should not target other clusters since Weka will not be mounted"
                )
            weka_buckets = []

    beaker_user = get_beaker_username()
    # Gantry writes the GCP credentials and exports GOOGLE_APPLICATION_CREDENTIALS
    # itself (see google_credentials_secret below), so it's not set here.
    env_vars: list[BeakerEnvVar] = []
    if weka_buckets:
        # Share a uv cache on Weka across jobs so wheels are downloaded (and
        # sdists like flash-attn are compiled) only once. Gantry's entrypoint
        # picks this up and flock-guards it against concurrent jobs.
        env_vars.append(
            BeakerEnvVar(
                name="UV_CACHE_DIR",
                value=f"/weka/{DEFAULT_OLMOEARTH_PRETRAIN_WEKA_BUCKET.bucket}/{PROJECT_NAME}/uv-cache",
            )
        )
    nccl_debug_env_vars = set_nccl_debug_env_vars(nccl_debug=nccl_debug)
    if nccl_debug_env_vars is not None:
        env_vars.extend(nccl_debug_env_vars)
    # Propagate the train module path to the experiment if set
    train_script_path = os.environ.get("TRAIN_SCRIPT_PATH")
    if train_script_path is not None:
        logger.info(f"Propagating train script path to experiment: {train_script_path}")
        env_vars.append(BeakerEnvVar(name="TRAIN_SCRIPT_PATH", value=train_script_path))
    # Propagate checkpoint sweep env vars to the experiment if set
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir is not None:
        logger.info(f"Propagating checkpoint dir to experiment: {checkpoint_dir}")
        env_vars.append(BeakerEnvVar(name="CHECKPOINT_DIR", value=checkpoint_dir))
    checkpoint_steps = os.environ.get("CHECKPOINT_STEPS")
    if checkpoint_steps is not None:
        logger.info(f"Propagating checkpoint steps to experiment: {checkpoint_steps}")
        env_vars.append(BeakerEnvVar(name="CHECKPOINT_STEPS", value=checkpoint_steps))
    # In-loop eval jobs source their tasks from the training run's own config
    # (rather than the shared catalog); propagate the flag so the remote job knows.
    loop_eval_from_train_config = os.environ.get("OE_LOOP_EVAL_FROM_TRAIN_CONFIG")
    if loop_eval_from_train_config is not None:
        logger.info("Propagating OE_LOOP_EVAL_FROM_TRAIN_CONFIG to experiment")
        env_vars.append(
            BeakerEnvVar(
                name="OE_LOOP_EVAL_FROM_TRAIN_CONFIG",
                value=loop_eval_from_train_config,
            )
        )
    # Propagate the finetune tag to the experiment if set
    finetune = os.environ.get("FINETUNE")
    if finetune is not None:
        logger.info(f"Propagating finetune tag to experiment: {finetune}")
        env_vars.append(BeakerEnvVar(name="FINETUNE", value=finetune))
    # Propagate the experiment key to the experiment if set
    experiment = os.environ.get("EXPERIMENT")
    if experiment is not None:
        logger.info(f"Propagating experiment key to experiment: {experiment}")
        env_vars.append(BeakerEnvVar(name="EXPERIMENT", value=experiment))
    embedding_diagnostics_only = os.environ.get("EMBEDDING_DIAGNOSTICS_ONLY")
    if embedding_diagnostics_only is not None:
        env_vars.append(
            BeakerEnvVar(
                name="EMBEDDING_DIAGNOSTICS_ONLY", value=embedding_diagnostics_only
            )
        )

    return OlmoEarthBeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=clusters,
        weka_buckets=weka_buckets,
        # torch 2.9.1 + CUDA 12.8, matching the torch pin in pyproject.toml.
        beaker_image=OLMoCoreBeakerImage.stable_cu128,
        num_nodes=1,
        num_gpus=1,
        shared_memory="256GiB",
        shared_filesystem=True,  # We only use Weka for now
        allow_dirty=False,
        priority="high",
        env_vars=env_vars,
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(
                name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"
            ),  # nosec
        ],
        # Gantry clones the (private) repo, authenticating with this workspace
        # secret, and writes these GCP credentials to a file, exporting
        # GOOGLE_APPLICATION_CREDENTIALS to point at it.
        gh_token_secret=f"{beaker_user}_GITHUB_TOKEN",
        google_credentials_secret="HELIOS_GCP_CREDENTIALS",
        # Gantry sets up the Python environment with uv; this replaces its
        # default install command so the environment matches the lockfile.
        system_python=False,
        install="uv sync --locked --all-extras",
        # breizhcrops is not part of the lockfile, so install it separately.
        post_setup="uv pip install breizhcrops==0.0.4.1",
    )


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build the common components for an experiment."""
    TRAINING_MODALITIES = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
        # Modality.WORLDCOVER.name,
        # Modality.LATLON.name,
        # Modality.SRTM.name,
        # Modality.OPENSTREETMAP_RASTER.name,
        # Modality.NAIP_10.name,
        # Modality.ERA5_10.name,
    ]
    if cmd == SubCmd.launch:
        cmd_to_launch = SubCmd.train
    elif cmd == SubCmd.launch_evaluate:
        cmd_to_launch = SubCmd.evaluate
    elif cmd == SubCmd.launch_prep:
        cmd_to_launch = SubCmd.prep
    elif cmd == SubCmd.launch_benchmark:
        cmd_to_launch = SubCmd.benchmark
    else:
        cmd_to_launch = cmd

    # Extract nccl_debug from overrides if present
    nccl_debug = extract_nccl_debug_from_overrides(overrides)
    # If we are running on a local cluster, we don't need to build a launch config as we may not have beaker access
    if local := cluster == LOCAL_CLUSTER_NAME:
        set_nccl_debug_env_vars(nccl_debug=nccl_debug, local=local)
        launch_config = None
    else:
        launch_config = build_launch_config(
            name=f"{run_name}-{cmd_to_launch}",
            cmd=[script, cmd_to_launch, run_name, cluster, *overrides],
            clusters=cluster,
            nccl_debug=nccl_debug,
        )
        # Set retries=2 for launch command
        if cmd == SubCmd.launch:
            launch_config.retries = 2
    root_dir = get_root_dir(cluster)

    beaker_user = get_beaker_username() or ANONYMOUS_USER
    if is_running_in_beaker() and beaker_user is None:
        raise ValueError(
            "Failed to get Beaker username. Make sure you are authenticated with Beaker if you are not running on a local cluster."
        )
    return CommonComponents(
        run_name=run_name,
        save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
        launch=launch_config,
        training_modalities=TRAINING_MODALITIES,
    )
