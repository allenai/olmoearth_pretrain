"""Common utiities for laucnhing experiments on beaker."""

import logging
import os

from olmo_core.internal.common import get_beaker_username
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerPriority,
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
BUDGET = "ai2/es-platform"
WORKSPACE = "ai2/earth-systems"
DEFAULT_BEAKER_IMAGE = f"petew/{OLMoCoreBeakerImage.stable_cu128}"
DEFAULT_FA3_BEAKER_IMAGE = "tylerr/olmo-core-tch271cu128-2025-11-25"
DEFAULT_UV_SYNC_CMD = "uv sync --locked --all-extras"
DEFAULT_FA3_UV_SYNC_CMD = "uv sync --locked --extra all-no-flash"
DEFAULT_FA3_EGG_PATH = (
    "/opt/conda/lib/python3.12/site-packages/"
    "flash_attn_3-3.0.0b1-py3.12-linux-x86_64.egg"
)

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


def extract_use_fa3_from_overrides(overrides: list[str]) -> bool:
    """Extract the use_fa3 flag from the overrides."""
    for override in overrides:
        if override.startswith("--common.use_fa3="):
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


def get_uv_sync_cmd(use_fa3: bool) -> str:
    """Get uv sync command based on whether FA3 is enabled."""
    if use_fa3:
        return DEFAULT_FA3_UV_SYNC_CMD
    return DEFAULT_UV_SYNC_CMD


def get_fa3_egg_path() -> str:
    """Get default FA3 egg path."""
    return DEFAULT_FA3_EGG_PATH


def build_launch_config(
    *,
    name: str,
    cmd: list[str],
    clusters: list[str] | str,
    task_name: str = "train",
    workspace: str = WORKSPACE,
    budget: str = BUDGET,
    nccl_debug: bool = False,
    use_fa3: bool = False,
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
    # Propagate the train module path to the experiment if set
    env_vars = [
        BeakerEnvVar(
            name="GOOGLE_APPLICATION_CREDENTIALS", value="/etc/gcp_credentials.json"
        ),
    ]
    nccl_debug_env_vars = set_nccl_debug_env_vars(nccl_debug=nccl_debug)
    if nccl_debug_env_vars is not None:
        env_vars.extend(nccl_debug_env_vars)
    # Propagate the train module path to the experiment if set
    train_script_path = os.environ.get("TRAIN_SCRIPT_PATH")
    if train_script_path is not None:
        logger.info(f"Propagating train script path to experiment: {train_script_path}")
        env_vars.append(BeakerEnvVar(name="TRAIN_SCRIPT_PATH", value=train_script_path))
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
    uv_sync_cmd = get_uv_sync_cmd(use_fa3=use_fa3)
    fa3_egg_path = get_fa3_egg_path()
    beaker_image = DEFAULT_FA3_BEAKER_IMAGE if use_fa3 else DEFAULT_BEAKER_IMAGE

    setup_steps = [
        # Write GCP credentials.
        'echo "$GCP_CREDENTIALS" > $GOOGLE_APPLICATION_CREDENTIALS',
        # Clone private repo.
        "conda install gh --channel conda-forge",
        # assumes that conda is installed, which is true for our beaker images.
        "gh auth status",
        "gh repo clone $REPO_URL .",
        'git checkout "$GIT_REF"',
        "git submodule update --init --recursive",
        "pip install uv",
        # so that we can use uv tools
        'export PATH="/root/.local/bin:$PATH" ',
        uv_sync_cmd,
        # activate the uv venv
        "venv_path=$(uv run python -c 'import sys; print(sys.executable)')",
        'source "$(dirname "$venv_path")/activate"',
        # explicitly install breizhcrops
        "uv pip install breizhcrops==0.0.4.1 ",
    ]
    if use_fa3:
        setup_steps.extend(
            [
                # Ensure FA3 from the base image is visible in the uv venv.
                f"python - <<'PY'\nimport site\nfrom pathlib import Path\nsp = Path(site.getsitepackages()[0])\npth = sp / '_fa3_only.pth'\nfa3_egg_path = {fa3_egg_path!r}\npth.write_text(f'{{fa3_egg_path}}\\n')\nprint('wrote', pth, '->', fa3_egg_path)\nPY",
                "uv run python -c 'import flash_attn_interface; print(\"FA3 import OK\")'",
            ]
        )
    setup_steps.extend(
        [
            # debugging - check torch version
            "uv pip show torch",
            # and then show the arch
            "uv run python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())'",
        ]
    )

    return OlmoEarthBeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=clusters,
        weka_buckets=weka_buckets,
        beaker_image=beaker_image,
        num_nodes=1,
        num_gpus=1,
        shared_memory="256GiB",
        shared_filesystem=True,  # We only use Weka for now
        allow_dirty=False,
        priority=BeakerPriority.high,
        env_vars=env_vars,
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(
                name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"
            ),  # nosec
            BeakerEnvSecret(name="GITHUB_TOKEN", secret=f"{beaker_user}_GITHUB_TOKEN"),  # nosec
            BeakerEnvSecret(name="GCP_CREDENTIALS", secret="HELIOS_GCP_CREDENTIALS"),  # nosec
        ],
        setup_steps=setup_steps,
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

    # Extract launch-relevant common overrides before CommonComponents exists.
    nccl_debug = extract_nccl_debug_from_overrides(overrides)
    use_fa3 = extract_use_fa3_from_overrides(overrides)
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
            use_fa3=use_fa3,
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
        use_fa3=use_fa3,
    )
