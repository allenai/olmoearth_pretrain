"""Common utiities for laucnhing experiments on beaker."""

import logging

from olmo_core.internal.common import get_beaker_username
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerPriority,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid

from helios.data.constants import Modality
from helios.internal.experiment import CommonComponents, SubCmd

logger = logging.getLogger(__name__)
BUDGET = "ai2/d5"
WORKSPACE = "ai2/earth-systems"

DEFAULT_HELIOS_WEKA_BUCKET = BeakerWekaBucket("dfive-default", "/weka/dfive-default")
PROJECT_NAME = "helios"

WEKA_CLUSTER_NAMES = ["jupiter", "saturn", "neptune", "ceres", "triton"]


def get_root_dir(cluster: str) -> str:
    """Get the root directory for the experiment.

    This is where the save_folder will be stored
    """
    if any(weka_cluster_name in cluster for weka_cluster_name in WEKA_CLUSTER_NAMES):
        root_dir = f"/weka/{DEFAULT_HELIOS_WEKA_BUCKET.bucket}/{PROJECT_NAME}"
    elif "augusta" in cluster:
        raise ValueError("Augusta is not supported yet")
    elif "local" in cluster:
        root_dir = "./local_output"
    else:
        raise ValueError(f"Cluster {cluster} is not supported")
    return root_dir


def build_launch_config(
    *,
    name: str,
    cmd: list[str],
    clusters: list[str] | str,
    task_name: str = "train",
    workspace: str = WORKSPACE,
    budget: str = BUDGET,
    nccl_debug: bool = False,
) -> BeakerLaunchConfig:
    """Build a launch config for a helios experiment.

    THis will be the default setup, any changes that are temporary should be overriden
    on the commandline
    """
    weka_buckets: list[BeakerWekaBucket] = [DEFAULT_HELIOS_WEKA_BUCKET]

    beaker_user = get_beaker_username()
    return BeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=clusters if isinstance(clusters, list) else [clusters],
        weka_buckets=weka_buckets,
        beaker_image=f"henryh/{OLMoCoreBeakerImage.stable}",  # we can all use the same image for now
        num_nodes=1,
        num_gpus=1,
        shared_memory="256GiB",
        shared_filesystem=True,  # We only use Weka for now
        allow_dirty=False,
        priority=BeakerPriority.high,
        env_vars=[
            BeakerEnvVar(name="NCCL_DEBUG", value="INFO" if nccl_debug else "WARN")
        ],
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            BeakerEnvSecret(
                name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"
            ),  # nosec
            BeakerEnvSecret(name="GITHUB_TOKEN", secret=f"{beaker_user}_GITHUB_TOKEN"),  # nosec
        ],
        setup_steps=[
            # Clone private repo.
            "conda install gh --channel conda-forge",
            # assumes that conda is installed, which is true for our beaker images.
            "gh auth status",
            "gh repo clone $REPO_URL .",
            'git checkout "$GIT_REF"',
            "git submodule update --init --recursive",
            # Setup python environment.
            "conda shell.bash activate base",
            "pip install -e '.[all]'",
            "pip install --upgrade beaker-py",
            # Quickly try a new version of PyTorch like this
            #  "pip install --upgrade --pre torch==2.6.0.dev20241112+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121",
            "pip freeze",
        ],
    )


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """Build the common components for an experiment."""
    # Variables to be changed per user
    SUPPORTED_MODALITIES = [
        Modality.SENTINEL2_L2A.name,
        Modality.LATLON.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
    ]

    cmd_to_launch = SubCmd.train
    if cmd == SubCmd.launch_prep:
        cmd_to_launch = SubCmd.prep

    launch_config = build_launch_config(
        name=f"{run_name}-{cmd_to_launch}",
        cmd=[script, cmd_to_launch, run_name, cluster, *overrides],
        clusters=cluster,
        nccl_debug=False,
    )
    root_dir = get_root_dir(cluster)
    beaker_user = get_beaker_username()
    return CommonComponents(
        run_name=run_name,
        save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
        supported_modality_names=SUPPORTED_MODALITIES,
        launch=launch_config,
    )
