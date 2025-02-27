"""Common utiities for laucnhing experiments on beaker."""

from olmo_core.internal.common import get_beaker_username
from olmo_core.io import is_url
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.utils import generate_uuid

BUDGET = "ai2/d5"
WORKSPACE = "ai2/earth-systems"

DEFAULT_HELIOS_WEKA_BUCKET = BeakerWekaBucket("dfive-default", "/weka/dfive-default")


def get_root_dir(cluster: str) -> str:
    root_dir: str = f"weka://{DEFAULT_HELIOS_WEKA_BUCKET.bucket}"
    if "jupiter" in cluster:
        root_dir = f"/weka/{DEFAULT_HELIOS_WEKA_BUCKET.bucket}"
    elif "augusta" in cluster:
        raise NotImplementedError("Augusta not supported yet")
    elif "local" in cluster:
        raise NotImplementedError("Local not supported yet")
    return root_dir


def build_launch_config(
    *,
    name: str,
    root_dir: str,
    cmd: list[str],
    clusters: list[str] | str,
    task_name: str = "train",
    workspace: str = WORKSPACE,
    budget: str = BUDGET,
    nccl_debug: bool = False,
) -> BeakerLaunchConfig:
    weka_buckets: list[BeakerWekaBucket] = []
    if root_dir.startswith("/weka/"):
        # I am pretty sure we check this cuz it might be augusta or something
        weka_buckets.append(DEFAULT_HELIOS_WEKA_BUCKET)

    beaker_user = get_beaker_username()

    return BeakerLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=clusters if isinstance(clusters, list) else [clusters],
        weka_buckets=weka_buckets,
        beaker_image=OLMoCoreBeakerImage.stable,
        num_nodes=1,
        num_gpus=8,
        shared_filesystem=not is_url(root_dir),
        allow_dirty=False,
        env_vars=[
            BeakerEnvVar(name="NCCL_DEBUG", value="INFO" if nccl_debug else "WARN")
        ],
        # TODO: These EnvSecrets might be pretty annoying to make EVERYONE  to se tup but worht standardizing
        env_secrets=[
            BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
            # TODO: Update to match the convention of name first
            BeakerEnvSecret(name="WANDB_API_KEY", secret="WANDB_API_KEY"),
            BeakerEnvSecret(name="GITHUB_PAT", secret="GITHUB_PAT"),
            # BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
            # BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
            # BeakerEnvSecret(name="SLACK_WEBHOOK_URL", secret="SLACK_WEBHOOK_URL"),
        ],
        setup_steps=[
            # Clone repo.
            "git clone 'https://$GITHUB_PAT@github.com/$GITHUB_REPOSITORY' .",
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
