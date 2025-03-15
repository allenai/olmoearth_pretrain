"""Common utiities for laucnhing experiments on beaker."""

import logging
from dataclasses import dataclass

from beaker import ExperimentSpec, RetrySpec, TaskResources, TaskSpec
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_beaker_username
from olmo_core.launch.beaker import (
    BeakerEnvSecret,
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerPriority,
    BeakerWekaBucket,
    OLMoCoreBeakerImage,
)
from olmo_core.launch.utils import ensure_repo
from olmo_core.utils import generate_uuid

from helios.data.constants import Modality
from helios.internal.experiment import CommonComponents, SubCmd

logger = logging.getLogger(__name__)
BUDGET = "ai2/d5"
WORKSPACE = "ai2/earth-systems"

DEFAULT_HELIOS_WEKA_BUCKET = BeakerWekaBucket("dfive-default", "/weka/dfive-default")
PROJECT_NAME = "helios"

WEKA_CLUSTER_NAMES = ["jupiter", "saturn", "neptune", "ceres", "triton"]


DEFAULT_SETUP_STEPS = (
    'git clone "$REPO_URL" .',
    'git checkout "$GIT_REF"',
    "git submodule update --init --recursive",
    "conda shell.bash activate base",
    "pip install -e '.[all]'",
    "pip freeze",
)


@dataclass
class HeliosLaunchConfig(BeakerLaunchConfig):
    """Testing extra functionality."""

    num_cpus: int = 40  # at least 32 per gpu

    def build_experiment_spec(
        self, torchrun: bool = True, entrypoint: str | None = None
    ) -> ExperimentSpec:
        """Get the Beaker experiment spec corresponding to this config instance."""
        # Get repository account, name, and current ref.
        github_account, github_repo, git_ref, is_public = ensure_repo(self.allow_dirty)

        if not is_public and self.setup_steps == DEFAULT_SETUP_STEPS:
            raise OLMoConfigurationError(
                "It looks like your repository is private and private repositories will require "
                "custom 'setup_steps' in order to clone the repo."
            )

        entrypoint_script = [
            "#!/usr/bin/env bash",
            "set -exuo pipefail",
            "[[ -d /var/lib/tcpxo/lib64 ]] && export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH",
            # Setup the kernel cache directory used by pytorch
            "mkdir -p /root/.cache/torch/kernels && export PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels",
            "mkdir -p /olmo-core-runtime",
            "cd /olmo-core-runtime",
            *self.setup_steps,
        ]

        if torchrun:
            if self.num_nodes > 1 and any(
                ["augusta" in cluster for cluster in self.clusters]
            ):
                entrypoint_script.append(
                    "BEAKER_REPLICA_RANK=$("
                    "python -m olmo_core.launch.reorder_ranks_in_gcp "
                    "${BEAKER_REPLICA_RANK} "
                    "${BEAKER_REPLICA_COUNT} "
                    "${BEAKER_LEADER_REPLICA_HOSTNAME}"
                    ")"
                )
                entrypoint_script.append(
                    "export BEAKER_REPLICA_RANK=$BEAKER_REPLICA_RANK"
                )
            entrypoint_script.append(" ".join(self._get_torchrun_cmd()) + ' "$@"')
        else:
            entrypoint = entrypoint or "python"
            entrypoint_script.append(f'{entrypoint} "$@"')

        entrypoint_dataset = self._create_script_dataset(
            "entrypoint.sh", entrypoint_script
        )

        task_spec = (
            TaskSpec.new(
                self.task_name,
                beaker_image=self.beaker.image.get(self.beaker_image).id,
                priority=self.priority,
                preemptible=self.preemptible,
                arguments=self.cmd,
                command=["bash", "/olmo-core/entrypoint.sh"],
                replicas=self.num_nodes if self.num_nodes > 1 else None,
                leader_selection=self.num_nodes > 1,
                host_networking=(
                    self.host_networking
                    if self.host_networking is not None
                    else (
                        self.num_nodes > 1
                        or any(["augusta" in cluster for cluster in self.clusters])
                    )
                ),
                propagate_failure=False if self.num_nodes > 1 else None,
                propagate_preemption=True if self.num_nodes > 1 else None,
                synchronized_start_timeout="90m" if self.num_nodes > 1 else None,
                resources=TaskResources(
                    gpu_count=self.num_gpus,
                    cpu_count=self.num_cpus,
                    shared_memory=self.shared_memory,
                ),
            )
            .with_dataset("/olmo-core", beaker=entrypoint_dataset.id)
            .with_constraint(cluster=self.clusters)
            .with_env_var(
                "REPO_URL", f"https://github.com/{github_account}/{github_repo}"
            )
            .with_env_var("GIT_REF", git_ref)
        )

        for name, val in self._get_env_vars():
            task_spec = task_spec.with_env_var(name=name, value=val)

        for env_secret in self.env_secrets or []:
            task_spec = task_spec.with_env_var(
                name=env_secret.name, secret=env_secret.secret
            )

        if self.nfs:
            task_spec = task_spec.with_dataset(
                "/net/nfs.cirrascale", host_path="/net/nfs.cirrascale"
            )
            task_spec = task_spec.with_dataset(
                "/net/nfs", host_path="/net/nfs.cirrascale"
            )

        if self.weka_buckets:
            for bucket in self.weka_buckets:
                task_spec = task_spec.with_dataset(bucket.mount, weka=bucket.bucket)

        return ExperimentSpec(
            description=self.description,
            budget=self.budget,
            tasks=[task_spec],
            retry=(
                None
                if not self.retries
                else RetrySpec(allowed_task_retries=self.retries)
            ),
        )


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
    return HeliosLaunchConfig(
        name=f"{name}-{generate_uuid()[:8]}",
        budget=budget,
        cmd=cmd,
        task_name=task_name,
        workspace=workspace,
        clusters=clusters if isinstance(clusters, list) else [clusters],
        weka_buckets=weka_buckets,
        beaker_image=f"henryh/{OLMoCoreBeakerImage.stable}",  # we can all use the same image for now
        num_nodes=1,
        num_gpus=8,
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
