"""Launch GCP Batch jobs to get Sentinel-2 L1C data."""

import argparse
import multiprocessing
import random
import uuid

import tqdm
from google.cloud import batch_v1
from rslearn.dataset import Dataset, Window
from upath import UPath

# Sentinel-2 L1C is only used in the res_10 group.
GROUP = "res_10"

# Relevant layers that would be ingested.
LAYER_NAMES = [
    "sentinel2_freq",
    "sentinel2_mo01",
    "sentinel2_mo02",
    "sentinel2_mo03",
    "sentinel2_mo04",
    "sentinel2_mo05",
    "sentinel2_mo06",
    "sentinel2_mo07",
    "sentinel2_mo08",
    "sentinel2_mo09",
    "sentinel2_mo10",
    "sentinel2_mo11",
    "sentinel2_mo12",
]


def is_window_pending(window: Window) -> bool:
    """Check if the window needs ingestion for Sentinel-2 L1C data.

    Args:
        window: the window to check.

    Returns:
        whether the window hasn't been ingested yet.
    """
    layer_datas = window.load_layer_datas()
    for layer_name in LAYER_NAMES:
        if layer_name not in layer_datas:
            # Not prepared, so doesn't need ingestion.
            continue
        layer_data = layer_datas[layer_name]
        if len(layer_data.serialized_item_groups) == 0:
            continue
        if not window.is_layer_completed(layer_name):
            return True
    return False


def launch_job(
    client: batch_v1.BatchServiceClient,
    image: str,
    project: str,
    region: str,
    ds_path: str,
    window_names: list[str],
) -> None:
    """Launch a Batch job that ingests the specified windows.

    Args:
        client: the Google Batch client to use to start the jobs.
        image: the Docker image URI on GCR.
        project: the GCP project to use.
        region: the GCP region to use.
        ds_path: the dataset path.
        window_names: names of the windows to ingest in this job.
    """
    # Define runnable.
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = image
    runnable.container.entrypoint = "python"
    runnable.container.commands = [
        "helios/dataset_creation/scripts/sentinel2_l1c/entrypoint.py",
        "--ds_path",
        ds_path,
        "--windows",
        ",".join(window_names),
    ]

    # Define single-runnable task.
    task = batch_v1.TaskSpec()
    task.runnables = [runnable]

    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 32000
    resources.memory_mib = 65536
    resources.boot_disk_mib = 1000000
    task.compute_resource = resources

    task.max_retry_count = 1
    task.max_run_duration = "10800s"

    group = batch_v1.TaskGroup()
    group.task_count = 1
    group.task_spec = task

    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = "e2-standard-32"
    policy.provisioning_model = batch_v1.AllocationPolicy.ProvisioningModel.SPOT
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    service_account = batch_v1.ServiceAccount()
    service_account.email = (
        "helios-dataset-creation@earthsystem-dev-c3po.iam.gserviceaccount.com"
    )
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]
    allocation_policy.service_account = service_account

    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy
    job.labels = {"env": "testing", "type": "container"}
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    task_uuid = str(uuid.uuid4())[0:16]
    create_request.job_id = f"helios-sentinel2-l1c-{task_uuid}"
    create_request.parent = f"projects/{project}/locations/{region}"

    client.create_job(create_request)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(
        description="Launch GCP Batch jobs to get Sentinel-2 L1C data",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to the rslearn dataset for dataset creation assuming /weka/dfive-default/ is mounted",
        required=True,
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Docker image URI on GCR to run",
        required=True,
    )
    parser.add_argument(
        "--project",
        type=str,
        help="GCP project to use",
        required=True,
    )
    parser.add_argument(
        "--region",
        type=str,
        help="GCP region to use",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers",
        default=32,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size, i.e., number of windows to ingest per GCP Batch job",
        default=50,
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        help="Maximum number of jobs to start",
        default=None,
    )
    args = parser.parse_args()

    # Check which windows are not done.
    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(
        groups=[GROUP], workers=args.workers, show_progress=True
    )
    p = multiprocessing.Pool(args.workers)
    is_pending_list = list(
        tqdm.tqdm(
            p.imap(is_window_pending, windows),
            desc="Checking pending windows",
            total=len(windows),
        )
    )

    pending_window_names: list[str] = []
    for window, is_pending in zip(windows, is_pending_list):
        if not is_pending:
            continue
        pending_window_names.append(window.name)
    print(f"got {len(pending_window_names)} pending windows")

    if len(pending_window_names) > args.max_jobs * args.batch_size:
        pending_window_names = random.sample(
            pending_window_names, args.max_jobs * args.batch_size
        )

    # Launch jobs for the pending windows.
    batches = []
    for i in range(0, len(pending_window_names), args.batch_size):
        batch = pending_window_names[i : i + args.batch_size]
        batches.append(batch)

    client = batch_v1.BatchServiceClient()
    for batch in tqdm.tqdm(batches, desc="Launching jobs"):
        launch_job(client, args.image, args.project, args.region, args.ds_path, batch)
