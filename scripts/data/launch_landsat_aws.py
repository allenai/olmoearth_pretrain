#!/usr/bin/env python3
"""Launch Landsat rslearn workers on EC2 spot instances, stage output to S3, sync to Weka.

Three subcommands:
    launch  -- spin up EC2 spot instances that run rslearn for landsat-only
    status  -- check how many shards have completed (reads S3 sentinels)
    sync    -- pull completed landsat layer data from S3 into a Weka rslearn dir

Defaults to on-demand instances (no reclaims). Pass --spot for cheaper runs at the risk
of losing progress if AWS reclaims the instance mid-job (EBS volumes are deleted on
termination, so there's no resume -- a reclaimed shard must be re-run from scratch).

The launch step bakes AWS credentials into EC2 user-data as env vars. No IAM instance
profiles or secrets managers needed -- just pass your key at launch time (or export
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY before running).

Usage:
    # Launch 20 spot instances for 640K-sample corpus
    python scripts/data/launch_landsat_aws.py launch \\
        --corpus /weka/.../studio_corpus_lonlats.json \\
        --rslearn-config olmoearth_pretrain/dataset_creation/rslearn_configs/corpus_v2.json \\
        --s3-bucket my-staging-bucket \\
        --num-instances 20

    # Check progress
    python scripts/data/launch_landsat_aws.py status \\
        --run-manifest /tmp/landsat_run_XXXX.json

    # Pull results to Weka (run this FROM a Weka machine)
    python scripts/data/launch_landsat_aws.py sync \\
        --run-manifest /tmp/landsat_run_XXXX.json \\
        --rslearn-dir /weka/.../dataset_creation/studio_corpus

IAM permissions needed by the credentials:
    ec2:RunInstances, ec2:TerminateInstances, ec2:DescribeInstances, ec2:CreateTags
    s3:PutObject, s3:GetObject, s3:ListBucket on the staging bucket
    s3:GetObject, s3:ListBucket on usgs-landsat (requester-pays)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Pinned s5cmd release. The corpus has 642,013 windows / many small landsat files;
# the default `aws s3 sync` (~10 workers) is far too slow. s5cmd defaults to 256
# concurrent workers and is typically 30-50x faster for this workload.
S5CMD_VERSION = "2.2.2"
S5CMD_URL = (
    f"https://github.com/peak/s5cmd/releases/download/v{S5CMD_VERSION}"
    f"/s5cmd_{S5CMD_VERSION}_Linux-64bit.tar.gz"
)

# studio_corpus_lonlats.json size; controls how the index space is sharded.
DEFAULT_TOTAL_WINDOWS = 642013

# Each "prefix" is corpus_NNNXXXX with a fixed 3-digit NNN, covering 10K windows.
WINDOWS_PER_PREFIX = 10000

USERDATA_TEMPLATE = r"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/landsat-worker.log) 2>&1
echo "=== Landsat worker shard {shard_id}/{num_shards} starting $(date) ==="

{aws_creds_block}export AWS_DEFAULT_REGION="{region}"

S3_INPUT="s3://{bucket}/{prefix}/inputs"
S3_OUTPUT="s3://{bucket}/{prefix}/windows"
SHARD_ID={shard_id}
NUM_SHARDS={num_shards}
WORKERS={workers}

# Install system deps
if command -v dnf &>/dev/null; then
    dnf install -y git gcc gcc-c++ python3-devel python3-pip \
        libffi-devel openssl-devel wget tar gzip || true
elif command -v yum &>/dev/null; then
    yum install -y git gcc gcc-c++ python3-devel python3-pip \
        libffi-devel openssl-devel wget || true
elif command -v apt-get &>/dev/null; then
    apt-get update && apt-get install -y git python3-dev python3-pip \
        libffi-dev wget || true
fi

# Install uv
export HOME="${{HOME:-/root}}"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone repo at pinned ref
git clone https://github.com/allenai/olmoearth_pretrain.git /opt/work
cd /opt/work
git checkout "{git_ref}"
uv python install 3.13
uv sync --locked --extra dataset-creation --python 3.13
source .venv/bin/activate

# Upgrade rslearn to git master for OLI-only scene filtering fix (PR #646)
uv pip install --no-deps 'rslearn[extra] @ git+https://github.com/allenai/rslearn.git@master'
echo "rslearn version: $(uv pip show rslearn | grep Version)"

# Fetch inputs from S3
aws s3 cp "$S3_INPUT/corpus.json" /tmp/corpus.json
aws s3 cp "$S3_INPUT/landsat_config.json" /tmp/landsat_config.json

# Local rslearn dataset dir on EBS
RSLEARN_DIR=/tmp/rslearn_landsat
mkdir -p "$RSLEARN_DIR"

# Run the standard corpus pipeline worker
python scripts/data/corpus_pipeline.py rslearn-worker \
    --corpus /tmp/corpus.json \
    --rslearn-dir "$RSLEARN_DIR" \
    --rslearn-config /tmp/landsat_config.json \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS" \
    --workers "$WORKERS" {max_samples_flag}

echo "=== rslearn worker done, uploading to S3 ==="

# Trap SIGTERM/SIGINT during upload so spot reclaims or cloud-init timeouts
# don't kill us mid-transfer. We defer the signal until upload finishes.
CAUGHT_SIGNAL=0
trap 'echo "SIGTERM caught, finishing upload before exit..."; CAUGHT_SIGNAL=1' TERM INT

# Upload all landsat layers in one parallel sync (much faster than per-layer loop)
cd "$RSLEARN_DIR"
aws s3 sync windows/ "$S3_OUTPUT/" \
    --exclude "*" --include "*/layers/landsat*/*" \
    --only-show-errors

# Signal completion
echo '{{"shard_id": '$SHARD_ID', "status": "done", "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}}' \
    | aws s3 cp - "$S3_OUTPUT/_COMPLETE_shard_$(printf '%05d' $SHARD_ID)"

echo "=== Upload complete ==="
if [ "$CAUGHT_SIGNAL" -eq 1 ]; then
    echo "Upload finished despite signal, exiting gracefully"
fi
{self_terminate_block}
"""


def _get_git_ref() -> str:
    """Get current git HEAD sha."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _get_aws_creds(require: bool = True) -> tuple[str, str, str]:
    """Read AWS credentials from environment (key, secret, optional session token).

    If require=False (instance profile mode), returns empty strings without error.
    """
    key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    token = os.environ.get("AWS_SESSION_TOKEN", "")
    if require and (not key_id or not secret):
        raise SystemExit(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set.\n"
            "Export them before running launch, or use --iam-instance-profile."
        )
    return key_id, secret, token


def _make_landsat_only_config(rslearn_config_path: str) -> dict:
    """Read a full rslearn config and return one with only the landsat layer."""
    with open(rslearn_config_path) as f:
        full_config = json.load(f)

    landsat_layer = None
    for name, layer in full_config.get("layers", {}).items():
        ds = layer.get("data_source", {})
        if "landsat" in ds.get("class_path", "").lower() or "landsat" in name.lower():
            landsat_layer = (name, layer)
            break

    if landsat_layer is None:
        raise SystemExit(f"No landsat layer found in {rslearn_config_path}")

    name, layer = landsat_layer
    if layer.get("data_source", {}).get("ingest") is not False:
        logger.warning("Setting ingest=false on landsat layer for EC2 efficiency")
        layer.setdefault("data_source", {})["ingest"] = False

    return {"layers": {name: layer}}


def cmd_launch(args: argparse.Namespace) -> None:
    """Launch EC2 spot instances for Landsat processing."""
    import boto3

    use_instance_profile = bool(args.iam_instance_profile)
    aws_key_id, aws_secret, aws_session_token = _get_aws_creds(
        require=not use_instance_profile
    )
    git_ref = _get_git_ref()

    if use_instance_profile:
        logger.info(
            f"Using IAM instance profile: {args.iam_instance_profile} (no baked-in credentials)"
        )
    else:
        logger.warning(
            "Baking AWS credentials into user-data (prefer --iam-instance-profile for production)"
        )

    logger.info(f"Git ref: {git_ref}")
    pricing = "spot" if args.spot else "on-demand"
    logger.info(f"Instances: {args.num_instances} x {args.instance_type} ({pricing})")
    logger.info(f"S3 staging: s3://{args.s3_bucket}/{args.s3_prefix}")

    landsat_config = _make_landsat_only_config(args.rslearn_config)
    logger.info(
        f"Landsat-only config: layer={list(landsat_config['layers'].keys())[0]}"
    )

    if not args.dry_run:
        s3 = boto3.client("s3", region_name=args.region)

        input_prefix = f"{args.s3_prefix}/inputs"
        with open(args.corpus, "rb") as f:
            s3.upload_fileobj(f, args.s3_bucket, f"{input_prefix}/corpus.json")
        logger.info(
            f"Uploaded corpus to s3://{args.s3_bucket}/{input_prefix}/corpus.json"
        )

        config_bytes = json.dumps(landsat_config, indent=2).encode()
        s3.put_object(
            Bucket=args.s3_bucket,
            Key=f"{input_prefix}/landsat_config.json",
            Body=config_bytes,
        )
        logger.info("Uploaded landsat-only config")
    else:
        logger.info("[DRY RUN] Would upload corpus + config to S3")

    # Launch instances
    ec2 = boto3.client("ec2", region_name=args.region)
    instance_ids = []
    run_id = f"landsat-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    if args.shard_range:
        start, end = args.shard_range.split("-")
        shard_ids = list(range(int(start), int(end) + 1))
    else:
        shard_ids = list(range(args.num_instances))

    logger.info(
        f"Launching shards: {shard_ids[0]}-{shard_ids[-1]} ({len(shard_ids)} instances, {args.num_instances} total shards)"
    )

    for shard_id in shard_ids:
        if use_instance_profile:
            aws_creds_block = "# Credentials provided by IAM instance profile\n"
        else:
            lines = [
                f'export AWS_ACCESS_KEY_ID="{aws_key_id}"',
                f'export AWS_SECRET_ACCESS_KEY="{aws_secret}"',
            ]
            if aws_session_token:
                lines.append(f'export AWS_SESSION_TOKEN="{aws_session_token}"')
            aws_creds_block = "\n".join(lines) + "\n"

        max_samples_flag = (
            f"--max-samples {args.max_samples}" if args.max_samples else ""
        )

        if args.no_self_terminate:
            self_terminate_block = (
                "echo 'Instance kept alive for debugging (--no-self-terminate)'"
            )
        else:
            self_terminate_block = (
                "# Self-terminate (IMDSv2)\n"
                'TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\\n'
                '    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")\n'
                'INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\\n'
                "    http://169.254.169.254/latest/meta-data/instance-id)\n"
                f'aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "{args.region}"'
            )

        userdata = USERDATA_TEMPLATE.format(
            shard_id=shard_id,
            num_shards=args.num_instances,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            region=args.region,
            git_ref=git_ref,
            workers=args.workers,
            aws_creds_block=aws_creds_block,
            max_samples_flag=max_samples_flag,
            self_terminate_block=self_terminate_block,
        )

        launch_kwargs: dict = {
            "ImageId": args.ami,
            "InstanceType": args.instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": userdata,
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": args.volume_size,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"{run_id}-shard-{shard_id:05d}"},
                        {"Key": "landsat-run-id", "Value": run_id},
                        {"Key": "shard-id", "Value": str(shard_id)},
                    ],
                }
            ],
        }

        if args.spot:
            launch_kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            }

        if args.key_name:
            launch_kwargs["KeyName"] = args.key_name
        if args.security_group_id:
            launch_kwargs["SecurityGroupIds"] = [args.security_group_id]
        if args.subnet_id:
            launch_kwargs["SubnetId"] = args.subnet_id
        if args.iam_instance_profile:
            launch_kwargs["IamInstanceProfile"] = {"Name": args.iam_instance_profile}

        if args.dry_run:
            logger.info(f"[DRY RUN] Would launch shard {shard_id}/{args.num_instances}")
            instance_ids.append(f"dry-run-{shard_id}")
            continue

        resp = ec2.run_instances(**launch_kwargs)
        iid = resp["Instances"][0]["InstanceId"]
        instance_ids.append(iid)
        logger.info(f"Launched shard {shard_id}: {iid}")

    # Save run manifest
    manifest = {
        "run_id": run_id,
        "git_ref": git_ref,
        "s3_bucket": args.s3_bucket,
        "s3_prefix": args.s3_prefix,
        "region": args.region,
        "num_instances": args.num_instances,
        "instance_type": args.instance_type,
        "instance_ids": instance_ids,
        "corpus": args.corpus,
        "rslearn_config": args.rslearn_config,
        "workers": args.workers,
        "max_samples": args.max_samples,
        "no_self_terminate": args.no_self_terminate,
        "spot": args.spot,
        "launched_at": datetime.now(UTC).isoformat(),
    }

    manifest_path = Path(f"/tmp/{run_id}.json")
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Run manifest saved to {manifest_path}")
    logger.info(f"Run ID: {run_id}")
    logger.info(
        f"Check status:  python {__file__} status --run-manifest {manifest_path}"
    )
    logger.info(
        f"Sync to Weka:  python {__file__} sync --run-manifest {manifest_path} "
        f"--rslearn-dir /weka/.../your_rslearn_dir"
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Check completion status of a Landsat EC2 run."""
    manifest = _load_manifest(args)

    bucket = manifest["s3_bucket"]
    prefix = manifest["s3_prefix"]
    num = manifest["num_instances"]

    # Check S3 completion sentinels
    result = subprocess.run(
        [
            "aws",
            "s3",
            "ls",
            f"s3://{bucket}/{prefix}/windows/_COMPLETE_shard_",
            "--region",
            manifest.get("region", "us-west-2"),
        ],
        capture_output=True,
        text=True,
    )

    completed = set()
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("_COMPLETE_shard_")
        if len(parts) == 2:
            try:
                completed.add(int(parts[1]))
            except ValueError:
                pass

    print(f"Run: {manifest['run_id']}")
    print(f"Completed: {len(completed)}/{num} shards")

    missing = sorted(set(range(num)) - completed)
    if missing:
        print(f"Missing shards: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    # Check EC2 instance states if we have instance IDs
    instance_ids = manifest.get("instance_ids", [])
    real_ids = [i for i in instance_ids if not i.startswith("dry-run")]
    if real_ids:
        try:
            import boto3

            ec2 = boto3.client("ec2", region_name=manifest.get("region", "us-west-2"))
            resp = ec2.describe_instances(InstanceIds=real_ids[:50])
            states: dict[str, int] = {}
            for res in resp["Reservations"]:
                for inst in res["Instances"]:
                    state = inst["State"]["Name"]
                    states[state] = states.get(state, 0) + 1
            print(f"Instance states: {states}")
        except Exception as e:
            print(f"(Could not check instance states: {e})")

    if len(completed) == num:
        print("\nAll shards complete! Run sync to pull data to Weka.")


def cmd_sync(args: argparse.Namespace) -> None:
    """Sync completed Landsat data from S3 to a Weka rslearn directory."""
    manifest = _load_manifest(args)

    bucket = manifest["s3_bucket"]
    prefix = manifest["s3_prefix"]
    rslearn_dir = args.rslearn_dir
    region = manifest.get("region", "us-west-2")

    s3_windows = f"s3://{bucket}/{prefix}/windows/"
    local_windows = f"{rslearn_dir}/windows/"

    logger.info(f"Syncing {s3_windows} -> {local_windows}")

    # Prefer s5cmd for speed if available
    s5cmd = shutil.which("s5cmd")
    if s5cmd:
        logger.info("Using s5cmd for parallel transfer")
        cmd = [
            s5cmd,
            "--log",
            "error",
            "sync",
            f"{s3_windows}*",
            local_windows,
        ]
    else:
        logger.info("Using aws s3 sync (install s5cmd for faster transfers)")
        cmd = [
            "aws",
            "s3",
            "sync",
            s3_windows,
            local_windows,
            "--only-show-errors",
            "--region",
            region,
            "--exclude",
            "_COMPLETE_*",
        ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Sync failed with exit code {result.returncode}")

    # Verify
    logger.info("Sync complete. Verifying...")
    count_cmd = f'find {rslearn_dir}/windows/res_10.0/ -path "*/layers/landsat*/completed" 2>/dev/null | wc -l'
    result = subprocess.run(count_cmd, shell=True, capture_output=True, text=True)
    count = int(result.stdout.strip() or "0")
    logger.info(f"Found {count} completed landsat layer groups across all windows")

    if args.cleanup:
        logger.info(f"Cleaning up S3 staging: s3://{bucket}/{prefix}/")
        subprocess.run(
            [
                "aws",
                "s3",
                "rm",
                f"s3://{bucket}/{prefix}/",
                "--recursive",
                "--only-show-errors",
                "--region",
                region,
            ]
        )
        logger.info("S3 cleanup done")


def _chunk_prefixes(prefixes: list[str], num_jobs: int) -> list[list[str]]:
    """Split prefixes into roughly equal contiguous chunks for num_jobs jobs."""
    n = len(prefixes)
    num_jobs = min(num_jobs, n)
    base, extra = divmod(n, num_jobs)
    chunks = []
    start = 0
    for i in range(num_jobs):
        size = base + (1 if i < extra else 0)
        chunks.append(prefixes[start : start + size])
        start += size
    return chunks


def _upsert_beaker_secret(
    workspace: str, name: str, value: str, dry_run: bool = False
) -> None:
    """Write or overwrite a Beaker secret via the CLI (reads value from stdin)."""
    if dry_run:
        logger.info(f"[DRY RUN] would write beaker secret {workspace}/{name}")
        return
    logger.info(f"Writing beaker secret {workspace}/{name}")
    subprocess.run(
        ["beaker", "secret", "write", "--workspace", workspace, name],
        input=value,
        text=True,
        check=True,
    )


def cmd_launch_sync(args: argparse.Namespace) -> None:
    """Launch N parallel Beaker jobs that sync S3 -> Weka using s5cmd.

    Splits the corpus index space into 3-digit prefix buckets
    (`corpus_NNN*`, NNN ∈ 000..064) and distributes them across jobs.

    AWS credentials are read from the local shell and uploaded as user-scoped
    Beaker secrets (`{user}_LANDSAT_SYNC_AWS_*`) — Beaker rejects credential-
    shaped names like `AWS_ACCESS_KEY_ID` when set as plaintext env vars.
    Re-running this command overwrites the secrets, so refreshing creds is a
    no-op for callers.
    """
    from olmo_core.internal.common import get_beaker_username
    from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar

    from olmoearth_pretrain.internal.common import WORKSPACE, build_launch_config

    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_token = os.environ.get("AWS_SESSION_TOKEN")
    if not aws_key or not aws_secret:
        raise SystemExit(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in the local shell"
        )
    if not aws_token:
        logger.warning(
            "AWS_SESSION_TOKEN not set — assuming long-lived IAM credentials"
        )

    beaker_user = get_beaker_username()
    secret_key = f"{beaker_user}_LANDSAT_SYNC_AWS_ACCESS_KEY_ID"
    secret_secret = f"{beaker_user}_LANDSAT_SYNC_AWS_SECRET_ACCESS_KEY"
    secret_token = f"{beaker_user}_LANDSAT_SYNC_AWS_SESSION_TOKEN"

    _upsert_beaker_secret(WORKSPACE, secret_key, aws_key, dry_run=args.dry_run)
    _upsert_beaker_secret(WORKSPACE, secret_secret, aws_secret, dry_run=args.dry_run)
    if aws_token:
        _upsert_beaker_secret(
            WORKSPACE, secret_token, aws_token, dry_run=args.dry_run
        )

    max_prefix_idx = (args.total_windows - 1) // WINDOWS_PER_PREFIX
    all_prefixes = [f"{i:03d}" for i in range(max_prefix_idx + 1)]
    job_chunks = _chunk_prefixes(all_prefixes, args.num_jobs)

    clusters = args.clusters if isinstance(args.clusters, list) else [args.clusters]
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    logger.info(
        f"Launching {len(job_chunks)} sync jobs over {len(all_prefixes)} prefixes "
        f"({args.total_windows} windows) on clusters {clusters}"
    )

    experiment_ids = []
    for job_idx, prefixes in enumerate(job_chunks):
        worker_cmd = [
            "scripts/data/launch_landsat_aws.py",
            "sync-worker",
            "--s3-bucket",
            args.s3_bucket,
            "--s3-prefix",
            args.s3_prefix,
            "--rslearn-dir",
            args.rslearn_dir,
            "--region",
            args.region,
            "--workers",
            str(args.workers),
            "--corpus-prefixes",
            *prefixes,
        ]

        config = build_launch_config(
            name=f"landsat-sync-{run_id}-{job_idx:02d}",
            cmd=worker_cmd,
            clusters=clusters,
            task_name="landsat-sync",
        )
        config.num_gpus = 0
        config.num_nodes = 1
        config.preemptible = False
        config.retries = 1
        # Replace the heavy --all-extras install with the dataset-creation extra.
        config.setup_steps = [
            s.replace(
                "uv sync --locked --all-extras",
                "uv sync --locked --extra dataset-creation",
            )
            for s in config.setup_steps
        ]
        # Install s5cmd at the end of setup; idempotent if already present.
        config.setup_steps.append(
            f"curl -sL {S5CMD_URL} | tar xz -C /usr/local/bin s5cmd "
            "&& /usr/local/bin/s5cmd version"
        )

        config.env_secrets.append(
            BeakerEnvSecret(name="AWS_ACCESS_KEY_ID", secret=secret_key)
        )
        config.env_secrets.append(
            BeakerEnvSecret(name="AWS_SECRET_ACCESS_KEY", secret=secret_secret)
        )
        if aws_token:
            config.env_secrets.append(
                BeakerEnvSecret(name="AWS_SESSION_TOKEN", secret=secret_token)
            )
        # AWS_DEFAULT_REGION isn't credential-shaped; plain env var is allowed.
        config.env_vars.append(
            BeakerEnvVar(name="AWS_DEFAULT_REGION", value=args.region)
        )

        if args.dry_run:
            logger.info(
                f"[DRY RUN] job {job_idx}: prefixes={prefixes} cmd={' '.join(worker_cmd)}"
            )
            continue

        logger.info(f"Launching job {job_idx}/{len(job_chunks)}: prefixes={prefixes}")
        eid = config.launch(torchrun=False, entrypoint="python")
        experiment_ids.append(eid)
        print(f"  https://beaker.org/ex/{eid}")

    logger.info(f"Launched {len(experiment_ids)} sync jobs")


def cmd_sync_worker(args: argparse.Namespace) -> None:
    """Sync a set of corpus_NNN* prefixes from S3 to Weka using s5cmd.

    Runs inside each Beaker job. Installs s5cmd if missing, then runs
    `s5cmd sync` once per prefix. s5cmd's `sync` is idempotent — re-running
    skips files already present with matching size.
    """
    s5cmd_path = shutil.which("s5cmd")
    if not s5cmd_path:
        logger.info("Installing s5cmd...")
        subprocess.check_call(
            ["bash", "-c", f"curl -sL {S5CMD_URL} | tar xz -C /usr/local/bin s5cmd"]
        )
        s5cmd_path = "/usr/local/bin/s5cmd"

    subprocess.check_call([s5cmd_path, "version"])

    dest = f"{args.rslearn_dir.rstrip('/')}/windows/res_10.0/"
    Path(dest).mkdir(parents=True, exist_ok=True)

    bucket = args.s3_bucket
    prefix = args.s3_prefix.strip("/")

    for cp in args.corpus_prefixes:
        s3_pattern = f"s3://{bucket}/{prefix}/windows/res_10.0/corpus_{cp}*"
        logger.info(f"[{cp}] syncing {s3_pattern} -> {dest}")
        t0 = time.time()
        cmd = [
            s5cmd_path,
            "--numworkers",
            str(args.workers),
            # --log info so retry/throttle messages surface; --log error hides
            # silent stalls when S3 throttles us at high aggregate concurrency.
            "--log",
            "info",
            "--stat",
            "sync",
            s3_pattern,
            dest,
        ]
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        if result.returncode != 0:
            raise SystemExit(
                f"s5cmd sync failed for corpus_{cp}* (exit={result.returncode}, "
                f"after {elapsed:.0f}s)"
            )
        logger.info(f"[{cp}] done in {elapsed:.0f}s")

    logger.info(f"All {len(args.corpus_prefixes)} prefixes synced.")


def _load_manifest(args: argparse.Namespace) -> dict:
    """Load run manifest from file or construct from args."""
    if args.run_manifest:
        with open(args.run_manifest) as f:
            return json.load(f)
    if args.s3_bucket and args.s3_prefix:
        return {
            "s3_bucket": args.s3_bucket,
            "s3_prefix": args.s3_prefix,
            "num_instances": getattr(args, "num_instances", 0),
            "region": getattr(args, "region", "us-west-2"),
            "run_id": "manual",
            "instance_ids": [],
        }
    raise SystemExit("Provide --run-manifest or --s3-bucket + --s3-prefix")


# -- Default AMI lookup --

SSM_AMI_PATHS = [
    # Deep Learning Base OSS (has Python, pip, system libs for GDAL/rasterio)
    "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64",
]


def _resolve_ami(region: str, ami: str | None) -> str:
    """Resolve AMI: use provided or query SSM for latest Amazon Linux 2023.

    Strongly recommended: pass --ami with a Deep Learning AMI that has
    Python 3.11+, GDAL, GEOS, and PROJ pre-installed. This avoids slow
    compilation of rasterio/fiona on a bare OS.

    Find one with:
        aws ec2 describe-images --region us-west-2 --owners amazon \\
            --filters "Name=name,Values=*Deep Learning Base OSS*" \\
            --query 'Images | sort_by(@, &CreationDate) | [-1].{ID:ImageId,Name:Name}'
    """
    if ami:
        return ami
    for ssm_path in SSM_AMI_PATHS:
        try:
            import boto3

            ssm = boto3.client("ssm", region_name=region)
            resp = ssm.get_parameter(Name=ssm_path)
            return resp["Parameter"]["Value"]
        except Exception:
            continue
    raise SystemExit(
        f"Could not auto-resolve AMI for {region}. Pass --ami explicitly.\n"
        f"Recommended: use a Deep Learning Base AMI with Python + GDAL pre-installed.\n"
        f"  aws ec2 describe-images --region {region} --owners amazon \\\n"
        f"    --filters 'Name=name,Values=*Deep Learning Base OSS*' \\\n"
        f"    --query 'Images | sort_by(@, &CreationDate) | [-1].{{ID:ImageId,Name:Name}}'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Landsat rslearn workers on EC2, stage to S3, sync to Weka.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- launch --
    p = sub.add_parser("launch", help="Launch EC2 spot instances for Landsat")
    p.add_argument("--corpus", required=True, help="Corpus JSON path (on Weka)")
    p.add_argument(
        "--rslearn-config",
        required=True,
        help="Full rslearn config (e.g. corpus_v2.json); landsat layer extracted automatically",
    )
    p.add_argument("--s3-bucket", required=True, help="S3 bucket for staging")
    p.add_argument(
        "--s3-prefix",
        default=None,
        help="S3 key prefix (default: auto-generated from timestamp)",
    )
    p.add_argument(
        "--num-instances",
        type=int,
        default=20,
        help="Total number of shards (determines how corpus is split)",
    )
    p.add_argument(
        "--shard-range",
        default=None,
        help="Launch only a subset of shards, e.g. '0-19' for first 20. Default: all shards.",
    )
    p.add_argument("--instance-type", default="c5.9xlarge")
    p.add_argument("--region", default="us-west-2")
    p.add_argument(
        "--ami", default=None, help="EC2 AMI ID (default: latest Amazon Linux 2023)"
    )
    p.add_argument("--key-name", default=None, help="EC2 key pair name for SSH access")
    p.add_argument("--security-group-id", default=None)
    p.add_argument("--subnet-id", default=None)
    p.add_argument("--iam-instance-profile", default=None)
    p.add_argument(
        "--workers", type=int, default=32, help="rslearn parallelism per instance"
    )
    p.add_argument(
        "--volume-size", type=int, default=200, help="Root EBS volume size in GB"
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit corpus to first N samples (for testing)",
    )
    p.add_argument(
        "--no-self-terminate",
        action="store_true",
        help="Keep instances alive after completion (for SSH debugging)",
    )
    p.add_argument(
        "--spot",
        action="store_true",
        help="Use spot instances (~3-5x cheaper but can be reclaimed mid-job, losing all progress)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print what would be launched"
    )
    p.set_defaults(func=cmd_launch)

    # -- status --
    p = sub.add_parser("status", help="Check completion status")
    p.add_argument("--run-manifest", default=None, help="Path to run manifest JSON")
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--s3-prefix", default=None)
    p.add_argument("--num-instances", type=int, default=0)
    p.add_argument("--region", default="us-west-2")
    p.set_defaults(func=cmd_status)

    # -- sync --
    p = sub.add_parser("sync", help="Sync Landsat data from S3 to Weka")
    p.add_argument("--run-manifest", default=None, help="Path to run manifest JSON")
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--s3-prefix", default=None)
    p.add_argument("--rslearn-dir", required=True, help="Weka rslearn dataset dir")
    p.add_argument("--region", default="us-west-2")
    p.add_argument(
        "--cleanup", action="store_true", help="Delete S3 staging data after sync"
    )
    p.set_defaults(func=cmd_sync)

    # -- launch-sync --
    p = sub.add_parser(
        "launch-sync",
        help="Launch N parallel Beaker jobs to sync S3 -> Weka with s5cmd",
    )
    p.add_argument("--s3-bucket", required=True)
    p.add_argument("--s3-prefix", required=True, help="e.g. landsat_runs/full_run")
    p.add_argument("--rslearn-dir", required=True, help="Weka rslearn dataset dir")
    p.add_argument("--region", default="us-west-2")
    p.add_argument(
        "--num-jobs",
        type=int,
        default=16,
        help="Number of parallel Beaker jobs (default 16, max 65 for default corpus)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=64,
        help="s5cmd --numworkers per job (default 64). At 16 jobs that's 1024 "
        "concurrent S3 connections total — higher values risk S3 throttling.",
    )
    p.add_argument(
        "--total-windows",
        type=int,
        default=DEFAULT_TOTAL_WINDOWS,
        help=f"Total corpus size (default {DEFAULT_TOTAL_WINDOWS})",
    )
    p.add_argument(
        "--clusters",
        nargs="+",
        default=["ai2/jupiter", "ai2/saturn"],
        help="Beaker clusters to target",
    )
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_launch_sync)

    # -- sync-worker --
    p = sub.add_parser(
        "sync-worker",
        help="(internal) sync one chunk of corpus_NNN* prefixes from S3 to Weka",
    )
    p.add_argument("--s3-bucket", required=True)
    p.add_argument("--s3-prefix", required=True)
    p.add_argument("--rslearn-dir", required=True)
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument(
        "--corpus-prefixes",
        nargs="+",
        required=True,
        help="3-digit corpus prefixes to sync, e.g. 000 001 002",
    )
    p.set_defaults(func=cmd_sync_worker)

    args = parser.parse_args()

    # Resolve AMI for launch
    if args.command == "launch":
        args.ami = _resolve_ami(args.region, args.ami)
        if args.s3_prefix is None:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            args.s3_prefix = f"landsat_runs/{ts}"
        logger.info(f"Using AMI: {args.ami}")

    args.func(args)


if __name__ == "__main__":
    main()
