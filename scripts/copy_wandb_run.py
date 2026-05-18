#!/usr/bin/env python3
r"""Copy metrics from one wandb run to another project, preserving step values.

Uses the sampledHistory GraphQL query directly, which is more reliable than
the SDK's history()/scan_history() methods for large runs.

Example usage::

    python3 scripts/copy_wandb_run.py \
        --src-entity eai-ai2 --src-project 2026_02_08_masked_neg \
        --run-id 0q8si8ko \
        --dest-entity eai-ai2 --dest-project 20260410_sigreg \
        --run-name base_token_masked_all
"""

import argparse
import json
import os
from collections import defaultdict

import requests
from tqdm import tqdm

import wandb

API_TIMEOUT = 120
GRAPHQL_URL = "https://api.wandb.ai/graphql"
SAMPLED_HISTORY_QUERY = """
query RunSampledHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $name) { sampledHistory(specs: $specs) }
  }
}
"""
METRIC_BATCH_SIZE = 5
DEFAULT_SAMPLES = 10_000


def _discover_metrics(run) -> list[str]:
    """Get all logged numeric metric keys from a run's summary."""
    skip = {"graph_0"}
    return sorted(
        k
        for k, v in run.summary.items()
        if not k.startswith("_") and k not in skip and isinstance(v, int | float)
    )


def _fetch_history(
    api_key: str,
    entity: str,
    project: str,
    run_id: str,
    metrics: list[str],
    samples: int = DEFAULT_SAMPLES,
) -> dict[int, dict[str, float]]:
    """Fetch metrics via sampledHistory GraphQL query in batches."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    step_data: dict[int, dict[str, float]] = defaultdict(dict)

    for i in tqdm(
        range(0, len(metrics), METRIC_BATCH_SIZE), desc="Fetching metric batches"
    ):
        batch = metrics[i : i + METRIC_BATCH_SIZE]
        spec = json.dumps({"keys": batch + ["_step"], "samples": samples})
        resp = requests.post(
            GRAPHQL_URL,
            headers=headers,
            json={
                "query": SAMPLED_HISTORY_QUERY,
                "variables": {
                    "project": project,
                    "entity": entity,
                    "name": run_id,
                    "specs": [spec],
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            tqdm.write(f"  Errors for {batch}: {data['errors']}")
            continue

        rows = data["data"]["project"]["run"]["sampledHistory"][0]
        for row in rows:
            step = int(row.get("_step", 0))
            for k, v in row.items():
                if k != "_step" and v is not None:
                    step_data[step][k] = v

        tqdm.write(f"  {batch}: {len(rows)} rows")

    return step_data


def copy_run(
    src_entity: str,
    src_project: str,
    run_id: str,
    dest_entity: str,
    dest_project: str,
    run_name: str | None = None,
    metrics: list[str] | None = None,
    samples: int = DEFAULT_SAMPLES,
    dry_run: bool = False,
):
    """Copy metrics from a source run to a new run in a destination project."""
    api_key = os.environ.get("WANDB_API_KEY") or wandb.Api().api_key
    api = wandb.Api(timeout=API_TIMEOUT)

    src_run_path = f"{src_entity}/{src_project}/{run_id}"
    print(f"Fetching source run: {src_run_path}")
    src_run = api.run(src_run_path)

    original_name = src_run.name
    original_config = {k: v for k, v in src_run.config.items() if not k.startswith("_")}
    original_tags = list(src_run.tags)

    print(f"Source run name: {original_name}")
    print(f"Tags: {original_tags}")

    if metrics is None:
        metrics = _discover_metrics(src_run)
        print(f"\nAuto-discovered {len(metrics)} numeric metrics from run summary")
    else:
        print(f"\n{len(metrics)} metrics specified")

    print(f"Metrics: {metrics}\n")

    step_data = _fetch_history(
        api_key, src_entity, src_project, run_id, metrics, samples
    )

    total_points = sum(len(d) for d in step_data.values())
    print(f"\nTotal: {total_points} data points across {len(step_data)} unique steps")

    if dry_run:
        print("\n[DRY RUN] Would create run with the above data. Exiting.")
        return

    dest_run_name = run_name or f"{original_name} (copied)"
    print(f"\nCreating run '{dest_run_name}' in {dest_entity}/{dest_project}")

    wandb.init(
        entity=dest_entity,
        project=dest_project,
        name=dest_run_name,
        config=original_config,
        tags=original_tags + ["copied-run"],
        notes=f"Copied from {src_run_path}",
    )

    sorted_steps = sorted(step_data.keys())
    print(f"Logging {len(sorted_steps)} steps...")

    for step in tqdm(sorted_steps, desc="Logging"):
        data = step_data[step]
        if data:
            wandb.log(data, step=step)

    wandb.finish()
    print(f"\nDone! Run created in {dest_entity}/{dest_project}")


def main():
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Copy metrics from one wandb run to another project"
    )
    parser.add_argument("--src-entity", required=True, help="Source entity/team")
    parser.add_argument("--src-project", required=True, help="Source project name")
    parser.add_argument("--run-id", required=True, help="Source run ID")
    parser.add_argument("--dest-entity", required=True, help="Destination entity/team")
    parser.add_argument(
        "--dest-project", required=True, help="Destination project name"
    )
    parser.add_argument(
        "--run-name", help="Name for the new run (default: original name + ' (copied)')"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specific metrics to copy (default: auto-discover all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Max samples per metric (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and display data without creating a new run",
    )

    args = parser.parse_args()

    copy_run(
        src_entity=args.src_entity,
        src_project=args.src_project,
        run_id=args.run_id,
        dest_entity=args.dest_entity,
        dest_project=args.dest_project,
        run_name=args.run_name,
        metrics=args.metrics,
        samples=args.samples,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
