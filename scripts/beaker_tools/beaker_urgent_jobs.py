#!/usr/bin/env python3
"""Show urgent jobs across all clusters for given beaker workspace(s).

Usage:
    python scripts/beaker_urgent_jobs.py [workspace_name ...]

Default workspaces: earth-systems
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

DEFAULT_WORKSPACES = ["earth-systems"]


def get_org_id(org_name: str = "ai2") -> str:
    result = subprocess.run(
        ["beaker", "organization", "get", org_name, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"beaker organization get failed: {result.stderr}")
    return json.loads(result.stdout)[0]["id"]

ACTIVE_STATUSES = {
    "STATUS_RUNNING",
    "STATUS_QUEUED",
    "STATUS_INITIALIZING",
    "STATUS_RESUMING",
    "STATUS_STARTING",
}


def beaker_rpc(method: str, payload: dict) -> dict:
    result = subprocess.run(
        ["beaker", "rpc", "call", method, json.dumps(payload), "--format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"beaker rpc {method} failed: {result.stderr}")
    return json.loads(result.stdout)


def resolve_workspace_id(name: str) -> str:
    if name.startswith("01") and len(name) > 20:
        return name
    result = subprocess.run(
        ["beaker", "workspace", "list", "ai2", "--format", "json", "--text", name],
        capture_output=True,
        text=True,
    )
    for ws in json.loads(result.stdout):
        if ws["name"] == name:
            return ws["id"]
    raise ValueError(f"Workspace '{name}' not found")


def resolve_username(user_id: str, cache: dict) -> str:
    if user_id not in cache:
        resp = beaker_rpc("GetUser", {"userId": user_id})
        cache[user_id] = resp.get("user", {}).get("name", user_id)
    return cache[user_id]


def fetch_workloads(workspace_id: str, org_id: str) -> list:
    """Fetch recent workloads (last 30 days covers all active jobs)."""
    since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    resp = beaker_rpc(
        "ListWorkloads",
        {
            "options": {
                "organizationId": org_id,
                "workloadType": "WORKLOAD_TYPE_EXPERIMENT",
                "workspaceId": workspace_id,
                "createdAfter": since,
                "pageSize": 1000,
            }
        },
    )
    return resp.get("workloads", [])


def main():
    workspaces = sys.argv[1:] or DEFAULT_WORKSPACES
    org_id = get_org_id()

    urgent_jobs = []
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    gpu_by_cluster: dict[str, int] = defaultdict(int)
    user_cache: dict[str, str] = {}

    for ws_name in workspaces:
        try:
            ws_id = resolve_workspace_id(ws_name)
        except ValueError as e:
            print(f"WARNING: {e}", file=sys.stderr)
            continue

        print(f"Fetching workloads from {ws_name}...", file=sys.stderr)
        workloads = fetch_workloads(ws_id, org_id)

        for w in workloads:
            exp = w.get("experiment", {})
            for task in exp.get("tasks", []):
                sd = task.get("systemDetails", {})
                priority = sd.get("priority", "")
                task_status = task.get("status", "unknown")

                if priority != "JOB_PRIORITY_URGENT":
                    continue
                if task_status not in ACTIVE_STATUSES:
                    continue

                clusters = []
                for pc in sd.get("placementConstraints", []):
                    if pc.get("type") == "JOB_PLACEMENT_CONSTRAINT_TYPE_CLUSTER":
                        clusters = [
                            v.replace("ai2/", "") for v in pc.get("values", [])
                        ]

                gpu_count = (
                    task.get("containerSpec", {})
                    .get("resourceRequest", {})
                    .get("gpuCount", 0)
                )
                cluster_str = ",".join(sorted(clusters)) if clusters else "any"

                author_id = exp.get("authorId", "")
                author = resolve_username(author_id, user_cache) if author_id else "?"

                urgent_jobs.append(
                    {
                        "name": exp.get("name", "?"),
                        "status": task_status.replace("STATUS_", ""),
                        "workspace": ws_name,
                        "clusters": cluster_str,
                        "gpus": gpu_count,
                        "created": task.get("created", ""),
                        "author": author,
                    }
                )

                stats[cluster_str][task_status.replace("STATUS_", "")] += 1
                gpu_by_cluster[cluster_str] += gpu_count

    # Print results
    print("=" * 90)
    print(f"URGENT JOBS SUMMARY — {len(urgent_jobs)} active urgent tasks")
    print("=" * 90)

    if not urgent_jobs:
        print("\nNo active urgent jobs found.")
        return

    # Summary by cluster group
    print(
        f"\n{'Cluster(s)':<50} {'Running':>8} {'Queued':>8} {'Init':>8} {'GPUs':>6}"
    )
    print("-" * 85)
    totals: dict[str, int] = defaultdict(int)
    total_gpus = 0
    for cluster in sorted(stats.keys()):
        s = stats[cluster]
        print(
            f"{cluster:<50} "
            f"{s.get('RUNNING', 0):>8} "
            f"{s.get('QUEUED', 0):>8} "
            f"{s.get('INITIALIZING', 0):>8} "
            f"{gpu_by_cluster[cluster]:>6}"
        )
        for status, count in s.items():
            totals[status] += count
        total_gpus += gpu_by_cluster[cluster]

    print("-" * 85)
    print(
        f"{'TOTAL':<50} "
        f"{totals.get('RUNNING', 0):>8} "
        f"{totals.get('QUEUED', 0):>8} "
        f"{totals.get('INITIALIZING', 0):>8} "
        f"{total_gpus:>6}"
    )

    # Detail listing
    print(f"\n{'Name':<55} {'Author':<15} {'Status':<15} {'GPUs':>5}  {'Workspace'}")
    print("-" * 110)
    for j in sorted(urgent_jobs, key=lambda x: (x["status"], x["name"])):
        name = j["name"][:54]
        print(f"{name:<55} {j['author']:<15} {j['status']:<15} {j['gpus']:>5}  {j['workspace']}")

    print()


if __name__ == "__main__":
    main()
