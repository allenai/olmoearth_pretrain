"""Launch one gantry job per v1.3 seagrass run (round 1 settings; see ../../2026-10-26-piperw-bathymetry-target/seagrass/launch.py).

Run from /weka/dfive-default/piperw/rslearn_projects (gantry uses that repo's pushed HEAD).
  python launch.py              # all runs in configs/runs.txt
  python launch.py RUN [RUN..]  # selected runs
"""

import subprocess  # nosec
import sys
from pathlib import Path

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/downstream/seagrass"
)
JOB = HERE / "seagrass_job.sh"


def launch(run: str, job: Path = JOB, args: list[str] | None = None) -> None:
    """Launch one seagrass run as a Beaker job."""
    cmd = [
        "gantry",
        "run",
        "--name",
        f"pw_seagrass_{run}",
        "--budget",
        "ai2/atec-olmoearth",
        "--workspace",
        "ai2/earth-systems",
        "--priority",
        "urgent",
        "--min-runtime",
        "8h",
        "--cluster",
        "ai2/saturn",
        "--cluster",
        "ai2/jupiter",
        "--cluster",
        "ai2/neptune",
        "--weka",
        "dfive-default:/weka/dfive-default",
        "--gpus",
        "1",
        "--shared-memory",
        "256GiB",
        "--beaker-image",
        "ai2/cuda12.8-ubuntu22.04-torch2.6.0",
        "--gh-token-secret",
        "PIPERW_GITHUB_TOKEN",
        "--env-secret",
        "WANDB_API_KEY=PIPERW_WANDB_TOKEN",
        "--no-python",
        "--allow-dirty",
        "--yes",
        "--",
        "bash",
        str(job),
        *(args or [run]),
    ]
    print("[launch]", run, flush=True)
    subprocess.run(cmd, check=True)  # nosec


if __name__ == "__main__":
    for run in sys.argv[1:] or (HERE / "configs" / "runs.txt").read_text().split():
        launch(run)
