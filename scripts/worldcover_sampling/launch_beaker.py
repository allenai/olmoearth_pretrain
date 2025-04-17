"""Launch compute_worldcover_histograms.py jobs on Beaker."""

import argparse
import os
import random
import uuid

import geopandas as gpd
from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    ExperimentSpec,
    Priority,
    TaskResources,
    TaskSpec,
)
from beaker.services.experiment import ExperimentClient
from tqdm import tqdm


def launch_worldcover_job(
    tile_names: list[str],
    out_dir: str,
    beaker_image: str,
    clusters: list[str],
    workspace: str = "ai2/earth-systems",
    budget: str = "ai2/d5",
    tile_size: int | None = None,
) -> None:
    """Launch worldcover job.

    Args:
        tile_names: the WorldCover tile names to process in this job.
        out_dir: the output directory to write the histogram CSVs.
        beaker_image: the Beaker image name.
        clusters: list of Beaker clusters to target.
        workspace: the Beaker workspace.
        budget: the Beaker budget.
        tile_size: the tile size.
    """
    unique_name = str(uuid.uuid4())[:8]
    description = f"compute_worldcover_histograms_{unique_name}"
    beaker = Beaker.from_env(default_workspace=workspace)

    weka_mount = DataMount(
        source=DataSource(weka="dfive-default"),
        mount_path="/weka/dfive-default",
    )

    command = [
        "python",
        "-u",
        "compute_worldcover_histograms.py",
        "--tile_names",
        ",".join(tile_names),
        "--out_dir",
        out_dir,
    ]
    if tile_size is not None:
        command.extend(
            [
                "--tile_size",
                str(tile_size),
            ]
        )

    spec = ExperimentSpec(
        budget=budget,
        description=description,
        tasks=[
            TaskSpec.new(
                name=description,
                beaker_image=beaker_image,
                command=command,
                constraints=Constraints(cluster=clusters),
                resources=TaskResources(gpu_count=1),
                priority=Priority.high,
                preemptible=True,
                datasets=[weka_mount],
            )
        ],
    )
    experiment = beaker.experiment.create(description, spec)
    experiment_client = ExperimentClient(beaker)
    print(f"Experiment created: {experiment.id}: {experiment_client.url(experiment)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Beaker jobs for computing WorldCover histograms",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="The directory to write histogram CSVs for each tile",
        required=True,
    )
    parser.add_argument(
        "--beaker_image",
        type=str,
        help="The Beaker image name",
        required=True,
    )
    parser.add_argument(
        "--clusters",
        type=str,
        help="Comma-separated list of Beaker clusters to target",
        default="ai2/jupiter-cirrascale-2,ai2/saturn-cirrascale,ai2/ceres-cirrascale",
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        help="Launch up to this many Beaker jobs",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Process this many tiles in each job",
        default=10,
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="the tile size",
        default=None,
    )
    args = parser.parse_args()

    # Read the grid that lists all the different ESA WorldCover files.
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    url = f"{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson"
    grid = gpd.read_file(url)

    # Check which tiles haven't been processed yet.
    needed_tile_names = []
    for tile_i in tqdm(range(len(grid)), desc="Identifying tiles that are needed"):
        tile_name = grid.iloc[tile_i]["ll_tile"]
        out_fname = os.path.join(args.out_dir, f"{tile_name}.csv")
        if os.path.exists(out_fname):
            print(f"Skipping {tile_name} since it is computed already")
            continue
        needed_tile_names.append(tile_name)

    batches = []
    for i in range(0, len(needed_tile_names), args.batch_size):
        batch = needed_tile_names[i : i + args.batch_size]
        batches.append(batch)

    # Launch the Beaker jobs.
    if args.max_jobs is not None and len(batches) > args.max_jobs:
        print(f"Sampling {args.max_jobs} jobs from {len(batches)} batches")
        batches = random.sample(batches, args.max_jobs)
    for batch in tqdm(batches, desc="Launching Beaker jobs"):
        launch_worldcover_job(
            tile_names=batch,
            out_dir=args.out_dir,
            beaker_image=args.beaker_image,
            clusters=args.clusters.split(","),
            tile_size=args.tile_size,
        )
