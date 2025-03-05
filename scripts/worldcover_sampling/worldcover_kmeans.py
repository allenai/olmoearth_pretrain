"""KNN worldcover sampling."""

import argparse

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def _find_clusters(
    tile_data: pd.DataFrame, target_tile: str, num_clusters_per_tile: int
) -> pd.DataFrame:
    if len(tile_data) < num_clusters_per_tile:
        print(
            f"{target_tile} has fewer than {num_clusters_per_tile} rows - returning all data"
        )
        return tile_data
    data = MinMaxScaler().fit_transform(tile_data.drop("tile_id", axis=1).values)
    kmeans = KMeans(
        n_clusters=num_clusters_per_tile, random_state=0, n_init="auto"
    ).fit(data)
    clusters = kmeans.predict(data)

    centroid_indices = []
    for i in range(num_clusters_per_tile):
        cluster_data = data[clusters == i]
        cluster_indices = np.argwhere(clusters == i)
        distances = norm(cluster_data - kmeans.cluster_centers_[i], axis=-1)
        closest_to_center = np.argmin(distances)
        centroid_indices.append(cluster_indices[closest_to_center].item())
    return tile_data.iloc[centroid_indices]


def _return_clusters_per_tile(
    all_data: pd.DataFrame,
    num_clusters_per_tile: int,
    num_tiles_to_process: int | None = None,
) -> pd.DataFrame:
    output_dfs = []
    count = 0
    for tile_id in tqdm(all_data.tile_id.unique()):
        tile_data = all_data[all_data.tile_id == tile_id]
        output_dfs.append(_find_clusters(tile_data, tile_id, num_clusters_per_tile))
        count += 1
        if (num_tiles_to_process is not None) and (count > num_tiles_to_process):
            return pd.concat(output_dfs)
    return pd.concat(output_dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Beaker jobs for computing WorldCover histograms",
    )
    parser.add_argument(
        "--csv_fname",
        type=str,
        help="File containing concatenated histogram CSVs computed by compute_worldcover_histograms.py",
        required=True,
    )
    parser.add_argument(
        "--subsampled_grid_path",
        type=str,
        help="Filename to write the per-tile clusters",
        required=True,
    )
    parser.add_argument(
        "--subsampled_global_grid_path",
        type=str,
        help="Filename to write the per-tile clusters",
        required=True,
    )
    args = parser.parse_args()

    grid = pd.read_csv(args.csv_fname)
    print(f"got {len(grid)} total tiles")

    print("finding per-tile clusters")
    output = _return_clusters_per_tile(
        grid, num_clusters_per_tile=50, num_tiles_to_process=None
    )
    output.to_csv(args.subsampled_grid_path)
    print(len(output))

    print("finding global clusters")
    output_global = _find_clusters(grid, "global", num_clusters_per_tile=150000)
    output_global.to_csv(args.subsampled_global_grid_path)
    print(len(output_global))
