"""Compute sample entropy."""

import csv
import random
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality

# lets start with these two modalities for now
MODALITIES = ["sentinel2_l2a", "worldcover"]
FIELDS = ["filename"] + MODALITIES


def process_worldcover(wc: np.ndarray) -> np.ndarray:
    """Process WC data."""
    wc[wc == 95] = 110
    wc = wc / 10  # now we should be to classes
    # keep missing values
    wc[wc == MISSING_VALUE / 10] = MISSING_VALUE
    return wc


def read_h5_file(h5_file_path: UPath) -> dict[str, Any]:
    """Read the h5 file."""
    sample_dict = {}
    with h5_file_path.open("rb") as f:
        with h5py.File(f, "r") as h5file:
            # timestamps should not be a floating string
            sample_dict = {k: v[()] for k, v in h5file.items() if k in MODALITIES}

            if Modality.WORLDCOVER.name in sample_dict:
                sample_dict[Modality.WORLDCOVER.name] = process_worldcover(
                    sample_dict[Modality.WORLDCOVER.name]
                )

    return sample_dict


def compute_histogram_entropy(modality_name: str, modality: np.ndarray) -> float:
    """Compute histogram entropy."""
    num_bins_and_range: dict[str, tuple[int, tuple[int, int]]] = {
        Modality.SENTINEL2_L2A.name: (100, (0, 10000)),
        Modality.WORLDCOVER.name: (12, (0, 12)),
    }
    num_bins, histogram_range = num_bins_and_range[modality_name]
    entropies = []
    for band in range(modality.shape[-1]):
        band_array = modality[..., band]
        band_array = band_array[band_array != MISSING_VALUE]
        entropies.append(
            entropy(np.histogram(band_array, range=histogram_range, bins=num_bins)[0])
        )
    return np.mean(entropies)


if __name__ == "__main__":
    NUM_FILES_TO_PROCESS = 2000000
    path_to_h5s = UPath(
        "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
    )
    save_filepath = UPath(f"sample_diversity_output_{NUM_FILES_TO_PROCESS}.csv")
    h5s_to_process = list(path_to_h5s.glob("*.h5"))
    print(f"Processing {len(h5s_to_process)} files")

    if save_filepath.exists():
        done_files = pd.read_csv(save_filepath).filename.values
    else:
        done_files = []
        with open(save_filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
            writer.writeheader()
    random.shuffle(h5s_to_process)
    for h5_file in tqdm(h5s_to_process[:NUM_FILES_TO_PROCESS]):
        if h5_file.name in done_files:
            continue
        elif h5_file.name == "normalizing_dict.h5":
            continue
        try:
            sample_dict = read_h5_file(h5_file)
            output = {"filename": h5_file.name}
            for key, val in sample_dict.items():
                output[key] = compute_histogram_entropy(key, val)

            with open(save_filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writerow(output)
        except Exception as e:
            print(e)
