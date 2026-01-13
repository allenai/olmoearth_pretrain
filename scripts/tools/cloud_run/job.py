"""Barebones script for generating embeddings."""

import argparse
import os
import time
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from google.cloud import storage
from torch.utils.data import DataLoader, Dataset

from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, Modality

GCLOUD_PROJECT = os.environ["GCLOUD_PROJECT"]
IN_BUCKET = os.environ["IN_BUCKET"]
OUT_BUCKET = os.environ["OUT_BUCKET"]
BATCH_SIZE = 128 * 128

# For querying bands from Google Earth Engine exported geotiffs
BANDS = {
    # Matches Modality.SENTINEL1.band_order: ['vv', 'vh']
    "sentinel1": ["VV", "VH"],
    # Matches Modality.SENTINEL2_L2A.band_order
    # ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09']
    "sentinel2": [
        "B2",
        "B3",
        "B4",
        "B8",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B11",
        "B12",
        "B1",
        "B9",
    ],
    # Matches Modality.LANDSAT.band_order
    # ['B8', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B9', 'B10', 'B11']
    "landsat": ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
}

client = storage.Client(project=GCLOUD_PROJECT)
in_bucket = client.bucket(IN_BUCKET)
out_bucket = client.bucket(OUT_BUCKET)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
model.eval()
model = model.encoder.to(device)
EMBEDDINGS_SIZE = model.project_and_aggregate.projection[0].out_features
print("Embedding size: ", EMBEDDINGS_SIZE)

computed = Normalizer(Strategy.COMPUTED)
predefined = Normalizer(Strategy.PREDEFINED)


class OlmoEarthGEEDataset(Dataset):
    """Dataset for iterating through Google Earth Engine based geotiff file."""

    def __init__(self, tif_path, timestamps):
        """Initializes dataset from a Google Earth Engine based geotiff file."""
        profile, input_dict = self._read_geotiff(tif_path, timestamps)
        self.profile = profile
        self.tensors = {
            "timestamps": torch.from_numpy(input_dict["timestamps"]),
            "sentinel2": torch.from_numpy(
                computed.normalize(Modality.SENTINEL2_L2A, input_dict["sentinel2"])
            ).float(),
            "sentinel1": torch.from_numpy(
                computed.normalize(Modality.SENTINEL1, input_dict["sentinel1"])
            ).float(),
            "landsat": torch.from_numpy(
                computed.normalize(Modality.LANDSAT, input_dict["landsat"])
            ).float(),
            "latlon": torch.from_numpy(
                predefined.normalize(Modality.LATLON, input_dict["latlon"])
            ).float(),
        }
        for k in self.tensors.keys():
            self.tensors[k].share_memory_()

    @staticmethod
    def _read_geotiff(tif_path, timestamps):
        with rio.open(tif_path) as src:
            profile = src.profile
            bands = src.descriptions
            height, width = src.height, src.width
            tile = src.read()

        height = tile.shape[1]
        width = tile.shape[2]
        num_pixels = height * width
        input_data = tile.reshape(len(bands), num_pixels)

        input_dict = {
            "timestamps": np.array([timestamps] * num_pixels),
            "latlon": input_data[
                [bands.index("latitude"), bands.index("longitude")]
            ].transpose(1, 0),
            "landsat": np.zeros(
                (num_pixels, 1, 1, len(timestamps), len(BANDS["landsat"]))
            ),
            "sentinel1": np.zeros(
                (num_pixels, 1, 1, len(timestamps), len(BANDS["sentinel1"]))
            ),
            "sentinel2": np.zeros(
                (num_pixels, 1, 1, len(timestamps), len(BANDS["sentinel2"]))
            ),
        }

        for i, key in enumerate(bands):
            if key == "latitude" or key == "longitude":
                continue
            modality, timestep_str, band = key.split("_")
            band_index = BANDS[modality].index(band)
            input_dict[modality][:, 0, 0, int(timestep_str), band_index] = input_data[i]
        return profile, input_dict

    def __len__(self):
        """Returns amount of pixels in geotiff file."""
        return self.profile["width"] * self.profile["height"]

    def __getitem__(self, idx):
        """Returns single pixel of data."""
        sample = {k: v[idx] for k, v in self.tensors.items()}
        return sample


def run_inference_on_tile(tile, timestamps):
    """Runs inference on all pixels in a tile."""
    print(tile)
    print("\n\tDownloading input data ...\t", end="")
    start = time.perf_counter()
    in_bucket.blob(tile).download_to_filename("in.tif")
    # Faster but not available inside docker container at the moment
    # subprocess.run(["gcloud", "storage", "cp", f"gs://{IN_BUCKET}/{tile}", "in.tif"], check=True)
    duration = time.perf_counter() - start
    print(f"{duration:.2f}s\t ✓")

    print("\n\tReading & normalizing data ...\t", end="")
    start = time.perf_counter()
    dataset = OlmoEarthGEEDataset("in.tif", timestamps)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )
    duration = time.perf_counter() - start
    print(f"{duration:.2f}s\t ✓")

    embeddings_list = []
    # Go through pixels in file in batches
    print("\tInference ...\t\t\t\t\t", end="")
    start = time.perf_counter()
    for batch in loader:
        gpu_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Create OlmoEarth sample
        masked_sample = MaskedOlmoEarthSample(
            timestamps=gpu_batch["timestamps"],
            sentinel2_l2a=gpu_batch["sentinel2"],
            sentinel1=gpu_batch["sentinel1"],
            landsat=gpu_batch["landsat"],
            latlon=gpu_batch["latlon"],
            sentinel2_l2a_mask=torch.zeros_like(gpu_batch["sentinel2"], device=device),
            sentinel1_mask=torch.zeros_like(gpu_batch["sentinel1"], device=device),
            landsat_mask=torch.zeros_like(gpu_batch["landsat"], device=device),
            latlon_mask=torch.zeros_like(gpu_batch["latlon"], device=device),
        )

        # Make predictions
        with torch.no_grad():
            preds = model(masked_sample, patch_size=1, fast_pass=True)
            preds_projected = model.project_and_aggregate(preds["tokens_and_masks"])
            embeddings = preds_projected.cpu().numpy()
        embeddings_list.append(embeddings)
    duration = time.perf_counter() - start
    print(f"{duration:.2f}s\t ✓")

    print("\tWriting to file ...\t\t\t", end="")
    start = time.perf_counter()
    profile = dataset.profile
    profile.update(
        count=EMBEDDINGS_SIZE, dtype="float32", compress="deflate", bigtiff="YES"
    )
    all_embeddings = np.concatenate(embeddings_list).transpose(1, 0)
    embeddings_reshaped = all_embeddings.reshape(
        EMBEDDINGS_SIZE, profile["height"], profile["width"]
    )
    with rio.open("out.tif", "w", **profile) as dst:
        dst.write(embeddings_reshaped.astype("float32"))
    duration = time.perf_counter() - start
    print(f"{duration:.2f}s\t ✓")

    print("\tUploading embeddings ...\t\t", end="")
    start = time.perf_counter()
    out_bucket.blob(tile).upload_from_filename("out.tif")
    # Faster but not available inside docker container at the moment
    # subprocess.run(["gcloud", "storage", "cp", "out.tif", f"gs://{OUT_BUCKET}/{tile}"], check=True)
    duration = time.perf_counter() - start
    print(f"{duration:.2f}s\t ✓")

    Path("in.tif").unlink()
    Path("out.tif").unlink()


def to_date_obj(d):
    """Converts date string to date object."""
    return dt.strptime(d, "%Y-%m-%d").date()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", required=True)
    parser.add_argument("-t", "--tiles", nargs="+", default=[], required=True)
    args = parser.parse_args()

    # Derive timestamps
    START_DATE, END_DATE = args.run.split("_")[-2:]
    timestamps_pd = pd.date_range(
        to_date_obj(START_DATE), to_date_obj(END_DATE), freq="MS"
    )[:-1]
    timestamps = [[t.day, t.month - 1, t.year] for t in timestamps_pd]

    for tile in args.tiles:
        run_inference_on_tile(f"{args.run}/{tile}", timestamps)
