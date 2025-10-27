# Barebones script for generating embeddings
from datetime import datetime as dt
from google.cloud import storage
from pathlib import Path
from rasterio.windows import Window

import argparse
import numpy as np
import os
import pandas as pd
import rasterio as rio
import json
import time
import torch

from torch.utils.data import default_collate
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

GCLOUD_PROJECT = os.environ["GCLOUD_PROJECT"]
IN_BUCKET = os.environ["IN_BUCKET"]
OUT_BUCKET = os.environ["OUT_BUCKET"]

BANDS = {
    "sentinel1":  ["VV", "VH"],
    "sentinel2":  ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
    "landsat":    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
}

MODEL_DIR = "latent_mim_tiny_shallow_decoder_lr2e-4_255000"
INFERENCE_WINDOW_SIZE = 100
EMBEDDINGS_SIZE = 192

client = storage.Client(project=GCLOUD_PROJECT)
in_bucket = client.bucket(IN_BUCKET)
out_bucket = client.bucket(OUT_BUCKET)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(f"{MODEL_DIR}/config.json", "r") as f:
    config_dict = json.load(f)
model_config = Config.from_dict(config_dict["model"])
model = model_config.build()
load_model_and_optim_state(f"{MODEL_DIR}/model_and_optim", model)
print("LOG: Loaded model.")
model.eval()
model = model.encoder
model = model.to(device)


# Helper function for data prep
def prepare_masked_olmo_earth_sample(tile, bands, timestamps, device=None):
    num_pixels = tile.shape[1] * tile.shape[2]
    input_data = tile.reshape(len(bands), num_pixels)

    # Fill input dict using geotiff data
    input_dict_raw = {
        "timestamps": np.array([timestamps] * num_pixels),
        "latlon":     input_data[[bands.index("latitude"), bands.index("longitude")]].transpose(1, 0),
        "landsat":    np.zeros((num_pixels, 1, 1, len(timestamps), len(BANDS["landsat"]))),
        "sentinel1":  np.zeros((num_pixels, 1, 1, len(timestamps), len(BANDS["sentinel1"]))),
        "sentinel2":  np.zeros((num_pixels, 1, 1, len(timestamps), len(BANDS["sentinel2"]))),
    }
    for i, key in enumerate(bands):
        if key == "latitude" or key == "longitude":
            continue
        modality, timestep_str, band = key.split("_")
        band_index = BANDS[modality].index(band)
        input_dict_raw[modality][:, 0, 0, int(timestep_str), band_index] = input_data[i]

    # Normalize input dict
    computed = Normalizer(Strategy.COMPUTED)
    predefined = Normalizer(Strategy.PREDEFINED)
    input_dict_normed = {
        "timestamps": input_dict_raw["timestamps"],
        "latlon":    predefined.normalize(Modality.LATLON, input_dict_raw["latlon"]).astype(np.float32),
        "landsat":   computed.normalize(Modality.LANDSAT, input_dict_raw["landsat"]).astype(np.float32),
        "sentinel1": computed.normalize(Modality.SENTINEL1, input_dict_raw["sentinel1"]).astype(np.float32),
        "sentinel2": computed.normalize(Modality.SENTINEL2_L2A, input_dict_raw["sentinel2"]).astype(np.float32),
    }

    # Prepared MaskedOlmoEarthSample
    masked_sample_dicts_list = []
    for i in range(num_pixels):
        sample = OlmoEarthSample(
            sentinel2_l2a=input_dict_normed["sentinel2"][i],
            sentinel1=input_dict_normed["sentinel1"][i],
            landsat=input_dict_normed["landsat"][i],
            timestamps=input_dict_normed["timestamps"][i],
            latlon=input_dict_normed["latlon"][i],
        )
        masked_sample = MaskedOlmoEarthSample.from_olmoearthsample(sample)
        masked_sample_dicts_list.append(masked_sample.as_dict(return_none=False))
    collated_sample = default_collate(masked_sample_dicts_list)
    collated_sample_to_device = {k: v.to(device) for k,v in collated_sample.items()}
    return MaskedOlmoEarthSample(**collated_sample_to_device)

    
def run_inference_on_tile(tile, timestamps):
    print(tile)
    print(f"\n\tDownloading input data ...\t", end="")
    start = time.perf_counter()
    in_bucket.blob(tile).download_to_filename("in.tif")
    duration = time.perf_counter() - start
    print(f"{duration:.2f}s\t ✓")

    with rio.open("in.tif") as src:
        profile = src.profile
        bands = src.descriptions
        height, width = src.height, src.width
        profile.update(count=EMBEDDINGS_SIZE, dtype="float32", compress="deflate", bigtiff="YES")

        with rio.open("out.tif", "w", **profile) as dst:
            for y in range(0, height, INFERENCE_WINDOW_SIZE):
                for x in range(0, width, INFERENCE_WINDOW_SIZE):
                    win = Window(x, y, min(INFERENCE_WINDOW_SIZE, width - x), min(INFERENCE_WINDOW_SIZE, height - y))
                    data = src.read(window=win)
                    masked_sample = prepare_masked_olmo_earth_sample(data, bands, timestamps, device)
                    with torch.no_grad():
                        preds = model(masked_sample, patch_size=1)
                    embeddings = preds["project_aggregated"].cpu().numpy().transpose(1, 0)
                    embeddings_reshaped = embeddings.reshape(embeddings.shape[0], data.shape[1], data.shape[2])
                    dst.write(embeddings_reshaped.astype("float32"), window=win)

    print(f"\tUploading embeddings ...\t", end="")
    start = time.perf_counter()
    out_bucket.blob(tile).upload_from_filename("out.tif")
    duration = time.perf_counter() - start
    print(f"{duration:.2f}s\t ✓")

    Path("in.tif").unlink()
    Path("out.tif").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run')
    parser.add_argument('-t', '--tiles', nargs='+', default=[])
    args = parser.parse_args()

    # Derive timestamps
    START_DATE, END_DATE = args.run.split("_")[-2:]
    to_date_obj = lambda d: dt.strptime(d, "%Y-%m-%d").date()
    timestamps_pd = pd.date_range(to_date_obj(START_DATE), to_date_obj(END_DATE), freq="MS")[:-1]
    timestamps = [[t.year, t.month - 1, t.day] for t in timestamps_pd]

    for tile in args.tiles:
        run_inference_on_tile(f"{args.run}/{tile}", timestamps)
