# Barebones server for generating embeddings
from datetime import datetime as dt
from google.cloud import storage
from pathlib import Path
from rasterio.windows import Window

import litserve as ls
import numpy as np
import pandas as pd
import rasterio as rio
import json
import torch

from torch.utils.data import default_collate
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

GCLOUD_PROJECT = "ai2-ivan"
IN_BUCKET = "ai2-ivan-helios-input-data"
OUT_BUCKET = "ai2-ivan-helios-output-data"

BANDS = {
    "sentinel1":  ["VV", "VH"],
    "sentinel2":  ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
    "landsat":    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
}

MODEL_DIR = "latent_mim_tiny_shallow_decoder_lr2e-4_255000"
MODEL_CONFIG = "latent_mim_tiny_shallow_decoder_lr2e-4_255000/config.json"

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

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        print("LOG: Entered setup()")

        client = storage.Client(project=GCLOUD_PROJECT)
        self.in_bucket = client.bucket(IN_BUCKET)
        self.out_bucket = client.bucket(OUT_BUCKET)

        self.inference_window_size = 100 # 10-12GB GPU memory requirements
        self.embeddings_size = 192

        with open(f"{MODEL_DIR}/config.json", "r") as f:
            config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])
        self.model = model_config.build()
        load_model_and_optim_state(
            "latent_mim_tiny_shallow_decoder_lr2e-4_255000/model_and_optim", self.model
        )
        print("LOG: Loaded model.")
        self.model.eval()
        self.model = self.model.encoder
        self.model = self.model.to(device)
        self.device = device


    def decode_request(self, request):
        print(f"LOG: Decoding request: {request}")

        # Download file from cloud storage
        in_gs_name = request["name"]
        in_gs_path = Path(in_gs_name)
        run_name = in_gs_path.parent.stem
        tile_name = in_gs_path.name
        input_tif_name = f"in_{tile_name}"
        self.in_bucket.blob(in_gs_name).download_to_filename(input_tif_name)
        print(f"LOG: Downloaded {IN_BUCKET}/{in_gs_name} to {input_tif_name}")
        
        # Generate timestamps from folder name
        date_folder = in_gs_path.parent.stem
        start_str, end_str = run_name.split("_")[:-2]
        start_date = dt.strptime(start_str, "%Y-%m-%d").date()
        end_date = dt.strptime(end_str, "%Y-%m-%d").date()
        timestamps_pd = pd.date_range(start_date, end_date, freq="MS")[:-1]
        timestamps = [[t.year, t.month - 1, t.day] for t in timestamps_pd]
        print(f"LOG: Generated {len(timestamps)} timestamps from folder name: {date_folder}")

        return {
            "run_name": run_name,
            "tile_name": tile_name,
            "timestamps": timestamps,
            "input_tif_name": input_tif_name
        }

    def predict(self, args):
         
        run_name = args.run_name
        tile_name = args.tile_name
        input_tif_name = args.input_tif_name
        output_tif_name = f"out_{tile_name}"
        with rio.open(input_tif_name) as src:
            profile = src.profile
            bands = src.descriptions
            height, width = src.height, src.width
            profile.update(count=self.embeddings_size, dtype="float32", compress="deflate", bigtiff="YES")

            with rio.open(output_tif_name, "w", **profile) as dst:
                for y in range(0, height, self.inference_window_size):
                    for x in range(0, width, self.inference_window_size):
                        win = Window(x, y, min(self.inference_window_size, width - x), min(self.inference_window_size, height - y))
                        data = src.read(window=win)
                        masked_sample = prepare_masked_olmo_earth_sample(data, bands, args.timestamps, self.device)
                        with torch.no_grad():
                            preds = self.model(masked_sample, patch_size=1)
                        embeddings = preds["project_aggregated"].cpu().numpy().transpose(1, 0)
                        embeddings_reshaped = embeddings.reshape(embeddings.shape[0], data.shape[1], data.shape[2])
                        dst.write(embeddings_reshaped.astype("float32"), window=win)

        out_gs_name = f"{run_name}/{tile_name}"
        self.out_bucket.blob(out_gs_name).upload_from_filename(output_tif_name)
        
        # Delete tifs once prediction is complete
        Path(input_tif_name).unlink()
        Path(output_tif_name).unlink()

        return out_gs_name

    def encode_response(self, out_gs_name):
        return {"name": out_gs_name, "bucket": OUT_BUCKET}


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8080)
