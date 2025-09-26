# Barebones server for generating embeddings
import json
import litserve as ls
import numpy as np
import pandas as pd
import rasterio as rio
import torch

from datetime import datetime
from helios.train.masking import MaskedHeliosSample
from helios.data.dataset import HeliosSample
from google.cloud import storage
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from pathlib import Path
from torch.utils.data import default_collate

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
IN_BUCKET = "ai2-ivan-helios-input-data"
OUT_BUCKET = "ai2-ivan-helios-output-data"

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        print("LOG: Entered setup()")

        client = storage.Client()
        self.in_bucket = client.bucket(IN_BUCKET)
        self.out_bucket = client.bucket(OUT_BUCKET)

        with open(
            "latent_mim_tiny_shallow_decoder_lr2e-4_255000/config.json", "r"
        ) as f:
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

    def decode_request(self, request):
        print(f"LOG: Decoding request: {request}")

        # Download file from cloud storage
        gs_name = request["name"]
        gs_path = Path(gs_name)
        input_tif_name = f"in_{gs_path.name}"
        self.in_bucket.blob(gs_name).download_to_filename(input_tif_name)
        print(f"LOG: Downloaded {IN_BUCKET}/{gs_name} to {input_tif_name}")
        
        # Generate timestamps from folder name
        date_folder = gs_path.parent.stem
        start_str, end_str = date_folder.split("_")
        start_date = datetime.strptime(start_str, "%Y-%m").date()
        end_date = datetime.strptime(end_str, "%Y-%m").date()
        timestamps_pd = pd.date_range(start_date, end_date, freq="MS")[:-1]
        timestamps = [[t.year, t.month - 1, t.day] for t in timestamps_pd]
        print(f"LOG: Generated {len(timestamps)} timestamps from folder name: {date_folder}")

        # Geotiff to input dictionary
        with rio.open(input_tif_name) as src:
            input_geotiff = src.read().astype(np.float32)
            batch_size = input_geotiff.shape[1] * input_geotiff.shape[2]
            geotiff_band_names = src.descriptions
            geotiff_profile = src.profile # Needed for writing geotiff later
        print(f"LOG: {input_tif_name} read in.")

        # Delete input tif once it is in memory
        Path(input_tif_name).unlink()

        flattened_data = input_geotiff.reshape(input_geotiff.shape[0], batch_size)
        timesteps = len(timestamps)

        input_dict = {
            "sentinel2_l2a": np.zeros((batch_size, 1, 1, timesteps, len(S2_BANDS)), dtype=np.float32),
            "sentinel1":  np.zeros((batch_size, 1, 1, timesteps, len(S1_BANDS)), dtype=np.float32),
            "timestamps": np.array([timestamps] * batch_size)
        }
        print(f"LOG: {input_tif_name} initialized input dict.")

        BANDS = {"sentinel1": S1_BANDS, "sentinel2_l2a": S2_BANDS}
        for i, key in enumerate(geotiff_band_names):
            key_parts = key.split("_")
            modality = "_".join(key_parts[:-2])
            timestep, band = int(key_parts[-2]), key_parts[-1]
            band_index = BANDS[modality].index(band)
            input_dict[modality][:, 0, 0, timestep, band_index] = flattened_data[i]
        print(f"LOG: {input_tif_name} filled input dict.")

        # Create MaskedHeliosSample from input dict
        masked_sample_dicts_list = []
        for i in range(batch_size):
            sample = HeliosSample(
                sentinel2_l2a=input_dict["sentinel2_l2a"][i],
                sentinel1=input_dict["sentinel1"][i],
                timestamps=input_dict["timestamps"][i]
            )
            masked_sample = MaskedHeliosSample.from_heliossample(sample)
            masked_sample_dicts_list.append(masked_sample.as_dict(return_none=False))

        collated_sample = default_collate(masked_sample_dicts_list)
        masked_helios_sample = MaskedHeliosSample(**collated_sample)  
        print(f"LOG: Created MaskedHeliosSample for batch_size: {batch_size}")
        return {
            "masked_helios_sample": masked_helios_sample,
            "geotiff_profile": geotiff_profile,
            "gs_name": gs_name,
            "gs_path": gs_path,
            
        }

    def predict(self, x):
        masked_helios_sample = x["masked_helios_sample"]
        geotiff_profile = x["geotiff_profile"]
        gs_name = x["gs_name"]
        gs_path = x["gs_path"]
        output_tif_name = f"out_{gs_path.name}"

        # Model forward pass
        with torch.no_grad():
            preds = self.model(masked_helios_sample, patch_size=1)
        print(f"LOG: Predictions made for {output_tif_name}")
        
        # Reshape embeddings
        embeddings = preds["project_aggregated"].numpy().transpose(1, 0)
        embeddings_size = embeddings.shape[0]
        embeddings_reshaped = embeddings.reshape(embeddings_size, geotiff_profile["width"], geotiff_profile["height"])
        print(f"LOG: {output_tif_name} embeddings shape: {embeddings_reshaped.shape}")

        # Save embeddings to a geotiff file
        geotiff_profile.update(count=embeddings_size, dtype=np.float32)
        with rio.open(output_tif_name, "w", **geotiff_profile) as dst:
            dst.write(embeddings_reshaped)
        print(f"LOG: {output_tif_name}: saved to file.")
        
        # Upload to Cloud Storage
        self.out_bucket.blob(gs_name).upload_from_filename(output_tif_name)
        print(f"LOG: {output_tif_name}: uploaded to {OUT_BUCKET}/{gs_name}.")

        # Delete output_tif once is has been uploaded
        Path(output_tif_name).unlink()

        return {
            "bucket": OUT_BUCKET,
            "name": gs_path
        }

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8080)
