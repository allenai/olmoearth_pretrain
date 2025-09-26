# Barebones server for generating embeddings
import json
import litserve as ls
import numpy as np
import torch

from helios.train.masking import MaskedHeliosSample
from helios.data.dataset import HeliosSample
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from torch.utils.data import default_collate


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        print("LOG: Entered setup()")
        with open(
            "latent_mim_tiny_shallow_decoder_lr2e-4_255000/config.json", "r"
        ) as f:
            config_dict = json.load(f)
        print("LOG: ✓ Loaded config.json.")
        model_config = Config.from_dict(config_dict["model"])
        print("LOG: ✓ Loaded config from_dict.")
        self.model = model_config.build()
        print("LOG: ✓ Built model from config.")
        load_model_and_optim_state(
            "latent_mim_tiny_shallow_decoder_lr2e-4_255000/model_and_optim", self.model
        )
        print("LOG: ✓ Loaded model and optim state.")
        self.model.eval()
        self.model = self.model.encoder
        self.model = self.model.to(device)

    def decode_request(self, request):
        print("LOG: Entered decode_request()")
        masked_sample_dicts_list = []
        batch_size = len(request["sentinel2_l2a"])
        print(f"LOG: ✓ Received request with batch_size: {batch_size}")
        for i in range(batch_size):
            sample = HeliosSample(
                sentinel2_l2a=np.array(request["sentinel2_l2a"][i], dtype=np.float32),
                timestamps=np.array(request["timestamps"][i]),
            )
            masked_sample = MaskedHeliosSample.from_heliossample(sample)
            masked_sample_dicts_list.append(masked_sample.as_dict(return_none=False))

        collated_sample = default_collate(masked_sample_dicts_list)
        masked_helios_sample = MaskedHeliosSample(**collated_sample)
        print(f"LOG: ✓ Created MaskedHeliosSample for batch_size: {batch_size}")
        return masked_helios_sample

    def predict(self, x):
        print("LOG: Entered predict()")
        with torch.no_grad():
            preds = self.model(x, patch_size=1)
        print("LOG: ✓ Predictions made.")
        project_aggregated = preds["project_aggregated"].numpy()
        print(f"LOG: ✓ Predictions shape: {project_aggregated.shape}")
        return project_aggregated.tolist()

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8080)
