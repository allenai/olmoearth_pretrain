"""Helios wrapper for Prithvi v2."""

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from huggingface_hub import hf_hub_download
from olmo_core.config import Config
from upath import UPath

from helios.data.constants import Modality
from helios.evals.models.prithviv2.prithvi_mae import PrithviMAE
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

# for Prithvi, true values are ["B02", "B03", "B04", "B05", "B06", "B07"]
PRITHVI_MEAN = [
    1087.0,
    1342.0,
    1433.0,
    2734.0,
    1958.0,
    1363.0,
]
PRITHVI_STD = [
    2248.0,
    2179.0,
    2178.0,
    1850.0,
    1242.0,
    1049.0,
]


class PrithviV2Models(StrEnum):
    """Names for different Prithvi models on torch hub."""

    VIT_300 = "Prithvi_EO_V2_300M"
    VIT_600 = "Prithvi_EO_V2_600M"


MODEL_TO_HF_INFO = {
    PrithviV2Models.VIT_300: {
        "hf_hub_id": f"ibm-nasa-geospatial/{PrithviV2Models.VIT_300.value}",
        "weights": f"{PrithviV2Models.VIT_300.value}.pt",
        "revision": "b2f2520ab889f42a25c5361ba18761fcb4ea44ad",
    },
    PrithviV2Models.VIT_600: {
        "hf_hub_id": f"ibm-nasa-geospatial/{PrithviV2Models.VIT_600.value}",
        "weights": f"{PrithviV2Models.VIT_600.value}.pt",
        "revision": "87f15784813828dc37aa3197a143cd4689e4d080",
    },
}


class PrithviV2(nn.Module):
    """Class containing the PrithviV2 model that can ingest MaskedHeliosSample objects."""

    supported_modalities = [Modality.SENTINEL2_L2A.name]

    def __init__(
        self,
        load_directory: str,
        size: PrithviV2Models,
        use_pretrained_normalizer: bool = True,
    ):
        """Initialize the PrithviV2 wrapper.

        Args:
            load_directory: The directory to load from
            size: one of VIT_300 or VIT_600
            use_pretrained_normalizer: Whether or not to apply prithvi pretraining normalization
        """
        super().__init__()

        hub_id = MODEL_TO_HF_INFO[size]["hf_hub_id"]
        revision = MODEL_TO_HF_INFO[size]["revision"]
        weights_path = MODEL_TO_HF_INFO[size]["weights"]

        if not (UPath(load_directory) / "config.json").exists():
            # even though we have a nosec here we actually follow the advice in
            # https://bandit.readthedocs.io/en/latest/plugins/b615_huggingface_unsafe_download.html
            # and pin the download to a specific commit, but our bandit can't tell because
            # "revision" is now a variable instead of a string
            _ = hf_hub_download(  # nosec
                local_dir=UPath(load_directory),
                repo_id=hub_id,
                filename="config.json",
                revision=revision,
            )
        with (UPath(load_directory) / "config.json").open("r") as f:
            config = yaml.safe_load(f)["pretrained_cfg"]

        config["num_frames"] = 1

        self.model = PrithviMAE(**config)

        if not (UPath(load_directory) / weights_path).exists():
            # even though we have a nosec here we actually follow the advice in
            # https://bandit.readthedocs.io/en/latest/plugins/b615_huggingface_unsafe_download.html
            # and pin the download to a specific commit, but our bandit can't tell because
            # "revision" is now a variable instead of a string
            _ = hf_hub_download(  # nosec
                local_dir=UPath(load_directory),
                repo_id=hub_id,
                filename=weights_path,
                revision=revision,
            )

        state_dict = torch.load(
            UPath(load_directory) / weights_path, map_location="cpu"
        )
        # discard fixed pos_embedding weight, following
        # https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/e4aabdc440c8ee703a749def8af5bf4700dee35b/inference.py#L362
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        self.model.load_state_dict(state_dict, strict=False)
        self.image_resolution = config["img_size"]
        self.bands = config["bands"]
        # patch size is a list [t, h, w], where h == w
        self.patch_size = config["patch_size"][-1]
        self.helios_s2_to_prithvi = [
            Modality.SENTINEL2_L2A.band_order.index(b) for b in self.bands
        ]
        self.use_pretrained_normalizer = use_pretrained_normalizer

    @staticmethod
    def normalize(data: torch.Tensor) -> torch.Tensor:
        """Normalize Prithvi input according to Prithvi stats."""
        # https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/inference.py#L140
        return (
            data - torch.tensor(PRITHVI_MEAN, dtype=data.dtype, device=data.device)
        ) / (torch.tensor(PRITHVI_STD, dtype=data.dtype, device=data.device))

    def _process_modality_data(self, data: torch.Tensor, modality: str) -> torch.Tensor:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]
            modality: What modality data is

        Returns:
            list of tensors of shape [B, C, H, W]
        """
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []

        # interpolate only accepts up to 4d tensors
        for i in range(t_dim):
            data_i = data[:, :, :, i, self.helios_s2_to_prithvi]
            if self.use_pretrained_normalizer:
                data_i = self.normalize(data_i)
            data_i = rearrange(data_i, "b h w c -> b c h w")

            new_height = (
                self.patch_size if original_height == 1 else self.image_resolution
            )

            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )

            data_list.append(data_i)

        return torch.stack(data_list, dim=2)

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
    ) -> torch.Tensor:
        """Prepare input for the Prithvi model from MaskedHeliosSample."""
        if len(masked_helios_sample.modalities) != 1:
            raise RuntimeError(
                f"Prithvi only supports one modality. Received {len(masked_helios_sample.modalities)}: {masked_helios_sample.modalities}"
            )
        modality = masked_helios_sample.modalities[0]
        if modality != Modality.SENTINEL2_L2A.name:
            raise RuntimeError(
                f"Prithvi only supports Sentinel2_L2A. Received {modality}"
            )

        data = getattr(masked_helios_sample, modality)
        return self._process_modality_data(data, modality)

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the satlas model."""
        processed_input = self.prepare_input(masked_helios_sample)
        t = processed_input.shape[2]
        output = self.model.forward_features(processed_input)[-1]
        # following
        # https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/prithvi_mae.py#L449
        # we remove the class token. This is also the approach they
        # take for classification: https://github.com/IBM/terratorch/blob/main/terratorch/models/scalar_output_model.py#L19
        output = output[:, 1:, :]
        side = math.isqrt(int((output).shape[1] / t))

        if not spatial_pool:
            # then we don't want to keep the spatial dimensions
            output = rearrange(
                output, "b (t h w) c -> b t (h w) c", h=side, w=side, t=t
            )
            output = output.mean(dim=2)
        else:
            # (t h w) following the unpatchify order
            output = rearrange(output, "b (t h w) c -> b t h w c", h=side, w=side, t=t)

        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output = output.mean(dim=1)
        elif pooling == PoolingType.MAX:
            output = torch.max(output, dim=1)[0]
        return output


@dataclass
class PrithviV2Config(Config):
    """olmo_core style config for PrithviV2 Wrapper."""

    load_directory: str = "/weka/dfive-default/helios/models/prithvi"
    size: PrithviV2Models = PrithviV2Models.VIT_300
    use_pretrained_normalizer: bool = True

    def build(self) -> PrithviV2:
        """Build the PrithviV2 model."""
        return PrithviV2(
            load_directory=self.load_directory,
            size=self.size,
            use_pretrained_normalizer=self.use_pretrained_normalizer,
        )
