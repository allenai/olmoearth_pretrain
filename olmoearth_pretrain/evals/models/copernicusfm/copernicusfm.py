"""Copernicus FM model."""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.models.copernicusfm.src.model_vit import vit_base_patch16
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


MODALITY_TO_WAVELENGTH_BANDWIDTHS: dict[str, dict[str, list]] = {
    # https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Bench/src/configs/dataset/cobench_eurosat_s2.yaml
    Modality.SENTINEL2_L2A.name: {
        "band_names": [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ],
        "band_wavelengths": [
            440,
            490,
            560,
            665,
            705,
            740,
            783,
            842,
            860,
            940,
            1370,
            1610,
            2190,
        ],
        "band_bandwidths": [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
    },
    # https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Bench/src/configs/dataset/cobench_eurosat_s1.yaml
    Modality.SENTINEL1.name: {
        "band_names": ["vv", "vh"],
        "band_wavelengths": [50000000, 50000000],
        "band_bandwidths": [1e9, 1e9],
    },
}

DEFAULT_LOAD_DIRECTORY = "/weka/dfive-default/helios/models/copernicusfm"


class CopernicusFM(torch.nn.Module):
    """Wrapper for Copernicus FM to ingest Masked OlmoEarth Pretrain Sample."""

    image_resolution = 224
    patch_size = 16
    input_mode = "spectral"
    supported_modalities = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
    supports_multiple_modalities_at_once = True

    def __init__(self, load_directory: str = DEFAULT_LOAD_DIRECTORY) -> None:
        """Initialize the Copernicus FM wrapper.

        Args:
            load_directory: The directory to load from
        """
        super().__init__()

        # global_pool=True so that we initialize the fc_norm layer
        self.model = vit_base_patch16(num_classes=10, global_pool=True)
        check_point = torch.load(
            UPath(load_directory) / "CopernicusFM_ViT_base_varlang_e100.pth"
        )
        if "model" in check_point:
            state_dict = check_point["model"]
        else:
            state_dict = check_point
        self.model.load_state_dict(state_dict, strict=False)

        # take MODALITY_TO_WAVELENGTH_BANDWIDTHS and rearrage it so that it has the same
        # ordering as the OlmoEarth Pretrain band orders, defined by Modality.band_order
        self.modality_to_wavelength_bandwidths = {}
        for modality in [Modality.SENTINEL2_L2A, Modality.SENTINEL1]:
            modality_name = modality.name
            wavelength_bandwidths = MODALITY_TO_WAVELENGTH_BANDWIDTHS[modality_name]
            wavelengths = []
            bandwidths = []
            for b in modality.band_order:
                cfm_idx = wavelength_bandwidths["band_names"].index(b)
                wavelengths.append(wavelength_bandwidths["band_wavelengths"][cfm_idx])
                bandwidths.append(wavelength_bandwidths["band_bandwidths"][cfm_idx])
            self.modality_to_wavelength_bandwidths[modality_name] = {
                "band_bandwidths": bandwidths,
                "band_wavelengths": wavelengths,
            }

    def _process_modality_data(
        self, data: torch.Tensor, modality: str
    ) -> list[torch.Tensor]:
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

        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")

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
        return data_list

    def prepare_input(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> tuple[list[torch.Tensor], list[int], list[int]]:
        """Prepare input for the CopernicusFM model from MaskedOlmoEarthSample."""
        wavelengths: list[int] = []
        bandwidths: list[int] = []
        all_processed_data: list[list[torch.Tensor]] = []
        for modality in masked_olmoearth_sample.modalities:
            if modality not in self.supported_modalities:
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported "
                    f"modalities list {self.supported_modalities}"
                )
                continue

            data = getattr(masked_olmoearth_sample, modality)

            if data is None:
                continue

            all_processed_data.append(self._process_modality_data(data, modality))
            wavelengths.extend(
                self.modality_to_wavelength_bandwidths[modality]["band_wavelengths"]
            )
            bandwidths.extend(
                self.modality_to_wavelength_bandwidths[modality]["band_bandwidths"]
            )

        concatenated_processed_data: list[torch.Tensor] = []
        for timestamp_data in zip(*all_processed_data):
            # concatenate along the channel dimension
            concatenated_processed_data.append(torch.cat(timestamp_data, dim=1))
        return concatenated_processed_data, wavelengths, bandwidths

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through CopernicusFM model."""
        # Prepare input
        per_timestep_inputs, wavelengths, bandwidths = self.prepare_input(
            masked_olmoearth_sample
        )
        output_features: list[torch.Tensor] = []
        for data in per_timestep_inputs:
            meta = torch.full(
                (1, 4), float("nan"), device=data.device
            )  # [lon, lat, delta_time, patch_token_area], assume unknown
            # "The embed tensor contains the encoded image features, which can be used for downstream tasks."
            _, timestep_output = self.model(
                data,
                meta,
                wavelengths,
                bandwidths,
                None,
                self.input_mode,
                self.patch_size,
            )
            if not spatial_pool:
                timestep_output = timestep_output.mean(dim=1)
                # apply the fc_norm layer
                timestep_output = self.model.fc_norm(timestep_output)
            else:
                # no norm, following
                # https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Bench/src/foundation_models/CopernicusFM/models_dwv_seg.py
                side = math.isqrt(timestep_output.shape[1])
                timestep_output = rearrange(
                    timestep_output, "b (h w) c -> b h w c", h=side, w=side
                )
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features


@dataclass
class CopernicusFMConfig(Config):
    """olmo_core style config for CopernicusFMConfig."""

    load_directory: str = DEFAULT_LOAD_DIRECTORY

    def build(self) -> CopernicusFM:
        """Build the CopernicusFM model."""
        return CopernicusFM(
            load_directory=self.load_directory,
        )
