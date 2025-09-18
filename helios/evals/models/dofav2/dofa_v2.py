"""Wrapper to run evals on the DOFA v2 model https://github.com/zhu-xlab/DOFA."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataclasses import dataclass
from einops import rearrange

from olmo_core.config import Config

from logging import getLogger

from helios.train.masking import MaskedHeliosSample
from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType

logger = getLogger(__name__)

# vh,vv
S1_MEAN = [166.36275909, 88.45542715]  # / 255.0
S1_STD = [64.83126309, 43.07350145]  # /255.0

S2_MEAN = [
    114.1099739,
    114.81779093,
    126.63977424,
    84.33539309,
    97.84789168,
    103.94461911,
    101.435633,
    72.32804172,
    56.66528851,
]
S2_STD = [
    77.84352553,
    69.96844919,
    67.42465279,
    64.57022983,
    61.72545487,
    61.34187099,
    60.29744676,
    47.88519516,
    42.55886798,
]
WAVE_LENGTHS_SENTINEL2 = [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]
WAVE_LENGTHS_SENTINEL1 = [5.405, 5.405]


def apply_normalization(data: torch.Tensor, modality: str) -> torch.Tensor:
    """Apply normalization to the data."""
    if modality == Modality.SENTINEL2_L2A.name:
        return transforms.Normalize(S2_MEAN, S2_STD)(data)
    elif modality == Modality.SENTINEL1.name:
        return transforms.Normalize(S1_MEAN, S1_STD)(data)
    else:
        raise ValueError(f"Unsupported modality: {modality}")


DOFA_S2_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b)
    for b in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
]
DOFA_S1_BANDS = [Modality.SENTINEL1.band_order.index(b) for b in ["vv", "vh"]]


# DOUBLE CHECK LIST
# AM I LOADING V2
# AM I APPLYING NORMALIZATION correctly
# Are the bands in the right order
class DOFAv2(nn.Module):
    """DOFA v2 model."""

    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]
    patch_size: int = 16
    base_resize: int = 224

    def __init__(
        self, torchhub_id: str = "vit_base_dofa", apply_normalization: bool = False
    ):
        super().__init__()
        self._load_model(torchhub_id)
        self.apply_normalization = apply_normalization

    def _load_model(self, torchhub_id: str):
        """Load the DOFA v2 model from torch hub."""
        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        for attempt in range(2):
            try:
                self.model = torch.hub.load(
                    "zhu-xlab/DOFA",
                    torchhub_id,  # The entry point defined in hubconf.py
                    pretrained=True,
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error loading DOFA v2 model: {e}. Retrying in 5 seconds..."
                )
                time.sleep(5)
        else:
            raise RuntimeError("Failed to load DOFA v2 model after retrying.")

    def get_wave_lengths(self, modality: str) -> list[float]:
        """Get the wave lengths for the modality."""
        if modality == Modality.SENTINEL2_L2A.name:
            return WAVE_LENGTHS_SENTINEL2
        elif modality == Modality.SENTINEL1.name:
            return WAVE_LENGTHS_SENTINEL1
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def _process_modality_data(
        self, data: torch.Tensor, modality: str
    ) -> list[tuple[torch.Tensor, list[float]]]:
        """Process the modality data for the DOFA v2 model."""
        if not modality in self.supported_modalities:
            raise ValueError(f"Unsupported modality: {modality}")

        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/dinov3 format
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []
        for i in range(t_dim):
            data_i = data[:, :, :, i, :]
            logger.info(f"Data shape prior to band subset: {data_i.shape}")

            if modality == Modality.SENTINEL2_L2A.name:
                data_i = data_i[..., DOFA_S2_BANDS]
            elif modality == Modality.SENTINEL1.name:
                data_i = data_i[..., DOFA_S1_BANDS]

            num_channels = data_i.shape[-1]
            if original_height > self.base_resize:
                new_height = original_height
            elif original_height <= self.base_resize and original_height > 1:
                new_height = self.base_resize
            else:
                new_height = self.patch_size
            # log shape prior to resize
            logger.info(f"Data shape prior to resize: {data_i.shape}")
            # TODO: check if this is correct
            # Rearrange for interpolating hw
            data_i = rearrange(data_i, "b h w c -> b c h w")
            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )
            logger.info(f"Data shape: {data_i.shape}")
            if self.apply_normalization:
                data_i = apply_normalization(data_i, modality)
            logger.info(f"Data shape after normalization: {data_i.shape}")
            data_list.append((data_i, self.get_wave_lengths(modality)))

        return data_list

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
    ) -> list[tuple[torch.Tensor, list[float]]]:
        """Prepare input for the dinov3 model from MaskedHeliosSample."""
        if len(masked_helios_sample.modalities) > 1:
            raise ValueError("Multiple modalities are not yet supported for this model")
        modality = masked_helios_sample.modalities[0]
        if modality not in self.supported_modalities:
            logger.warning(
                f"Skipping modality {modality} as it is not in the supported modalities list {self.supported_modalities}"
            )
            raise ValueError(f"Unsupported modality: {modality}")

        data = getattr(masked_helios_sample, modality)

        # Process the modality data
        processed_data = self._process_modality_data(data, modality)
        return processed_data

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
    ) -> torch.Tensor:
        per_timestep_dofa_inputs_and_wave_lengths = self.prepare_input(
            masked_helios_sample
        )
        # potentially will need to add a flag for segmentation
        output_features = []
        for dofa_input, wave_lengths in per_timestep_dofa_inputs_and_wave_lengths:
            timestep_output = self.model.forward_features(dofa_input, wave_lengths)
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features


@dataclass
class DOFAv2Config(Config):
    """Config for the DOFA v2 model."""

    torchhub_id: str = "vit_base_dofa"
    apply_normalization: bool = False

    def build(self) -> "DOFAv2":
        return DOFAv2(
            torchhub_id=self.torchhub_id, apply_normalization=self.apply_normalization
        )
