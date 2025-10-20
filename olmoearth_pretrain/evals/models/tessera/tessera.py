"""Tessera foundation model https://github.com/ucam-eo/tessera ."""

import logging
import os
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from einops import reduce, repeat
from olmo_core.config import Config
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.models.tessera.tessera_model import build_inference_model
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

# Normalization stats copied from https://github.com/ucam-eo/tessera/blob/a883aa12392eb9fc237ae4c29824318760e138a2/tessera_infer/src/datasets/ssl_dataset.py
# Mean and variance
S2_BAND_MEAN = np.array(
    [
        1711.0938,
        1308.8511,
        1546.4543,
        3010.1293,
        3106.5083,
        2068.3044,
        2685.0845,
        2931.5889,
        2514.6928,
        1899.4922,
    ],
    dtype=np.float32,
)
S2_BAND_STD = np.array(
    [
        1926.1026,
        1862.9751,
        1803.1792,
        1741.7837,
        1677.4543,
        1888.7862,
        1736.3090,
        1715.8104,
        1514.5199,
        1398.4779,
    ],
    dtype=np.float32,
)
S1_BAND_MEAN = np.array([5484.0407, 3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334, 1726.0670], dtype=np.float32)


# Tessera band order based on https://github.com/ucam-eo/tessera/blob/a883aa12392eb9fc237ae4c29824318760e138a2/tessera_preprocessing/s2_fast_processor.py#L51
HELIOS_SENTINEL2_TESSERA_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b)
    for b in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
    if b in Modality.SENTINEL2_L2A.band_order
]
HELIOS_SENTINEL1_TESSERA_BANDS = [
    Modality.SENTINEL1.band_order.index(b)
    for b in ["vv", "vh"]
    if b in Modality.SENTINEL1.band_order
]


# Only makes sense for pixel time series data
class Tessera(nn.Module):
    """Wrapper for the Tessera model that can ingest MaskedOlmoEarthSample objects."""

    # Pixel time series data
    patch_size: int = 1
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]
    required_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]
    requires_timeseries: bool = True
    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        checkpoint_path: str | None = None,
        use_pretrained_normalizer: bool = True,
    ):
        """Initialize the Tessera wrapper.

        Args:
            checkpoint_path: Path to Tessera model checkpoint (optional)
            use_pretrained_normalizer: Whether to apply normalization to input data
            input_size: Input image size expected by the model
        """
        super().__init__()
        self.use_pretrained_normalizer = use_pretrained_normalizer
        self.checkpoint_path = checkpoint_path

        # Initialize model - placeholder for now
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str | None = None) -> None:
        """Load the Tessera model."""
        # TODO: Implement actual Tessera model loading
        model = build_inference_model()
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
        # remove the _orig_mod prefix from all the keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # remove all keys starting with "projector."
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("projector.")
        }
        model.load_state_dict(state_dict)
        logger.info(f"Loaded Tessera model from {checkpoint_path}")
        self.model = model

    def calculate_day_of_year(self, timestamp: torch.Tensor) -> torch.Tensor:
        """Calculate day of year from timestamp.

        Args:
            timestamp: Tensor of shape (..., 3) where last dim is [day, month, year]

        Returns:
            Tensor of same shape as input without last dim, with day of year as int
        """
        # timestamp[..., 0] = day, timestamp[..., 1] = month, timestamp[..., 2] = year
        day = timestamp[..., 0]
        month = timestamp[..., 1]
        year = timestamp[..., 2]

        # Days in months for non-leap years
        days_in_month = torch.tensor(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], device=timestamp.device
        )

        # Check for leap year: (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)

        # Cumulative days at the start of each month (0 for Jan, 31 for Feb, etc.)
        cum_days = torch.cat(
            [
                torch.zeros(1, device=timestamp.device, dtype=days_in_month.dtype),
                days_in_month.cumsum(0)[:-1],
            ]
        )

        # Get cumulative days for the given month
        # month is 1-based (Jan=1), so subtract 1 for indexing
        month_idx = month.long() - 1
        cum_days_for_month = cum_days[month_idx]

        # Add 1 if leap year and month > 2
        leap_day = (is_leap & (month > 2)).long()

        doy = cum_days_for_month + day + leap_day
        return doy

    @staticmethod
    def standardize(
        data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Standardize the data."""
        return data.sub_(mean).div_(std)

    def prepare_input(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare input for the Tessera model from MaskedOlmoEarthSample."""
        if (s2_x := masked_olmoearth_sample.sentinel2_l2a) is None or (
            s1_x := masked_olmoearth_sample.sentinel1
        ) is None:
            raise ValueError("Tessera does not support single modality input")

        doy = self.calculate_day_of_year(masked_olmoearth_sample.timestamps)
        doy = repeat(doy, "b t  -> b  h w t 1", h=s2_x.shape[1], w=s2_x.shape[2])

        s2_x = s2_x[..., HELIOS_SENTINEL2_TESSERA_BANDS]
        s1_x = s1_x[..., HELIOS_SENTINEL1_TESSERA_BANDS]

        if self.use_pretrained_normalizer:
            s2_x = self.standardize(
                s2_x,
                torch.tensor(S2_BAND_MEAN, device=s2_x.device),
                torch.tensor(S2_BAND_STD, device=s2_x.device),
            )
            s1_x = self.standardize(
                s1_x,
                torch.tensor(S1_BAND_MEAN, device=s1_x.device),
                torch.tensor(S1_BAND_STD, device=s1_x.device),
            )

        s2_x = torch.cat([s2_x, doy], dim=-1)
        s1_x = torch.cat([s1_x, doy], dim=-1)
        return s2_x, s1_x

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through Tessera model for classification."""
        # Prepare input
        s2_x, s1_x = self.prepare_input(masked_olmoearth_sample)
        b, h, w, _, _ = s2_x.shape
        # Create an output tensor with the same shape as s2_x except the last dim, and set the last dim to the latent dim
        output_shape = tuple([b, h, w, self.model.latent_dim])
        output_features = torch.zeros(*output_shape, device=s2_x.device)

        for i, j in product(range(h), range(w)):
            s2_x_ij = s2_x[:, i, j, :, :]
            s1_x_ij = s1_x[:, i, j, :, :]
            output_features_ij = self.model(s2_x_ij, s1_x_ij)
            output_features[:, i, j, :] = output_features_ij

        if not spatial_pool:
            output_features = reduce(output_features, "b ... d -> b d", pooling)
        # Tessera pools across modality and time within the architecture
        # so we need to return the output features as is
        return output_features


@dataclass
class TesseraConfig(Config):
    """olmo_core style config for Tessera."""

    checkpoint_path: str | None = None
    use_pretrained_normalizer: bool = True

    def build(self) -> "Tessera":
        """Build the Tessera model from this config."""
        if self.checkpoint_path is None:
            default_weka_path = (
                "/weka/dfive-default/helios/models/tessera/tessera_checkpoint.pt"
            )
            # if the file exists use it otherwise give instructions to download it and ovveride to point to the local path
            if os.path.exists(default_weka_path):
                logger.info(
                    f"Using default Tessera checkpoint from {default_weka_path}"
                )
                self.checkpoint_path = default_weka_path
            else:
                logger.info(
                    f"Tessera checkpoint not found at {default_weka_path},\
                     please download it and override the checkpoint_path with \
                      --model.checkpoint_path=<path_to_checkpoint>"
                )
                self.checkpoint_path = default_weka_path
        return Tessera(
            checkpoint_path=self.checkpoint_path,
            use_pretrained_normalizer=self.use_pretrained_normalizer,
        )
