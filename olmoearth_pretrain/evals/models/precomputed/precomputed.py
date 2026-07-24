"""Baseline for precomputed embedding products (e.g. AlphaEarth / Google Satellite Embeddings).

Products like AlphaEarth are distributed as embeddings only, with no runnable
model. They are baked into eval dataset stores as data modalities (see
``Modality.GSE``), and this baseline reads them off the sample so the shared
eval harness (probes, KNN, metrics) treats them exactly like any forward-pass
model's embeddings.
"""

import logging
from dataclasses import dataclass, field

import torch
from einops import reduce
from torch import nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


class PrecomputedEmbedding(nn.Module):
    """Reads a precomputed embedding modality off the sample instead of running a model.

    The embedding product must be present on the eval dataset as the modality
    named by ``modality`` (baked in offline, e.g. by the embedding
    materializer); samples whose modality field is missing raise rather than
    silently degrading.
    """

    # Embeddings are per-pixel rasters, so the "token grid" is the pixel grid.
    patch_size: int = 1
    requires_timeseries: bool = False
    supports_multiple_modalities_at_once: bool = True

    def __init__(self, modality: str = Modality.GSE.name):
        """Initialize the precomputed embedding reader.

        Args:
            modality: Name of the Modality carrying the embedding product
                (e.g. "GSE" for AlphaEarth / Google Satellite Embeddings).
        """
        super().__init__()
        valid_names = Modality.names()
        if modality not in valid_names:
            raise ValueError(
                f"Unknown modality '{modality}'. Expected one of {sorted(valid_names)}"
            )
        self.modality = modality
        self.supported_modalities: list[str] = [modality]
        self.required_modalities: list[str] = [modality]
        # This module has no real weights, but the eval harness assumes models
        # have at least one parameter (device resolution via
        # `next(model.parameters())`, optimizer construction in the experiment
        # scaffolding), so keep a single dummy parameter.
        self._device_anchor = nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> torch.device:
        """Device the module has been moved to."""
        return self._device_anchor.device

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Return the precomputed embeddings carried by the sample.

        Args:
            masked_olmoearth_sample: Sample carrying the embedding modality
                with shape (B, H, W, T, C).
            pooling: Pooling used to collapse space when ``spatial_pool`` is False.
            spatial_pool: If True, keep the spatial grid and return (B, H, W, C);
                otherwise pool over space and return (B, C).
        """
        embeddings = getattr(masked_olmoearth_sample, self.modality, None)
        if embeddings is None:
            raise ValueError(
                f"Sample is missing precomputed embedding modality "
                f"'{self.modality}'. Confirm the eval dataset has this modality "
                f"baked in (see the embedding materializer) and that the task's "
                f"input_modalities includes it."
            )
        embeddings = torch.as_tensor(embeddings).float()
        if embeddings.ndim == 5:
            # Embedding products are annual (T=1); average defensively if a
            # store ever carries multiple timesteps.
            embeddings = reduce(embeddings, "b h w t c -> b h w c", "mean")
        if embeddings.ndim != 4:
            raise ValueError(
                f"Expected (B, H, W, T, C) or (B, H, W, C) for modality "
                f"'{self.modality}', got shape {tuple(embeddings.shape)}"
            )
        if not spatial_pool:
            embeddings = reduce(embeddings, "b h w c -> b c", pooling)
        return embeddings


@dataclass
class PrecomputedEmbeddingConfig(Config):
    """olmo_core style config for precomputed embedding baselines."""

    modality: str = field(default_factory=lambda: Modality.GSE.name)

    def build(self) -> "PrecomputedEmbedding":
        """Build the PrecomputedEmbedding reader from this config."""
        return PrecomputedEmbedding(modality=self.modality)
