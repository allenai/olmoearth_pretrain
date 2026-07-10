"""Model that adds the pixel decoders on top of the latent-MIM setup.

:class:`DualResLatentMIM` is a standalone model (it mirrors the small
:class:`~olmoearth_pretrain.nn.latent_mim.LatentMIM` body rather than subclassing it)
that runs the usual latent-MIM coarse path -- encoder, predictor/decoder, optional
coarse reconstructor, EMA target encoder -- and additionally runs the two
pixel-representation heads:

* :class:`~olmoearth_pretrain.nn.pixel_decoder.PixelReconstructionDecoder` (SSL), and
* :class:`~olmoearth_pretrain.nn.pixel_decoder.PixelMapProbe` (auxiliary supervised).

Its ``forward`` returns the standard latent-MIM 5-tuple **plus** the (weighted) pixel
loss as a 6th element, so the train module adds it to the coarse loss directly -- no
stashed attribute. The coarse decoder is untouched; it still cross-attends only to the
pooled per-patch main representations.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.dual_res_encoder import (
    DualResEncoder,
    PixelModalityState,
)
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.pixel_decoder import (
    PixelMapProbe,
    PixelReconstructionDecoder,
)
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


class DualResLatentMIM(nn.Module, DistributedMixins):
    """Latent-MIM model with additional pixel reconstruction + map-probe heads."""

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: DualResEncoder,
        decoder: nn.Module,
        reconstructor: nn.Module | None = None,
        pixel_reconstruction_decoder: PixelReconstructionDecoder | None = None,
        pixel_map_probe: PixelMapProbe | None = None,
        pixel_recon_weight: float = 1.0,
        pixel_map_weight: float = 1.0,
    ) -> None:
        """Initialize the dual-resolution latent-MIM model.

        Args:
            encoder: The dual-resolution encoder.
            decoder: The coarse latent-MIM predictor/decoder.
            reconstructor: Optional coarse MAE reconstructor.
            pixel_reconstruction_decoder: Optional SSL pixel reconstruction head.
            pixel_map_probe: Optional auxiliary supervised map probe.
            pixel_recon_weight: Weight for the pixel reconstruction loss.
            pixel_map_weight: Weight for the map-probe loss.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.pixel_reconstruction_decoder = pixel_reconstruction_decoder
        self.pixel_map_probe = pixel_map_probe
        self.pixel_recon_weight = pixel_recon_weight
        self.pixel_map_weight = pixel_map_weight

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        TokensAndMasks | None,
        dict[str, Any],
        Tensor | None,
    ]:
        """Forward pass: coarse latent-MIM path + pixel heads.

        Returns the latent-MIM 5-tuple plus the (weighted) pixel loss (or ``None``).
        """
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        # The pixel branch is not a coarse-decoder kwarg -- pop it before unpacking.
        pixel_branch = output_dict.pop("pixel_branch", {})
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        extra_metrics: dict[str, Any] = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats

        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )

        pixel_loss = self._pixel_loss(pixel_branch, x, patch_size, extra_metrics)

        return (
            latent,
            decoded,
            latent_projected_and_pooled,
            reconstructed,
            extra_metrics,
            pixel_loss,
        )

    def _pixel_loss(
        self,
        pixel_branch: dict[str, PixelModalityState],
        x: MaskedOlmoEarthSample,
        patch_size: int,
        extra_metrics: dict[str, Any],
    ) -> Tensor | None:
        total: Tensor | None = None
        components: dict[str, float] = {}
        if self.pixel_reconstruction_decoder is not None:
            recon = self.pixel_recon_weight * self.pixel_reconstruction_decoder(
                pixel_branch, x, patch_size
            )
            total = recon if total is None else total + recon
            components["recon"] = float(recon.detach())
        if self.pixel_map_probe is not None:
            mapl = self.pixel_map_weight * self.pixel_map_probe(
                pixel_branch, x, patch_size
            )
            total = mapl if total is None else total + mapl
            components["map"] = float(mapl.detach())
        if components:
            extra_metrics["pixel_loss"] = components
        return total

    def apply_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Apply FSDP to the model."""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config: dict[str, Any] = dict(mesh=dp_mesh, mp_policy=mp_policy)

        self.encoder.apply_fsdp(**fsdp_config)
        self.decoder.apply_fsdp(**fsdp_config)
        self.target_encoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        if self.pixel_reconstruction_decoder is not None:
            fully_shard(self.pixel_reconstruction_decoder, **fsdp_config)
        if self.pixel_map_probe is not None:
            fully_shard(self.pixel_map_probe, **fsdp_config)
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the coarse path (pixel decoders left eager)."""
        self.encoder.apply_compile()
        self.decoder.apply_compile()
        self.target_encoder.apply_compile()


@dataclass
class DualResLatentMIMConfig(LatentMIMConfig):
    """Configuration for :class:`DualResLatentMIM`.

    Adds the pixel reconstruction + map-probe heads on top of :class:`LatentMIMConfig`.
    ``encoder_config`` must be a
    :class:`~olmoearth_pretrain.nn.dual_res_encoder.DualResEncoderConfig`.
    """

    pixel_reconstruction: bool = True
    pixel_recon_depth: int = 2
    pixel_recon_num_heads: int = 4
    pixel_recon_mlp_ratio: float = 4.0
    pixel_recon_weight: float = 1.0
    pixel_recon_location_ratio: float = 1.0
    map_targets: dict[str, str] = field(default_factory=dict)
    map_num_classes: dict[str, int] = field(default_factory=dict)
    pixel_map_weight: float = 1.0

    def build(self) -> "DualResLatentMIM":
        """Build the dual-resolution latent-MIM model."""
        self.validate()
        encoder = self.encoder_config.build()
        if not isinstance(encoder, DualResEncoder):
            raise ValueError("DualResLatentMIM requires a DualResEncoderConfig encoder")
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        pixel_reconstruction_decoder = None
        if self.pixel_reconstruction:
            pixel_reconstruction_decoder = PixelReconstructionDecoder(
                supported_modality_names=encoder.supported_modality_names,
                pixel_embedding_size=encoder.pixel_embedding_size,
                num_heads=self.pixel_recon_num_heads,
                mlp_ratio=self.pixel_recon_mlp_ratio,
                depth=self.pixel_recon_depth,
                location_ratio=self.pixel_recon_location_ratio,
                tokenization_config=encoder.tokenization_config,
            )
        pixel_map_probe = None
        if self.map_targets:
            pixel_map_probe = PixelMapProbe(
                pixel_embedding_size=encoder.pixel_embedding_size,
                map_targets=self.map_targets,
                num_classes=self.map_num_classes,
            )
        return DualResLatentMIM(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            pixel_reconstruction_decoder=pixel_reconstruction_decoder,
            pixel_map_probe=pixel_map_probe,
            pixel_recon_weight=self.pixel_recon_weight,
            pixel_map_weight=self.pixel_map_weight,
        )
