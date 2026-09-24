"""Copernicus-FM baseline wrapper.

Copernicus-FM (Wang et al., https://github.com/zhu-xlab/Copernicus-FM) is a ViT
whose patch embedding is produced by a *dynamic hypernetwork*: instead of a fixed
stem per sensor, it generates patch-embedding weights from each band's central
wavelength and bandwidth. That makes it sensor-agnostic, but it is a
SINGLE-IMAGE encoder -- its `meta_info` time term encodes when one image was
taken, not a sequence. PLANTEUR is a 12-step monthly-mosaic series, so we run the
encoder per timestep and pool over time, the same convention `dinov3.py` uses for
the other single-image baselines.

Weights: https://huggingface.co/wangyi111/Copernicus-FM
"""

from dataclasses import dataclass, field
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from upath import UPath

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = getLogger(__name__)

# Sentinel-2 L2A in the OlmoEarth modality band order, which the eval pipeline
# feeds (see Modality.SENTINEL2_L2A in data/constants.py):
#   B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09   -- 12 bands, NOT 10.
# Central wavelength / bandwidth per band are Copernicus-FM's own constants
# (Copernicus-Bench/src/configs/dataset/cobench_lcz_s2.yaml gives the 10-band
# subset and their 13-band list supplies B01/B09), reordered to match.
S2_WAVELENGTHS: list[float] = [
    490,
    560,
    665,
    842,
    705,
    740,
    783,
    860,
    1610,
    2190,
    443,
    945,
]
S2_BANDWIDTHS: list[float] = [
    65,
    35,
    30,
    115,
    15,
    15,
    20,
    20,
    90,
    180,
    20,
    20,
]

# Sentinel-1 GRD (VV, VH) -- microwave, expressed as wavelength in nm for the
# same Fourier expansion. C-band ~5.405 GHz => ~55.5 mm.
S1_WAVELENGTHS: list[float] = [55465.0, 55465.0]
S1_BANDWIDTHS: list[float] = [100.0, 100.0]

_MODALITY_SPECTRA: dict[str, tuple[list[float], list[float]]] = {
    Modality.SENTINEL2_L2A.name: (S2_WAVELENGTHS, S2_BANDWIDTHS),
    Modality.SENTINEL1.name: (S1_WAVELENGTHS, S1_BANDWIDTHS),
}


class CopernicusFMWrapper(nn.Module):
    """Wrap CopernicusFMViT to the olmoearth eval-baseline interface."""

    # forward() iterates every modality present on the sample, so a task may
    # request more than one at a time (the evaluator gates multi-modality tasks
    # on this flag).
    supports_multiple_modalities_at_once = True

    supported_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]

    def __init__(
        self,
        load_directory: str,
        model_size: str = "base",
        kernel_size: int = 16,
        image_resolution: int = 224,
        use_pretrained_normalizer: bool = True,
    ) -> None:
        """Load a pretrained Copernicus-FM ViT from ``load_directory``.

        Args:
            load_directory: directory holding the CopernicusFM_ViT_*.pth weights.
            model_size: "base" or "large".
            kernel_size: patch-embedding kernel the hypernetwork generates for.
            image_resolution: side length each frame is bilinearly resized to
                before encoding; defaults to the 224x224 pretraining size.
            use_pretrained_normalizer: use the model's own band statistics.
        """
        super().__init__()
        self.kernel_size = kernel_size
        # The evaluator reads model.patch_size to build the token -> label
        # mapping (evaluator_callback.py: `if hasattr(model, "patch_size")`),
        # falling back to the task's patch_size when absent. Copernicus-FM
        # emits one token per kernel_size x kernel_size tile, so without this
        # the evaluator assumed the task's per-pixel patch_size=1 while the
        # model actually returns a (kernel_size)-strided grid.
        self.patch_size = kernel_size
        # Copernicus-FM was pretrained at 224x224 (model_vit.py img_size=224),
        # and its positional embeddings are interpolated from that 14x14 grid.
        # Following the house convention (croma/panopticon/clay), every frame is
        # bilinearly resized to the pretraining resolution before encoding.
        self.image_resolution = image_resolution
        self.use_pretrained_normalizer = use_pretrained_normalizer

        # imported lazily: the upstream repo is vendored on weka, not installed
        from olmoearth_pretrain.evals.models.copernicusfm.model_vit import (
            vit_base_patch16,
            vit_large_patch16,
        )

        builders = {"base": vit_base_patch16, "large": vit_large_patch16}
        if model_size not in builders:
            raise ValueError(
                f"unknown model_size {model_size!r}, want {list(builders)}"
            )

        ckpt_path = (
            UPath(load_directory) / f"CopernicusFM_ViT_{model_size}_varlang_e100.pth"
        )
        if not ckpt_path.exists():
            raise RuntimeError(f"Missing Copernicus-FM checkpoint: {ckpt_path}")

        # return_intermediate exposes the patch-token grid; forward_features
        # alone returns only the CLS token, which a segmentation probe cannot use.
        depth = {"base": 12, "large": 24}[model_size]
        self.model = builders[model_size](
            num_classes=0,
            global_pool=False,
            return_intermediate=True,
            intermediate_indices=[depth - 1],
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(
                f"Copernicus-FM: {len(missing)} missing keys, e.g. {missing[:5]}"
            )
        if unexpected:
            logger.warning(
                f"Copernicus-FM: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}"
            )
        self.model.eval()

    @property
    def embedding_size(self) -> int:
        """Width of the encoder's token embeddings."""
        return int(self.model.cls_token.shape[-1])

    def forward(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Encode each timestep independently, then pool over time.

        Copernicus-FM has no temporal axis, so a 12-step series is encoded frame
        by frame; pooling over the frames yields one embedding per pixel/patch.
        """
        per_timestep: list[torch.Tensor] = []
        for modality_name, (waves, bandwidths) in _MODALITY_SPECTRA.items():
            data = getattr(masked_olmoearth_sample, modality_name, None)
            if data is None:
                continue
            # stored as (b, h, w, t, c) -- see dinov3.py for the same unpacking
            num_timesteps = data.shape[3]
            for i in range(num_timesteps):
                frame = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")
                original_height = frame.shape[-2]
                new_height = (
                    self.kernel_size if original_height == 1 else self.image_resolution
                )
                frame = F.interpolate(
                    frame,
                    size=(new_height, new_height),
                    mode="bilinear",
                    align_corners=False,
                )
                meta_info = torch.full(
                    (frame.shape[0], 4), float("nan"), device=frame.device
                )  # lon/lat/time/area unknown -> model falls back to learned tokens
                _cls, intermediates = self.model.forward_features(
                    frame,
                    meta_info=meta_info,
                    wave_list=waves,
                    bandwidth=bandwidths,
                    language_embed=None,
                    input_mode="spectral",
                    kernel_size=self.kernel_size,
                )
                # (B, C, Hp, Wp) -> (B, Hp, Wp, C)
                feats = intermediates[-1].permute(0, 2, 3, 1).contiguous()
                per_timestep.append(feats)

        if not per_timestep:
            raise ValueError(
                "Copernicus-FM got no supported modality; expected one of "
                f"{list(_MODALITY_SPECTRA)}"
            )

        stacked = torch.stack(per_timestep, dim=0)
        if pooling == PoolingType.MAX:
            return stacked.max(dim=0).values
        return stacked.mean(dim=0)


@dataclass
class CopernicusFMConfig(Config):
    """olmo_core style config for CopernicusFMWrapper."""

    load_directory: str = "/weka/dfive-default/helios/models/copernicusfm"
    model_size: str = "base"
    kernel_size: int = 16
    image_resolution: int = 224
    use_pretrained_normalizer: bool = True
    _unused: list[str] = field(default_factory=list)

    def build(self) -> CopernicusFMWrapper:
        """Build the Copernicus-FM model."""
        return CopernicusFMWrapper(
            load_directory=self.load_directory,
            model_size=self.model_size,
            kernel_size=self.kernel_size,
            image_resolution=self.image_resolution,
            use_pretrained_normalizer=self.use_pretrained_normalizer,
        )
