"""Generate text-embedding targets for map modalities.

Replaces target-encoder outputs with pre-computed semantic text embeddings
for discrete/categorical map modalities (WorldCover, OSM raster, WorldCereal).
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks

logger = logging.getLogger(__name__)

# Modalities where each pixel is a single discrete class ID stored in one band.
# Un-normalize by multiplying by (max - min) + min, then round to int.
_DISCRETE_MODALITIES = {"worldcover"}

# Modalities with multiple binary/soft channels, one embedding per channel.
_MULTICHANNEL_MODALITIES = {"openstreetmap_raster", "worldcereal"}

# Normalization ranges from predefined.json (min, max) used to recover raw values.
_NORM_RANGES: dict[str, tuple[float, float]] = {
    "worldcover": (0.0, 100.0),
    "openstreetmap_raster": (0.0, 1.0),
    "worldcereal": (0.0, 100.0),
}


class TextEmbeddingTargetGenerator(nn.Module):
    """Produces frozen text-embedding targets for map modalities.

    For discrete modalities (WorldCover): pixels → class IDs → per-class embeddings,
    area-weighted average within each patch.

    For multi-channel modalities (OSM, WorldCereal): average channel activations
    within each patch, then weight-combine the per-channel embeddings.
    """

    def __init__(self, embeddings_path: str, modalities: list[str]):
        """Initialize with pre-computed embeddings.

        Args:
            embeddings_path: path to .pt file from precompute_text_embeddings.py
            modalities: list of modality names to generate text targets for
        """
        super().__init__()
        self.modalities = modalities
        data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.embedding_dim = 0

        for modality in modalities:
            if modality not in data:
                raise ValueError(
                    f"No pre-computed embeddings for modality '{modality}' "
                    f"in {embeddings_path}"
                )
            mod_data = data[modality]
            embs = mod_data["embeddings"]  # [num_classes, D]
            self.embedding_dim = embs.shape[-1]

            if modality in _DISCRETE_MODALITIES:
                class_ids = mod_data["class_ids"]  # [num_classes]
                max_id = class_ids.max().item()
                # Dense lookup table: index by raw class ID → embedding
                dense = torch.zeros(max_id + 1, embs.shape[-1])
                for i, cid in enumerate(class_ids):
                    dense[cid] = embs[i]
                self.register_buffer(f"{modality}_lookup", dense, persistent=False)
            else:
                # Multi-channel: embeddings indexed by channel order
                self.register_buffer(f"{modality}_embeddings", embs, persistent=False)

        # Freeze all parameters (there shouldn't be any, but just in case)
        for p in self.parameters():
            p.requires_grad = False

    def _compute_discrete_targets(
        self, modality: str, raw_data: Tensor, patch_size: int
    ) -> Tensor:
        """Compute targets for a discrete single-band modality (e.g. WorldCover).

        Args:
            modality: modality name
            raw_data: normalized raster [B, H, W, T=1, bands=1]
            patch_size: spatial patch size

        Returns:
            [B, P_H, P_W, 1, 1, D] text embedding targets
        """
        norm_min, norm_max = _NORM_RANGES[modality]
        lookup = getattr(self, f"{modality}_lookup")
        max_class_id = lookup.shape[0] - 1

        # Un-normalize: raw_class = norm * (max - min) + min
        class_ids = (raw_data * (norm_max - norm_min) + norm_min).round().long()
        # [B, H, W, 1, 1] → [B, H, W]
        class_ids = class_ids[:, :, :, 0, 0]
        class_ids = class_ids.clamp(0, max_class_id)

        # Patchify: [B, P_H, P_W, p*p]
        class_ids = rearrange(
            class_ids,
            "b (ph p1) (pw p2) -> b ph pw (p1 p2)",
            p1=patch_size,
            p2=patch_size,
        )

        # Look up embeddings: [B, P_H, P_W, p*p, D]
        patch_embs = lookup[class_ids]

        # Area-weighted average then L2-normalize: [B, P_H, P_W, D]
        targets = F.normalize(patch_embs.mean(dim=-2), dim=-1)

        # Add back T and band_set dims: [B, P_H, P_W, 1, 1, D]
        return targets[:, :, :, None, None, :]

    def _compute_multichannel_targets(
        self, modality: str, raw_data: Tensor, patch_size: int
    ) -> Tensor:
        """Compute targets for a multi-channel modality (e.g. OSM raster, WorldCereal).

        Args:
            modality: modality name
            raw_data: normalized raster [B, H, W, T=1, C]
            patch_size: spatial patch size

        Returns:
            [B, P_H, P_W, 1, 1, D] text embedding targets
        """
        embs = getattr(self, f"{modality}_embeddings")  # [C, D]

        # [B, H, W, 1, C] → [B, H, W, C]
        data = raw_data[:, :, :, 0, :]

        # Patchify and average across pixels: [B, P_H, P_W, C]
        data = rearrange(
            data,
            "b (ph p1) (pw p2) c -> b ph pw (p1 p2) c",
            p1=patch_size,
            p2=patch_size,
        )
        avg = data.mean(dim=-2)  # [B, P_H, P_W, C]

        # Weight-combine channel embeddings: [B, P_H, P_W, D]
        targets = torch.einsum("bpqc,cd->bpqd", avg.float(), embs.float())
        targets = F.normalize(targets, dim=-1)

        # Add back T and band_set dims: [B, P_H, P_W, 1, 1, D]
        return targets[:, :, :, None, None, :]

    def compute_targets(
        self, modality: str, raw_data: Tensor, patch_size: int
    ) -> Tensor:
        """Compute text embedding targets for a modality.

        Args:
            modality: modality name
            raw_data: normalized raster data from the batch
            patch_size: spatial patch size for patchification

        Returns:
            [B, P_H, P_W, T, BandSets, D] text embedding targets
        """
        if modality in _DISCRETE_MODALITIES:
            return self._compute_discrete_targets(modality, raw_data, patch_size)
        elif modality in _MULTICHANNEL_MODALITIES:
            return self._compute_multichannel_targets(modality, raw_data, patch_size)
        else:
            raise ValueError(f"Unknown text-target modality type: {modality}")

    @torch.no_grad()
    def replace_targets(
        self,
        target_output: TokensAndMasks,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> TokensAndMasks:
        """Replace target-encoder outputs with text-embedding targets.

        For each modality in self.modalities that is present in the batch,
        replaces the corresponding field in target_output.

        Args:
            target_output: target encoder output (TokensAndMasks)
            batch: the raw (possibly masked) batch with pixel-level data
            patch_size: spatial patch size

        Returns:
            Updated TokensAndMasks with text embedding targets for map modalities.
        """
        # Ensure buffers are on the same device as the batch data
        device = target_output.device
        if next(self.buffers()).device != device:
            self.to(device)

        updates: dict[str, Tensor] = {}
        for modality in self.modalities:
            raw_data = getattr(batch, modality, None)
            if raw_data is None:
                continue

            spec = Modality.get(modality)
            effective_patch_size = patch_size * spec.image_tile_size_factor
            targets = self.compute_targets(modality, raw_data, effective_patch_size)

            existing = getattr(target_output, modality, None)
            if existing is not None:
                targets = targets.to(dtype=existing.dtype, device=existing.device)

            updates[modality] = targets

        return target_output._replace(**updates)


@dataclass
class TextEmbeddingTargetConfig(Config):
    """Configuration for text embedding targets.

    Args:
        embeddings_path: path to pre-computed .pt file
        modalities: list of modality names to use text targets for
    """

    embeddings_path: str = ""
    modalities: list[str] = field(default_factory=list)

    def build(self) -> TextEmbeddingTargetGenerator | None:
        """Build the text embedding target generator."""
        if not self.embeddings_path or not self.modalities:
            return None
        return TextEmbeddingTargetGenerator(self.embeddings_path, self.modalities)
