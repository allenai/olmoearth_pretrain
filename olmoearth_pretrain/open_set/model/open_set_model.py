"""Full open-set segmentation model.

Wires:
- A frozen OlmoEarth encoder
- A text-conditioned cross-attention decoder
- A dot-product classification head over a (text-class, image-pixel) embedding pair

The forward returns per-pixel binary logits at a caller-specified target
size. Output is a single channel (one binary mask per text query).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.open_set.model.cross_attn_decoder import (
    CrossAttnDecoder,
    CrossAttnDecoderConfig,
)
from olmoearth_pretrain.open_set.model.encoder_wrapper import (
    FrozenOlmoEarthEncoder,
    load_encoder_from_distributed_checkpoint,
)

logger = getLogger(__name__)

# Preference order for which spatial modality's grid we use to produce the
# pixel-level dot-product output. We pick the first one that is present on
# the batch — so when the modality subsampler keeps only Landsat, we use
# Landsat's grid.
DEFAULT_REFERENCE_MODALITIES: tuple[str, ...] = (
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    "naip_10",
    "naip",
)


@dataclass
class OpenSetSegmenterConfig(Config):
    """Configuration for :class:`OpenSetSegmenter`.

    Attributes:
        decoder_config: The cross-attention decoder configuration.
        text_dim: Dimensionality of the text encoder's embeddings.
        head_dim: Dimensionality of the joint pixel/class embedding space the
            dot-product head operates in. Defaults to the decoder's ``dim``.
        reference_modalities: Preference order for the spatial modality whose
            patch grid drives the output resolution. The first present
            modality wins.
        upsample_mode: Upsampling mode for going from patch grid to output
            mask resolution.
    """

    decoder_config: CrossAttnDecoderConfig = field(
        default_factory=CrossAttnDecoderConfig
    )
    text_dim: int = 1152
    head_dim: int | None = None
    reference_modalities: tuple[str, ...] = DEFAULT_REFERENCE_MODALITIES
    upsample_mode: str = "bilinear"

    def build(
        self,
        encoder: FrozenOlmoEarthEncoder,
    ) -> OpenSetSegmenter:
        """Build the open-set segmenter."""
        return OpenSetSegmenter(self, encoder=encoder)


@dataclass
class OpenSetModelConfig(Config):
    """Top-level model config used by the olmo-core ``train(config)`` flow.

    Calling ``build()`` (no args) loads the OlmoEarth encoder from
    ``checkpoint_path``, wraps it in :class:`FrozenOlmoEarthEncoder`, and
    constructs the :class:`OpenSetSegmenter`. This matches the contract
    expected by ``OlmoEarthExperimentConfig.model.build()`` so that no
    additional plumbing is needed at launch time.

    Attributes:
        checkpoint_path: Path to a distributed olmo-core checkpoint
            (``step{N}/`` containing ``config.json`` + ``model_and_optim/``)
            or to a consolidated checkpoint (``config.json`` + ``weights.pth``).
        decoder_config: Cross-attention decoder configuration.
        text_dim: Dimensionality of the text encoder's embeddings (must match
            the encoder configured on the train module).
        head_dim: Joint pixel/class embedding dim. Defaults to decoder dim.
        reference_modalities: Spatial-modality preference order.
        upsample_mode: Mode passed to ``F.interpolate`` when downsampling.
        trainable_encoder: If True, do *not* freeze the encoder. Default False.
    """

    checkpoint_path: str = ""
    decoder_config: CrossAttnDecoderConfig = field(
        default_factory=CrossAttnDecoderConfig
    )
    text_dim: int = 1152
    head_dim: int | None = None
    reference_modalities: tuple[str, ...] = DEFAULT_REFERENCE_MODALITIES
    upsample_mode: str = "bilinear"
    trainable_encoder: bool = False

    def validate(self) -> None:
        """Validate the model config."""
        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path is required — pass it via "
                "--model.checkpoint_path=/path/to/step{N}"
            )

    def build(self) -> OpenSetSegmenter:
        """Load the frozen encoder and wire up the segmenter."""
        self.validate()
        encoder = load_encoder_from_distributed_checkpoint(self.checkpoint_path)
        frozen = FrozenOlmoEarthEncoder(encoder, trainable=self.trainable_encoder)
        segmenter_config = OpenSetSegmenterConfig(
            decoder_config=self.decoder_config,
            text_dim=self.text_dim,
            head_dim=self.head_dim,
            reference_modalities=self.reference_modalities,
            upsample_mode=self.upsample_mode,
        )
        return segmenter_config.build(frozen)


class OpenSetSegmenter(nn.Module):
    """Encoder (frozen) + cross-attention decoder + dot-product head.

    Each refined image token predicts a ``max_patch_size × max_patch_size``
    grid of pixel embeddings. The grid unfolds to
    ``(P_H × max_patch_size, P_W × max_patch_size)`` per image, where
    ``P_H = H_image / actual_patch_size``. So:

    - When ``actual_patch_size == max_patch_size``: output matches image
      resolution natively. No interpolation is applied at the loss.
    - When ``actual_patch_size < max_patch_size``: output is at higher than
      image resolution and gets bilinearly downsampled to ``target_size``.

    This mirrors the ``SupervisionHead`` pattern on ``gabi/supervision`` —
    output is *never* upsampled to reach the target.
    """

    def __init__(
        self,
        config: OpenSetSegmenterConfig,
        encoder: FrozenOlmoEarthEncoder,
        max_patch_size: int | None = None,
    ) -> None:
        """Initialize the open-set segmenter.

        Args:
            config: Segmenter configuration.
            encoder: Frozen OlmoEarth encoder.
            max_patch_size: Override the encoder's ``max_patch_size``. Useful
                for testing with stub encoders. Defaults to
                ``encoder.max_patch_size``.
        """
        super().__init__()
        self.config = config
        self.encoder = encoder

        self.decoder: CrossAttnDecoder = config.decoder_config.build(
            image_dim=encoder.embedding_dim,
            text_dim=config.text_dim,
        )

        head_dim = config.head_dim or config.decoder_config.dim
        self.head_dim = head_dim
        self.max_patch_size = (
            max_patch_size if max_patch_size is not None else encoder.max_patch_size
        )
        # Each token predicts a max_patch_size² grid of head_dim-dim embeddings.
        self.pixel_proj = nn.Linear(
            self.decoder.output_dim, head_dim * self.max_patch_size**2
        )
        self.text_proj = nn.Linear(config.text_dim, head_dim)

    @property
    def reference_modalities(self) -> tuple[str, ...]:
        """Preference order for the spatial output reference modality."""
        return self.config.reference_modalities

    def _select_reference_modality(self, shapes: dict[str, tuple[int, ...]]) -> str:
        """Pick the first preferred spatial modality that is present."""
        for name in self.reference_modalities:
            if name in shapes and Modality.get(name).is_spatial:
                return name
        # Fallback: any spatial modality present.
        for name in shapes:
            if Modality.get(name).is_spatial:
                return name
        raise RuntimeError(
            "No spatial modality present on the batch — cannot produce a mask. "
            f"Available modalities: {list(shapes)}"
        )

    @staticmethod
    def _slice_modality(
        flat_tokens: torch.Tensor,
        shapes: dict[str, tuple[int, ...]],
        modality: str,
    ) -> torch.Tensor:
        """Slice ``flat_tokens`` to recover the named modality's tokens.

        Returns a tensor of shape ``[B, *shapes[modality], D]``.
        """
        offset = 0
        d = flat_tokens.shape[-1]
        for name, shape in shapes.items():
            n = 1
            for s in shape:
                n *= s
            if name == modality:
                slice_ = flat_tokens[:, offset : offset + n]
                return slice_.reshape(flat_tokens.shape[0], *shape, d)
            offset += n
        raise KeyError(f"Modality {modality!r} not found in shapes")

    @staticmethod
    def _to_pixel_grid(modality_tokens: torch.Tensor) -> torch.Tensor:
        """Collapse temporal and bandset dims, return ``[B, P_H, P_W, D]``.

        ``modality_tokens`` is the per-modality reshape coming from
        ``_slice_modality``. We accept either:
            ``[B, P_H, P_W, T, BS, D]`` — spatio-temporal modality, or
            ``[B, P_H, P_W, BS, D]`` — spatial-only modality (no time axis).
        """
        if modality_tokens.ndim == 6:
            return modality_tokens.mean(dim=(3, 4))
        if modality_tokens.ndim == 5:
            return modality_tokens.mean(dim=3)
        if modality_tokens.ndim == 4:
            return modality_tokens
        raise ValueError(
            f"Unexpected reference-modality token shape {tuple(modality_tokens.shape)}"
        )

    def forward(
        self,
        sample: MaskedOlmoEarthSample,
        patch_size: int,
        text_tokens: torch.Tensor,
        text_pooled: torch.Tensor,
        text_attn_mask: torch.Tensor | None = None,
        target_size: tuple[int, int] | None = None,
        reference_modality: str | None = None,
    ) -> torch.Tensor:
        """Run the full open-set forward.

        Args:
            sample: Masked input sample (encoder is frozen so masks are not
                used for MIM-style training; in practice the caller will pass
                an unmasked sample).
            patch_size: Patch size to use for tokenization.
            text_tokens: ``[N_classes, L, D_text]`` per-token text embeddings.
                ``N_classes`` is the number of distinct text queries this
                forward should answer (positives + hard negatives across the
                batch). All ``N_classes`` masks are produced for every image
                in the batch.
            text_pooled: ``[N_classes, D_text]`` pooled text embeddings.
            text_attn_mask: Optional ``[N_classes, L]`` mask for text padding.
            target_size: ``(H_out, W_out)`` of the output mask. Defaults to
                the patch grid dimensions (no upsampling).
            reference_modality: Override which modality's spatial grid to use.
                Defaults to the first present modality from
                ``self.reference_modalities``.

        Returns:
            Per-pixel binary logits of shape ``[N_classes, B, H_out, W_out]``.
        """
        encoder_tokens, context_mask, shapes = self.encoder(sample, patch_size)
        # encoder_tokens: [B, N_enc, D_enc], context_mask: [B, N_enc] bool

        b = encoder_tokens.shape[0]
        n_classes = text_tokens.shape[0]

        # Replicate image tokens across class queries; replicate text tokens
        # across batch. Resulting batch dim is B * N_classes — the decoder
        # then sees one (image, class) pair per row.
        image_tokens_rep = encoder_tokens.unsqueeze(0).expand(n_classes, -1, -1, -1)
        image_tokens_rep = rearrange(image_tokens_rep, "c b n d -> (c b) n d")
        context_mask_rep = context_mask.unsqueeze(0).expand(n_classes, -1, -1)
        context_mask_rep = rearrange(context_mask_rep, "c b n -> (c b) n")

        text_tokens_rep = text_tokens.unsqueeze(1).expand(-1, b, -1, -1)
        text_tokens_rep = rearrange(text_tokens_rep, "c b l d -> (c b) l d")
        if text_attn_mask is not None:
            text_attn_mask_rep = text_attn_mask.unsqueeze(1).expand(-1, b, -1)
            text_attn_mask_rep = rearrange(text_attn_mask_rep, "c b l -> (c b) l")
        else:
            text_attn_mask_rep = None

        refined = self.decoder(
            image_tokens=image_tokens_rep,
            text_tokens=text_tokens_rep,
            text_attn_mask=text_attn_mask_rep,
            image_attn_mask=context_mask_rep,
        )  # [(C*B), N_enc, D_dec]

        # Slice the reference modality and pool to a 2D grid.
        ref_name = reference_modality or self._select_reference_modality(shapes)
        ref_tokens = self._slice_modality(refined, shapes, ref_name)
        ref_grid = self._to_pixel_grid(ref_tokens)  # [(C*B), P_H, P_W, D_dec]

        # Each token predicts a max_patch_size² grid of pixel embeddings.
        # Unfold spatial dims so we end up at (P_H*mps, P_W*mps) resolution.
        mps = self.max_patch_size
        raw = self.pixel_proj(ref_grid)  # [(C*B), P_H, P_W, head_dim * mps²]
        pixel_emb = rearrange(
            raw,
            "n ph pw (d i j) -> n (ph i) (pw j) d",
            d=self.head_dim,
            i=mps,
            j=mps,
        )  # [(C*B), P_H*mps, P_W*mps, head_dim]
        out_h, out_w = pixel_emb.shape[1], pixel_emb.shape[2]

        # text_pooled: [N_classes, D_text]; project then expand across batch.
        text_emb = self.text_proj(text_pooled)  # [N_classes, head_dim]
        text_emb = text_emb.unsqueeze(1).expand(-1, b, -1)
        text_emb = rearrange(text_emb, "c b d -> (c b) d")

        logits = torch.einsum("nhwd,nd->nhw", pixel_emb, text_emb)
        # logits: [(C*B), P_H*mps, P_W*mps]

        # Downsample-only: when actual_patch_size < max_patch_size we are at
        # finer-than-target resolution and need to come down to target_size.
        # When actual_patch_size == max_patch_size the shapes already match.
        if target_size is not None and target_size != (out_h, out_w):
            if target_size[0] > out_h or target_size[1] > out_w:
                raise ValueError(
                    f"target_size {target_size} exceeds the model's native "
                    f"output resolution ({out_h}, {out_w}). The supervision "
                    "pattern requires target_size <= P_H*max_patch_size — "
                    "this happens automatically when actual_patch_size "
                    "<= max_patch_size."
                )
            logits = F.interpolate(
                logits.unsqueeze(1).float(),
                size=target_size,
                mode=self.config.upsample_mode,
                align_corners=False
                if self.config.upsample_mode == "bilinear"
                else None,
            ).squeeze(1)

        # Reshape back to [N_classes, B, H, W].
        logits = rearrange(logits, "(c b) h w -> c b h w", c=n_classes, b=b)
        return logits
