"""Simple set up of latent predictor."""

import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHead,
    SupervisionHeadConfig,
)
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


class FrozenTargetProjection(nn.Module):
    """Frozen projection-only target encoder.

    When every modality exits at depth 0 and the target is never EMA-updated
    (``ema_decay=(1.0, 1.0)``), the full target-encoder copy is dead weight: the
    encoder's ``forward`` skips ``apply_attn`` entirely and the target is just the
    frozen initial projection. This module deepcopies only the pieces that
    exit-0 actually runs (``patch_embeddings`` + optional ``embedding_projector``),
    so the transformer blocks are never copied, sharded, all-gathered, or saved.

    ``project_aggregated`` is intentionally not computed: both latent-MIM train
    modules only consume ``tokens_and_masks`` from the target output.
    """

    def __init__(self, encoder: nn.Module):
        """Copy and freeze the projection submodules of ``encoder``."""
        super().__init__()
        self.patch_embeddings = deepcopy(encoder.patch_embeddings)
        self.embedding_projector = deepcopy(encoder.embedding_projector)
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute exit-0 targets: patch embeddings + optional projector."""
        if token_exit_cfg is not None and any(
            exit_depth > 0 for exit_depth in token_exit_cfg.values()
        ):
            raise ValueError(
                "FrozenTargetProjection only supports token_exit_cfg with all "
                f"exit depths 0, got {token_exit_cfg}. Use the full target "
                "encoder (projection_only_target=False) for deeper exits."
            )
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)
        output = TokensAndMasks(**patchified_tokens_and_masks)
        if self.embedding_projector is not None:
            output = self.embedding_projector(output)
        return {"tokens_and_masks": output}


class LatentMIM(nn.Module, DistributedMixins):
    """Latent MIM Style."""

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: torch.nn.Module | None = None,
        supervision_head: SupervisionHead | None = None,
        projection_supervision_heads: dict[int, SupervisionHead] | None = None,
        projection_only_target: bool = False,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: Optional reconstructor for auto-encoding.
            supervision_head: Optional supervision head for direct supervision
                of decode-only modalities from decoder output.
            projection_supervision_heads: Optional per-prefix supervision heads
                reading the encoder's DETACHED low-dim register projection
                (``projected_registers``) instead of the register grid, keyed by
                Matryoshka prefix width (head ``d`` reads
                ``projected_registers[..., :d]``). Their gradients train the
                projection (and themselves) only -- never the encoder. Requires the
                encoder's ``register_projection_dims``.
            projection_only_target: If True, the target encoder is only the frozen
                initial projection (patch embeddings + optional embedding projector)
                instead of a full copy of the encoder. Only valid when all token
                exit depths are 0 and the target is never EMA-updated
                (ema_decay=(1.0, 1.0)).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.supervision_head = supervision_head
        # ModuleDict keys must be strings; keep prefix widths as str(dim).
        self.projection_supervision_heads = (
            nn.ModuleDict(
                {str(dim): head for dim, head in projection_supervision_heads.items()}
            )
            if projection_supervision_heads
            else None
        )
        if projection_only_target:
            self.target_encoder: nn.Module = FrozenTargetProjection(self.encoder)
        else:
            self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        TokensAndMasks | None,
        dict[str, Any],
        dict[str, torch.Tensor] | None,
        dict[str, Any] | None,
    ]:
        """Forward pass for the Latent MIM Style.

        Returns:
            latent: embeddings from encoder
            decoded: predictions from decoder for masked tokens
            latent_projected_and_pooled: pooled tokens for contrastive loss
            reconstructed: MAE predictions if enabled
            extra_metrics: additional metrics to log
            supervision_preds: per-modality supervision predictions (or None)
            projection_outputs: detached-student outputs when the encoder has a
                register projection (else None): the teacher ``registers``, the
                student ``projected_registers``, and the student's
                ``supervision_preds`` (or None). Consumed by the train module's
                distillation / projection-supervision losses. The two grids are
                FLATTENED to ``[B, N, D]`` here: every consumer (uniformity, Gram,
                cosine) is relational over cells and takes a token sequence.
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        # The decoder reads only the registers; the student projection is for the
        # train module's losses (and evals), never a decoder input.
        projected_registers = decoder_kwargs.pop("projected_registers", None)
        extra_metrics = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )

        supervision_preds = None
        if self.supervision_head is not None:
            if getattr(self.supervision_head, "register_supervision", False):
                # The encoder already hands back the grid as [B, n_h, n_w, D], which
                # is the shape the heads expect.
                supervision_preds = self.supervision_head(
                    decoded, x, register_grid=decoder_kwargs.get("registers")
                )
            else:
                supervision_preds = self.supervision_head(decoded, x)

        # Surfaced whenever a register bottleneck ran, not only when a distillation
        # student exists: losses that shape the register grid itself (e.g. the
        # uniformity term) apply to non-distilled models too, and the grid is
        # otherwise visible only inside decoder_kwargs.
        registers = decoder_kwargs.get("registers")
        projection_outputs: dict | None = None
        if registers is not None:
            projection_outputs = {
                "registers": rearrange(registers, "b h w d -> b (h w) d"),
                "projected_registers": None,
                "supervision_preds": None,
            }
        if projected_registers is not None:
            projection_supervision_preds = None
            if self.projection_supervision_heads is not None:
                # Same grid as the registers (the student mirrors the primary's grid),
                # already shaped [B, n_h, n_w, d] by the encoder. Head d reads the first
                # d dims (the Matryoshka prefix), so every listed width is supervised as
                # a self-sufficient embedding. The student input is already detached
                # inside the encoder, so these gradients stop at the projection.
                projected_grid = projected_registers
                projection_supervision_preds = {
                    dim_str: head(
                        decoded, x, register_grid=projected_grid[..., : int(dim_str)]
                    )
                    for dim_str, head in self.projection_supervision_heads.items()
                }
            assert projection_outputs is not None, (
                "a projection student cannot exist without a register bottleneck"
            )
            projection_outputs["projected_registers"] = rearrange(
                projected_registers, "b h w d -> b (h w) d"
            )
            projection_outputs["supervision_preds"] = projection_supervision_preds

        return (
            latent,
            decoded,
            latent_projected_and_pooled,
            reconstructed,
            extra_metrics,
            supervision_preds,
            projection_outputs,
        )

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
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        self.encoder.apply_fsdp(**fsdp_config)
        self.decoder.apply_fsdp(**fsdp_config)
        if isinstance(self.target_encoder, FrozenTargetProjection):
            # Tiny frozen module: shard as a single unit (one all-gather per step)
            # instead of the per-block wrapping a full encoder copy would get.
            fully_shard(self.target_encoder, **fsdp_config)
        else:
            self.target_encoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        if self.supervision_head is not None:
            fully_shard(self.supervision_head, **fsdp_config)
        if self.projection_supervision_heads is not None:
            for head in self.projection_supervision_heads.values():
                fully_shard(head, **fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Applying torch.compile to the model")
        self.encoder.apply_compile()
        logger.info("Applied torch.compile to the encoder")
        self.decoder.apply_compile()
        logger.info("Applied torch.compile to the decoder")
        if hasattr(self.target_encoder, "apply_compile"):
            self.target_encoder.apply_compile()
            logger.info("Applied torch.compile to the target encoder")
        if self.supervision_head is not None:
            self.supervision_head = torch.compile(self.supervision_head)
            logger.info("Applied torch.compile to the supervision head")
        if self.projection_supervision_heads is not None:
            for dim_str, head in self.projection_supervision_heads.items():
                self.projection_supervision_heads[dim_str] = torch.compile(head)
            logger.info("Applied torch.compile to the projection supervision heads")


@dataclass
class LatentMIMConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: Config
    decoder_config: Config
    reconstructor_config: Config | None = None
    supervision_head_config: SupervisionHeadConfig | None = None
    # Where the (register) supervision heads attach when the encoder has a detached
    # register projection: "registers" (default -- the register grid, gradients shape
    # the encoder/bottleneck as before), "projection" (only the detached low-dim
    # student -- the encoder gets NO supervision gradient), or "both" (two separate
    # heads, one per source). Only meaningful with register_supervision=True; sources
    # other than "registers" require encoder_config.register_projection_dim.
    supervision_source: str = "registers"
    # Scales the PROJECTION supervision heads' weights relative to the register
    # head's, decoupling the two. Without it both are built from one
    # supervision_head_config, so supervision_source="both" forces the student head
    # to the teacher's weight -- and at that weight it cost 2-5 mIoU on the projected
    # PASTIS probes, while 0.1x was roughly neutral. With a w1 register head, 0.1
    # here is the w0p1 arm and 0.01 the w0p01 arm. None = same weight as the
    # register head (the previous behaviour).
    projection_supervision_weight_scale: float | None = None
    projection_only_target: bool = False

    def validate(self) -> None:
        """Validate the configuration."""
        if (
            self.encoder_config.supported_modalities
            != self.decoder_config.supported_modalities
        ):
            raise ValueError("Encoder and decoder must support the same modalities")
        if (
            self.encoder_config.max_sequence_length
            != self.decoder_config.max_sequence_length
        ):
            raise ValueError(
                "Encoder and decoder must have the same max sequence length"
            )
        encoder_output_size = (
            self.encoder_config.output_embedding_size
            or self.encoder_config.embedding_size
        )
        if encoder_output_size != self.decoder_config.encoder_embedding_size:
            raise ValueError("Encoder embedding size must be consistent!")
        encoder_uses_registers = getattr(
            self.encoder_config, "use_register_bottleneck", False
        )
        decoder_uses_registers = getattr(
            self.decoder_config, "use_register_bottleneck", False
        )
        if encoder_uses_registers != decoder_uses_registers:
            raise ValueError(
                "use_register_bottleneck must match between encoder and decoder"
            )
        if encoder_uses_registers:
            # The decoder cross-attends the grid the encoder SHIPS. With
            # ``register_output_dim`` the bottleneck runs internally at
            # ``register_dim`` and projects down on output, so it is the projected
            # width the decoder must match -- not the internal one.
            encoder_register_dim = (
                getattr(self.encoder_config, "register_output_dim", None)
                or self.encoder_config.register_dim
                or (self.encoder_config.embedding_size // 2)
            )
            if self.decoder_config.register_dim != encoder_register_dim:
                raise ValueError(
                    "decoder_config.register_dim "
                    f"({self.decoder_config.register_dim}) must match the encoder's "
                    f"shipped register dim ({encoder_register_dim})"
                )
        if (
            self.supervision_head_config is not None
            and getattr(self.supervision_head_config, "register_supervision", False)
            and not encoder_uses_registers
        ):
            raise ValueError(
                "register_supervision requires the encoder register bottleneck"
            )
        if (
            self.projection_supervision_weight_scale is not None
            and self.supervision_source == "registers"
        ):
            raise ValueError(
                "projection_supervision_weight_scale has no effect with "
                "supervision_source='registers' (no projection heads exist to "
                "scale); use 'both' or 'projection'"
            )
        if self.projection_supervision_weight_scale is not None and (
            self.projection_supervision_weight_scale < 0
        ):
            raise ValueError(
                "projection_supervision_weight_scale must be non-negative, got "
                f"{self.projection_supervision_weight_scale}"
            )
        if self.supervision_source not in ("registers", "projection", "both"):
            raise ValueError(
                "supervision_source must be 'registers', 'projection' or 'both', "
                f"got {self.supervision_source!r}"
            )
        if self.supervision_source != "registers":
            if self.supervision_head_config is None or not getattr(
                self.supervision_head_config, "register_supervision", False
            ):
                raise ValueError(
                    "supervision_source='projection'/'both' requires a "
                    "supervision_head_config with register_supervision=True"
                )
            if getattr(self.encoder_config, "register_projection_dims", None) is None:
                raise ValueError(
                    "supervision_source='projection'/'both' requires "
                    "encoder_config.register_projection_dims (the detached student "
                    "the projection heads read)"
                )

    def _projection_supervision_head_config(self) -> SupervisionHeadConfig:
        """The supervision head config the PROJECTION heads are built from.

        The per-modality weights are already ``base_weight * TASK_TYPE_WEIGHTS``, so
        scaling them uniformly is exactly a change of base weight with the
        classification/regression balance left intact.
        """
        assert self.supervision_head_config is not None
        if self.projection_supervision_weight_scale is None:
            return self.supervision_head_config
        scale = self.projection_supervision_weight_scale
        return replace(
            self.supervision_head_config,
            modality_configs={
                name: replace(modality, weight=modality.weight * scale)
                for name, modality in (
                    self.supervision_head_config.modality_configs.items()
                )
            },
        )

    def build(self) -> "LatentMIM":
        """Build the Latent Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        supervision_head = None
        projection_supervision_heads = None
        if self.supervision_head_config is not None:
            if getattr(self.supervision_head_config, "register_supervision", False):
                # Heads read the register grid, so embedding_dim is the width that grid
                # is SHIPPED at -- register_output_dim when the bottleneck projects its
                # output down, otherwise the internal register width.
                embedding_dim = (
                    getattr(self.encoder_config, "register_output_dim", None)
                    or self.encoder_config.register_dim
                    or (self.encoder_config.embedding_size // 2)
                )
                if self.supervision_source in ("registers", "both"):
                    supervision_head = self.supervision_head_config.build(
                        embedding_dim=embedding_dim,
                        max_patch_size=self.encoder_config.max_patch_size,
                    )
                if self.supervision_source in ("projection", "both"):
                    # SEPARATE heads (their own parameters), one per Matryoshka
                    # prefix width, each reading the first d dims of the detached
                    # student grid.
                    projection_head_config = self._projection_supervision_head_config()
                    projection_supervision_heads = {
                        dim: projection_head_config.build(
                            embedding_dim=dim,
                            max_patch_size=self.encoder_config.max_patch_size,
                        )
                        for dim in self.encoder_config.register_projection_dims
                    }
            else:
                output_embed_size = getattr(
                    self.decoder_config, "output_embedding_size", None
                )
                embedding_dim = (
                    output_embed_size
                    if output_embed_size is not None
                    else self.encoder_config.embedding_size
                )
                supervision_head = self.supervision_head_config.build(
                    embedding_dim=embedding_dim,
                    max_patch_size=self.encoder_config.max_patch_size,
                )
        return LatentMIM(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            supervision_head=supervision_head,
            projection_supervision_heads=projection_supervision_heads,
            projection_only_target=self.projection_only_target,
        )
