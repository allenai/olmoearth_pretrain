"""Simple set up of latent predictor."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)
from torch.distributed.tensor import DTensor

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
        projection_only_target: bool = False,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: Optional reconstructor for auto-encoding.
            supervision_head: Optional supervision head for direct supervision
                of decode-only modalities from decoder output.
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
    ]:
        """Forward pass for the Latent MIM Style.

        Returns:
            latent: embeddings from encoder
            decoded: predictions from decoder for masked tokens
            latent_projected_and_pooled: pooled tokens for contrastive loss
            reconstructed: MAE predictions if enabled
            extra_metrics: additional metrics to log
            supervision_preds: per-modality supervision predictions (or None)
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        extra_metrics = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats
        # Log the learned per-read residual gates (when enabled) keyed by encoder read
        # depth, so we can see whether the multi-depth reads stay distributed or collapse
        # toward the final layer. Tiny (one scalar per read); full_tensor() gathers the
        # FSDP-sharded parameter (a collective all ranks hit, since the flag is uniform).
        register_bottleneck = getattr(self.encoder, "register_bottleneck", None)
        if register_bottleneck is not None and getattr(
            register_bottleneck, "learned_read_weighting", False
        ):
            gates = register_bottleneck.read_gates.detach()
            if isinstance(gates, DTensor):
                gates = gates.full_tensor()
            read_layers = register_bottleneck.read_layers or list(
                range(1, gates.numel() + 1)
            )
            extra_metrics["register_read_gates"] = {
                str(layer): gates[i].item() for i, layer in enumerate(read_layers)
            }
        # Log the per-depth contribution norms of the fused read source (when enabled),
        # keyed by encoder read depth. In the learned arm, drift of these norms (e.g.
        # collapse onto the final layer) is the signal the A/B exists to measure; in the
        # uniform arm they are ~constant by construction. Activations, not parameters, so
        # no FSDP gather is needed.
        if (
            register_bottleneck is not None
            and getattr(register_bottleneck, "fused_read", None) is not None
            and register_bottleneck.last_read_source_norms is not None
        ):
            source_norms = register_bottleneck.last_read_source_norms
            assert register_bottleneck.read_layers is not None
            extra_metrics["register_read_source_norms"] = {
                str(layer): source_norms[i].item()
                for i, layer in enumerate(register_bottleneck.read_layers)
            }
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )

        supervision_preds = None
        # Expose the encoder's spatial latent grid (the Perceiver/register bottleneck
        # output, time/modality collapsed) for heads that probe it directly: the
        # register-supervision heads below and the open-set probe (see
        # olmoearth_pretrain.nn.open_set_latent_mim), which reads it via
        # ``last_register_grid`` after the forward pass.
        register_grid = None
        registers = decoder_kwargs.get("registers")
        if register_bottleneck is not None and registers is not None:
            n_h, n_w = register_bottleneck.register_grid
            register_grid = registers.reshape(
                registers.shape[0], n_h, n_w, registers.shape[-1]
            )
        self.last_register_grid = register_grid
        if self.supervision_head is not None:
            if getattr(self.supervision_head, "register_supervision", False):
                # Supervise the register grid directly ([B, n_h, n_w, D]).
                supervision_preds = self.supervision_head(
                    decoded, x, register_grid=register_grid
                )
            else:
                supervision_preds = self.supervision_head(decoded, x)

        return (
            latent,
            decoded,
            latent_projected_and_pooled,
            reconstructed,
            extra_metrics,
            supervision_preds,
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


@dataclass
class LatentMIMConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: Config
    decoder_config: Config
    reconstructor_config: Config | None = None
    supervision_head_config: SupervisionHeadConfig | None = None
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
            encoder_register_dim = self.encoder_config.register_dim or (
                self.encoder_config.embedding_size // 2
            )
            if self.decoder_config.register_dim != encoder_register_dim:
                raise ValueError(
                    "decoder_config.register_dim "
                    f"({self.decoder_config.register_dim}) must match the encoder "
                    f"register dim ({encoder_register_dim})"
                )
        if (
            self.supervision_head_config is not None
            and getattr(self.supervision_head_config, "register_supervision", False)
            and not encoder_uses_registers
        ):
            raise ValueError(
                "register_supervision requires the encoder register bottleneck"
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
        if self.supervision_head_config is not None:
            if getattr(self.supervision_head_config, "register_supervision", False):
                # Heads read the register grid, so embedding_dim is the register dim.
                embedding_dim = self.encoder_config.register_dim or (
                    self.encoder_config.embedding_size // 2
                )
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
            projection_only_target=self.projection_only_target,
        )
