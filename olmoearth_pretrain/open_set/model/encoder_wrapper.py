"""Frozen OlmoEarth encoder wrapper.

Loads an encoder from either:

- a distributed olmo-core checkpoint directory (``step{N}/`` with
  ``config.json`` next to a ``model_and_optim/`` subdir), or
- a path containing ``config.json`` + ``weights.pth`` (the consolidated
  format used by ``model_loader.load_model_from_path``).

The full LatentMIM-style training model is loaded, then we strip everything
but the encoder, freeze its parameters, and expose a small forward that
returns the spatial token sequence + a context mask suitable for use as
cross-attention K/V.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

from olmoearth_pretrain.config import Config, require_olmo_core
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.model_loader import (
    CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    patch_legacy_encoder_config,
)

logger = logging.getLogger(__name__)

# Directory inside a distributed checkpoint that holds the model + optim shards.
DIST_CHECKPOINT_SUBDIR = "model_and_optim"


def _looks_like_distributed_checkpoint(path: Path) -> bool:
    """Distributed checkpoints place model shards under ``model_and_optim/``."""
    return (path / DIST_CHECKPOINT_SUBDIR).is_dir()


def _looks_like_consolidated_checkpoint(path: Path) -> bool:
    """The consolidated format ships ``weights.pth`` next to ``config.json``."""
    return (path / WEIGHTS_FILENAME).is_file()


def load_encoder_from_distributed_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load an OlmoEarth pretraining checkpoint and return its encoder.

    Args:
        checkpoint_path: Directory of a distributed olmo-core checkpoint
            (e.g. ``.../step370000/``) or of a consolidated checkpoint
            (with ``config.json`` + ``weights.pth``).
        device: Where to place the encoder.

    Returns:
        The encoder ``nn.Module`` extracted from the loaded model. Parameters
        are *not* yet frozen — caller is expected to wrap with
        :class:`FrozenOlmoEarthEncoder` (which handles freezing + eval mode).
    """
    checkpoint_path = Path(checkpoint_path)
    config_fpath = checkpoint_path / CONFIG_FILENAME
    if not config_fpath.is_file():
        raise FileNotFoundError(
            f"No {CONFIG_FILENAME} found at {checkpoint_path}. "
            "This must be a checkpoint directory (distributed or consolidated)."
        )

    with config_fpath.open() as f:
        config_dict = json.load(f)
    config_dict = patch_legacy_encoder_config(config_dict)
    model_config = Config.from_dict(config_dict["model"])
    full_model = model_config.build()

    if _looks_like_distributed_checkpoint(checkpoint_path):
        require_olmo_core("Loading distributed checkpoints (model_and_optim/ format)")
        # Lazy import — only available with the [training] extras.
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        load_model_and_optim_state(
            str(checkpoint_path / DIST_CHECKPOINT_SUBDIR), full_model
        )
        logger.info(
            "Loaded distributed checkpoint from %s",
            checkpoint_path / DIST_CHECKPOINT_SUBDIR,
        )
    elif _looks_like_consolidated_checkpoint(checkpoint_path):
        with (checkpoint_path / WEIGHTS_FILENAME).open("rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        full_model.load_state_dict(state_dict)
        logger.info(
            "Loaded consolidated checkpoint from %s",
            checkpoint_path / WEIGHTS_FILENAME,
        )
    else:
        raise FileNotFoundError(
            f"Could not find either {DIST_CHECKPOINT_SUBDIR}/ or "
            f"{WEIGHTS_FILENAME} under {checkpoint_path}."
        )

    if not hasattr(full_model, "encoder"):
        raise AttributeError(
            f"Loaded model {type(full_model).__name__} has no .encoder attribute. "
            "This wrapper expects an encoder/decoder pretraining model."
        )
    encoder = full_model.encoder
    encoder = encoder.to(device)
    return encoder


class FrozenOlmoEarthEncoder(nn.Module):
    """Frozen wrapper around an OlmoEarth encoder.

    The encoder is set to eval mode and its parameters have ``requires_grad``
    set to False. Forward passes run under ``torch.no_grad()`` by default so
    activations do not retain a grad graph; set ``trainable=True`` at
    construction time to disable both freezing and the no-grad context
    (useful for partial fine-tuning experiments later).

    The forward returns a flat ``[B, N, D]`` token sequence concatenating
    every spatial input modality across time and band-set, plus a
    ``[B, N]`` boolean mask which is True for tokens the cross-attention
    decoder should attend to (i.e. tokens that the encoder actually saw and
    that are not missing).
    """

    def __init__(
        self,
        encoder: nn.Module,
        trainable: bool = False,
    ) -> None:
        """Initialize the wrapper around an existing encoder."""
        super().__init__()
        self.encoder = encoder
        self.trainable = trainable
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True) -> FrozenOlmoEarthEncoder:
        """Force eval mode when frozen, otherwise behave normally."""
        if not self.trainable:
            return super().train(False)
        return super().train(mode)

    @property
    def embedding_dim(self) -> int:
        """Encoder output embedding size."""
        return int(self.encoder.embedding_size)

    @property
    def min_patch_size(self) -> int:
        """Smallest patch size the underlying encoder supports.

        Proxied to the wrapped encoder so that olmo-core's
        ``OlmoEarthTrainModule.on_attach`` validation
        (``data_loader.min_patch_size == model.encoder.min_patch_size``)
        sees the right value through ``OpenSetSegmenter.encoder``.
        """
        return int(self.encoder.min_patch_size)

    @property
    def max_patch_size(self) -> int:
        """Largest patch size the underlying encoder supports."""
        return int(self.encoder.max_patch_size)

    def _encoder_forward(
        self, sample: MaskedOlmoEarthSample, patch_size: int
    ) -> dict[str, object]:
        """Run the underlying encoder, optionally under ``no_grad``."""
        if self.trainable:
            return self.encoder(sample, patch_size=patch_size, fast_pass=True)
        with torch.no_grad():
            return self.encoder(sample, patch_size=patch_size, fast_pass=True)

    def forward(
        self,
        sample: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, ...]]]:
        """Run the encoder and produce decoder-ready tokens.

        Args:
            sample: Masked input sample.
            patch_size: Patch size to use for tokenization. Must be in the
                range ``[encoder.min_patch_size, encoder.max_patch_size]``.

        Returns:
            Tuple of:
                - tokens: ``[B, N, D]`` flat token sequence across all spatial
                  input modalities.
                - context_mask: ``[B, N]`` bool mask, True for tokens the
                  decoder should attend to (excludes MISSING and TARGET-only
                  tokens).
                - shapes: dict mapping each spatial modality name to its
                  per-modality token shape (without the batch and embedding
                  dims), useful for reshaping to a 2D grid downstream.
        """
        output_dict = self._encoder_forward(sample, patch_size)
        tokens_and_masks: TokensAndMasks = output_dict["tokens_and_masks"]  # type: ignore[assignment]

        flat_tokens: list[torch.Tensor] = []
        flat_masks: list[torch.Tensor] = []
        shapes: dict[str, tuple[int, ...]] = {}

        for modality_name in tokens_and_masks.modalities:
            modality_spec = Modality.get(modality_name)
            if not modality_spec.is_spatial:
                # Non-spatial modalities (latlon, era5_10) carry context but
                # should not contribute "image" tokens for the decoder. We
                # still pass them as cross-attention K/V via the same
                # concatenation — they are global context.
                pass
            tokens = getattr(tokens_and_masks, modality_name)
            mask = getattr(
                tokens_and_masks,
                tokens_and_masks.get_masked_modality_name(modality_name),
            )
            if tokens is None or mask is None:
                continue
            shapes[modality_name] = tuple(tokens.shape[1:-1])
            flat_tokens.append(rearrange(tokens, "b ... d -> b (...) d"))
            flat_masks.append(rearrange(mask, "b ... -> b (...)"))

        if not flat_tokens:
            raise RuntimeError("Encoder produced no tokens for this batch.")

        tokens = torch.cat(flat_tokens, dim=1)  # [B, N, D]
        masks = torch.cat(flat_masks, dim=1)  # [B, N]
        context_mask = masks == MaskValue.ONLINE_ENCODER.value
        return tokens, context_mask, shapes

    def selected_modality_tokens(
        self,
        tokens: torch.Tensor,
        shapes: dict[str, tuple[int, ...]],
        modality: str,
    ) -> torch.Tensor:
        """Slice out the tokens contributed by one modality.

        Args:
            tokens: The flat ``[B, N, D]`` tensor returned by ``forward``.
            shapes: The ``shapes`` dict returned by ``forward``.
            modality: Modality whose tokens to extract.

        Returns:
            ``[B, *shape, D]`` — the original spatial-temporal-bandset token
            tensor for that modality.
        """
        if modality not in shapes:
            raise KeyError(f"Modality {modality!r} not present in this encoder output")
        offset = 0
        d = tokens.shape[-1]
        for name, shape in shapes.items():
            n_modality = 1
            for s in shape:
                n_modality *= s
            if name == modality:
                slice_ = tokens[:, offset : offset + n_modality]
                return slice_.reshape(tokens.shape[0], *shape, d)
            offset += n_modality
        raise KeyError(f"Modality {modality!r} not found in shapes")  # pragma: no cover


def freeze_parameters(modules: Iterable[nn.Module]) -> None:
    """Set ``requires_grad = False`` on every parameter under ``modules``."""
    for module in modules:
        for p in module.parameters():
            p.requires_grad = False
