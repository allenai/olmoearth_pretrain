"""Chunked, decoder-only projection targets give the same loss as full targets."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import FrozenTargetProjection
from olmoearth_pretrain.train.loss import (
    CompactDecoderTargets,
    ModalityPatchDiscriminationMaskedNegativesVec,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.train.train_module.latent_mim import compact_decoder_targets

MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
T, PX, PS, D = 5, 8, 2, 32


def _sample(batch: int, decode_frac: float = 0.4) -> MaskedOlmoEarthSample:
    """S2 (3 band sets) + S1 with per-token ENCODER / DECODER / MISSING masks."""
    g = torch.Generator().manual_seed(0)
    fields: dict = {}
    n = PX // PS
    for name in MODALITIES:
        bands = Modality.get(name).num_bands
        u = torch.rand(batch, n, n, T, generator=g)
        tok = torch.where(
            u < decode_frac,
            MaskValue.DECODER.value,
            torch.where(
                u < 0.9, MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value
            ),
        )
        pix = tok.repeat_interleave(PS, 1).repeat_interleave(PS, 2)
        fields[name] = torch.randn(batch, PX, PX, T, bands, generator=g)
        fields[f"{name}_mask"] = pix[..., None].expand(-1, -1, -1, -1, bands).clone()
    fields["timestamps"] = torch.tensor([[[1, t, 2020] for t in range(T)]]).expand(
        batch, -1, -1
    )
    return MaskedOlmoEarthSample(**fields)


def _target() -> FrozenTargetProjection:
    torch.manual_seed(0)
    encoder = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=D,
        num_heads=2,
        depth=1,
        max_patch_size=4,
        min_patch_size=1,
    ).build()
    return FrozenTargetProjection(encoder)


def _predictions(full: TokensAndMasks) -> TokensAndMasks:
    """Random predictions in the targets' layout, with the targets' masks."""
    g = torch.Generator().manual_seed(1)
    fields: dict = {}
    for name in full.modalities:
        value = getattr(full, name)
        if value is None:
            continue
        fields[name] = torch.randn(value.shape, generator=g)
        mask_name = full.get_masked_modality_name(name)
        fields[mask_name] = getattr(full, mask_name)
    return TokensAndMasks(**fields)


@pytest.mark.parametrize("chunk", [1, 2, 5])
def test_compact_targets_give_the_same_loss(chunk: int) -> None:
    """Loss on chunked decoder-only targets == loss on the full targets."""
    target = _target()
    sample = _sample(batch=5)
    with torch.no_grad():
        full = target(sample, patch_size=PS)["tokens_and_masks"]
    predictions = _predictions(full)
    compact = compact_decoder_targets(target, sample, PS, None, predictions, chunk)
    assert isinstance(compact, CompactDecoderTargets)
    loss = ModalityPatchDiscriminationMaskedNegativesVec(tau=0.1)
    torch.testing.assert_close(
        loss.compute(predictions, compact), loss.compute(predictions, full)
    )
    # Only decoder positions are kept: K = the batch's largest decoder count.
    _, masks = predictions.flatten_tokens_and_masks_per_modality()
    for tokens, mask in zip(compact.tokens, masks):
        expected = int((mask == MaskValue.DECODER.value).sum(dim=-1).max())
        assert tokens.shape[:2] == (5, max(expected, 1))
