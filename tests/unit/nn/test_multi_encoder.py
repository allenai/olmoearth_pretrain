"""Tests for the MultiEncoder (per-modality towers fused in the shared trunk / pooler)."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue, OlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import (
    AttentionPoolerConfig,
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.multi_encoder import MultiEncoder, MultiEncoderConfig
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import PatchDiscriminationLoss
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskingConfig,
    ModalityDropoutMaskingStrategy,
    RandomWithDecodeMaskingStrategy,
)

INPUTS = ["sentinel2_l2a", "sentinel1"]
MAPS = ["worldcover"]
MODALITIES = INPUTS + MAPS
B, H, W, T = 2, 8, 8, 3
PATCH_SIZE = 4
EMBED = 16
POOLER_DIM = 8


def _timestamps(b: int, t: int) -> torch.Tensor:
    days = torch.randint(1, 28, (b, t, 1), dtype=torch.long)
    months = torch.randint(1, 13, (b, t, 1), dtype=torch.long)
    years = torch.full((b, t, 1), 2022, dtype=torch.long)
    return torch.cat([days, months, years], dim=-1)


def _raw_sample(b: int = B) -> OlmoEarthSample:
    return OlmoEarthSample(
        sentinel2_l2a=torch.randn(b, H, W, T, Modality.SENTINEL2_L2A.num_bands),
        sentinel1=torch.randn(b, H, W, T, Modality.SENTINEL1.num_bands),
        worldcover=torch.randn(b, H, W, Modality.WORLDCOVER.num_bands),
        timestamps=_timestamps(b, T),
    )


def _masked_sample(b: int = B) -> MaskedOlmoEarthSample:
    strategy = RandomWithDecodeMaskingStrategy(
        encode_ratio=0.5, decode_ratio=0.5, only_decode_modalities=MAPS
    )
    return strategy.apply_mask(_raw_sample(b), patch_size=PATCH_SIZE)


def _encoder_kwargs(**overrides: object) -> dict:
    kwargs: dict = dict(
        supported_modality_names=MODALITIES,
        embedding_size=EMBED,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=8,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        attention_pooler_config=AttentionPoolerConfig(
            pooler_dim=POOLER_DIM, pooler_depth=1
        ),
    )
    kwargs.update(overrides)
    return kwargs


def _decoder_config() -> PredictorConfig:
    return PredictorConfig(
        supported_modality_names=MODALITIES,
        encoder_embedding_size=EMBED,
        decoder_embedding_size=EMBED,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_attention_pooler=True,
        pooler_dim=POOLER_DIM,
    )


# --------------------------------------------------------------------------- config


def test_default_groups_are_one_per_modality() -> None:
    """With tower_groups=None every supported modality gets its own tower."""
    enc = MultiEncoderConfig(**_encoder_kwargs(tower_depth=1)).build()
    assert enc.tower_groups == {m: [m] for m in MODALITIES}
    assert set(enc.towers.keys()) == set(MODALITIES)
    assert all(len(t) == 1 for t in enc.towers.values())


@pytest.mark.parametrize(
    "groups",
    [
        {"a": ["sentinel2_l2a", "sentinel1"], "b": ["sentinel1", "worldcover"]},
        {"a": ["sentinel2_l2a", "sentinel1", "worldcover", "landsat"]},
        {"a": [], "b": MODALITIES},
    ],
)
def test_bad_groups_rejected(groups: dict) -> None:
    """Overlapping, unknown or empty groups are rejected at validation."""
    with pytest.raises(ValueError):
        MultiEncoderConfig(**_encoder_kwargs(tower_groups=groups)).validate()


def test_ungrouped_modalities_pass_through() -> None:
    """Modalities in no group get no tower and are left untouched by the tower stage."""
    torch.manual_seed(0)
    x = _masked_sample()
    groups = {"s2": ["sentinel2_l2a"], "s1": ["sentinel1"]}  # worldcover ungrouped
    enc = MultiEncoderConfig(
        **_encoder_kwargs(depth=0, tower_depth=1, tower_groups=groups)
    ).build()
    assert set(enc.towers.keys()) == {"s2", "s1"}
    plain = EncoderConfig(**_encoder_kwargs(depth=0)).build()
    plain.load_state_dict(
        {k: v for k, v in enc.state_dict().items() if k in plain.state_dict()}
    )
    enc.eval(), plain.eval()
    with torch.no_grad():
        out = enc(x, patch_size=PATCH_SIZE)["tokens_and_masks"]
        ref = plain(x, patch_size=PATCH_SIZE)["tokens_and_masks"]
    torch.testing.assert_close(out.worldcover, ref.worldcover)
    assert not torch.allclose(out.sentinel2_l2a, ref.sentinel2_l2a)


def test_flash_attn_rejected() -> None:
    """Towers do not support flash-attn packing; the config says so."""
    with pytest.raises(ValueError):
        MultiEncoderConfig(**_encoder_kwargs(use_flash_attn=True)).validate()


# --------------------------------------------------------------------------- forward


def test_zero_towers_matches_plain_encoder() -> None:
    """tower_depth=0 without fusion embeddings is exactly the plain Encoder."""
    torch.manual_seed(0)
    x = _masked_sample()
    plain = EncoderConfig(**_encoder_kwargs()).build().eval()
    multi = MultiEncoderConfig(
        **_encoder_kwargs(tower_depth=0, fusion_embeddings=False)
    ).build()
    assert isinstance(multi, MultiEncoder)
    missing, unexpected = multi.load_state_dict(plain.state_dict(), strict=False)
    assert not missing and not unexpected
    multi.eval()
    with torch.no_grad():
        out_plain = plain(x, patch_size=PATCH_SIZE)
        out_multi = multi(x, patch_size=PATCH_SIZE)
    torch.testing.assert_close(
        out_plain["location_tokens"], out_multi["location_tokens"]
    )
    for m in INPUTS:
        torch.testing.assert_close(
            getattr(out_plain["tokens_and_masks"], m),
            getattr(out_multi["tokens_and_masks"], m),
        )


@pytest.mark.parametrize("fusion_depth", [0, 2])
@pytest.mark.parametrize(
    "groups", [None, {"optical": ["sentinel2_l2a", "worldcover"], "sar": ["sentinel1"]}]
)
def test_forward_shapes_and_grads(fusion_depth: int, groups: dict | None) -> None:
    """Location-token grid shape and gradients through towers and pooler."""
    torch.manual_seed(0)
    x = _masked_sample()
    enc = MultiEncoderConfig(
        **_encoder_kwargs(depth=fusion_depth, tower_depth=2, tower_groups=groups)
    ).build()
    enc.train()
    out = enc(x, patch_size=PATCH_SIZE)
    h_p, w_p = H // PATCH_SIZE, W // PATCH_SIZE
    assert out["location_tokens"].shape == (B, h_p, w_p, POOLER_DIM)
    assert out["tokens_and_masks"].sentinel2_l2a.shape[:3] == (B, h_p, w_p)
    out["location_tokens"].sum().backward()
    for group, tower in enc.towers.items():
        if not any(m in INPUTS for m in enc.tower_groups[group]):
            continue  # a decode-only group never has visible tokens, so no grads
        assert tower[0].attn.q.weight.grad is not None, group
        assert enc.fusion_embeddings[group].grad is not None, group
    assert enc.attention_pooler is not None
    assert enc.attention_pooler.learned_query.grad is not None


def test_towers_do_not_mix_modalities_before_fusion() -> None:
    """With no shared blocks, a modality's patch tokens depend only on its own input."""
    torch.manual_seed(0)
    x = _masked_sample()
    enc = MultiEncoderConfig(**_encoder_kwargs(depth=0, tower_depth=2)).build().eval()
    with torch.no_grad():
        ref = enc(x, patch_size=PATCH_SIZE)
        assert x.sentinel1 is not None
        perturbed = x._replace(sentinel1=x.sentinel1 + 5.0)
        out = enc(perturbed, patch_size=PATCH_SIZE)
    tm_ref, tm_out = ref["tokens_and_masks"], out["tokens_and_masks"]
    torch.testing.assert_close(tm_ref.sentinel2_l2a, tm_out.sentinel2_l2a)
    # ... while the S1 tower and the fused location tokens do see the change.
    assert not torch.allclose(tm_ref.sentinel1, tm_out.sentinel1)
    assert not torch.allclose(ref["location_tokens"], out["location_tokens"])


def test_fully_dropped_modality_runs() -> None:
    """A modality with no ONLINE_ENCODER token anywhere in the batch is skipped cleanly."""
    torch.manual_seed(0)
    x = _masked_sample()
    assert x.sentinel1_mask is not None
    s1_mask = x.sentinel1_mask.clone()
    s1_mask[s1_mask == MaskValue.ONLINE_ENCODER.value] = MaskValue.DECODER.value
    x = x._replace(sentinel1_mask=s1_mask)
    enc = MultiEncoderConfig(**_encoder_kwargs(depth=1, tower_depth=1)).build()
    out = enc(x, patch_size=PATCH_SIZE)
    assert torch.isfinite(out["location_tokens"]).all()
    out["location_tokens"].sum().backward()


def test_refuses_nonzero_token_exit() -> None:
    """Only exit depth 0 (projection-only target) is supported."""
    x = _masked_sample()
    enc = MultiEncoderConfig(**_encoder_kwargs(tower_depth=1)).build()
    with pytest.raises(NotImplementedError):
        enc(x, patch_size=PATCH_SIZE, token_exit_cfg={m: 1 for m in MODALITIES})
    # Exit depth 0 everywhere (the projection-only target) is fine.
    enc(x.unmask(), patch_size=PATCH_SIZE, token_exit_cfg={m: 0 for m in MODALITIES})


# --------------------------------------------------------------------------- end to end


def test_latent_mim_with_multi_encoder() -> None:
    """LatentMIM end to end with a MultiEncoder: forward, target, loss, backward."""
    torch.manual_seed(0)
    x = _masked_sample()
    model = LatentMIMConfig(
        encoder_config=MultiEncoderConfig(**_encoder_kwargs(depth=0, tower_depth=2)),
        decoder_config=_decoder_config(),
        projection_only_target=True,
    ).build()
    assert isinstance(model.encoder, MultiEncoder)
    _, decoded, _, _, _, _, _ = model.forward(x, PATCH_SIZE)
    with torch.no_grad():
        target_dict = model.target_encoder.forward(
            x.unmask(),
            patch_size=PATCH_SIZE,
            token_exit_cfg={m: 0 for m in MODALITIES},
        )
        target, _, _ = unpack_encoder_output(target_dict)
    loss = PatchDiscriminationLoss().compute(decoded, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.towers["sentinel1"][0].attn.q.weight.grad is not None
    assert model.decoder.location_tokens_to_decoder_embed.weight.grad is not None


# --------------------------------------------------------------------------- masking


def _dropout_strategy(rate: float, **kw: object) -> ModalityDropoutMaskingStrategy:
    strategy = MaskingConfig(
        strategy_config={
            "type": "modality_dropout",
            "base_strategy": {
                "type": "random_with_decode",
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
                "only_decode_modalities": MAPS,
            },
            "dropout_modalities": INPUTS,
            "dropout_rate": rate,
            **kw,
        }
    ).build()
    assert isinstance(strategy, ModalityDropoutMaskingStrategy)
    return strategy


def _online_count(masked: MaskedOlmoEarthSample, modality: str) -> torch.Tensor:
    mask = getattr(masked, MaskedOlmoEarthSample.get_masked_modality_name(modality))
    return (mask == MaskValue.ONLINE_ENCODER.value).flatten(1).sum(1)


def test_modality_dropout_drops_whole_modalities_and_keeps_an_input() -> None:
    """Rate 1 / max 1 drops exactly one input modality per sample, never both."""
    torch.manual_seed(0)
    np.random.seed(0)
    strategy = _dropout_strategy(rate=1.0, max_dropped=1)
    masked = strategy.apply_mask(_raw_sample(b=16), patch_size=PATCH_SIZE)
    s2 = _online_count(masked, "sentinel2_l2a")
    s1 = _online_count(masked, "sentinel1")
    # Exactly one input modality is dropped per sample (rate 1, max_dropped 1), and the
    # other always keeps something for the online encoder.
    assert ((s2 == 0) ^ (s1 == 0)).all()
    assert (s2 + s1 > 0).all()
    # Dropped tokens became DECODER targets, not MISSING.
    s1_mask = torch.as_tensor(masked.sentinel1_mask)
    s2_mask = torch.as_tensor(masked.sentinel2_l2a_mask)
    wc_mask = torch.as_tensor(masked.worldcover_mask)
    assert bool((s1_mask != MaskValue.MISSING.value).all())
    assert bool((s2_mask != MaskValue.MISSING.value).all())
    # Decode-only maps are untouched.
    assert bool((wc_mask == MaskValue.DECODER.value).all())


def test_modality_dropout_never_empties_the_online_encoder() -> None:
    """Even at rate 1 every sample keeps at least one visible token."""
    np.random.seed(1)
    torch.manual_seed(1)
    strategy = _dropout_strategy(rate=1.0)  # tries to drop every input
    masked = strategy.apply_mask(_raw_sample(b=16), patch_size=PATCH_SIZE)
    total = _online_count(masked, "sentinel2_l2a") + _online_count(masked, "sentinel1")
    assert (total > 0).all()


def test_modality_dropout_rate_zero_is_identity() -> None:
    """Rate 0 reproduces the wrapped strategy's masks exactly."""
    sample = _raw_sample()
    np.random.seed(2)
    torch.manual_seed(2)
    base = RandomWithDecodeMaskingStrategy(0.5, 0.5, MAPS).apply_mask(
        sample, patch_size=PATCH_SIZE
    )
    np.random.seed(2)
    torch.manual_seed(2)
    wrapped = _dropout_strategy(rate=0.0).apply_mask(sample, patch_size=PATCH_SIZE)
    for m in MODALITIES:
        name = MaskedOlmoEarthSample.get_masked_modality_name(m)
        torch.testing.assert_close(getattr(base, name), getattr(wrapped, name))


def test_modality_dropout_propagates_ratios() -> None:
    """encode/decode ratios come from the wrapped strategy."""
    strategy = _dropout_strategy(rate=0.3)
    assert strategy.encode_ratio == 0.5
    assert strategy.decode_ratio == 0.5
