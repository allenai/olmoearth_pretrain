"""Unit tests for the Set-Latent Perceiver (SLP) encoder.

Ported from ``earthy/tests/test_perceiver.py`` and adapted to the repo's
``MaskedOlmoEarthSample`` data type. Each test encodes a review-hardened
invariant from ``docs/perceiver_encoder_spec.md`` S6.
"""

from typing import Any

import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.set_latent_perceiver import (
    SetLatentPerceiver,
    SetLatentPerceiverConfig,
    _ensure_nonempty,
    contrastive_loss,
    soft_target_contrastive_loss,
)

# Band counts for the modalities used in these tests.
_NUM_BANDS = {"sentinel2_l2a": 12, "sentinel1": 2}


def _make_sample(
    b: int = 2, h: int = 16, w: int = 16, t: int = 3, year: int = 2021
) -> MaskedOlmoEarthSample:
    """Build a synthetic MaskedOlmoEarthSample with S2 + S1 modalities."""
    torch.manual_seed(0)
    ts = torch.zeros(b, t, 3, dtype=torch.float32)
    for ti in range(t):
        ts[:, ti, 0] = 1  # day
        ts[:, ti, 1] = ti % 12  # month (zero-indexed)
        ts[:, ti, 2] = year
    return MaskedOlmoEarthSample(
        timestamps=ts,
        sentinel2_l2a=torch.randn(b, h, w, t, _NUM_BANDS["sentinel2_l2a"]) * 1000
        + 1400,
        sentinel1=torch.rand(b, h, w, t, _NUM_BANDS["sentinel1"]) * 0.2,
        latlon=torch.tensor([[40.0, -100.0], [10.0, 20.0], [0.0, 0.0]])[:b],
    )


def _model(**overrides: Any) -> SetLatentPerceiver:
    """Small SLP for fast CPU tests."""
    kwargs: dict[str, Any] = dict(
        supported_modality_names=["sentinel2_l2a", "sentinel1"],
        dim=64,
        heads=4,
        latents=16,
        nested_latents=(8, 16),
        self_depth_per_read=1,
        level2_depth=1,
        decoder_depth=1,
        target_dim=32,
        cond_dropout=0.5,
    )
    kwargs.update(overrides)
    return SetLatentPerceiver(**kwargs)


def test_forward_backward_and_output_shapes() -> None:
    """Forward+backward runs and returns finite loss with sane metrics."""
    model = _model()
    loss, metrics = model(_make_sample(), mask_seed=3)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["target_count"] > 0
    assert set(metrics["group_losses"]) <= {
        f"{m}__bs{i}" for m in ("sentinel2_l2a", "sentinel1") for i in range(3)
    }
    assert metrics["num_groups"] == 4  # S2 has 3 band sets, S1 has 1


def test_missing_group_trains_and_only_its_tokenizer_lacks_grads() -> None:
    """A batch with an absent modality trains; only its params lack grads.

    DDP handles the unused params via find_unused_parameters, FSDP2 natively —
    there is deliberately no 0*sum(p) grad anchor, which would make the loss a
    DTensor under FSDP2.
    """
    model = _model()
    sample = _make_sample()._replace(sentinel1=None)  # S1 absent from the batch

    loss, _ = model(sample, mask_seed=1)
    loss.backward()
    assert torch.isfinite(loss)

    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert all("sentinel1" in n for n in missing)
    assert any("sentinel1" in n for n, _ in model.named_parameters())


def test_masked_tokens_cannot_influence_latents() -> None:
    """No-leak: a masked token's content cannot reach the latents."""
    model = _model()
    model.train()
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(7)
    per_group, b, _ = model._tokenize(_make_sample(), generator=None)
    masks = model._sample_masks(per_group, b, device, generator)
    visible = {name: g["valid"] & ~masks[name] for name, g in per_group.items()}

    flat = torch.cat(
        [(g["content"] + g["meta"]).flatten(1, 3) for g in per_group.values()], dim=1
    )
    flat_vis = torch.cat([visible[name].flatten(1, 3) for name in per_group], dim=1)
    masked_idx = (~flat_vis[0]).nonzero()[0, 0]

    with torch.no_grad():
        base = model._encode_set(flat, ~flat_vis, 16)
        poked = flat.clone()
        poked[0, masked_idx] += 1e6  # garbage in a masked token
        after = model._encode_set(poked, ~flat_vis, 16)
    assert torch.allclose(base, after, atol=1e-4)


def test_single_timestep_sample_still_trains() -> None:
    """Temporal scope falls back to the global pool for T=1 samples."""
    model = _model(contrast_scope="temporal")
    loss, metrics = model(_make_sample(t=1), mask_seed=5)
    loss.backward()

    assert metrics["target_count"] > 0
    assert len(metrics["group_losses"]) > 0
    assert torch.isfinite(loss) and loss.item() != 0.0


def test_encode_returns_grid_features_and_is_deterministic() -> None:
    """encode() returns the anchor grid and is deterministic at eval."""
    model = _model()
    model.eval()
    sample = _make_sample()
    with torch.no_grad():
        features = model.encode(sample)
        again = model.encode(sample)
    assert features.shape == (2, 2, 2, 64)  # 16px / 8px patch = 2x2 grid
    assert torch.isfinite(features).all()
    assert torch.equal(features, again)  # no dropout at eval


def test_nested_k_sampling_and_eval_capacity() -> None:
    """Nested-K samples the configured prefixes in train, full pool at eval."""
    model = _model()
    gen = torch.Generator().manual_seed(0)
    model.train()
    ks = {model._sample_k(gen, torch.device("cpu")) for _ in range(32)}
    assert ks <= {8, 16} and len(ks) == 2
    model.eval()
    assert model._sample_k(None, torch.device("cpu")) == 16


def test_target_tokenizers_are_frozen() -> None:
    """Frozen random-projection targets never receive gradient."""
    model = _model()
    loss, _ = model(_make_sample(), mask_seed=2)
    loss.backward()
    assert all(not p.requires_grad for p in model.target_tokenizers.parameters())
    assert all(p.grad is None for p in model.target_tokenizers.parameters())


def test_vit_base_scale_default_config() -> None:
    """The default config is ViT-B scale (~88M encoder)."""
    model = SetLatentPerceiver()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = (
        params
        - sum(p.numel() for p in model.decoder_blocks.parameters())
        - sum(p.numel() for p in model.head.parameters())
    )
    assert 70e6 < encoder_params < 120e6, (
        f"encoder params {encoder_params / 1e6:.1f}M not ViT-B scale"
    )


def test_soft_target_infonce_tolerates_duplicate_targets() -> None:
    """Soft labels absorb near-duplicate targets; dedup keeps the exact floor."""
    torch.manual_seed(0)
    base = torch.randn(4, 32)
    exact = base.repeat_interleave(8, dim=0)
    soft, correct, total = soft_target_contrastive_loss(
        exact.clone(), exact, 0.1, 0.1, 64
    )
    assert soft.item() < 0.05
    assert correct == total == 32

    near = exact + 1e-3 * torch.randn_like(exact)
    pred_soft = near.clone().requires_grad_(True)
    soft_near, _, _ = soft_target_contrastive_loss(pred_soft, near, 0.1, 0.1, 64)
    soft_near.backward()
    pred_hard = near.clone().requires_grad_(True)
    hard_near, _, _ = contrastive_loss(pred_hard, near, 0.1, 64)
    hard_near.backward()
    assert torch.isfinite(soft_near)
    assert pred_soft.grad.norm().item() < 0.1 * pred_hard.grad.norm().item()


def test_global_soft_scope_used_by_default_model() -> None:
    """The default objective is global soft-target InfoNCE."""
    model = _model(contrast_scope="global")
    assert model.soft_targets
    loss, metrics = model(_make_sample(t=2), mask_seed=9)
    loss.backward()
    assert torch.isfinite(loss) and metrics["target_count"] > 0


def test_cloud_mask_pools_any_cloudy_pixel_per_token() -> None:
    """Any cloudy pixel in a token's stored footprint excludes it from targets."""
    model = _model(supported_modality_names=["sentinel2_l2a"])
    sample = _make_sample(t=1)
    per_group, b, device = model._tokenize(sample, generator=None)
    # One cloudy pixel in the first token's 8x8 footprint; far token stays clean.
    cloud = torch.zeros(b, 1, 16, 16, dtype=torch.bool)
    cloud[:, :, 0, 0] = True
    masks = {name: g["valid"].clone() for name, g in per_group.items()}
    masks = model._apply_cloud_mask(masks, per_group, {"sentinel2_l2a": cloud})
    for name in masks:
        assert not masks[name][:, :, 0, 0].any()  # token over the cloudy pixel excluded
        assert masks[name][:, :, -1, -1].all()  # far corner unaffected


def test_cloud_mask_handles_non_multiple_grid() -> None:
    """Cloud grids whose H/W are not a multiple of patch_px pad up like the data."""
    model = _model(supported_modality_names=["sentinel2_l2a"])
    sample = _make_sample(t=1, h=20, w=20)  # pads to 24 -> 3x3 tokens
    per_group, b, device = model._tokenize(sample, generator=None)
    cloud = torch.zeros(b, 1, 20, 20, dtype=torch.bool)
    cloud[:, :, 19, 19] = True  # cloudy pixel in the padded edge token
    masks = {name: g["valid"].clone() for name, g in per_group.items()}
    masks = model._apply_cloud_mask(masks, per_group, {"sentinel2_l2a": cloud})
    for name in masks:
        assert masks[name].shape[-2:] == (3, 3)
        assert not masks[name][:, :, 2, 2].any()  # edge token excluded
        assert masks[name][:, :, 0, 0].all()  # clean token unaffected


def test_no_parameter_is_dead_weight() -> None:
    """Every trainable param gets a real (structural) gradient."""
    model = _model(cond_dropout=0.0)
    loss, _ = model(_make_sample(), mask_seed=11)
    loss.backward()
    zero_grad = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad
        and p.grad is not None
        and p.grad.abs().max().item() == 0.0
        and "latent_pool" not in n  # nested-K prefixes idle some steps
    ]
    assert zero_grad == [], f"dead-weight parameters: {zero_grad}"


def test_k_seed_is_rank_free_and_decoupled_from_masks() -> None:
    """Same k_seed with different mask seeds -> identical K draw (rank-free)."""
    model = _model()
    model.train()
    sample = _make_sample()
    ks = [model(sample, mask_seed=ms, k_seed=1234)[1]["k"] for ms in (1, 2, 3)]
    assert len(set(ks)) == 1


def test_per_group_metrics_reported() -> None:
    """Per-group loss/count metrics cover every present group."""
    model = _model()
    _, metrics = model(_make_sample(), mask_seed=4)
    assert set(metrics["group_correct"]) == {
        "sentinel2_l2a__bs0",
        "sentinel2_l2a__bs1",
        "sentinel2_l2a__bs2",
        "sentinel1__bs0",
    }


def test_encode_global_is_pooled_vector() -> None:
    """encode_global returns a single pooled (B, D) vector."""
    model = _model()
    model.eval()
    with torch.no_grad():
        feats = model.encode_global(_make_sample())
    assert feats.shape == (2, 64)


def test_missing_modality_is_skipped_not_crashed() -> None:
    """A modality absent from the sample is skipped without crashing."""
    model = _model()
    sample = _make_sample()._replace(sentinel1=None)
    loss, metrics = model(sample, mask_seed=3)
    loss.backward()
    assert not any(name.startswith("sentinel1") for name in metrics["group_losses"])
    assert torch.isfinite(loss)


def test_null_absolute_time_is_dropped_on_tokens() -> None:
    """cond_dropout=1.0 nulls absolute time: absolute year no longer affects meta."""
    model = _model(cond_dropout=1.0)
    model.train()
    a = _make_sample(year=2021)
    c = _make_sample(year=2005)
    tok_a, _, _ = model._tokenize(a, generator=torch.Generator().manual_seed(0))
    tok_c, _, _ = model._tokenize(c, generator=torch.Generator().manual_seed(0))
    diff = max(
        float((tok_a[name]["meta"] - tok_c[name]["meta"]).abs().max()) for name in tok_a
    )
    assert diff < 1e-3  # absolute time nulled; identical intra-sample spacing


def test_trained_years_nulls_out_of_range_dates() -> None:
    """Out-of-range absolute dates are nulled (not extrapolated)."""
    model = _model(trained_years=(0.0, 1.0), cond_dropout=0.0)  # 2020-2021
    model.eval()
    a = _make_sample(year=2018)
    c = _make_sample(year=2010)
    tok_a, _, _ = model._tokenize(a, generator=None)
    tok_c, _, _ = model._tokenize(c, generator=None)
    nulled_diff = max(
        float((tok_a[name]["meta"] - tok_c[name]["meta"]).abs().max()) for name in tok_a
    )
    assert nulled_diff < 1e-3

    model.trained_years = None
    tok_a2, _, _ = model._tokenize(a, generator=None)
    tok_c2, _, _ = model._tokenize(c, generator=None)
    open_diff = max(
        float((tok_a2[name]["meta"] - tok_c2[name]["meta"]).abs().max())
        for name in tok_a2
    )
    assert open_diff > 1e-2  # absolute-time channel now differs by the year gap


def test_nonempty_fallback_searches_all_groups() -> None:
    """The visible/target guarantee is satisfied from another group if the first is invalid."""
    model = _model(mask_family_probs=(0.0, 1.0, 0.0), timestep_mask_prob=1.0)
    sample = _make_sample(t=1)
    assert sample.sentinel2_l2a is not None
    s2 = sample.sentinel2_l2a.clone()
    s2[0] = MISSING_VALUE  # first modality fully invalid for sample 0
    sample = sample._replace(sentinel2_l2a=s2)

    per_group, b, device = model._tokenize(sample, generator=None)
    generator = torch.Generator().manual_seed(0)
    masks = model._sample_masks(per_group, b, device, generator)
    visible = {name: g["valid"] & ~masks[name] for name, g in per_group.items()}
    _ensure_nonempty(masks, visible, per_group)

    for i in range(b):
        assert any(v[i].any() for v in visible.values()), f"sample {i} fully key-masked"
        assert any(m[i].any() for m in masks.values()), f"sample {i} has no targets"


def test_single_valid_token_sample_keeps_a_visible_token() -> None:
    """With one valid token, the target-fallback must not consume it."""
    model = _model(mask_family_probs=(1.0, 0.0, 0.0), token_mask_prob=0.0)
    sample = _make_sample(t=1)
    assert sample.sentinel2_l2a is not None and sample.sentinel1 is not None
    s2 = torch.full_like(sample.sentinel2_l2a, MISSING_VALUE)
    s1 = torch.full_like(sample.sentinel1, MISSING_VALUE)
    s2[0, :8, :8] = 1.0  # exactly one valid 8x8 token (group bs0..bs2), sample 0
    sample = sample._replace(sentinel2_l2a=s2, sentinel1=s1)

    per_group, b, device = model._tokenize(sample, generator=None)
    generator = torch.Generator().manual_seed(0)
    masks = model._sample_masks(per_group, b, device, generator)
    visible = {name: g["valid"] & ~masks[name] for name, g in per_group.items()}
    _ensure_nonempty(masks, visible, per_group)

    total_visible = sum(int(v[0].sum()) for v in visible.values())
    assert total_visible >= 1, "the sole valid token was consumed as a target"


def test_encode_without_latlon_uses_trained_null_channel() -> None:
    """Unknown georef -> trained null (never fabricated); a real latlon differs."""
    model = _model()
    model.eval()
    sample = _make_sample()
    absent = sample._replace(latlon=None)
    nan_latlon = sample._replace(latlon=torch.full_like(sample.latlon, float("nan")))
    with torch.no_grad():
        from_absent = model.encode(absent)
        from_nan = model.encode(nan_latlon)
        from_real = model.encode(sample)
    assert torch.allclose(from_absent, from_nan, atol=1e-6)
    assert not torch.allclose(from_absent, from_real, atol=1e-4)


def test_config_build_roundtrips() -> None:
    """The config builds the model and serializes/deserializes (old-ckpt safe)."""
    cfg = SetLatentPerceiverConfig(
        supported_modality_names=["sentinel2_l2a", "sentinel1"],
        dim=64,
        heads=4,
        latents=16,
        nested_latents=(8, 16),
        self_depth_per_read=1,
        level2_depth=1,
        decoder_depth=1,
        target_dim=32,
    )
    model = cfg.build()
    assert isinstance(model, SetLatentPerceiver)
    d = cfg.as_dict(exclude_none=True, include_class_name=True)
    rebuilt = SetLatentPerceiverConfig.from_dict(d)
    assert rebuilt.dim == 64 and rebuilt.supported_modality_names == [
        "sentinel2_l2a",
        "sentinel1",
    ]
