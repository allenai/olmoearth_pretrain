"""Tests for subsampled pixel-resolution MIM targets (``nn/pixel_targets.py``).

Covers the three pieces the ``rc_pixtgt_pix512`` arms rely on:

* ``gather_pixels`` keeps exactly the drawn pixel of every token cell;
* a shifted decoder query lands on the coordinate of the per-pixel latent it targets;
* the train module's pixel-target forward runs end to end on a small per-pixel
  latent model, with the same decode-query count as the patch-target forward.
"""

from unittest.mock import patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.flexi_vit import (
    CompositeEncodings,
    EncoderConfig,
    PerceiverConfig,
    PredictorConfig,
    build_pixel_latent_positions,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.pixel_targets import (
    PooledPixelQueries,
    gather_pixels,
    gather_pooled_pixels,
    offsets_to_query_shift,
    sample_independent_pixel_offsets,
    sample_pixel_offsets,
    sample_pooled_pixel_queries,
    spatial_token_grid,
    token_decode_mask,
)
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskingConfig,
    MaskValue,
)
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

B, H, W, T = 2, 8, 8, 2
MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]


def _make_sample() -> MaskedOlmoEarthSample:
    """S2 + latlon sample; the top-left 4x4 block is decoded at t=0."""
    torch.manual_seed(1234)
    num_bands = Modality.SENTINEL2_L2A.num_bands
    mask = torch.zeros(B, H, W, T, num_bands, dtype=torch.long)
    mask[:, 0:4, 0:4, 0, :] = MaskValue.DECODER.value
    mask[:, 4:8, 0:4, 1, :] = MaskValue.TARGET_ENCODER_ONLY.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, num_bands),
        sentinel2_l2a_mask=mask,
        latlon=torch.randn(B, Modality.LATLON.num_bands),
        latlon_mask=torch.zeros(B, Modality.LATLON.num_bands, dtype=torch.long),
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


# v1.3 tokenizes Sentinel-2 as one band set (scripts/official/v1_2/base.py).
S2_ONE_BANDSET = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
        )
    }
)


def _model_config(
    tokenization_config: TokenizationConfig | None = None,
) -> LatentMIMConfig:
    """Small rc_pix512-shaped model with a projection-only target.

    Per-pixel random-stride Perceiver latents and a 2D-RoPE decoder over them.
    """
    encoder_config = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        tokenization_config=tokenization_config,
        perceiver_config=PerceiverConfig(
            register_dim=16,
            latent_depth=2,
            pixel_latents=True,
            random_latent_stride=True,
            max_latents=32,
            eval_latent_stride=1,
        ),
    )
    decoder_config = PredictorConfig(
        supported_modality_names=MODALITIES,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_perceiver=True,
        register_dim=16,
        tokenization_config=tokenization_config,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        projection_only_target=True,
    )


def test_gather_pixels_keeps_the_drawn_pixel() -> None:
    """Cell (i, j) of the gathered field is pixel (i*p + o_r, j*p + o_c)."""
    sample = _make_sample()
    patch_size = 4
    grid = spatial_token_grid(sample, patch_size)
    assert grid == (H // patch_size, W // patch_size)
    offsets = sample_pixel_offsets(B, grid, patch_size, torch.device("cpu"))
    gathered = gather_pixels(sample, offsets, patch_size)
    assert gathered.sentinel2_l2a is not None and sample.sentinel2_l2a is not None
    assert gathered.sentinel2_l2a.shape == (
        B,
        *grid,
        T,
        Modality.SENTINEL2_L2A.num_bands,
    )
    assert (
        gathered.sentinel2_l2a_mask is not None
        and sample.sentinel2_l2a_mask is not None
    )
    for b in range(B):
        for i in range(grid[0]):
            for j in range(grid[1]):
                r = i * patch_size + int(offsets[b, i, j, 0])
                c = j * patch_size + int(offsets[b, i, j, 1])
                assert torch.equal(
                    gathered.sentinel2_l2a[b, i, j], sample.sentinel2_l2a[b, r, c]
                )
                assert torch.equal(
                    gathered.sentinel2_l2a_mask[b, i, j],
                    sample.sentinel2_l2a_mask[b, r, c],
                )
    # Non-spatial modalities pass through untouched.
    assert gathered.latlon is sample.latlon


@pytest.mark.parametrize("patch_size", [2, 4])
def test_shifted_query_lands_on_its_pixel_latent(patch_size: int) -> None:
    """The decoder's shifted query coordinate equals the drawn pixel's latent (stride 1)."""
    torch.manual_seed(0)
    model = _model_config().build()
    decoder = model.decoder
    h_p, w_p = H // patch_size, W // patch_size
    offsets = sample_pixel_offsets(B, (h_p, w_p), patch_size, torch.device("cpu"))
    shift = offsets_to_query_shift(offsets, patch_size)
    gsd_ratio = (
        CompositeEncodings.calculate_gsd_ratio(10, patch_size)
        * decoder.rope_coordinate_scale
    )
    tokens = torch.zeros(B, h_p, w_p, T, 1, 16)
    query_positions = decoder._build_2d_rope_positions_for_modality(
        modality_name="sentinel2_l2a",
        modality=Modality.SENTINEL2_L2A,
        tokens=tokens,
        gsd_ratio=gsd_ratio,
        query_pixel_shift=shift,
    )
    register_positions = build_pixel_latent_positions(
        B, (H, W), patch_size, gsd_ratio, torch.device("cpu"), stride=1
    ).view(B, H, W, 2)
    for b in range(B):
        for i in range(h_p):
            for j in range(w_p):
                r = i * patch_size + int(offsets[b, i, j, 0])
                c = j * patch_size + int(offsets[b, i, j, 1])
                for t in range(T):
                    torch.testing.assert_close(
                        query_positions[b, i, j, t, 0], register_positions[b, r, c]
                    )


def test_zero_shift_is_the_patch_query_and_a_shift_moves_it() -> None:
    """A zero shift reproduces the unshifted decoder; a real shift changes it."""
    torch.manual_seed(0)
    model = _model_config().build()
    model.eval()
    sample = _make_sample()
    patch_size = 2
    grid = (H // patch_size, W // patch_size)
    with torch.no_grad():
        base = model.forward(sample, patch_size)[1].sentinel2_l2a
        zero = model.forward(
            sample, patch_size, query_pixel_shift=torch.zeros(B, *grid, 2)
        )[1].sentinel2_l2a
        shifted = model.forward(
            sample,
            patch_size,
            query_pixel_shift=torch.full((B, *grid, 2), 0.25),
        )[1].sentinel2_l2a
    assert base is not None and zero is not None and shifted is not None
    torch.testing.assert_close(zero, base)
    assert not torch.allclose(shifted, base)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_train_module_pixel_target_forward(patch_size: int) -> None:
    """model_forward with pixel targets: finite loss, gradients, same query count."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config().build()
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = config.build(model, device=torch.device("cpu"))
    sample = _make_sample()
    loss, _latent, decoded, target_output, _metrics = train_module.model_forward(
        sample, patch_size, train_module.token_exit_cfg
    )
    assert torch.isfinite(loss)
    # One query and one target per token, exactly as with patch targets.
    assert decoded.sentinel2_l2a is not None and target_output.sentinel2_l2a is not None
    assert decoded.sentinel2_l2a.shape[:3] == (B, H // patch_size, W // patch_size)
    assert target_output.sentinel2_l2a.shape[:-1] == decoded.sentinel2_l2a.shape[:-1]
    loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_pixel_targets_require_projection_target() -> None:
    """A full target encoder cannot be projected per pixel: refuse it at build time."""
    model_config = _model_config()
    model_config.projection_only_target = False
    model = model_config.build()
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        with pytest.raises(ValueError, match="projection_only_target"):
            config.build(model, device=torch.device("cpu"))


# --- independent draw: one pixel per token -----------------------------------------------


def test_independent_gather_draws_per_timestep_and_per_modality() -> None:
    """Multitemporal modalities get a pixel per (cell, timestep); static ones per cell."""
    torch.manual_seed(0)
    base = _make_sample()
    num_wc = Modality.WORLDCOVER.num_bands
    sample = base._replace(
        worldcover=torch.randn(B, H, W, 1, num_wc),
        worldcover_mask=torch.zeros(B, H, W, 1, num_wc, dtype=torch.long),
    )
    patch_size = 4
    offsets = sample_independent_pixel_offsets(sample, patch_size, torch.device("cpu"))
    h_p, w_p = H // patch_size, W // patch_size
    assert set(offsets) == {"sentinel2_l2a", "worldcover"}
    assert offsets["sentinel2_l2a"].shape == (B, h_p, w_p, T, 2)
    assert offsets["worldcover"].shape == (B, h_p, w_p, 2)
    gathered = gather_pixels(sample, offsets, patch_size)
    assert gathered.sentinel2_l2a is not None and sample.sentinel2_l2a is not None
    assert gathered.worldcover is not None and sample.worldcover is not None
    s2_off, wc_off = offsets["sentinel2_l2a"], offsets["worldcover"]
    for b in range(B):
        for i in range(h_p):
            for j in range(w_p):
                for t in range(T):
                    r = i * patch_size + int(s2_off[b, i, j, t, 0])
                    c = j * patch_size + int(s2_off[b, i, j, t, 1])
                    assert torch.equal(
                        gathered.sentinel2_l2a[b, i, j, t],
                        sample.sentinel2_l2a[b, r, c, t],
                    )
                r = i * patch_size + int(wc_off[b, i, j, 0])
                c = j * patch_size + int(wc_off[b, i, j, 1])
                assert torch.equal(
                    gathered.worldcover[b, i, j], sample.worldcover[b, r, c]
                )
    assert gathered.latlon is sample.latlon


@pytest.mark.parametrize("patch_size", [2, 4])
def test_per_timestep_shift_lands_on_each_timesteps_pixel(patch_size: int) -> None:
    """A [B, h, w, T, 2] shift moves each timestep's query to that timestep's pixel."""
    torch.manual_seed(0)
    model = _model_config().build()
    decoder = model.decoder
    h_p, w_p = H // patch_size, W // patch_size
    offsets = torch.randint(0, patch_size, (B, h_p, w_p, T, 2))
    gsd_ratio = (
        CompositeEncodings.calculate_gsd_ratio(10, patch_size)
        * decoder.rope_coordinate_scale
    )
    query_positions = decoder._build_2d_rope_positions_for_modality(
        modality_name="sentinel2_l2a",
        modality=Modality.SENTINEL2_L2A,
        tokens=torch.zeros(B, h_p, w_p, T, 1, 16),
        gsd_ratio=gsd_ratio,
        query_pixel_shift=offsets_to_query_shift(offsets, patch_size),
    )
    latent_positions = build_pixel_latent_positions(
        B, (H, W), patch_size, gsd_ratio, torch.device("cpu"), stride=1
    ).view(B, H, W, 2)
    for b in range(B):
        for i in range(h_p):
            for j in range(w_p):
                for t in range(T):
                    r = i * patch_size + int(offsets[b, i, j, t, 0])
                    c = j * patch_size + int(offsets[b, i, j, t, 1])
                    torch.testing.assert_close(
                        query_positions[b, i, j, t, 0], latent_positions[b, r, c]
                    )


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_train_module_independent_draw_forward(patch_size: int) -> None:
    """The independent draw runs end to end with the same query/target count."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config(S2_ONE_BANDSET).build()
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
        pixel_target_draw="independent",
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = config.build(model, device=torch.device("cpu"))
    loss, _latent, decoded, target_output, _metrics = train_module.model_forward(
        _make_sample(), patch_size, train_module.token_exit_cfg
    )
    assert torch.isfinite(loss)
    assert decoded.sentinel2_l2a is not None and target_output.sentinel2_l2a is not None
    assert target_output.sentinel2_l2a.shape[:-1] == decoded.sentinel2_l2a.shape[:-1]
    loss.backward()


def test_independent_draw_refuses_multiple_bandsets() -> None:
    """With several band sets a token's pixels would be ambiguous: refuse, don't guess."""
    model: LatentMIM = _model_config().build()  # default tokenization: 3 S2 band sets
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
        pixel_target_draw="independent",
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = config.build(model, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="one band set"):
        train_module.model_forward(_make_sample(), 2, train_module.token_exit_cfg)


def test_unknown_pixel_target_draw_is_refused() -> None:
    """A typo in the draw name fails at build time, not silently as 'shared'."""
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
        pixel_target_draw="random",
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        with pytest.raises(ValueError, match="pixel_target_draw"):
            config.build(_model_config().build(), device=torch.device("cpu"))


# --- pooled draw: targets drawn from all masked pixels -----------------------------------


def _spatial_sample() -> MaskedOlmoEarthSample:
    """S2 + WorldCover only (the pooled draw refuses non-spatial modalities).

    Masks sit on 4x4 blocks so every tested patch size sees whole tokens; three S2
    tokens and two WorldCover tokens are decoded at patch size 4.
    """
    base = _make_sample()
    num_s2 = Modality.SENTINEL2_L2A.num_bands
    s2_mask = torch.zeros(B, H, W, T, num_s2, dtype=torch.long)
    for rows, cols, t in (
        (slice(0, 4), slice(0, 4), 0),
        (slice(4, 8), slice(4, 8), 1),
        (slice(0, 4), slice(4, 8), 1),
    ):
        s2_mask[:, rows, cols, t] = MaskValue.DECODER.value
    num_wc = Modality.WORLDCOVER.num_bands
    wc_mask = torch.zeros(B, H, W, 1, num_wc, dtype=torch.long)
    wc_mask[:, 4:8, 4:8] = MaskValue.DECODER.value
    wc_mask[:, 0:4, 0:4] = MaskValue.DECODER.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=base.sentinel2_l2a,
        sentinel2_l2a_mask=s2_mask,
        worldcover=torch.randn(B, H, W, 1, num_wc),
        worldcover_mask=wc_mask,
        timestamps=base.timestamps,
    )


def _pooled_model() -> LatentMIM:
    config = _model_config(S2_ONE_BANDSET)
    names = [Modality.SENTINEL2_L2A.name, Modality.WORLDCOVER.name]
    config.encoder_config.supported_modality_names = names
    assert config.decoder_config is not None
    config.decoder_config.supported_modality_names = names
    return config.build()


@pytest.mark.parametrize("patch_size", [2, 4])
def test_pooled_draw_counts_and_units(patch_size: int) -> None:
    """As many slots as masked tokens; each a distinct pixel of a masked token."""
    torch.manual_seed(0)
    sample = _spatial_sample()
    max_per_token = 0
    saw_empty_token = False
    for _ in range(40):
        pooled = sample_pooled_pixel_queries(sample, patch_size, torch.device("cpu"))
        for name, slots in pooled.items():
            decode = token_decode_mask(sample, name, patch_size)
            counts = decode.flatten(1).sum(1)
            assert torch.equal(slots.valid.sum(1), counts)
            for b in range(B):
                picked = slots.token_index[b][slots.valid[b]]
                pix = slots.pixel[b][slots.valid[b]]
                assert (pix >= 0).all() and (pix < patch_size).all()
                assert decode[b, picked[:, 0], picked[:, 1], picked[:, 2]].all()
                units = {tuple(u) for u in torch.cat([picked, pix], 1).tolist()}
                assert len(units) == len(picked)  # without replacement
                per_token: dict[tuple, int] = {}
                for tok in picked.tolist():
                    per_token[tuple(tok)] = per_token.get(tuple(tok), 0) + 1
                max_per_token = max(max_per_token, max(per_token.values(), default=0))
                saw_empty_token |= len(per_token) < int(counts[b])
    # The point of the pooled draw: footprints get zero or several targets.
    assert max_per_token >= 2
    assert saw_empty_token


def test_gather_pooled_pixels_picks_each_slot() -> None:
    """Slot q of sample b holds pixel (i*p + r, j*p + c) at timestep t."""
    torch.manual_seed(0)
    sample = _spatial_sample()
    patch_size = 4
    pooled = sample_pooled_pixel_queries(sample, patch_size, torch.device("cpu"))
    gathered = gather_pooled_pixels(sample, pooled, patch_size)
    for name, slots in pooled.items():
        field, picked = getattr(sample, name), getattr(gathered, name)
        assert picked.shape[:4] == (B, slots.valid.shape[1], 1, 1)
        for b in range(B):
            for q in range(slots.valid.shape[1]):
                i, j, t = slots.token_index[b, q].tolist()
                r, c = slots.pixel[b, q].tolist()
                assert torch.equal(
                    picked[b, q, 0, 0],
                    field[b, i * patch_size + r, j * patch_size + c, t],
                )


def test_forward_pooled_matches_the_standard_decoder() -> None:
    """One slot per masked token at the shared pixel == the shared-draw decoder."""
    torch.manual_seed(0)
    model = _pooled_model().eval()
    sample = _spatial_sample()
    patch_size = 4
    h_p, w_p = H // patch_size, W // patch_size
    offsets = sample_pixel_offsets(B, (h_p, w_p), patch_size, torch.device("cpu"))
    pooled = {}
    for name in sample.modalities:
        decode = token_decode_mask(sample, name, patch_size)
        tok = decode.nonzero()  # [N, 4] (b, i, j, t), same count per sample here
        per_sample = [tok[tok[:, 0] == b, 1:] for b in range(B)]
        n = max(len(x) for x in per_sample)
        index = torch.zeros(B, n, 3, dtype=torch.long)
        valid = torch.zeros(B, n, dtype=torch.bool)
        for b, x in enumerate(per_sample):
            index[b, : len(x)] = x
            valid[b, : len(x)] = True
        pixel = offsets[torch.arange(B).view(-1, 1), index[..., 0], index[..., 1]]
        pooled[name] = PooledPixelQueries(token_index=index, pixel=pixel, valid=valid)
    with torch.no_grad():
        standard = model.forward(
            sample,
            patch_size,
            query_pixel_shift=offsets_to_query_shift(offsets, patch_size),
        )[1]
        flat = model.forward(sample, patch_size, pooled_queries=pooled)[1]
    for name, slots in pooled.items():
        ref, got = getattr(standard, name), getattr(flat, name)
        assert ref is not None and got is not None
        for b in range(B):
            for q in range(slots.valid.shape[1]):
                if not slots.valid[b, q]:
                    continue
                i, j, t = slots.token_index[b, q].tolist()
                torch.testing.assert_close(got[b, q, 0, 0, 0], ref[b, i, j, t, 0])


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_train_module_pooled_draw_forward(patch_size: int) -> None:
    """The pooled draw trains: finite loss, gradients, one target per masked token."""
    torch.manual_seed(0)
    model = _pooled_model()
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
        pixel_target_draw="pooled",
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = config.build(model, device=torch.device("cpu"))
    sample = _spatial_sample()
    loss, _latent, decoded, target_output, _metrics = train_module.model_forward(
        sample, patch_size, train_module.token_exit_cfg
    )
    assert torch.isfinite(loss)
    for name in sample.modalities:
        pred, tgt = getattr(decoded, name), getattr(target_output, name)
        assert pred is not None and tgt is not None
        assert tgt.shape[:-1] == pred.shape[:-1]
        if patch_size > 1:
            n_decoded = token_decode_mask(sample, name, patch_size).flatten(1).sum(1)
            pred_mask = getattr(decoded, f"{name}_mask")
            assert torch.equal(
                (pred_mask == MaskValue.DECODER.value).flatten(1).sum(1), n_decoded
            )
    loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
