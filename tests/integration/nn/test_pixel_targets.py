"""Tests for pixel-resolution MIM targets (``nn/pixel_targets.py``).

Every draw produces :class:`PixelQueries` slots, decoded by
``Predictor.forward_pixel_queries`` and scored against ``gather_query_pixels``:

* the samplers: slot counts, which tokens and pixels each draw may name;
* a slot's query lands on the coordinate of the per-pixel latent it targets, and
  decodes exactly as the standard decoder when its pixel is the patch center;
* the train module's pixel-target forward runs end to end for every draw.
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
    Perceiver,
    PerceiverConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.pixel_targets import (
    PIXEL_TARGET_DRAWS,
    PixelQueries,
    gather_query_pixels,
    pixel_center_shift,
    sample_pixel_offsets,
    sample_pixel_queries,
    shared_pixel_queries,
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
MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.WORLDCOVER.name]
CPU = torch.device("cpu")

# v1.3 tokenizes Sentinel-2 as one band set (scripts/official/v1_2/base.py); pixel
# targets require it.
S2_ONE_BANDSET = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
        )
    }
)


def _sample(height: int = H, width: int = W) -> MaskedOlmoEarthSample:
    """S2 + WorldCover; decoded tokens sit on 4x4 blocks (whole tokens at ps 2 and 4).

    At patch size 4: three S2 tokens and two WorldCover tokens are decoded.
    """
    torch.manual_seed(1234)
    num_s2 = Modality.SENTINEL2_L2A.num_bands
    s2_mask = torch.zeros(B, height, width, T, num_s2, dtype=torch.long)
    for rows, cols, t in (
        (slice(0, 4), slice(0, 4), 0),
        (slice(4, 8), slice(4, 8), 1),
        (slice(0, 4), slice(4, 8), 1),
    ):
        s2_mask[:, rows, cols, t] = MaskValue.DECODER.value
    num_wc = Modality.WORLDCOVER.num_bands
    wc_mask = torch.zeros(B, height, width, 1, num_wc, dtype=torch.long)
    wc_mask[:, 4:8, 4:8] = MaskValue.DECODER.value
    wc_mask[:, 0:4, 0:4] = MaskValue.DECODER.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, height, width, T, num_s2),
        sentinel2_l2a_mask=s2_mask,
        worldcover=torch.randn(B, height, width, 1, num_wc),
        worldcover_mask=wc_mask,
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def _model_config(
    tokenization_config: TokenizationConfig | None = S2_ONE_BANDSET,
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


def _train_module_config(**kwargs: object) -> LatentMIMTrainModuleConfig:
    return LatentMIMTrainModuleConfig(
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
        **kwargs,  # type: ignore[arg-type]
    )


def _check_slots(
    sample: MaskedOlmoEarthSample,
    queries: dict[str, PixelQueries],
    patch_size: int,
) -> None:
    """Valid slots name decoded tokens and in-footprint pixels, one per masked token."""
    assert set(queries) == set(sample.modalities)
    for name, slots in queries.items():
        decode = token_decode_mask(sample, name, patch_size)
        assert torch.equal(slots.valid.sum(1), decode.flatten(1).sum(1))
        for b in range(B):
            picked = slots.token_index[b][slots.valid[b]]
            pix = slots.pixel[b][slots.valid[b]]
            assert (pix >= 0).all() and (pix < patch_size).all()
            assert decode[b, picked[:, 0], picked[:, 1], picked[:, 2]].all()


# --- samplers -----------------------------------------------------------------------


@pytest.mark.parametrize("patch_size", [2, 4])
def test_shared_draw_one_slot_per_token_at_its_cells_pixel(patch_size: int) -> None:
    """Every decoded token gets one slot, at the pixel drawn for its cell."""
    sample = _sample()
    grid = spatial_token_grid(sample, patch_size)
    offsets = sample_pixel_offsets(B, grid, patch_size, CPU)
    queries = shared_pixel_queries(sample, patch_size, offsets)
    _check_slots(sample, queries, patch_size)
    for name, slots in queries.items():
        for b in range(B):
            picked = slots.token_index[b][slots.valid[b]]
            # Each decoded token exactly once.
            assert len({tuple(x) for x in picked.tolist()}) == len(picked)
            for q in range(int(slots.valid[b].sum())):
                i, j, _ = slots.token_index[b, q].tolist()
                assert torch.equal(slots.pixel[b, q], offsets[b, i, j])


@pytest.mark.parametrize("patch_size", [2, 4])
def test_independent_draw_one_slot_per_token(patch_size: int) -> None:
    """One slot per decoded token; the tokens of a cell can point at different pixels."""
    torch.manual_seed(0)
    sample = _sample()
    differs = False
    for _ in range(20):
        queries = sample_pixel_queries(sample, patch_size, "independent", CPU)
        _check_slots(sample, queries, patch_size)
        s2 = queries["sentinel2_l2a"]
        for b in range(B):
            picked = s2.token_index[b][s2.valid[b]]
            assert len({tuple(x) for x in picked.tolist()}) == len(picked)
        # Cell (0, 0) is decoded in S2 at t=0 and in WorldCover: compare their pixels.
        wc = queries["worldcover"]
        s2_first = s2.pixel[:, 0]
        wc_first = wc.pixel[:, 0]
        assert torch.equal(s2.token_index[:, 0, :2], wc.token_index[:, 0, :2])
        differs |= not torch.equal(s2_first, wc_first)
    assert differs


@pytest.mark.parametrize("patch_size", [2, 4])
def test_pooled_draw_counts_and_units(patch_size: int) -> None:
    """As many slots as masked tokens; each a distinct pixel of a masked token."""
    torch.manual_seed(0)
    sample = _sample()
    max_per_token = 0
    saw_empty_token = False
    for _ in range(40):
        queries = sample_pixel_queries(sample, patch_size, "pooled", CPU)
        _check_slots(sample, queries, patch_size)
        for name, slots in queries.items():
            counts = token_decode_mask(sample, name, patch_size).flatten(1).sum(1)
            for b in range(B):
                picked = slots.token_index[b][slots.valid[b]]
                pix = slots.pixel[b][slots.valid[b]]
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


def test_samplers_refuse_non_spatial_modalities_and_unknown_draws() -> None:
    """A non-spatial token has no footprint to draw a pixel from."""
    sample = _sample()._replace(
        latlon=torch.randn(B, Modality.LATLON.num_bands),
        latlon_mask=torch.zeros(B, Modality.LATLON.num_bands, dtype=torch.long),
    )
    for draw in PIXEL_TARGET_DRAWS:
        with pytest.raises(ValueError, match="spatial modalities only"):
            sample_pixel_queries(sample, 2, draw, CPU)
    with pytest.raises(ValueError, match="draw"):
        sample_pixel_queries(_sample(), 2, "random", CPU)


def test_gather_query_pixels_picks_each_slot() -> None:
    """Slot q of sample b holds pixel (i*p + r, j*p + c) at timestep t."""
    torch.manual_seed(0)
    sample = _sample()
    patch_size = 4
    queries = sample_pixel_queries(sample, patch_size, "pooled", CPU)
    gathered = gather_query_pixels(sample, queries, patch_size)
    for name, slots in queries.items():
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


# --- decoder --------------------------------------------------------------------------


@pytest.mark.parametrize("patch_size", [2, 4])
def test_shifted_query_lands_on_its_pixel_latent(patch_size: int) -> None:
    """Patch coordinate + pixel_center_shift = the pixel's stride-1 latent coordinate."""
    gsd_ratio = CompositeEncodings.calculate_gsd_ratio(10, patch_size)
    h_p, w_p = H // patch_size, W // patch_size
    latents = Perceiver.build_pixel_latent_positions(
        1, (H, W), patch_size, gsd_ratio, CPU, stride=1
    ).view(H, W, 2)
    for i in range(h_p):
        for j in range(w_p):
            for r in range(patch_size):
                for c in range(patch_size):
                    shift = pixel_center_shift(torch.tensor([r, c]), patch_size)
                    query = torch.tensor([i, j], dtype=torch.float32) * gsd_ratio
                    torch.testing.assert_close(
                        query + shift * gsd_ratio,
                        latents[i * patch_size + r, j * patch_size + c],
                    )


def test_center_pixel_slot_is_the_standard_decoder() -> None:
    """A center-pixel slot decodes exactly as the standard decoder.

    At an odd patch size the center pixel has zero shift, so the slot path decodes
    every masked token as the standard decoder does.
    """
    torch.manual_seed(0)
    model = _model_config().build().eval()
    patch_size = 3
    sample = _sample(height=6, width=6)
    grid = spatial_token_grid(sample, patch_size)
    center = torch.full((B, *grid, 2), patch_size // 2)
    queries = shared_pixel_queries(sample, patch_size, center)
    with torch.no_grad():
        standard = model.forward(sample, patch_size)[1]
        slots_out = model.forward(sample, patch_size, pixel_queries=queries)[1]
    for name, slots in queries.items():
        ref, got = getattr(standard, name), getattr(slots_out, name)
        assert ref is not None and got is not None
        assert slots.valid.any()
        for b in range(B):
            for q in range(slots.valid.shape[1]):
                if not slots.valid[b, q]:
                    continue
                i, j, t = slots.token_index[b, q].tolist()
                torch.testing.assert_close(got[b, q, 0, 0, 0], ref[b, i, j, t, 0])


def test_slots_decode_independently() -> None:
    """A slot's output depends only on its own (token, pixel).

    Reordering and duplicating slots does not change it; moving its pixel does.
    """
    torch.manual_seed(0)
    model = _model_config().build().eval()
    sample = _sample()
    patch_size = 4
    queries = sample_pixel_queries(sample, patch_size, "pooled", CPU)
    s2 = queries["sentinel2_l2a"]
    perm = torch.arange(s2.valid.shape[1]).flip(0)
    reordered = dict(queries)
    reordered["sentinel2_l2a"] = PixelQueries(
        token_index=torch.cat([s2.token_index[:, perm], s2.token_index[:, :1]], 1),
        pixel=torch.cat([s2.pixel[:, perm], (s2.pixel[:, :1] + 1) % patch_size], 1),
        valid=torch.cat([s2.valid[:, perm], s2.valid[:, :1]], 1),
    )
    with torch.no_grad():
        out = model.forward(sample, patch_size, pixel_queries=queries)[1].sentinel2_l2a
        out2 = model.forward(sample, patch_size, pixel_queries=reordered)[1]
    assert out is not None and out2.sentinel2_l2a is not None
    n = s2.valid.shape[1]
    torch.testing.assert_close(out2.sentinel2_l2a[:, :n], out[:, perm])
    # The appended slot is slot 0's token at a different pixel.
    assert not torch.allclose(out2.sentinel2_l2a[:, n], out[:, 0])


# --- train module ---------------------------------------------------------------------


@pytest.mark.parametrize("draw", PIXEL_TARGET_DRAWS)
@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_train_module_pixel_target_forward(draw: str, patch_size: int) -> None:
    """Finite loss and gradients; one target per masked token for every draw."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config().build()
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = _train_module_config(pixel_target_draw=draw).build(
            model, device=CPU
        )
    sample = _sample()
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


def test_pixel_targets_refuse_multiple_bandsets() -> None:
    """With several band sets a token's pixels would be ambiguous: refuse, don't guess."""
    model: LatentMIM = _model_config(tokenization_config=None).build()  # 3 S2 band sets
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = _train_module_config().build(model, device=CPU)
    with pytest.raises(ValueError, match="one band set"):
        train_module.model_forward(_sample(), 2, train_module.token_exit_cfg)


def test_pixel_targets_require_projection_target() -> None:
    """A full target encoder cannot be projected per pixel: refuse it at build time."""
    model_config = _model_config()
    model_config.projection_only_target = False
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        with pytest.raises(ValueError, match="projection_only_target"):
            _train_module_config().build(model_config.build(), device=CPU)


def test_unknown_pixel_target_draw_is_refused() -> None:
    """A typo in the draw name fails at build time, not silently as 'shared'."""
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        with pytest.raises(ValueError, match="pixel_target_draw"):
            _train_module_config(pixel_target_draw="random").build(
                _model_config().build(), device=CPU
            )
