"""Integration tests for the pixel-representation decoders (reconstruction + map probe)."""

import logging

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.nn.dual_res_encoder import (
    DualResEncoder,
    DualResEncoderConfig,
)
from olmoearth_pretrain.nn.pixel_decoder import (
    PixelMapProbe,
    PixelReconstructionDecoder,
    build_recon_groupings,
)
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig

logger = logging.getLogger(__name__)

EMB = 32
PIXEL_EMB = 16
PATCH_SIZE = 4
S2 = Modality.SENTINEL2_L2A.name
WC = Modality.WORLDCOVER_ONEHOT.name
SRTM = Modality.SRTM.name
OSM = Modality.OPENSTREETMAP_RASTER.name
WORLDCEREAL = Modality.WORLDCEREAL.name
B, H, W, T = 2, 8, 8, 12
WC_CLASSES = Modality.get(WC).num_bands


def _encoder() -> DualResEncoder:
    return DualResEncoderConfig(
        supported_modality_names=[S2],
        embedding_size=EMB,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=T,
        pixel_embedding_size=PIXEL_EMB,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
    ).build()


def _sample(with_maps: bool = False) -> MaskedOlmoEarthSample:
    kwargs = dict(
        sentinel2_l2a=torch.randn(B, H, W, T, Modality.SENTINEL2_L2A.num_bands),
        timestamps=torch.stack(
            [
                torch.randint(1, 28, (B, T)),
                torch.randint(1, 12, (B, T)),
                torch.full((B, T), 2023),
            ],
            dim=-1,
        ),
    )
    if with_maps:
        labels = torch.randint(0, WC_CLASSES, (B, H, W, 1))
        kwargs["worldcover_onehot"] = torch.nn.functional.one_hot(
            labels, WC_CLASSES
        ).float()
        kwargs["srtm"] = torch.randn(B, H, W, 1, 1)
        kwargs["openstreetmap_raster"] = (
            torch.rand(B, H, W, 1, Modality.get(OSM).num_bands) > 0.5
        ).float()
        kwargs["worldcereal"] = (
            torch.rand(B, H, W, 1, Modality.get(WORLDCEREAL).num_bands) > 0.5
        ).float()
    sample = OlmoEarthSample(**kwargs)
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
        }
    ).build()
    return strategy.apply_mask(sample, patch_size=PATCH_SIZE)


def test_build_recon_groupings_only_decodes_with_online_context() -> None:
    """DECODER units are kept only in location groups that have ONLINE context."""
    torch.manual_seed(0)
    # Keys at locations 3 and 7; queries at 3 (kept), 7 (kept) and 5 (dropped).
    key_loc = torch.tensor([3, 7, 3])
    key_t = torch.tensor([0, 1, 2])
    query_loc = torch.tensor([3, 5, 7])
    query_t = torch.tensor([1, 0, 0])
    rg = build_recon_groupings(key_loc, key_t, query_loc, query_t, t_mult=4)
    assert rg.num_groups == 2
    assert rg.num_decode == 2  # the location-5 query is dropped (no online context)
    assert rg.keep.tolist() == [True, False, True]
    assert rg.max_kt == 2  # location 3 has two keys


def test_build_recon_groupings_location_subsampling() -> None:
    """location_ratio keeps a subset of locations with keys/queries consistent."""
    torch.manual_seed(0)
    n_loc = 40
    key_loc = torch.arange(n_loc).repeat_interleave(3)
    key_t = torch.arange(key_loc.numel()) % 4
    query_loc = torch.arange(n_loc)
    query_t = torch.zeros(n_loc, dtype=torch.long)
    rg = build_recon_groupings(
        key_loc, key_t, query_loc, query_t, t_mult=4, location_ratio=0.5
    )
    assert rg.num_groups == 20
    assert rg.num_decode == 20  # one query per kept location
    assert int(rg.keep.sum()) == 20
    # Every kept key's location is a kept-query location.
    kept_key_locs = set(key_loc[rg.key_order].tolist())
    kept_query_locs = set(query_loc[rg.keep].tolist())
    assert kept_key_locs == kept_query_locs
    assert rg.max_kt == 3


def test_reconstruction_decoder_trains_encoder_and_heads() -> None:
    """Reconstruction MSE is finite and its grads reach the encoder + decoder heads."""
    model = _encoder()
    dec = PixelReconstructionDecoder(
        supported_modality_names=[S2],
        pixel_embedding_size=PIXEL_EMB,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
    )
    x = _sample()
    out = model(x, patch_size=PATCH_SIZE)
    loss = dec(out["pixel_branch"], x, PATCH_SIZE)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in dec.blocks.parameters())
    assert dec.mask_tokens[S2].grad is not None
    # Grad flows back into the encoder's pixel branch.
    assert model.pixel_self_blocks is not None
    assert any(p.grad is not None for p in model.pixel_self_blocks.parameters())


def test_reconstruction_uses_cross_modality_context() -> None:
    """A modality with encode/decode band-set roles is reconstructed from the OTHER one.

    ``random_time_with_decode`` with two present band sets makes one encode-only
    (ONLINE + TARGET, no DECODER) and one decode-only (DECODER + TARGET, no ONLINE).
    The old per-modality decoder had zero reconstructable queries in this regime; the
    cross-modality decoder must reconstruct the decode-role modality's pixels from the
    encode-role modality's ONLINE reps at the same locations.
    """
    torch.manual_seed(0)
    s1 = Modality.SENTINEL1.name
    # Production regime: one band set per modality (as in scripts/vnext/dual_res),
    # which is exactly what makes same-modality reconstruction context impossible.
    tok = TokenizationConfig(
        overrides={
            S2: ModalityTokenization(band_groups=[Modality.get(S2).band_order]),
            s1: ModalityTokenization(band_groups=[Modality.get(s1).band_order]),
        }
    )
    encoder = DualResEncoderConfig(
        supported_modality_names=[S2, s1],
        embedding_size=EMB,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=T,
        tokenization_config=tok,
        pixel_embedding_size=PIXEL_EMB,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
    ).build()
    dec = PixelReconstructionDecoder(
        supported_modality_names=[S2, s1],
        pixel_embedding_size=PIXEL_EMB,
        num_heads=4,
        mlp_ratio=2.0,
        depth=1,
        tokenization_config=tok,
    )
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, Modality.SENTINEL2_L2A.num_bands),
        sentinel1=torch.randn(B, H, W, T, Modality.SENTINEL1.num_bands),
        timestamps=torch.stack(
            [
                torch.randint(1, 28, (B, T)),
                torch.randint(1, 12, (B, T)),
                torch.full((B, T), 2023),
            ],
            dim=-1,
        ),
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
        },
        tokenization_config=tok,
    ).build()
    x = strategy.apply_mask(sample, patch_size=PATCH_SIZE)

    # The role split holds: neither modality has both ONLINE and DECODER units in the
    # same instance (this is exactly the regime that silenced the old decoder).
    from olmoearth_pretrain.datatypes import MaskValue

    for m in (S2, s1):
        mask = getattr(x, f"{m}_mask")
        online_i = (mask == MaskValue.ONLINE_ENCODER.value).flatten(1).any(1)
        decode_i = (mask == MaskValue.DECODER.value).flatten(1).any(1)
        assert not bool((online_i & decode_i).any()), (
            f"expected {m} to be encode- or decode-only per instance"
        )

    out = encoder(x, patch_size=PATCH_SIZE)
    loss = dec(out["pixel_branch"], x, PATCH_SIZE)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0, "cross-modality reconstruction found no queries"
    loss.backward()
    # Both modalities' mask tokens are exercised across the batch (each instance
    # decodes one of the two), and gradient reaches the encoder pixel branch.
    assert any(
        dec.mask_tokens[m].grad is not None and dec.mask_tokens[m].grad.abs().sum() > 0
        for m in (S2, s1)
    )
    assert encoder.pixel_self_blocks is not None
    assert any(p.grad is not None for p in encoder.pixel_self_blocks.parameters())


def test_map_probe_ce_mse_and_bce() -> None:
    """The map probe produces finite CE + MSE + BCE losses and trains through reps."""
    model = _encoder()
    probe = PixelMapProbe(
        pixel_embedding_size=PIXEL_EMB,
        map_targets={WC: "ce", SRTM: "mse", OSM: "bce", WORLDCEREAL: "bce"},
    )
    x = _sample(with_maps=True)
    out = model(x, patch_size=PATCH_SIZE)
    loss = probe(out["pixel_branch"], x, PATCH_SIZE)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in probe.heads.parameters())
    assert model.pixel_self_blocks is not None
    assert any(p.grad is not None for p in model.pixel_self_blocks.parameters())


def test_reconstruction_finite_when_no_decode_targets() -> None:
    """With an all-ONLINE sample (no decode units) the recon loss is finite (keep-alive)."""
    model = _encoder()
    dec = PixelReconstructionDecoder(
        supported_modality_names=[S2], pixel_embedding_size=PIXEL_EMB, depth=1
    )
    x = _sample().unmask()  # no DECODER tokens remain
    out = model(x, patch_size=PATCH_SIZE)
    loss = dec(out["pixel_branch"], x, PATCH_SIZE)
    assert torch.isfinite(loss)
    loss.backward()
    # Keep-alive: heads still receive (zero) gradient so FSDP collectives fire.
    assert all(p.grad is not None for head in dec.heads[S2] for p in head.parameters())
