"""Per-modality calendar codes from packed timestamps (v1.3 + per-modality DOY)."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.per_modality_timestamps import (
    PACKED_TIMESTAMP_WIDTH,
    PER_MODALITY_TIMESTAMP_MODALITIES,
    pack_per_modality_timestamps,
)
from olmoearth_pretrain.nn.encodings import timestamps_to_days
from olmoearth_pretrain.nn.flexi_vit import (
    CompositeEncodings,
    EncoderConfig,
    FlexiVitBase,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.masking import MaskingConfig

S2, S1, LANDSAT = PER_MODALITY_TIMESTAMP_MODALITIES
MODS = [S2, S1, Modality.WORLDCOVER.name]


def _stamps(b: int, t: int, offset_days: int = 0) -> torch.Tensor:
    day = torch.arange(t).repeat(b, 1) * 7 % 27 + 1 + offset_days
    month = torch.arange(t).repeat(b, 1) % 12
    year = torch.full((b, t), 2022)
    return torch.stack([day, month, year], dim=-1).long()


def _packed(shared: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [shared, shared, s1, shared], dim=-1
    )  # layout (S2, S1, Landsat)


def _encodings(time_code: str = "doy", dropout: float = 0.0) -> CompositeEncodings:
    ce = CompositeEncodings(
        embedding_size=48,
        supported_modalities=[Modality.get(m) for m in MODS],
        random_channel_embeddings=True,
    )
    ce.configure_time_code(time_code, dropout)
    return ce


def test_layout_matches_packer() -> None:
    shared = np.ones((4, 3), dtype=np.int64)
    s1 = shared * 2
    np.testing.assert_array_equal(
        pack_per_modality_timestamps(shared, {S1: s1}),
        _packed(torch.from_numpy(shared), torch.from_numpy(s1)).numpy(),
    )
    assert PACKED_TIMESTAMP_WIDTH == 12


@pytest.mark.parametrize("time_code", ["month", "doy"])
def test_unpacked_and_trivially_packed_codes_equal_old_code(time_code: str) -> None:
    ce = _encodings(time_code).eval()
    shared = _stamps(2, 5)
    old = ce.compute_time_code(shared)
    for ts in (shared, _packed(shared, shared)):
        codes = ce.compute_time_codes(ts, MODS)
        for m in MODS:
            torch.testing.assert_close(codes[m], old, rtol=0, atol=0)


def test_s1_code_uses_its_own_dates() -> None:
    ce = _encodings("doy").eval()
    shared = _stamps(2, 5)
    s1 = _stamps(2, 5, offset_days=1)  # S1 one day later than S2 in every slot
    codes = ce.compute_time_codes(_packed(shared, s1), MODS)
    torch.testing.assert_close(codes[S2], ce.compute_time_code(shared), rtol=0, atol=0)
    torch.testing.assert_close(codes[S1], ce.compute_time_code(s1), rtol=0, atol=0)
    torch.testing.assert_close(codes["worldcover"], codes[S2], rtol=0, atol=0)
    assert not torch.allclose(codes[S1], codes[S2])


def test_month_code_resolves_only_months() -> None:
    """With v1.3's month code a few-day shift inside the month changes nothing."""
    ce = _encodings("month").eval()
    shared = _stamps(1, 5)
    codes = ce.compute_time_codes(_packed(shared, _stamps(1, 5, offset_days=1)), MODS)
    torch.testing.assert_close(codes[S1], codes[S2], rtol=0, atol=0)


def test_dropout_mask_shared_across_modalities() -> None:
    ce = _encodings("doy", dropout=0.5).train()
    shared = _stamps(64, 3)
    s1 = _stamps(64, 3, offset_days=3)
    torch.manual_seed(0)
    codes = ce.compute_time_codes(_packed(shared, s1), MODS)
    dropped = {m: (codes[m] == 0).all(dim=(1, 2)) for m in MODS}
    assert 0 < int(dropped[S2].sum()) < 64
    assert torch.equal(dropped[S2], dropped[S1]) and torch.equal(dropped[S2], dropped["worldcover"])
    # same RNG draw as the old single-code path when timestamps are unpacked
    torch.manual_seed(0)
    old = ce.compute_time_code(shared)
    torch.manual_seed(0)
    new = ce.compute_time_codes(shared, MODS)
    torch.testing.assert_close(new[S2], old, rtol=0, atol=0)


def test_apply_encodings_s1_tokens_get_s1_code() -> None:
    ce = _encodings("doy").eval()
    shared = _stamps(2, 4)
    s1 = _stamps(2, 4, offset_days=5)
    tokens = {
        S2: torch.zeros(2, 3, 3, 4, Modality.SENTINEL2_L2A.num_band_sets, 48),
        S1: torch.zeros(2, 3, 3, 4, Modality.SENTINEL1.num_band_sets, 48),
    }
    out_packed = ce(tokens, _packed(shared, s1), patch_size=4)
    out_shared = ce(tokens, shared, patch_size=4)
    n = 12
    torch.testing.assert_close(out_packed[S2], out_shared[S2], rtol=0, atol=0)
    # only the calendar dims [2n, 3n) of S1 tokens change
    diff = (out_packed[S1] - out_shared[S1]).abs().amax(dim=(0, 1, 2, 3, 4))
    assert diff[2 * n : 3 * n].max() > 0
    assert diff[: 2 * n].max() == 0 and diff[3 * n :].max() == 0
    # and the change is exactly (S1 code - shared code) on every patch
    delta = (out_packed[S1] - out_shared[S1])[..., 0, 2 * n : 3 * n]
    expected = (ce.compute_time_code(s1) - ce.compute_time_code(shared))[:, None, None]
    torch.testing.assert_close(delta, expected.expand_as(delta))


def test_3d_rope_uses_per_modality_days() -> None:
    model = FlexiVitBase(
        embedding_size=32,
        num_heads=2,
        mlp_ratio=2.0,
        depth=1,
        drop_path=0.0,
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.SENTINEL1],
        max_sequence_length=12,
        position_encoding="rope_3d",
    )
    shared = torch.tensor([[[1, 0, 2023], [1, 1, 2023], [1, 6, 2023]]])
    s1 = torch.tensor([[[5, 0, 2023], [9, 1, 2023], [2, 6, 2023]]])
    tokens = {S2: torch.zeros(1, 1, 1, 3, 1, 32), S1: torch.zeros(1, 1, 1, 3, 1, 32)}
    masks = {f"{S2}_mask": torch.zeros(1, 1, 1, 3, 1), f"{S1}_mask": torch.zeros(1, 1, 1, 3, 1)}
    pos = model.build_rope_positions(
        tokens_only_dict=tokens,
        original_masks_dict=masks,
        patch_size=4,
        input_res=10,
        timestamps=_packed(shared, s1),
    )
    expected = torch.cat([timestamps_to_days(shared)[0], timestamps_to_days(s1)[0]])
    torch.testing.assert_close(
        torch.sort(pos[0, :, 0]).values,
        torch.sort(expected * model.rope_temporal_coordinate_scale).values,
    )
    # unpacked timestamps: both modalities on the shared days, as before
    pos_old = model.build_rope_positions(
        tokens_only_dict=tokens,
        original_masks_dict=masks,
        patch_size=4,
        input_res=10,
        timestamps=shared,
    )
    pos_triv = model.build_rope_positions(
        tokens_only_dict=tokens,
        original_masks_dict=masks,
        patch_size=4,
        input_res=10,
        timestamps=_packed(shared, shared),
    )
    torch.testing.assert_close(pos_old, pos_triv, rtol=0, atol=0)


def _sample(seed: int, packed: bool) -> OlmoEarthSample:
    g = torch.Generator().manual_seed(seed)
    h = w = 8
    t = 4
    shared = _stamps(1, t)[0].numpy()
    s1 = _stamps(1, t, offset_days=3 + seed)[0].numpy()
    ts = pack_per_modality_timestamps(shared, {S1: s1}) if packed else shared
    return OlmoEarthSample(
        sentinel2_l2a=torch.rand(h, w, t, Modality.SENTINEL2_L2A.num_bands, generator=g).numpy(),
        sentinel1=torch.rand(h, w, t, Modality.SENTINEL1.num_bands, generator=g).numpy(),
        worldcover=torch.rand(h, w, 1, Modality.WORLDCOVER.num_bands, generator=g).numpy(),
        timestamps=ts,
    )


def test_tiny_latent_mim_forward_backward_with_packed_timestamps() -> None:
    """Collate + v1.3 masking + encoder (3D RoPE mixed, DOY) + decoder on packed stamps."""
    enc = EncoderConfig(
        supported_modality_names=MODS,
        embedding_size=48,
        max_patch_size=8,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=12,
        position_encoding="rope_3d_mixed",
        time_code="doy",
        time_code_dropout=0.3,
    )
    dec = PredictorConfig(
        supported_modality_names=MODS,
        encoder_embedding_size=48,
        decoder_embedding_size=48,
        depth=1,
        mlp_ratio=1.0,
        num_heads=2,
        max_sequence_length=12,
        drop_path=0.0,
        output_embedding_size=None,
        position_encoding="rope",
        time_code="doy",
    )
    torch.manual_seed(0)
    model = LatentMIMConfig(encoder_config=enc, decoder_config=dec).build()
    masking = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ["worldcover"],
        }
    ).build()
    outs = {}
    for packed in (True, False):
        torch.manual_seed(1)
        np.random.seed(1)
        patch_size, batch = collate_single_masked_batched(
            [(2, _sample(i, packed)) for i in range(3)], transform=None, masking_strategy=masking
        )
        assert batch.timestamps.shape[-1] == (PACKED_TIMESTAMP_WIDTH if packed else 3)
        model.train()
        torch.manual_seed(2)
        out = model(batch, patch_size)
        latent, decoded = out[0], out[1]
        loss = sum(
            getattr(decoded, m).float().pow(2).mean()
            for m in decoded.modalities
            if getattr(decoded, m) is not None
        ) + sum(getattr(latent, m).float().pow(2).mean() for m in latent.modalities)
        model.zero_grad()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        outs[packed] = latent
    # S1 latents differ once S1 carries its own dates; masks are identical
    assert torch.equal(outs[True].sentinel1_mask, outs[False].sentinel1_mask)
    assert not torch.allclose(outs[True].sentinel1, outs[False].sentinel1)
