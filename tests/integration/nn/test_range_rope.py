"""Interval-valued (sinc-gated) mixed 3D RoPE and its use for latents and reads."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.encodings import (
    apply_3d_mixed_rope,
    init_3d_mixed_rope_freqs,
)
from olmoearth_pretrain.nn.flexi_vit import Encoder, PerceiverConfig
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig, JointLatentTransformer
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def test_extent_gate_equals_the_average_rotation_over_the_interval() -> None:
    """The gated rotation IS the mean of point rotations over the interval.

    Sample many times uniformly across ``[c - W/2, c + W/2]``, rotate the same vector
    at each, average: that must match one gated rotation at the centre with extent W.
    Spatial coordinates are held fixed so only the temporal axis is integrated.
    """
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 3, 16
    freqs = init_3d_mixed_rope_freqs(D, H, base=100.0)
    # Make the temporal frequencies span "barely turns" to "several turns" per window.
    freqs[0] = torch.linspace(0.05, 12.0, H * (D // 2)).reshape(H, D // 2)
    x = torch.randn(B, H, N, D)
    centre = torch.tensor([[[3.0, 1.0, 2.0], [7.5, 0.0, 5.0], [-2.0, 4.0, 4.0]]])
    width = torch.tensor([[1.0, 2.5, 0.0]])
    gated = apply_3d_mixed_rope(x, centre, freqs, extent=width)

    grid = torch.linspace(-0.5, 0.5, 4001)  # fine uniform grid over the interval
    acc = torch.zeros_like(x)
    for u in grid:
        pos = centre.clone()
        pos[..., 0] = centre[..., 0] + u * width
        acc += apply_3d_mixed_rope(x, pos, freqs)
    averaged = acc / len(grid)
    torch.testing.assert_close(gated, averaged, atol=2e-3, rtol=1e-3)
    # Zero extent is exactly the plain rotation.
    torch.testing.assert_close(
        gated[:, :, 2], apply_3d_mixed_rope(x, centre, freqs)[:, :, 2]
    )


def test_extent_gate_attenuates_fine_pairs_and_keeps_coarse_ones() -> None:
    """Fine pairs are silenced, coarse pairs kept.

    A pair turning >= 1 time across the window is (nearly) silenced; a pair turning a
    small fraction of a turn keeps (nearly) its length.
    """
    B, H, N, D = 1, 1, 1, 8
    freqs = torch.zeros(3, H, D // 2)
    freqs[0, 0] = torch.tensor(
        [0.1, 1.0, 2 * torch.pi, 4 * torch.pi]
    )  # turns/unit width
    x = torch.ones(B, H, N, D)
    out = apply_3d_mixed_rope(x, torch.zeros(B, N, 3), freqs, extent=torch.ones(B, N))
    pair_norms = out.reshape(B, H, N, D // 2, 2).norm(dim=-1)[0, 0, 0]
    ref = torch.ones(D // 2) * (2**0.5)  # each pair started at length sqrt(2)
    assert pair_norms[0] > 0.99 * ref[0]  # 0.05 turns: untouched
    assert 0.8 * ref[1] < pair_norms[1] < 0.99 * ref[1]  # half a radian: mild
    assert pair_norms[2] < 1e-5  # exactly one turn: gone
    assert pair_norms[3] < 1e-5  # two turns: gone


def test_joint_latent_time_range_runs_and_differs_from_point_latents() -> None:
    """Interval latents run end to end on a 3D encoder and change the registers."""
    torch.manual_seed(0)

    def build(latent_time_range: bool) -> Encoder:
        return Encoder(
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.SENTINEL1],
            embedding_size=32,
            max_patch_size=4,
            min_patch_size=1,
            num_heads=4,
            mlp_ratio=2.0,
            max_sequence_length=12,
            depth=0,
            drop_path=0.0,
            position_encoding="rope_3d_mixed",
            perceiver_config=JointLatentConfig(
                register_dim=32, joint_depth=2, latent_time_range=latent_time_range
            ),
        )

    torch.manual_seed(1)
    point = build(False).eval()
    torch.manual_seed(1)
    interval = build(True).eval()
    interval.load_state_dict(point.state_dict())
    assert isinstance(interval.perceiver, JointLatentTransformer)
    assert interval.perceiver.latent_time_range
    B, H, W, T = 2, 8, 8, 4
    nb2, nb1 = Modality.SENTINEL2_L2A.num_bands, Modality.SENTINEL1.num_bands
    timestamps = torch.tensor([[[1, m, 2020] for m in (0, 3, 6, 9)]] * B).long()
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, nb2),
        sentinel2_l2a_mask=torch.full(
            (B, H, W, T, nb2), MaskValue.ONLINE_ENCODER.value
        ),
        sentinel1=torch.randn(B, H, W, T, nb1),
        sentinel1_mask=torch.full((B, H, W, T, nb1), MaskValue.ONLINE_ENCODER.value),
        timestamps=timestamps,
    )
    with torch.no_grad():
        r_point = point(sample, patch_size=2, input_res=10)["registers"]
        r_interval = interval(sample, patch_size=2, input_res=10)["registers"]
    assert r_interval.shape == (B, 4, 4, 32)
    assert torch.isfinite(r_interval).all()
    assert not torch.allclose(r_point, r_interval)
    with pytest.raises(ValueError, match="rope_3d_mixed"):
        Encoder(
            supported_modalities=[Modality.SENTINEL2_L2A],
            embedding_size=32,
            max_patch_size=4,
            min_patch_size=1,
            num_heads=4,
            mlp_ratio=2.0,
            max_sequence_length=12,
            depth=0,
            drop_path=0.0,
            position_encoding="rope",
            perceiver_config=JointLatentConfig(register_dim=32, latent_time_range=True),
        )


def test_perceiver_read_time_range_runs() -> None:
    """Interval register queries run on the time-aware Perceiver reads.

    The option is refused without read_time_rope.
    """
    torch.manual_seed(0)
    encoder = Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A],
        embedding_size=32,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=4,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=0,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        perceiver_config=PerceiverConfig(
            register_dim=32,
            latent_depth=2,
            attn_dim=32,
            read_time_rope=True,
            read_time_range=True,
        ),
    ).eval()
    assert encoder.perceiver is not None and encoder.perceiver.read_time_range
    B, H, W, T = 2, 8, 8, 3
    nb2 = Modality.SENTINEL2_L2A.num_bands
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, nb2),
        sentinel2_l2a_mask=torch.full(
            (B, H, W, T, nb2), MaskValue.ONLINE_ENCODER.value
        ),
        timestamps=torch.tensor([[[1, m, 2020] for m in (0, 4, 8)]] * B).long(),
    )
    with torch.no_grad():
        out = encoder(sample, patch_size=2, input_res=10)
    assert out["registers"].shape == (B, 4, 4, 32)
    assert torch.isfinite(out["registers"]).all()
    with pytest.raises(ValueError, match="read_time_rope"):
        PerceiverConfig(register_dim=32, read_time_range=True).validate(
            encoder_num_heads=4, position_encoding="rope_3d_mixed"
        )
