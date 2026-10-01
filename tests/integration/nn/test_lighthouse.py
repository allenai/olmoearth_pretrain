"""Tests for Lighthouse inference (``nn/lighthouse.py``), on the dense CPU path."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig, JointLatentTransformer
from olmoearth_pretrain.nn.lighthouse import (
    BLOCK,
    LighthouseSettings,
    _build_layout,
    lighthouse_dense_mask,
    lighthouse_reach_px,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def _encoder(joint_depth: int = 2) -> Encoder:
    torch.manual_seed(0)
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
            register_dim=32,
            joint_depth=joint_depth,
            latent_reads_all=True,
            pixel_latents=True,
            eval_latent_stride=1,
            student_dims=[8],
            student_output_norm=True,
        ),
    ).eval()


def _sample(H: int, W: int, T: int = 3, seed: int = 1) -> MaskedOlmoEarthSample:
    g = torch.Generator().manual_seed(seed)
    nb2 = Modality.SENTINEL2_L2A.num_bands
    nb1 = Modality.SENTINEL1.num_bands
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(1, H, W, T, nb2, generator=g),
        sentinel2_l2a_mask=torch.full(
            (1, H, W, T, nb2), MaskValue.ONLINE_ENCODER.value
        ),
        sentinel1=torch.randn(1, H, W, T, nb1, generator=g),
        sentinel1_mask=torch.full((1, H, W, T, nb1), MaskValue.ONLINE_ENCODER.value),
        timestamps=torch.tensor([[[1, 0, 2020], [1, 3, 2020], [1, 6, 2020]]]).long(),
    )


def _crop(
    sample: MaskedOlmoEarthSample, r0: int, r1: int, c0: int, c1: int
) -> MaskedOlmoEarthSample:
    d = sample.as_dict()
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = v if k == "timestamps" else v[:, r0:r1, c0:c1]
    return MaskedOlmoEarthSample(**out)


def _run(
    encoder: Encoder,
    sample: MaskedOlmoEarthSample,
    patch_size: int,
    fov_px: int | None,
    origin: tuple[int, int] = (0, 0),
) -> tuple[torch.Tensor, torch.Tensor]:
    perceiver = encoder.perceiver
    assert isinstance(perceiver, JointLatentTransformer)
    perceiver.lighthouse = (
        LighthouseSettings(fov_px=fov_px, dense=True, seq_chunk=97, origin_px=origin)
        if fov_px is not None
        else None
    )
    with torch.no_grad():
        out = encoder(sample, patch_size=patch_size, input_res=10, fast_pass=True)
    perceiver.lighthouse = None
    return out["student_registers"][0], out["registers"][0]


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_one_window_domain_reproduces_the_stock_forward(patch_size: int) -> None:
    """With the domain exactly one FOV wide, every FOV is the whole domain."""
    encoder = _encoder()
    sample = _sample(8, 8)
    stu_ref, reg_ref = _run(encoder, sample, patch_size, None)
    stu, reg = _run(encoder, sample, patch_size, fov_px=8)
    torch.testing.assert_close(reg, reg_ref, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(stu, stu_ref, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("patch_size", [1, 2])
def test_chunk_with_the_reach_as_halo_is_exact(patch_size: int) -> None:
    """A chunk carrying the receptive-field reach as halo reproduces its core."""
    encoder = _encoder(joint_depth=2)
    fov_px = 4 * patch_size
    reach = lighthouse_reach_px(fov_px, patch_size, joint_depth=2, max_patch_size=4)
    sample = _sample(36, 36)
    full, _ = _run(encoder, sample, patch_size, fov_px)
    core = (16, 20)  # pixels [16, 20) on both axes
    lo, hi = core[0] - reach, core[1] + reach
    chunk, _ = _run(
        encoder, _crop(sample, lo, hi, lo, hi), patch_size, fov_px, origin=(lo, lo)
    )
    c = slice(core[0] - lo, core[1] - lo)
    torch.testing.assert_close(
        chunk[c, c], full[core[0] : core[1], core[0] : core[1]], atol=2e-5, rtol=1e-4
    )
    # A halo short of the reach is NOT exact (the reach is real, not an over-estimate).
    lo2, hi2 = core[0] - patch_size, core[1] + patch_size
    short, _ = _run(
        encoder,
        _crop(sample, lo2, hi2, lo2, hi2),
        patch_size,
        fov_px,
        origin=(lo2, lo2),
    )
    c2 = slice(core[0] - lo2, core[1] - lo2)
    assert not torch.allclose(
        short[c2, c2], full[core[0] : core[1], core[0] : core[1]], atol=1e-4
    )


@pytest.mark.parametrize(
    ("n_h", "n_w", "fov", "group", "per_cell", "lat_per_axis"),
    [(13, 10, 4, 4, 5, 1), (9, 12, 4, 3, 7, 2), (20, 20, 8, 4, 3, 1)],
)
def test_layout_block_lists_cover_every_allowed_pair_and_key_counts(
    n_h: int, n_w: int, fov: int, group: int, per_cell: int, lat_per_axis: int
) -> None:
    """Every allowed (q, kv) pair sits in a listed block; queries see trained counts."""
    rng = np.random.default_rng(0)
    cells = np.repeat(np.arange(n_h * n_w), per_cell)
    token_cells = cells[rng.permutation(cells.size)]  # encoder order is arbitrary
    lr = np.arange(n_h * lat_per_axis) // lat_per_axis
    lc = np.arange(n_w * lat_per_axis) // lat_per_axis
    latent_cells = (lr[:, None] * n_w + lc[None, :]).reshape(-1)
    lay = _build_layout(
        token_cells, latent_cells, n_h, n_w, fov, group, torch.device("cpu")
    )
    dense = lighthouse_dense_mask(lay, fov)
    listed = torch.zeros(lay.length // BLOCK, lay.length // BLOCK, dtype=torch.bool)
    for qb in range(listed.shape[0]):
        listed[qb, lay.kv_indices[0, 0, qb, : lay.kv_num_blocks[0, 0, qb]].long()] = (
            True
        )
    covered = listed.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    assert not (dense & ~covered).any()

    valid, is_lat = lay.valid, lay.is_latent
    keys = dense & valid[None, :]
    lat_keys = (keys & is_lat[None, :]).sum(1)
    tok_keys = (keys & ~is_lat[None, :]).sum(1)
    n_lat_fov = fov * fov * lat_per_axis**2
    assert (lat_keys[valid] == n_lat_fov).all()  # every query: the FOV's latents
    assert (tok_keys[valid & is_lat] == fov * fov * per_cell).all()
    assert (tok_keys[valid & ~is_lat] == per_cell).all()  # tokens: own cell
