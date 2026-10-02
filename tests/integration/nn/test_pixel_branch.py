"""Tests for the latent-resolution thin conv pixel branch (``nn/pixel_branch.py``).

Covers the ``rc_tconv_*_pix512`` arms:

* the ONLINE-only cell pooling at every latent stride;
* init equivalence with the branch-free ``rc_pix512``-shaped encoder;
* the leakage guard (values at non-ONLINE pixels, or a fully masked band set, never
  reach the latent init), for the plain and the mask-normalized convolutions;
* the mask-normalized convolution's independence from masked neighbours;
* the locality of the space / space_time / time conv steps;
* the per-modality (``modality_concat``) register init;
* the latent grid following the drawn stride in training and gradient into the branch.

The encoder-level tests run every branch variant in ``VARIANTS``.
"""

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, PerceiverConfig
from olmoearth_pretrain.nn.pixel_branch import (
    PixelRegisterBranch,
    ThinConvStep,
    pool_online_pixels,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

REGISTER_DIM = 16
PIXEL_DIM = 16
B, H, W, T = 2, 8, 8, 2
MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LATLON.name,
]


@dataclass(frozen=True)
class Variant:
    """One pixel-branch configuration (the ``rc_tconv_*`` arms)."""

    mask_normalized: bool = False
    mixing: str | None = None
    register_pool: str | None = None


VARIANTS = {
    "space": Variant(),
    "space_mnorm": Variant(mask_normalized=True),
    "space_time_concat": Variant(mixing="space_time", register_pool="modality_concat"),
    "time_concat": Variant(mixing="time", register_pool="modality_concat"),
}
variants = pytest.mark.parametrize(
    "variant", list(VARIANTS.values()), ids=list(VARIANTS)
)


def _build_encoder(variant: Variant | None = None, seed: int = 0) -> Encoder:
    """Small rc_pix512-shaped encoder, with the pixel branch when ``variant`` is set."""
    torch.manual_seed(seed)
    perceiver = PerceiverConfig(
        register_dim=REGISTER_DIM,
        latent_depth=2,
        pixel_latents=True,
        random_latent_stride=True,
        max_latents=32,
        eval_latent_stride=1,
    )
    if variant is not None:
        perceiver.pixel_branch_type = "thinconv"
        perceiver.pixel_branch_dim = PIXEL_DIM
        perceiver.pixel_branch_depth = 2
        if variant.mask_normalized:
            perceiver.pixel_branch_mask_normalized = True
        perceiver.pixel_branch_mixing = variant.mixing
        perceiver.pixel_branch_register_pool = variant.register_pool
    return EncoderConfig(
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
        perceiver_config=perceiver,
    ).build()


def _make_sample() -> MaskedOlmoEarthSample:
    """S2 + S1 + latlon; masks on 4x4 blocks so every patch size sees whole tokens.

    S1 is a fully decoded band set (as a decode band set of ``random_time_with_decode``).
    """
    torch.manual_seed(1234)
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    s1_bands = Modality.SENTINEL1.num_bands
    s2_mask = torch.zeros(B, H, W, T, s2_bands, dtype=torch.long)
    s2_mask[:, 0:4, 0:4, 0] = MaskValue.DECODER.value
    s2_mask[:, 4:8, 0:4, 1] = MaskValue.TARGET_ENCODER_ONLY.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn(B, H, W, T, s1_bands),
        sentinel1_mask=torch.full(
            (B, H, W, T, s1_bands), MaskValue.DECODER.value, dtype=torch.long
        ),
        latlon=torch.randn(B, Modality.LATLON.num_bands),
        latlon_mask=torch.zeros(B, Modality.LATLON.num_bands, dtype=torch.long),
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def _perturb_masked(sample: MaskedOlmoEarthSample) -> MaskedOlmoEarthSample:
    """Copy of the sample with every non-ONLINE value of S2 and S1 perturbed."""
    updates = {}
    for name in (Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name):
        data = getattr(sample, name)
        mask = getattr(sample, sample.get_masked_modality_name(name))
        hidden = (mask != MaskValue.ONLINE_ENCODER.value).any(dim=-1, keepdim=True)
        assert hidden.any()
        updates[name] = data + 100.0 * torch.randn_like(data) * hidden
    return sample._replace(**updates)


def _open(encoder: Encoder) -> PixelRegisterBranch:
    """Open the zero-init handoff so the branch actually contributes."""
    assert encoder.pixel_branch is not None
    torch.manual_seed(7)
    nn.init.normal_(encoder.pixel_branch.to_register.weight, std=0.05)
    return encoder.pixel_branch


def _eval_registers(
    encoder: Encoder, sample: MaskedOlmoEarthSample, patch_size: int
) -> torch.Tensor:
    encoder.eval()
    with torch.no_grad():
        return encoder(sample, patch_size=patch_size, input_res=10)["registers"]


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_pool_online_pixels(stride: int) -> None:
    """Cells average their ONLINE pixels; cells with none are zero and flagged off."""
    torch.manual_seed(0)
    x = torch.randn(B, H, W, T, 3)
    online = torch.rand(B, H, W, T) > 0.3
    online[:, :stride, :stride, 0] = False  # one cell with no ONLINE pixel
    pooled, cell_online = pool_online_pixels(x, online, stride)
    assert pooled.shape == (B, H // stride, W // stride, T, 3)
    for b in range(B):
        for i in range(H // stride):
            for j in range(W // stride):
                for t in range(T):
                    rows = slice(i * stride, (i + 1) * stride)
                    cols = slice(j * stride, (j + 1) * stride)
                    m = online[b, rows, cols, t]
                    if m.any():
                        expected = x[b, rows, cols, t][m].mean(0)
                        torch.testing.assert_close(pooled[b, i, j, t], expected)
                        assert cell_online[b, i, j, t]
                    else:
                        assert not cell_online[b, i, j, t]
                        assert torch.equal(pooled[b, i, j, t], torch.zeros(3))


@variants
def test_init_equivalence(variant: Variant) -> None:
    """At init the branch encoder equals the branch-free one exactly."""
    plain = _build_encoder()
    branch = _build_encoder(variant)
    missing, unexpected = branch.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(k.startswith("pixel_branch.") for k in missing)
    assert branch.pixel_branch is not None
    assert not branch.pixel_branch.to_register.weight.any()
    sample = _make_sample()
    for patch_size in (1, 2, 4):
        regs_plain = _eval_registers(plain, sample, patch_size)
        regs_branch = _eval_registers(branch, sample, patch_size)
        assert regs_plain.shape == (B, H, W, REGISTER_DIM)
        assert torch.equal(regs_plain, regs_branch)


@variants
@pytest.mark.parametrize("stride", [1, 2, 4])
def test_branch_ignores_masked_values(variant: Variant, stride: int) -> None:
    """Values at non-ONLINE pixels (incl. the fully decoded S1) never reach the init."""
    encoder = _build_encoder(variant)
    branch = _open(encoder)
    sample = _make_sample()
    with torch.no_grad():
        init_a = branch(sample, 4, stride)
        init_b = branch(_perturb_masked(sample), 4, stride)
    assert init_a is not None and init_b is not None
    assert init_a.shape == (B, (H // stride) * (W // stride), REGISTER_DIM)
    assert init_a.abs().sum() > 0
    assert torch.equal(init_a, init_b)


@variants
def test_masked_band_set_contributes_nothing(variant: Variant) -> None:
    """Dropping the fully decoded S1 leaves the latent init unchanged."""
    encoder = _build_encoder(variant)
    branch = _open(encoder)
    sample = _make_sample()
    without_s1 = sample._replace(sentinel1=None, sentinel1_mask=None)
    with torch.no_grad():
        torch.testing.assert_close(branch(sample, 4, 2), branch(without_s1, 4, 2))


@variants
def test_cell_with_no_online_unit_gets_zero_init(variant: Variant) -> None:
    """A cell masked at every timestep starts from the bare learned latent."""
    encoder = _build_encoder(variant)
    branch = _open(encoder)
    sample = _make_sample()
    assert sample.sentinel2_l2a_mask is not None
    s2_mask = sample.sentinel2_l2a_mask.clone()
    s2_mask[:, 0:4, 0:4] = MaskValue.DECODER.value
    sample = sample._replace(sentinel2_l2a_mask=s2_mask)
    with torch.no_grad():
        init = branch(sample, 4, 1)
    assert init is not None
    init = init.view(B, H, W, REGISTER_DIM)
    assert not init[:, 0:4, 0:4].any()
    assert init[:, 4:, 4:].abs().sum() > 0


def test_mask_normalized_conv_ignores_masked_neighbours() -> None:
    """A partial conv step equals a conv over the window's ONLINE cells, times k^2/count."""
    torch.manual_seed(0)
    branch = PixelRegisterBranch(
        [Modality.SENTINEL2_L2A.name],
        register_dim=REGISTER_DIM,
        pixel_dim=PIXEL_DIM,
        num_steps=1,
        mask_normalized=True,
        grad_checkpointing=False,
    )
    step = branch.steps[0]
    assert isinstance(step, ThinConvStep) and step.dwconv is not None
    frames = torch.randn(1, 1, 5, 5, PIXEL_DIM)
    valid = torch.ones(1, 1, 5, 5, 1)
    valid[0, 0, 2, 2] = 0
    scale = branch._partial_conv_scale(valid)
    out = step(frames, valid, scale)
    # Reference at the centre's right neighbour (2, 3): weighted sum over its ONLINE
    # window cells, rescaled by 9 / 8.
    y = step.norm(frames)[0, 0]
    w = step.dwconv.weight[:, 0]  # [D, 3, 3]
    acc = torch.zeros(PIXEL_DIM)
    for di in range(3):
        for dj in range(3):
            r, c = 2 - 1 + di, 3 - 1 + dj
            if valid[0, 0, r, c, 0] > 0:
                acc = acc + w[:, di, dj] * y[r, c]
    expected = frames[0, 0, 2, 3] + step.mlp(acc * 9 / 8 + step.dwconv.bias)
    torch.testing.assert_close(out[0, 0, 2, 3], expected)


def test_space_step_is_the_ported_per_frame_unit() -> None:
    """``"space"`` is ``x + mlp(dwconv2d(norm(x)))`` on every frame independently."""
    torch.manual_seed(0)
    step = ThinConvStep(PIXEL_DIM, 3, 2.0, mixing="space")
    assert step.dwconv is not None and step.dwconv_t is None
    frames = torch.randn(2, 3, 5, 5, PIXEL_DIM)
    out = step(frames)
    for n in range(2):
        for t in range(3):
            x = frames[n, t][None]  # [1, h, w, D]
            y = step.dwconv(step.norm(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            torch.testing.assert_close(out[n, t], (x + step.mlp(y))[0])


@pytest.mark.parametrize("mixing", ["space", "space_time", "time"])
def test_step_locality(mixing: str) -> None:
    """Perturbing one (t=0, cell) reaches exactly the cells and timesteps each mixing reads.

    Kernels 3 x 3 in space and 3 in time, one step: ``space`` stays in frame 0 within the
    3 x 3 window; ``time`` stays at the cell and reaches t=1; ``space_time`` reaches the
    window at t=0 and t=1. Nothing reaches t=2 or another series.
    """
    torch.manual_seed(0)
    step = ThinConvStep(PIXEL_DIM, 3, 2.0, mixing=mixing)
    frames = torch.randn(2, 4, 5, 5, PIXEL_DIM)
    bumped = frames.clone()
    bumped[0, 0, 2, 2] += torch.randn(PIXEL_DIM)  # not constant: LayerNorm removes that
    with torch.no_grad():
        changed = (step(bumped) - step(frames)).abs().amax(dim=-1) > 1e-6
    expected = torch.zeros(2, 4, 5, 5, dtype=torch.bool)
    spatial = slice(1, 4) if mixing != "time" else slice(2, 3)
    timesteps = [0] if mixing == "space" else [0, 1]
    for t in timesteps:
        expected[0, t, spatial, spatial] = True
    assert torch.equal(changed, expected)


def _concat_slots(branch: PixelRegisterBranch, sample: MaskedOlmoEarthSample) -> Tensor:
    """The ``[B, h, w, M * Dp]`` input of the register projection (stride 1)."""
    captured: list[Tensor] = []
    handle = branch.to_register.register_forward_hook(
        lambda _m, inputs, _o: captured.append(inputs[0])
    )
    with torch.no_grad():
        branch(sample, 4, 1)
    handle.remove()
    return captured[0]


def test_modality_concat_slots() -> None:
    """Each modality fills its own slot; a modality with no ONLINE unit gives zeros."""
    encoder = _build_encoder(VARIANTS["space_time_concat"])
    branch = _open(encoder)
    assert branch.pixel_modality_names == [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]
    assert branch.to_register.in_features == 2 * PIXEL_DIM
    sample = _make_sample()  # S1 fully decoded
    slots = _concat_slots(branch, sample)
    assert slots.shape == (B, H, W, 2 * PIXEL_DIM)
    assert slots[..., :PIXEL_DIM].abs().amax(dim=-1).gt(0).all()
    assert not slots[..., PIXEL_DIM:].any()
    # With S1 ONLINE everywhere its slot is filled, and S2's slot is unchanged.
    assert sample.sentinel1_mask is not None
    online_s1 = sample._replace(
        sentinel1_mask=torch.full_like(
            sample.sentinel1_mask, MaskValue.ONLINE_ENCODER.value
        )
    )
    slots_s1 = _concat_slots(branch, online_s1)
    assert slots_s1[..., PIXEL_DIM:].abs().amax(dim=-1).gt(0).all()
    torch.testing.assert_close(slots_s1[..., :PIXEL_DIM], slots[..., :PIXEL_DIM])


@variants
def test_training_grid_follows_stride_and_branch_gets_gradient(
    variant: Variant,
) -> None:
    """In training the latent grid follows the drawn stride; the branch is on the graph."""
    encoder = _build_encoder(variant)
    branch = _open(encoder)
    encoder.train()
    sample = _make_sample()
    shapes = set()
    torch.manual_seed(3)
    for _ in range(8):
        regs = encoder(sample, patch_size=4, input_res=10)["registers"]
        shapes.add(tuple(regs.shape[1:3]))
    # max_latents=32 on an 8x8 image: strides 2 (4x4 = 16 latents) and 4 (2x2) fit.
    assert shapes <= {(4, 4), (2, 2)} and len(shapes) == 2
    regs = encoder(sample, patch_size=4, input_res=10)["registers"]
    (regs * torch.randn_like(regs)).sum().backward()
    grads = [p.grad for p in branch.embed.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_config_validation() -> None:
    """The branch needs pixel latents; its settings need the branch."""
    with pytest.raises(ValueError, match="pixel_latents"):
        PerceiverConfig(register_dim=8, pixel_branch_type="thinconv").validate(
            encoder_num_heads=2, position_encoding="rope"
        )
    with pytest.raises(ValueError, match="pixel_branch_type"):
        PerceiverConfig(
            register_dim=8, pixel_latents=True, pixel_branch_type="conv"
        ).validate(encoder_num_heads=2, position_encoding="rope")
    needs_type: list[dict[str, Any]] = [
        {"pixel_branch_mask_normalized": True},
        {"pixel_branch_mixing": "time"},
        {"pixel_branch_time_kernel": 3},
        {"pixel_branch_register_pool": "modality_concat"},
    ]
    for setting in needs_type:
        with pytest.raises(ValueError, match="pixel_branch_type"):
            PerceiverConfig(register_dim=8, pixel_latents=True, **setting).validate(
                encoder_num_heads=2, position_encoding="rope"
            )
    invalid: list[tuple[dict[str, Any], str]] = [
        ({"pixel_branch_mixing": "spacetime"}, "pixel_branch_mixing"),
        ({"pixel_branch_time_kernel": 2}, "pixel_branch_time_kernel"),
        ({"pixel_branch_register_pool": "concat"}, "pixel_branch_register_pool"),
        (
            {"pixel_branch_mixing": "time", "pixel_branch_mask_normalized": True},
            "mask_normalized",
        ),
    ]
    for setting, match in invalid:
        with pytest.raises(ValueError, match=match):
            PerceiverConfig(
                register_dim=8,
                pixel_latents=True,
                pixel_branch_type="thinconv",
                **setting,
            ).validate(encoder_num_heads=2, position_encoding="rope")
