"""Local smoke test for the 3D RoPE temporal-coordinate plumbing.

Run directly without launching a job:

    python scripts/official/v1_1/smoke_temporal_rope.py

What this verifies (no GPU required):

  1. ``timestamps_to_days`` produces sensible day counts and calendar deltas.
  2. ``apply_3d_axial_rope`` honors a separate ``temporal_base``.
  3. The encoder under ``rope_3d`` rotates query/keys with calendar deltas
     (different timestamps -> different output features) while preserving
     vector norms (RoPE invariant).
  4. Switching to ``rope_3d_mixed`` works end-to-end and gradients flow
     into the learnable 3-vec frequencies.
  5. The predictor cross-attention also accepts 3D RoPE positions.

Each check prints a one-line status. Exits non-zero on any failure.
"""

from __future__ import annotations

import sys
import traceback

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.encodings import (
    apply_3d_axial_rope,
    axial_3d_dim_split,
    timestamps_to_days,
)
from olmoearth_pretrain.nn.flexi_vit import Encoder, Predictor, TokensAndMasks
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def _check(name: str, cond: bool, detail: str = "") -> None:
    """Print PASS/FAIL line for a single check; raise on failure."""
    status = "PASS" if cond else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not cond:
        raise AssertionError(name)


def check_timestamps_to_days() -> None:
    """Verify the days-since-2000 computation on hand-picked dates."""
    timestamps = torch.tensor(
        [
            [1, 0, 2000],  # 2000-01-01
            [1, 1, 2000],  # 2000-02-01 -> 31
            [15, 7, 2023],  # 2023-08-15
        ],
        dtype=torch.long,
    )
    days = timestamps_to_days(timestamps)
    _check(
        "timestamps_to_days(2000-01-01) == 0",
        abs(days[0].item() - 0.0) < 1e-3,
        f"got {days[0].item():.3f}",
    )
    _check(
        "timestamps_to_days(2000-02-01) == 31",
        abs(days[1].item() - 31.0) < 1e-3,
        f"got {days[1].item():.3f}",
    )
    expected_2023 = 23 * 365.25 + 212 + 14
    _check(
        "timestamps_to_days(2023-08-15) ≈ expected",
        abs(days[2].item() - expected_2023) < 1.0,
        f"got {days[2].item():.3f}, expected {expected_2023:.3f}",
    )


def check_temporal_base_changes_only_temporal_slice() -> None:
    """Confirm temporal_base only rotates the t chunk, leaving spatial intact."""
    x = torch.randn(1, 2, 4, 16)
    positions = torch.tensor([[[1.0, 0.0, 0.0]] * 4])
    out_default = apply_3d_axial_rope(
        x, positions, base=10000.0, temporal_dim_frac=0.25
    )
    out_separate = apply_3d_axial_rope(
        x, positions, base=10000.0, temporal_dim_frac=0.25, temporal_base=1000.0
    )
    d_t, _, _ = axial_3d_dim_split(16, 0.25)
    _check(
        "spatial slice unchanged when only temporal_base differs",
        torch.allclose(out_default[..., d_t:], out_separate[..., d_t:]),
    )
    _check(
        "temporal slice changes when temporal_base differs",
        not torch.allclose(out_default[..., :d_t], out_separate[..., :d_t]),
    )
    _check(
        "norms preserved under apply_3d_axial_rope",
        torch.allclose(out_separate.norm(dim=-1), x.norm(dim=-1), atol=1e-5),
    )


def _make_sample(timestamps: torch.Tensor) -> MaskedOlmoEarthSample:
    """Build a minimal (S2 + latlon) masked sample for the given timestamps."""
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    latlon_bands = Modality.LATLON.num_bands
    B, H, W, T = 1, 8, 8, timestamps.shape[1]
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        sentinel2_l2a_mask=torch.zeros(B, H, W, T, s2_bands, dtype=torch.long),
        latlon=torch.randn(B, latlon_bands),
        latlon_mask=torch.zeros(B, latlon_bands, dtype=torch.long),
        timestamps=timestamps,
    )


def check_encoder_rope_3d_uses_real_deltas() -> None:
    """Encoder forward should change when timestamps shift by real days."""
    torch.manual_seed(0)
    encoder = Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        embedding_size=32,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        spatial_pos_encoding="rope_3d",
        rope_temporal_base=1000.0,
    )
    encoder.eval()

    # Same seed for input data, only timestamps differ.
    torch.manual_seed(123)
    sample_close = _make_sample(
        torch.tensor([[[1, 0, 2023], [1, 1, 2023]]], dtype=torch.long)
    )
    torch.manual_seed(123)
    sample_far = _make_sample(
        torch.tensor([[[1, 0, 2023], [1, 6, 2023]]], dtype=torch.long)
    )

    with torch.inference_mode():
        out_close, _, _ = unpack_encoder_output(
            encoder.forward(sample_close, patch_size=4, input_res=10)
        )
        out_far, _, _ = unpack_encoder_output(
            encoder.forward(sample_far, patch_size=4, input_res=10)
        )
    diff = (out_close.sentinel2_l2a - out_far.sentinel2_l2a).abs().max().item()
    _check(
        "rope_3d encoder output shifts with timestamp deltas",
        diff > 1e-4,
        f"max abs diff = {diff:.4e}",
    )


def check_encoder_rope_3d_mixed_grad_flow() -> None:
    """rope_3d_mixed should backprop into the learnable per-head freqs."""
    torch.manual_seed(0)
    encoder = Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        embedding_size=32,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        spatial_pos_encoding="rope_3d_mixed",
        rope_temporal_coordinate_scale=1.0 / 30.0,
    )
    sample = _make_sample(
        torch.tensor([[[1, 0, 2023], [1, 6, 2023]]], dtype=torch.long)
    )
    out, _, _ = unpack_encoder_output(
        encoder.forward(sample, patch_size=4, input_res=10)
    )
    out.sentinel2_l2a.sum().backward()
    grads = []
    for blk in encoder.blocks:
        g = blk.attn.rope_mixed_freqs.grad
        if g is not None:
            grads.append(g)
            assert torch.isfinite(g).all()
    _check(
        "rope_3d_mixed: at least one block has finite freq grads",
        len(grads) > 0,
        f"found grads in {len(grads)}/{len(encoder.blocks)} blocks",
    )
    total = sum(g.abs().sum().item() for g in grads)
    _check(
        "rope_3d_mixed: total freq grad mass > 0", total > 0, f"sum |g| = {total:.4e}"
    )


def check_predictor_rope_3d_cross_attn() -> None:
    """Predictor cross-attention should accept 3D RoPE positions and backprop."""
    torch.manual_seed(0)
    predictor = Predictor(
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        encoder_embedding_size=32,
        decoder_embedding_size=32,
        depth=2,
        mlp_ratio=2.0,
        num_heads=2,
        max_sequence_length=12,
        drop_path=0.0,
        spatial_pos_encoding="rope_3d",
        rope_temporal_base=1000.0,
    )
    s2_band_sets = len(Modality.SENTINEL2_L2A.band_sets)
    latlon_band_sets = len(Modality.LATLON.band_sets)
    B, H, W, T, D = 1, 2, 2, 2, 32
    s2 = torch.randn(B, H, W, T, s2_band_sets, D, requires_grad=True)
    s2_mask = torch.zeros(B, H, W, T, s2_band_sets, dtype=torch.float32)
    s2_mask[:, 0, 0, :, :] = MaskValue.DECODER.value
    latlon = torch.randn(B, latlon_band_sets, D, requires_grad=True)
    latlon_mask = torch.zeros(B, latlon_band_sets, dtype=torch.float32)
    timestamps = torch.tensor([[[1, 0, 2023], [1, 6, 2023]]], dtype=torch.long)

    out = predictor.forward(
        TokensAndMasks(
            sentinel2_l2a=s2,
            sentinel2_l2a_mask=s2_mask,
            latlon=latlon,
            latlon_mask=latlon_mask,
        ),
        timestamps,
        patch_size=4,
        input_res=10,
    )
    out.sentinel2_l2a.sum().backward()
    _check(
        "predictor rope_3d: q-projection grad is non-empty",
        predictor.blocks[0].attn.q.weight.grad is not None
        and predictor.blocks[0].attn.q.weight.grad.abs().sum().item() > 0,
    )


def main() -> None:
    """Run all smoke checks; exit non-zero if any fail."""
    checks = [
        ("timestamps_to_days", check_timestamps_to_days),
        ("temporal_base", check_temporal_base_changes_only_temporal_slice),
        ("encoder rope_3d real-deltas", check_encoder_rope_3d_uses_real_deltas),
        ("encoder rope_3d_mixed grad flow", check_encoder_rope_3d_mixed_grad_flow),
        ("predictor rope_3d cross-attn", check_predictor_rope_3d_cross_attn),
    ]
    failures: list[str] = []
    for name, fn in checks:
        try:
            fn()
        except Exception:
            failures.append(name)
            traceback.print_exc()

    print()
    if failures:
        print(f"FAILED: {len(failures)} of {len(checks)} checks: {failures}")
        sys.exit(1)
    print(f"OK: all {len(checks)} smoke checks passed.")


if __name__ == "__main__":
    main()
