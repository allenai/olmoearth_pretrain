"""Local smoke test for the Perceiver latent-bottleneck encoder.

Run directly without launching a job:

    python scripts/vnext/perceiver/smoke_perceiver.py

What this verifies (no GPU required):

  1. PerceiverEncoder forward produces per-modality outputs with baseline
     shapes, and every modality/bandset views the same fused dense map
     (strict bottleneck).
  2. The latent anchor grid has the expected strided size.
  3. Gradients flow end-to-end (patch embeds -> read blocks -> latent trunk
     -> read-out -> predictor head), including the learnable mixed-RoPE
     frequencies and the latent/query seed tokens.
  4. The frozen-target path (token_exit_cfg all zeros) still bypasses
     attention entirely.
  5. The eval fast_pass path runs.

Each check prints a one-line status. Exits non-zero on any failure.
"""

from __future__ import annotations

import sys
import traceback

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.perceiver import (
    PerceiverEncoder,
    PerceiverPredictor,
)
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

MODALITIES = [Modality.SENTINEL2_L2A, Modality.WORLDCOVER]
D = 32
PATCH = 4


def _check(name: str, cond: bool, detail: str = "") -> None:
    """Print PASS/FAIL line for a single check; raise on failure."""
    status = "PASS" if cond else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not cond:
        raise AssertionError(name)


def _make_sample(
    batch_size: int = 2, height: int = 8, timesteps: int = 4
) -> MaskedOlmoEarthSample:
    """Build a minimal (S2 + worldcover) masked sample.

    S2 gets a mix of ONLINE_ENCODER and DECODER tokens (last timestep fully
    masked, emulating time masking); worldcover is decode-only.
    """
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    wc_bands = Modality.WORLDCOVER.num_bands
    b, h, w, t = batch_size, height, height, timesteps
    s2_mask = torch.full(
        (b, h, w, t, s2_bands), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
    )
    s2_mask[:, :, :, -1] = MaskValue.DECODER.value
    timestamps = (
        torch.stack(
            [
                torch.tensor([15, month, 2023], dtype=torch.long)
                for month in range(timesteps)
            ]
        )
        .unsqueeze(0)
        .repeat(b, 1, 1)
    )
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(b, h, w, t, s2_bands),
        sentinel2_l2a_mask=s2_mask,
        worldcover=torch.randn(b, h, w, wc_bands),
        worldcover_mask=torch.full(
            (b, h, w, wc_bands), MaskValue.DECODER.value, dtype=torch.long
        ),
        timestamps=timestamps,
    )


def _make_encoder(**overrides: object) -> PerceiverEncoder:
    """Small PerceiverEncoder for smoke checks."""
    kwargs: dict = dict(
        supported_modalities=MODALITIES,
        embedding_size=D,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        rope_temporal_coordinate_scale=1.0 / 30.0,
        latent_stride_hw=2,
        latent_stride_t=2,
        num_reads=2,
        readout_depth=1,
    )
    kwargs.update(overrides)
    return PerceiverEncoder(**kwargs)


def _make_predictor() -> PerceiverPredictor:
    """Matching attention-free predictor head."""
    return PerceiverPredictor(
        supported_modalities=MODALITIES,
        encoder_embedding_size=D,
        decoder_embedding_size=D,
        depth=0,
        head_depth=2,
        mlp_ratio=2.0,
        num_heads=2,
        max_sequence_length=12,
        position_encoding="rope_3d_mixed",
        rope_temporal_coordinate_scale=1.0 / 30.0,
    )


def check_shapes_and_bottleneck() -> None:
    """Output shapes match baseline conventions; all views share the dense map."""
    torch.manual_seed(0)
    encoder = _make_encoder()
    encoder.train()
    sample = _make_sample()
    out, _, _ = unpack_encoder_output(encoder.forward(sample, patch_size=PATCH))

    s2 = out.sentinel2_l2a
    wc = out.worldcover
    ph = 8 // PATCH
    s2_bandsets = len(Modality.SENTINEL2_L2A.band_sets)
    _check(
        "s2 output shape [B,PH,PW,T,bs,D]",
        tuple(s2.shape) == (2, ph, ph, 4, s2_bandsets, D),
        f"got {tuple(s2.shape)}",
    )
    _check(
        "worldcover output has same leading grid",
        wc.shape[0] == 2 and wc.shape[1] == ph and wc.shape[2] == ph,
        f"got {tuple(wc.shape)}",
    )
    _check(
        "bandsets view the same fused token",
        torch.allclose(s2[..., 0, :], s2[..., -1, :]),
    )
    # Static modality reads temporal plane 0 of the same dense map.
    wc_plane = wc[..., 0, :] if wc.ndim == 6 else wc
    while wc_plane.ndim > 4:
        wc_plane = wc_plane[..., 0, :]
    _check(
        "static modality equals dense plane t=0",
        torch.allclose(wc_plane.reshape(2, ph, ph, D), s2[:, :, :, 0, 0, :]),
    )


def check_latent_grid_size() -> None:
    """Anchored latent count follows the strides."""
    encoder = _make_encoder()
    dims = {"sentinel2_l2a": (2, 2, 2, 4, 3, D)}
    b, h, w, t = encoder._grid_dims(dims)
    _check("grid dims inferred", (b, h, w, t) == (2, 2, 2, 4))
    rows = torch.arange(1, dtype=torch.float32)
    cols = torch.arange(1, dtype=torch.float32)
    slots = torch.arange(0, 4, 2, dtype=torch.long)
    timestamps = _make_sample().timestamps
    content, coords = encoder._build_query_grid(
        encoder.latent_token, rows, cols, slots, timestamps, 1.0, 2, torch.device("cpu")
    )
    _check(
        "latent grid content/coords shapes",
        tuple(content.shape) == (2, 2, D) and tuple(coords.shape) == (2, 2, 3),
        f"content {tuple(content.shape)}, coords {tuple(coords.shape)}",
    )
    _check(
        "temporal anchor coords are scaled days",
        coords[0, 0, 0] > 200,  # ~2023 in scaled months since 2000
        f"t coord = {coords[0, 0, 0].item():.2f}",
    )


def check_grad_flow() -> None:
    """Backprop reaches every new component and the patch embeddings."""
    torch.manual_seed(0)
    encoder = _make_encoder()
    predictor = _make_predictor()
    encoder.train()
    predictor.train()
    sample = _make_sample()
    out_dict = encoder.forward(sample, patch_size=PATCH)
    latent, pooled, _ = unpack_encoder_output(out_dict)
    decoded = predictor.forward(latent, sample.timestamps, patch_size=PATCH)
    loss = decoded.sentinel2_l2a.float().pow(2).mean() + pooled.float().pow(2).mean()
    loss.backward()

    named = dict(encoder.named_parameters())
    named.update({f"pred.{k}": v for k, v in predictor.named_parameters()})
    targets = {
        "latent seed": "latent_token",
        "readout seed": "readout_query_token",
        "read block q": "read_blocks.0.attn.q.weight",
        "read block mixed freqs": "read_blocks.0.attn.rope_mixed_freqs",
        "latent trunk q": "blocks.0.attn.q.weight",
        "readout block q": "readout_blocks.0.attn.q.weight",
        "predictor head mlp": "pred.head_mlps.0.fc1.weight",
        "predictor output": "pred.to_output_embed.weight",
    }
    for label, pname in targets.items():
        param = named[pname]
        ok = param.grad is not None and param.grad.abs().sum().item() > 0
        _check(f"grad flows into {label}", ok)

    patch_grads = [
        p.grad.abs().sum().item()
        for n, p in encoder.patch_embeddings.named_parameters()
        if p.grad is not None
    ]
    _check(
        "grad flows into patch embeddings",
        len(patch_grads) > 0 and sum(patch_grads) > 0,
    )


def check_frozen_target_path() -> None:
    """token_exit_cfg all zeros bypasses attention (frozen projection targets)."""
    torch.manual_seed(0)
    encoder = _make_encoder()
    encoder.eval()
    sample = _make_sample()
    exit_cfg = {m.name: 0 for m in MODALITIES}
    with torch.inference_mode():
        out, _, _ = unpack_encoder_output(
            encoder.forward(sample.unmask(), patch_size=PATCH, token_exit_cfg=exit_cfg)
        )
    s2_bandsets = len(Modality.SENTINEL2_L2A.band_sets)
    _check(
        "frozen-target path returns patchified tokens",
        tuple(out.sentinel2_l2a.shape) == (2, 2, 2, 4, s2_bandsets, D),
        f"got {tuple(out.sentinel2_l2a.shape)}",
    )
    # Bandsets must differ here (separate projections), unlike the dense map.
    _check(
        "frozen-target bandsets are independent projections",
        not torch.allclose(out.sentinel2_l2a[..., 0, :], out.sentinel2_l2a[..., 1, :]),
    )


def check_fast_pass_eval() -> None:
    """fast_pass eval path (no mask removal) runs and matches shapes."""
    torch.manual_seed(0)
    encoder = _make_encoder()
    encoder.eval()
    sample = _make_sample().unmask()
    with torch.inference_mode():
        out_dict = encoder.forward(sample, patch_size=PATCH, fast_pass=True)
    out = out_dict["tokens_and_masks"]
    _check(
        "fast_pass output shape",
        tuple(out.sentinel2_l2a.shape)[:4] == (2, 2, 2, 4),
        f"got {tuple(out.sentinel2_l2a.shape)}",
    )
    _check("fast_pass skips pooled output", "project_aggregated" not in out_dict)


def main() -> None:
    """Run all smoke checks; exit non-zero if any fail."""
    checks = [
        ("shapes + strict bottleneck", check_shapes_and_bottleneck),
        ("latent grid size", check_latent_grid_size),
        ("grad flow", check_grad_flow),
        ("frozen-target path", check_frozen_target_path),
        ("fast_pass eval", check_fast_pass_eval),
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
