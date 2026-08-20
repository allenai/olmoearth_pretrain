"""Smoke test for ERA5 objective B (reconstruction).

Tests the forward pass and loss computation for:
  1. B-only: reconstruction objective alone
  2. A+B: both supervised and reconstruction objectives
  3. A-only: supervised only (regression check)

Does NOT require distributed training, GPUs, or real data — uses
synthetic tensors with the correct shapes.

Usage:
    cd /weka/dfive-default/hadriens/olmoearth_pretrain
    source ./.venv/bin/activate
    python scripts/era5_supervised/v0/smoke_test_objective_b.py
"""

from __future__ import annotations

import sys

import torch

from olmoearth_pretrain.data.constants import ERA5_INPUT_SEQUENCE_LENGTH, Modality
from olmoearth_pretrain.data.multi_task_era5_dataset import (
    Era5SslBatch,
    Era5SupervisedBatch,
)
from olmoearth_pretrain.nn.era5_decoder import Era5TimeQueryDecoderConfig
from olmoearth_pretrain.nn.era5_encoder import Era5DailyEncoderConfig
from olmoearth_pretrain.nn.transforms.era5_corruption import (
    SwtHaloSpanMaskPolicy,
    SwtNaiveMaskPolicy,
    corrupt_era5_swt,
)
from olmoearth_pretrain.nn.transforms.era5_swt import (
    StationaryWaveletTransform1d,
    multiscale_swt_loss,
    swt_band_supports,
)
from olmoearth_pretrain.train.train_module.era5_multiobjective import (
    Era5MultiObjectiveModelConfig,
    ReconstructionObjectiveConfig,
    SupervisedObjectiveConfig,
    SupervisedTaskConfig,
    _parse_recon_mode,
)

T = ERA5_INPUT_SEQUENCE_LENGTH  # 448
V = Modality.ERA5L_DAY_10.num_bands  # 14
D = 128  # small for smoke test
B = 4
SWT_BUFFER = 83


def _make_timestamps(device: torch.device) -> torch.Tensor:
    timestamps = torch.zeros(B, T, 3, dtype=torch.long, device=device)
    timestamps[..., 0] = torch.arange(1, T + 1).unsqueeze(0)
    timestamps[..., 1] = (timestamps[..., 0] - 1) * 12 // 365
    timestamps[..., 2] = 2020
    return timestamps


def _make_batch(device: torch.device = torch.device("cpu")) -> Era5SupervisedBatch:
    """Create a synthetic ERA5 supervised batch."""
    era5 = torch.randn(B, T, V, device=device)
    timestamps = _make_timestamps(device)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    return Era5SupervisedBatch(
        era5=era5,
        timestamps=timestamps,
        valid_mask=torch.ones(B, T, V, dtype=torch.bool, device=device),
        labels=labels,
        task_name="smoke_task",
    )


def _make_ssl_batch(device: torch.device = torch.device("cpu")) -> Era5SslBatch:
    """Create a synthetic ERA5 SSL batch (no label, no S2)."""
    era5 = torch.randn(B, T, V, device=device)
    timestamps = _make_timestamps(device)
    return Era5SslBatch(
        era5=era5,
        timestamps=timestamps,
        valid_mask=torch.ones(B, T, V, dtype=torch.bool, device=device),
        task_name="smoke_ssl_task",
    )


# Relative to the repo root (run from ``.../olmoearth_pretrain``).
SWT_STATS_REL = "scripts/era5_supervised/v0/norm_configs/swt_input_stats.json"


def _swt_encoder_config() -> Era5DailyEncoderConfig:
    """SWT-input encoder config (reconstruction requires is_swt_input)."""
    return Era5DailyEncoderConfig(
        embedding_size=D,
        depth=2,
        num_heads=4,
        max_sequence_length=T,
        modality_name=Modality.ERA5L_DAY_10.name.lower(),
        use_mask_embed=True,
        use_conv_stem=True,
        is_swt_input=True,
        swt_input_stats_path=SWT_STATS_REL,
    )


def test_corruption():
    """Test that both SWT masking policies produce valid mask pairs."""
    print("--- test_corruption ---")
    ts = SWT_BUFFER
    n_bands = 7
    supports = swt_band_supports([0, 1, 2, 3, 4, 5], include_approx=True)
    device = torch.device("cpu")

    # Naive budget policy.
    naive = corrupt_era5_swt(
        B, T, V, n_bands, supports, ts, device, SwtNaiveMaskPolicy(budget=0.7)
    )
    assert naive.band_mask.shape == (B, T, V * n_bands)
    assert naive.raw_loss_mask.shape == (B, T, V)
    assert not naive.band_mask[:, :ts, :].any(), "Buffer band-masked (naive)"
    assert not naive.raw_loss_mask[:, :ts, :].any(), "Buffer loss-masked (naive)"
    frac = naive.band_mask[:, ts:, :].float().mean().item()
    print(f"  Naive band-mask fraction: {frac:.3f} (budget 0.7)")

    # Halo-corrected span policy: loss mask is a strict subset of the band mask.
    halo = corrupt_era5_swt(
        B, T, V, n_bands, supports, ts, device, SwtHaloSpanMaskPolicy()
    )
    assert not halo.band_mask[:, :ts, :].any(), "Buffer band-masked (halo)"
    assert not halo.raw_loss_mask[:, :ts, :].any(), "Buffer loss-masked (halo)"
    band_any = halo.band_mask.view(B, T, V, n_bands).any(dim=-1)
    assert (halo.raw_loss_mask & ~band_any).sum() == 0, "Loss mask outside band mask"
    assert halo.raw_loss_mask.sum() < band_any.sum(), "Halo added no hidden positions"
    print("  Halo loss-mask is a strict subset of band-mask: OK")
    print("  PASS")


def test_swt():
    """Test SWT forward + loss with Haar 6-level causal and target_start cropping."""
    print("--- test_swt ---")
    x = torch.randn(B, T, V)
    x_hat = x + 0.1 * torch.randn_like(x)
    swt = StationaryWaveletTransform1d(num_channels=V, max_levels=6)
    levels = [0, 1, 2, 3, 4, 5]
    ts = SWT_BUFFER
    t_win = T - ts

    # Full-length SWT (no cropping)
    bands_full = swt(x.transpose(1, 2), levels=levels, target_start=0)
    assert len(bands_full) == 6, f"Expected 6 levels, got {len(bands_full)}"
    for i, (approx, detail) in enumerate(bands_full):
        assert approx.shape == (B, V, T), f"Level {i} approx shape: {approx.shape}"
        assert detail.shape == (B, V, T), f"Level {i} detail shape: {detail.shape}"

    # Cropped SWT (target window only)
    bands_crop = swt(x.transpose(1, 2), levels=levels, target_start=ts)
    assert len(bands_crop) == 6
    for i, (approx, detail) in enumerate(bands_crop):
        assert approx.shape == (B, V, t_win), (
            f"Level {i} cropped approx shape: {approx.shape}, expected [B,V,{t_win}]"
        )
        assert detail.shape == (B, V, t_win)
        # Cropped output should match the tail of the full output
        assert torch.allclose(approx, bands_full[i][0][:, :, ts:]), (
            f"Level {i}: cropped approx does not match full[ts:]"
        )
        assert torch.allclose(detail, bands_full[i][1][:, :, ts:]), (
            f"Level {i}: cropped detail does not match full[ts:]"
        )
    print("  Cropped SWT matches full SWT tail: OK")

    # SWT loss (uses full sequence, no cropping -- standalone helper)
    loss, metrics = multiscale_swt_loss(x_hat, x, swt, levels=levels)
    assert loss.ndim == 0, f"Loss should be scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    print(f"  SWT loss: {loss.item():.6f}")
    for k, v in metrics.items():
        print(f"  {k}: {v.item():.6f}")
    print("  PASS")


def test_decoder_forward():
    """Test the decoder produces correct output shape."""
    print("--- test_decoder_forward ---")
    decoder_cfg = Era5TimeQueryDecoderConfig(
        embedding_size=D,
        depth=2,
        num_heads=4,
        max_sequence_length=T,
        num_output_channels=V,
    )
    decoder = decoder_cfg.build()

    encoder_cfg = Era5DailyEncoderConfig(
        embedding_size=D,
        depth=2,
        num_heads=4,
        max_sequence_length=T,
        modality_name=Modality.ERA5L_DAY_10.name.lower(),
    )
    encoder = encoder_cfg.build()

    batch = _make_batch()
    with torch.no_grad():
        out = encoder(
            era5=batch.era5,
            timestamps=batch.timestamps,
            valid_mask=batch.valid_mask,
        )
        x_hat = decoder(
            tokens=out["tokens"],
            timestamps=batch.timestamps,
        )
    assert x_hat.shape == (B, T, V), f"Expected ({B}, {T}, {V}), got {x_hat.shape}"
    assert torch.isfinite(x_hat).all(), "Non-finite values in x_hat"
    print(f"  x_hat shape: {x_hat.shape}")
    print("  PASS")


def test_b_only():
    """B-only: reconstruction objective produces finite loss, correct shapes."""
    print("--- test_b_only (reconstruction only) ---")
    model_cfg = Era5MultiObjectiveModelConfig(
        encoder_config=_swt_encoder_config(),
        supervised_objective=None,
        reconstruction_objective=ReconstructionObjectiveConfig(
            decoder=Era5TimeQueryDecoderConfig(
                embedding_size=D,
                depth=1,
                num_heads=4,
                max_sequence_length=T,
                num_output_channels=V,
            ),
            swt_levels=[0, 1],
            swt_lambda=0.1,
        ),
    )
    model = model_cfg.build()
    assert len(model.objective_list) == 1
    obj = model.objective_list[0]
    assert obj.name == "reconstruction"

    batch = _make_batch()
    loss, metrics = obj.compute(model.encoder, batch)
    assert loss.ndim == 0, f"Loss should be scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    print(f"  Total reconstruction loss: {loss.item():.6f}")
    for k, v in metrics.items():
        print(f"  {k}: {v.item():.6f}")

    # Check backward works
    loss.backward()
    encoder_grad = sum(
        p.grad.abs().sum().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    )
    assert encoder_grad > 0, "No gradients flowed to encoder"
    print(f"  Encoder grad norm (sum): {encoder_grad:.4f}")
    print("  PASS")


def test_ssl_batch_dispatch():
    """SSL batch: reconstruction fires, supervised does not; loss/grads flow."""
    print("--- test_ssl_batch_dispatch (Era5SslBatch) ---")
    model_cfg = Era5MultiObjectiveModelConfig(
        encoder_config=_swt_encoder_config(),
        supervised_objective=SupervisedObjectiveConfig(
            tasks=[
                SupervisedTaskConfig(
                    name="smoke_task",
                    task_type="classification",
                    num_classes=2,
                )
            ],
        ),
        reconstruction_objective=ReconstructionObjectiveConfig(
            decoder=Era5TimeQueryDecoderConfig(
                embedding_size=D,
                depth=1,
                num_heads=4,
                max_sequence_length=T,
                num_output_channels=V,
            ),
            swt_levels=[0, 1],
            swt_lambda=0.1,
        ),
    )
    model = model_cfg.build()
    objectives = {obj.name: obj for obj in model.objective_list}

    ssl_batch = _make_ssl_batch()

    # Reconstruction must apply to an SSL batch; supervised must not.
    assert objectives["reconstruction"].applies_to(ssl_batch), (
        "reconstruction should apply to Era5SslBatch"
    )
    assert not objectives["supervised"].applies_to(ssl_batch), (
        "supervised should NOT apply to Era5SslBatch"
    )
    print("  applies_to dispatch: OK")

    # Reconstruction forward/backward over the SSL batch.
    loss, metrics = objectives["reconstruction"].compute(model.encoder, ssl_batch)
    assert loss.ndim == 0, f"Loss should be scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    loss.backward()
    encoder_grad = sum(
        p.grad.abs().sum().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    )
    assert encoder_grad > 0, "No gradients flowed to encoder from SSL batch"
    print(f"  SSL reconstruction loss: {loss.item():.6f}")
    print(f"  Encoder grad norm (sum): {encoder_grad:.4f}")

    # Generic batch helpers preserve the SSL type and slice/move tensors.
    micro = ssl_batch.microbatch(0, 2)
    assert isinstance(micro, Era5SslBatch) and len(micro) == 2, (
        f"microbatch broke type/len: {type(micro).__name__}, {len(micro)}"
    )
    moved = ssl_batch.to_device(torch.device("cpu"))
    assert isinstance(moved, Era5SslBatch), "to_device broke SSL batch type"
    print("  microbatch/to_device preserve type: OK")
    print("  PASS")


def test_a_plus_b():
    """A+B: both objectives fire on the same batch, gradients flow."""
    print("--- test_a_plus_b (supervised + reconstruction) ---")
    model_cfg = Era5MultiObjectiveModelConfig(
        encoder_config=_swt_encoder_config(),
        supervised_objective=SupervisedObjectiveConfig(
            tasks=[
                SupervisedTaskConfig(
                    name="smoke_task",
                    task_type="classification",
                    num_classes=2,
                )
            ],
        ),
        reconstruction_objective=ReconstructionObjectiveConfig(
            decoder=Era5TimeQueryDecoderConfig(
                embedding_size=D,
                depth=1,
                num_heads=4,
                max_sequence_length=T,
                num_output_channels=V,
            ),
            swt_levels=[0],
            swt_lambda=0.1,
        ),
    )
    model = model_cfg.build()
    assert len(model.objective_list) == 2
    names = {obj.name for obj in model.objective_list}
    assert names == {"supervised", "reconstruction"}, f"Got: {names}"

    batch = _make_batch()

    # Both objectives should apply
    for obj in model.objective_list:
        assert obj.applies_to(batch), f"{obj.name} should apply to batch"

    # Run both
    total_loss = torch.zeros(())
    for obj in model.objective_list:
        loss, metrics = obj.compute(model.encoder, batch)
        assert torch.isfinite(loss), f"{obj.name} loss not finite"
        total_loss = total_loss + loss * obj.weight
        print(f"  {obj.name} loss: {loss.item():.6f}")

    total_loss.backward()
    encoder_grad = sum(
        p.grad.abs().sum().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    )
    assert encoder_grad > 0, "No gradients flowed to encoder"
    print(f"  Combined loss: {total_loss.item():.6f}")
    print(f"  Encoder grad norm (sum): {encoder_grad:.4f}")
    print("  PASS")


def test_a_only():
    """A-only: supervised objective still works (regression check)."""
    print("--- test_a_only (supervised only) ---")
    model_cfg = Era5MultiObjectiveModelConfig(
        encoder_config=Era5DailyEncoderConfig(
            embedding_size=D,
            depth=2,
            num_heads=4,
            max_sequence_length=T,
            modality_name=Modality.ERA5L_DAY_10.name.lower(),
        ),
        supervised_objective=SupervisedObjectiveConfig(
            tasks=[
                SupervisedTaskConfig(
                    name="smoke_task",
                    task_type="classification",
                    num_classes=2,
                )
            ],
        ),
        reconstruction_objective=None,
    )
    model = model_cfg.build()
    assert len(model.objective_list) == 1
    obj = model.objective_list[0]
    assert obj.name == "supervised"

    batch = _make_batch()
    loss, metrics = obj.compute(model.encoder, batch)
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    loss.backward()
    print(f"  Supervised loss: {loss.item():.6f}")
    print("  PASS")


def test_recon_mode_gating():
    """Test that group_recon_mode correctly gates raw and SWT loss channels."""
    print("--- test_recon_mode_gating ---")
    swt_levels = [0, 1, 2]

    # _parse_recon_mode unit tests (returns include_raw, detail_levels, lowpass)
    inc_raw, lvls, inc_lp = _parse_recon_mode("raw_plus_all_swt", swt_levels)
    assert inc_raw is True and lvls == [0, 1, 2] and inc_lp is False

    inc_raw, lvls, inc_lp = _parse_recon_mode("raw_plus_no_fast_swt", swt_levels)
    assert inc_raw is True and lvls == [1, 2] and inc_lp is False

    inc_raw, lvls, inc_lp = _parse_recon_mode("raw_plus_slow_swt", swt_levels)
    assert inc_raw is True and lvls == [2] and inc_lp is False

    inc_raw, lvls, inc_lp = _parse_recon_mode("lowpass_plus_slow_swt", swt_levels)
    assert inc_raw is False and lvls == [2] and inc_lp is True
    print("  _parse_recon_mode: OK")

    # End-to-end: B-only with default recon modes produces finite loss
    model_cfg = Era5MultiObjectiveModelConfig(
        encoder_config=_swt_encoder_config(),
        reconstruction_objective=ReconstructionObjectiveConfig(
            decoder=Era5TimeQueryDecoderConfig(
                embedding_size=D,
                depth=1,
                num_heads=4,
                max_sequence_length=T,
                num_output_channels=V,
            ),
            swt_levels=[0, 1, 2],
            swt_lambda=0.1,
        ),
    )
    model = model_cfg.build()
    obj = model.objective_list[0]
    batch = _make_batch()
    loss, metrics = obj.compute(model.encoder, batch)
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    assert "reconstruction/raw_loss" in metrics
    assert "reconstruction/swt_level_0_loss" in metrics
    assert "reconstruction/swt_level_1_loss" in metrics
    assert "reconstruction/swt_level_2_loss" in metrics
    loss.backward()
    print(f"  End-to-end loss: {loss.item():.6f}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v.item():.6f}")
    print("  PASS")


def test_halo_span_masking():
    """Test SwtHaloSpanMaskPolicy end-to-end: finite loss + grads."""
    print("--- test_halo_span_masking ---")
    batch = _make_batch()

    model_cfg = Era5MultiObjectiveModelConfig(
        encoder_config=_swt_encoder_config(),
        reconstruction_objective=ReconstructionObjectiveConfig(
            decoder=Era5TimeQueryDecoderConfig(
                embedding_size=D,
                depth=1,
                num_heads=4,
                max_sequence_length=T,
                num_output_channels=V,
            ),
            mask_policy=SwtHaloSpanMaskPolicy(),
            swt_levels=[0, 1],
            swt_lambda=0.1,
        ),
    )
    model = model_cfg.build()
    obj = model.objective_list[0]
    loss, metrics = obj.compute(model.encoder, batch)
    assert loss.ndim == 0, f"Loss should be scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    # Halos hide strictly more band elements than are supervised.
    assert (
        metrics["reconstruction/masked_fraction"].item()
        < metrics["reconstruction/band_masked_fraction"].item()
    ), "Halo band mask should exceed the supervised (span-only) loss mask"
    print(f"  Halo recon loss: {loss.item():.6f}")
    print(f"  Loss-mask fraction:  {metrics['reconstruction/masked_fraction']:.4f}")
    print(
        f"  Band-mask fraction:  {metrics['reconstruction/band_masked_fraction']:.4f}"
    )

    loss.backward()
    encoder_grad = sum(
        p.grad.abs().sum().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    )
    assert encoder_grad > 0, "No gradients flowed to encoder"
    print(f"  Encoder grad norm (sum): {encoder_grad:.4f}")
    print("  PASS")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"T={T}, V={V}, D={D}, B={B}\n")

    tests = [
        test_corruption,
        test_swt,
        test_decoder_forward,
        test_b_only,
        test_ssl_batch_dispatch,
        test_a_plus_b,
        test_a_only,
        test_recon_mode_gating,
        test_halo_span_masking,
    ]
    failed = []
    for test_fn in tests:
        try:
            test_fn()
            print()
        except Exception as e:
            print(f"  FAIL: {e}\n")
            failed.append(test_fn.__name__)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All smoke tests passed.")
        sys.exit(0)
