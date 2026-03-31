#!/usr/bin/env python3
"""Benchmark static-shape encoder export and TensorRT compilation.

Tests the full pipeline: load → quantize → wrap → torch.export → TRT compile → benchmark.

Usage:
    # Verify static wrapper matches original (no TRT needed)
    python scripts/benchmark_trt_export.py --model-id OLMOEARTH_V1_NANO --skip-trt

    # Full pipeline with TensorRT
    python scripts/benchmark_trt_export.py --model-id OLMOEARTH_V1_BASE --precision fp8

    # Benchmark all precisions
    python scripts/benchmark_trt_export.py --model-id OLMOEARTH_V1_BASE --precision fp32 fp8 fp4
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import torch.utils.cpp_extension as torch_cpp_ext
    _orig = torch_cpp_ext.load
    def _patched(*a, **kw):
        f = list(kw.get("extra_cuda_cflags", []))
        f.append("-DUSE_CUDA")
        kw["extra_cuda_cflags"] = f
        return _orig(*a, **kw)
    torch_cpp_ext.load = _patched

import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.export import StaticOlmoEarthEncoder, verify_export
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.quantization import check_modelopt_available, quantize_model


def make_calib_fn(patch_size=2, spatial_size=64, device="cuda"):
    """Create calibration function for quantization."""
    def calib_fn(m):
        m.eval()
        s2 = Modality.SENTINEL2_L2A
        for _ in range(16):
            data = torch.randn(4, spatial_size, spatial_size, 1, s2.num_bands, device=device)
            mask = torch.full(
                (4, spatial_size, spatial_size, 1, s2.num_band_sets),
                MaskValue.ONLINE_ENCODER.value, dtype=torch.long, device=device,
            )
            ts = torch.zeros(4, 1, 3, dtype=torch.long, device=device)
            sample = MaskedOlmoEarthSample(sentinel2_l2a=data, sentinel2_l2a_mask=mask, timestamps=ts)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    m(sample, patch_size=patch_size)
    return calib_fn


def benchmark_latency(model, x, ts, warmup=20, iters=100):
    """Benchmark inference latency."""
    model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x, ts)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x, ts)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return (elapsed / iters) * 1000  # ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark TRT export")
    parser.add_argument("--model-id", default="OLMOEARTH_V1_NANO", choices=[m.name for m in ModelID])
    parser.add_argument("--precision", type=str, nargs="+", default=["fp32"], choices=["fp32", "fp8", "fp4"])
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument("--skip-trt", action="store_true", help="Skip TensorRT compilation")
    parser.add_argument("--batch-sizes", type=str, default="1,4", help="Batch sizes for benchmark")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    device = "cuda"

    print("=" * 60)
    print("  OlmoEarth Static Export Benchmark")
    print(f"  Model: {args.model_id}")
    print(f"  Precisions: {args.precision}")
    print(f"  Patch size: {args.patch_size}, Spatial: {args.spatial_size}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Skip TRT: {args.skip_trt}")
    print("=" * 60)

    # Load model
    print(f"\nLoading {args.model_id}...")
    model = load_model_from_id(ModelID[args.model_id], load_weights=True).to(device).eval()
    encoder = model.encoder

    results = {}

    for precision in args.precision:
        print(f"\n{'=' * 60}")
        print(f"  {precision.upper()}")
        print("=" * 60)

        # Quantize if needed
        if precision == "fp32":
            enc = encoder
        else:
            if not check_modelopt_available():
                print("  nvidia-modelopt not available, skipping")
                continue
            enc = copy.deepcopy(encoder)
            calib_fn = make_calib_fn(args.patch_size, args.spatial_size, device)
            enc = quantize_model(enc, calib_fn, precision=precision)

        # Build static wrapper
        print("  Building static wrapper...")
        static_enc = StaticOlmoEarthEncoder(
            enc, patch_size=args.patch_size, spatial_size=args.spatial_size,
        ).to(device).eval()

        # Verify
        print("  Verifying output match...")
        sim = verify_export(enc, static_enc, num_samples=5, device=device)
        print(f"  Cosine similarity: {sim:.6f}")

        # torch.export
        print("  Running torch.export...")
        s2 = Modality.SENTINEL2_L2A
        dummy_x = torch.randn(1, args.spatial_size, args.spatial_size, 1, s2.num_bands, device=device)
        dummy_ts = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
        try:
            exported = torch.export.export(static_enc, (dummy_x, dummy_ts))
            print("  torch.export: SUCCESS")
        except Exception as e:
            print(f"  torch.export: FAILED — {e}")
            exported = None

        # Benchmark static encoder (eager)
        print("\n  Latency (static encoder, eager):")
        for bs in batch_sizes:
            x = torch.randn(bs, args.spatial_size, args.spatial_size, 1, s2.num_bands, device=device)
            ts = torch.zeros(bs, 1, 3, dtype=torch.long, device=device)
            try:
                lat = benchmark_latency(static_enc, x, ts)
                print(f"    bs={bs}: {lat:.1f} ms ({bs/lat*1000:.0f} samples/s)")
                results.setdefault(precision, {})[bs] = {"eager_ms": lat}
            except RuntimeError as e:
                if "CUDA_HOME" in str(e) or "MX" in str(e):
                    print(f"    bs={bs}: skipped (CUDA toolkit needed)")
                else:
                    raise

        # TensorRT compilation
        if not args.skip_trt and exported is not None:
            print("\n  Compiling with TensorRT...")
            try:
                import torch_tensorrt
                for bs in batch_sizes:
                    x = torch.randn(bs, args.spatial_size, args.spatial_size, 1, s2.num_bands, device=device)
                    ts = torch.zeros(bs, 1, 3, dtype=torch.long, device=device)

                    compiled = torch_tensorrt.dynamo.compile(
                        exported,
                        inputs=[
                            torch_tensorrt.Input(shape=x.shape, dtype=x.dtype),
                            torch_tensorrt.Input(shape=ts.shape, dtype=ts.dtype),
                        ],
                    )

                    # Verify TRT output
                    with torch.no_grad():
                        trt_out = compiled(x, ts)
                        eager_out = static_enc(x, ts)
                    trt_sim = F.cosine_similarity(trt_out.flatten(), eager_out.flatten(), dim=0).item()
                    print(f"    bs={bs}: TRT cosine sim = {trt_sim:.6f}")

                    # Benchmark TRT
                    lat = benchmark_latency(compiled, x, ts)
                    print(f"    bs={bs}: TRT latency = {lat:.1f} ms ({bs/lat*1000:.0f} samples/s)")
                    if bs in results.get(precision, {}):
                        results[precision][bs]["trt_ms"] = lat

            except ImportError:
                print("  torch_tensorrt not installed, skipping")
            except Exception as e:
                print(f"  TRT compilation failed: {e}")

        if precision != "fp32":
            del enc
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Precision':<10} {'Batch':>6} {'Eager(ms)':>10} {'TRT(ms)':>10} {'Speedup':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for prec, bs_results in results.items():
        for bs, r in bs_results.items():
            eager = r.get("eager_ms", 0)
            trt = r.get("trt_ms")
            trt_str = f"{trt:.1f}" if trt else "N/A"
            speedup = f"{eager/trt:.1f}x" if trt else "N/A"
            print(f"  {prec.upper():<10} {bs:>6} {eager:>10.1f} {trt_str:>10} {speedup:>8}")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
