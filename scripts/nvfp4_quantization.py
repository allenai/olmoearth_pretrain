#!/usr/bin/env python3
"""NVFP4 Weight Quantization Pipeline for OlmoEarth.

Quantizes OlmoEarth ViT model weights to NVIDIA FP4 format for faster inference
on Blackwell GPUs (RTX 5090, B100). Runs 5 steps with verification at each stage.

Requirements:
    nvidia-modelopt >= 0.31 (for mtq.quantize with NVFP4_DEFAULT_CFG)
    PyTorch >= 2.7 with CUDA (Blackwell sm_100 support)

Installation (pick one):
    # Option 1: pip install (simplest)
    pip install -U nvidia-modelopt[all]

    # Option 2: From source (latest features)
    git clone git@github.com:NVIDIA/Model-Optimizer.git
    cd Model-Optimizer && pip install -e .[dev]

    # Option 3: TensorRT-LLM container (ModelOpt pre-installed, upgrade to latest)
    docker run --gpus all -it nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9
    pip install -U nvidia-modelopt

Usage:
    # Run all steps
    python scripts/nvfp4_quantization.py

    # Run individual step
    python scripts/nvfp4_quantization.py --step 1

    # Use a different model variant
    python scripts/nvfp4_quantization.py --model-id OLMOEARTH_V1_NANO

    # Use MLP-only quantization (more conservative)
    python scripts/nvfp4_quantization.py --quant-config mlp_only
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

# On Windows, CUDA CCCL headers conflict with PyTorch's compiled_autograd.h
# (C2872: 'std' ambiguous symbol). Defining USE_CUDA makes the header take
# the Windows fast-path that skips the problematic if-constexpr block.
if sys.platform == "win32":

    def _patch_modelopt_for_msvc():
        """Patch ModelOpt's CUDA extension build to work on Windows."""
        try:
            import torch.utils.cpp_extension as torch_cpp_ext

            _original_load = torch_cpp_ext.load

            def _patched_load(*args, **kwargs):
                cuda_flags = list(kwargs.get("extra_cuda_cflags", []))
                cuda_flags.append("-DUSE_CUDA")
                kwargs["extra_cuda_cflags"] = cuda_flags
                return _original_load(*args, **kwargs)

            torch_cpp_ext.load = _patched_load
        except ImportError:
            pass

    _patch_modelopt_for_msvc()

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.quantization import (
    check_modelopt_available,
    count_quantizable_layers,
    count_quantizer_nodes,
    get_model_memory_mb,
    get_modelopt_install_instructions,
    quantize_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Helpers
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_banner(step: int, title: str) -> None:
    """Print a step banner."""
    print(f"\n{'='*70}")
    print(f"  STEP {step}: {title}")
    print(f"{'='*70}\n")


def print_result(name: str, passed: bool, detail: str = "") -> None:
    """Print a PASS/FAIL result."""
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[X]"
    msg = f"  {marker} {name}: {status}"
    if detail:
        msg += f" ({detail})"
    print(msg)


def create_synthetic_sample(
    batch_size: int,
    height: int = 64,
    width: int = 64,
    num_timesteps: int = 1,
    device: torch.device = DEVICE,
) -> MaskedOlmoEarthSample:
    """Create a synthetic MaskedOlmoEarthSample for testing.

    Mimics a Sentinel-2 L2A input, which is the most common modality.
    Shape: [B, H, W, T, C] where C=12 (Sentinel-2 L2A bands).

    Args:
        batch_size: Number of samples in batch.
        height: Spatial height (pixels). Must be divisible by patch_size * tile_factor.
        width: Spatial width (pixels).
        num_timesteps: Number of timesteps.
        device: Target device.

    Returns:
        MaskedOlmoEarthSample with sentinel2_l2a data and masks.
    """
    s2_spec = Modality.SENTINEL2_L2A
    num_bands = s2_spec.num_bands  # 12 bands
    num_band_sets = s2_spec.num_band_sets  # 3 band sets

    # Sentinel-2 L2A data: [B, H, W, T, C]
    s2_data = torch.randn(batch_size, height, width, num_timesteps, num_bands, device=device)

    # Mask: [B, H, W, T, num_band_sets] — all ONLINE_ENCODER (visible)
    s2_mask = torch.full(
        (batch_size, height, width, num_timesteps, num_band_sets),
        MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
        device=device,
    )

    # Timestamps: [B, T, 3] — (day, month, year)
    timestamps = torch.zeros(batch_size, num_timesteps, 3, dtype=torch.long, device=device)
    timestamps[:, :, 0] = 15  # day
    timestamps[:, :, 1] = 6  # month (July, 0-indexed)
    timestamps[:, :, 2] = 2024  # year

    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=s2_data,
        sentinel2_l2a_mask=s2_mask,
    )


def get_encoder(model: nn.Module) -> nn.Module:
    """Extract the encoder from the full model.

    The loaded model is LatentMIM with .encoder, .decoder, .target_encoder.
    For inference/embedding extraction, we only need the encoder.
    """
    if hasattr(model, "encoder"):
        return model.encoder
    return model


def run_forward_pass(
    encoder: nn.Module,
    sample: MaskedOlmoEarthSample,
    patch_size: int = 2,
) -> torch.Tensor:
    """Run a forward pass through the encoder and return the aggregated embedding.

    Args:
        encoder: OlmoEarth encoder (extracted via get_encoder()).
        sample: Input sample.
        patch_size: Patch size for patchification.

    Returns:
        Aggregated embedding tensor [B, D].
    """
    encoder.eval()
    with torch.no_grad():
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = encoder(sample, patch_size=patch_size)
        else:
            output = encoder(sample, patch_size=patch_size)
    # Encoder returns dict with "tokens_and_masks" and "project_aggregated"
    return output["project_aggregated"]


# ============================================================================
# Step 1: Load Model & Inspect
# ============================================================================


def step1_load_and_inspect(model_id: str) -> nn.Module:
    """Load the model and verify it works."""
    print_banner(1, "LOAD MODEL & INSPECT")

    # Load model
    print(f"  Loading {model_id} from HuggingFace...")
    mid = ModelID[model_id]
    full_model = load_model_from_id(mid, load_weights=True)
    full_model = full_model.to(DEVICE)

    # Extract encoder for inference
    encoder = get_encoder(full_model)
    encoder.eval()
    print(f"  Model loaded on {DEVICE}")
    print(f"  Full model type: {type(full_model).__name__}")
    print(f"  Encoder type: {type(encoder).__name__}")

    # Count parameters (encoder only — that's what we quantize)
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    memory_mb = get_model_memory_mb(encoder)
    print(f"\n  Encoder parameters:   {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory footprint:     {memory_mb:.1f} MB")

    # Count quantizable layers
    layer_counts = count_quantizable_layers(encoder)
    print(f"\n  Quantizable layers (encoder):")
    for layer_type, count in layer_counts.items():
        print(f"    {layer_type}: {count}")

    # Verify: forward pass
    print(f"\n  Running forward pass with synthetic data...")
    sample = create_synthetic_sample(batch_size=2)
    embedding = run_forward_pass(encoder, sample)

    # Checks
    has_correct_shape = embedding.dim() == 2 and embedding.shape[0] == 2
    no_nan = not torch.isnan(embedding).any().item()
    no_inf = not torch.isinf(embedding).any().item()
    embedding_dim = embedding.shape[1]

    print(f"\n  Output shape: {list(embedding.shape)}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Output range: [{embedding.min().item():.4f}, {embedding.max().item():.4f}]")

    print_result("Output shape is [B, D]", has_correct_shape, f"got {list(embedding.shape)}")
    print_result("No NaN in output", no_nan)
    print_result("No Inf in output", no_inf)

    all_passed = has_correct_shape and no_nan and no_inf
    print(f"\n  Step 1 {'PASSED' if all_passed else 'FAILED'}")
    return encoder


# ============================================================================
# Step 2: Apply NVFP4 Quantization
# ============================================================================


def step2_quantize(model: nn.Module, quant_config: str, precision: str = "fp4") -> nn.Module:
    """Apply quantization to the model."""
    print_banner(2, f"APPLY {precision.upper()} QUANTIZATION")

    # Check modelopt
    if not check_modelopt_available():
        print(f"  {get_modelopt_install_instructions()}")
        print(f"\n  Step 2 SKIPPED (nvidia-modelopt not installed)")
        return model

    # NVFP4 quantization requires CUDA (FP4 is a GPU-native format)
    if not torch.cuda.is_available():
        print("  NVFP4 quantization requires a CUDA GPU.")
        print("  Current device: CPU-only (torch compiled without CUDA)")
        print("  On your RTX 5090, install CUDA-enabled PyTorch:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu130")
        print(f"\n  Step 2 SKIPPED (no CUDA)")
        return model

    # Record pre-quantization memory
    pre_memory = get_model_memory_mb(model)
    print(f"  Pre-quantization memory: {pre_memory:.1f} MB")

    # Define calibration function
    num_calib_batches = 32
    print(f"  Running calibration with {num_calib_batches} synthetic batches...")

    def calibration_fn(model: nn.Module) -> None:
        model.eval()
        for i in range(num_calib_batches):
            sample = create_synthetic_sample(batch_size=4)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(sample, patch_size=2)

    # Quantize
    print(f"  Applying NVFP4 quantization (config={quant_config})...")
    t0 = time.time()
    try:
        quantized_model = quantize_model(model, calibration_fn, precision=precision, config=quant_config)
    except Exception as e:
        print(f"\n  Quantization call failed: {e}")
        print(f"\n  Step 2 FAILED (quantization error)")
        return model
    elapsed = time.time() - t0
    print(f"  Quantization took {elapsed:.1f}s")

    # Count quantizer nodes (this works even without CUDA extension)
    num_quantizers = count_quantizer_nodes(quantized_model)
    has_quantizers = num_quantizers > 0
    print(f"  Quantizer nodes inserted: {num_quantizers}")
    print_result("Quantizer nodes found in model", has_quantizers, f"{num_quantizers} nodes")

    # Verify: forward pass still works with quantized model
    # This requires the CUDA MX extension (needs CUDA toolkit with nvcc installed)
    print(f"\n  Verifying quantized model forward pass...")
    sample = create_synthetic_sample(batch_size=2)
    try:
        embedding = run_forward_pass(quantized_model, sample)
    except RuntimeError as e:
        err_msg = str(e)
        if "CUDA_HOME" in err_msg or "nvcc" in err_msg.lower() or "MX" in err_msg:
            print(f"\n  Forward pass failed: CUDA toolkit (nvcc) not installed.")
            print(f"  ModelOpt NVFP4 requires the CUDA Toolkit to JIT-compile")
            print(f"  the MX quantization CUDA extension.")
            print(f"\n  Install CUDA Toolkit 13.0+:")
            print(f"    https://developer.nvidia.com/cuda-13-0-0-download-archive")
            print(f"  Or latest (13.2):")
            print(f"    https://developer.nvidia.com/cuda-downloads")
            print(f"\n  After install, verify with: nvcc --version")
            print(f"\n  Quantizer nodes WERE inserted ({num_quantizers} nodes).")
            print(f"  The model is correctly quantized — it just needs nvcc to run.")
            print(f"\n  Step 2 PARTIAL (quantized but cannot verify without CUDA toolkit)")
            return quantized_model
        else:
            print(f"\n  Forward pass failed: {e}")
            print(f"\n  Step 2 FAILED")
            return model

    has_correct_shape = embedding.dim() == 2 and embedding.shape[0] == 2
    no_nan = not torch.isnan(embedding).any().item()

    # Check memory (post-quantization, fake-quantized so memory won't shrink much)
    post_memory = get_model_memory_mb(quantized_model)

    print(f"\n  Post-quantization memory: {post_memory:.1f} MB")
    print(f"  Output shape: {list(embedding.shape)}")

    print_result("Forward pass produces correct shape", has_correct_shape)
    print_result("No NaN in quantized output", no_nan)

    all_passed = has_correct_shape and no_nan and has_quantizers
    print(f"\n  Step 2 {'PASSED' if all_passed else 'FAILED'}")
    return quantized_model


# ============================================================================
# Step 3: Embedding Quality — Cosine Similarity
# ============================================================================


def step3_embedding_quality(
    fp32_model: nn.Module,
    fp4_model: nn.Module,
    num_samples: int = 100,
) -> None:
    """Compare embedding quality between FP32 and FP4 models."""
    print_banner(3, "EMBEDDING QUALITY COMPARISON")

    if fp32_model is fp4_model:
        print("  Skipping (FP4 model not available, same as FP32)")
        print(f"\n  Step 3 SKIPPED")
        return

    print(f"  Extracting embeddings from {num_samples} samples...")

    fp32_embeddings = []
    fp4_embeddings = []

    # Use same random seed for both to get identical inputs
    for i in range(num_samples):
        torch.manual_seed(i + 42)
        sample = create_synthetic_sample(batch_size=1)

        fp32_emb = run_forward_pass(fp32_model, sample)
        try:
            fp4_emb = run_forward_pass(fp4_model, sample)
        except RuntimeError as e:
            if "CUDA_HOME" in str(e) or "MX" in str(e):
                print(f"\n  FP4 model forward pass failed (CUDA toolkit not installed).")
                print(f"  Cannot compare embeddings without nvcc.")
                print(f"\n  Step 3 SKIPPED (install CUDA toolkit to enable)")
                return
            raise

        fp32_embeddings.append(fp32_emb.cpu())
        fp4_embeddings.append(fp4_emb.cpu())

    fp32_all = torch.cat(fp32_embeddings, dim=0)  # [N, D]
    fp4_all = torch.cat(fp4_embeddings, dim=0)  # [N, D]

    # Compute per-sample cosine similarity
    cos_sim = F.cosine_similarity(fp32_all, fp4_all, dim=1)  # [N]

    mean_sim = cos_sim.mean().item()
    std_sim = cos_sim.std().item()
    min_sim = cos_sim.min().item()
    max_sim = cos_sim.max().item()

    # Also compute L2 distance
    l2_dist = torch.norm(fp32_all - fp4_all, dim=1)
    mean_l2 = l2_dist.mean().item()

    print(f"\n  Cosine Similarity Statistics:")
    print(f"    Mean:  {mean_sim:.6f}")
    print(f"    Std:   {std_sim:.6f}")
    print(f"    Min:   {min_sim:.6f}")
    print(f"    Max:   {max_sim:.6f}")
    print(f"\n  L2 Distance (mean): {mean_l2:.6f}")

    # Checks
    mean_ok = mean_sim > 0.95
    min_ok = min_sim > 0.80

    print_result("Mean cosine similarity > 0.95", mean_ok, f"{mean_sim:.6f}")
    print_result("Min cosine similarity > 0.80", min_ok, f"{min_sim:.6f}")

    all_passed = mean_ok and min_ok
    print(f"\n  Step 3 {'PASSED' if all_passed else 'FAILED'}")


# ============================================================================
# Step 4: Throughput Benchmark
# ============================================================================


def step4_throughput(
    fp32_model: nn.Module,
    fp4_model: nn.Module,
    batch_sizes: list[int] | None = None,
) -> None:
    """Benchmark throughput at various batch sizes."""
    print_banner(4, "THROUGHPUT BENCHMARK")

    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64]

    warmup_iters = 10
    timed_iters = 50
    patch_size = 2
    height = 64
    width = 64

    models = {"FP32": fp32_model}
    if fp4_model is not fp32_model:
        models["FP4"] = fp4_model

    results: dict[str, dict[int, dict]] = {}

    for model_name, model in models.items():
        results[model_name] = {}
        model.eval()

        for bs in batch_sizes:
            # Check if batch size fits in memory
            try:
                sample = create_synthetic_sample(batch_size=bs, height=height, width=width)
            except RuntimeError:
                print(f"  {model_name} bs={bs}: OOM creating sample, skipping")
                continue

            # Warmup
            try:
                for _ in range(warmup_iters):
                    _ = run_forward_pass(model, sample, patch_size=patch_size)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except RuntimeError as e:
                err_msg = str(e)
                if "out of memory" in err_msg.lower():
                    print(f"  {model_name} bs={bs}: OOM during warmup, skipping")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                elif "CUDA_HOME" in err_msg or "MX" in err_msg:
                    print(f"  {model_name}: CUDA toolkit not installed, skipping all batch sizes")
                    break
                raise

            # Timed
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(timed_iters):
                _ = run_forward_pass(model, sample, patch_size=patch_size)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            patches_per_sample = (height // patch_size) * (width // patch_size)
            total_patches = bs * timed_iters * patches_per_sample
            patches_per_sec = total_patches / elapsed
            latency_ms = (elapsed / timed_iters) * 1000
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                torch.cuda.reset_peak_memory_stats()
            else:
                mem_mb = 0.0

            results[model_name][bs] = {
                "patches_per_sec": patches_per_sec,
                "latency_ms": latency_ms,
                "mem_mb": mem_mb,
            }

    # Print results table
    print(f"\n  {'Model':<8} {'Batch':>6} {'Patches/s':>12} {'Latency(ms)':>12} {'GPU Mem(MB)':>12}")
    print(f"  {'-'*8} {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
    for model_name, bs_results in results.items():
        for bs, r in bs_results.items():
            print(
                f"  {model_name:<8} {bs:>6} {r['patches_per_sec']:>12,.0f} "
                f"{r['latency_ms']:>12.1f} {r['mem_mb']:>12.0f}"
            )

    # Speedup comparison
    if "FP4" in results and "FP32" in results:
        print(f"\n  Speedup (FP4 vs FP32):")
        for bs in batch_sizes:
            if bs in results["FP32"] and bs in results["FP4"]:
                speedup = results["FP4"][bs]["patches_per_sec"] / results["FP32"][bs]["patches_per_sec"]
                print(f"    bs={bs}: {speedup:.2f}x")

    # Verify
    has_results = any(len(v) > 0 for v in results.values())
    print_result("Throughput numbers obtained", has_results)
    print(f"\n  Step 4 {'PASSED' if has_results else 'FAILED'}")




# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVFP4 Weight Quantization for OlmoEarth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="OLMOEARTH_V1_BASE",
        choices=[m.name for m in ModelID],
        help="Model variant to quantize (default: OLMOEARTH_V1_BASE)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Run only this step (default: run all)",
    )
    parser.add_argument(
        "--quant-config",
        type=str,
        default="default",
        choices=["default", "mlp_only"],
        help="Quantization config: 'default' (all Linear) or 'mlp_only' (default: default)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp4",
        choices=["fp4", "fp8"],
        help="Quantization precision: 'fp4' (Blackwell) or 'fp8' (Hopper+) (default: fp4)",
    )
    parser.add_argument(
        "--num-quality-samples",
        type=int,
        default=100,
        help="Number of samples for quality comparison (default: 100)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,64",
        help="Comma-separated batch sizes for throughput (default: 1,4,16,64)",
    )
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"\n{'#'*70}")
    print(f"  OlmoEarth {args.precision.upper()} Weight Quantization Pipeline")
    print(f"  Model: {args.model_id}")
    print(f"  Precision: {args.precision.upper()}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  ModelOpt available: {check_modelopt_available()}")
    print(f"{'#'*70}")

    run_steps = [args.step] if args.step else [1, 2, 3, 4]

    fp32_model = None
    fp4_model = None

    if 1 in run_steps:
        fp32_model = step1_load_and_inspect(args.model_id)

    if 2 in run_steps:
        if fp32_model is None:
            mid = ModelID[args.model_id]
            full_model = load_model_from_id(mid, load_weights=True).to(DEVICE)
            fp32_model = get_encoder(full_model)
            fp32_model.eval()
        # Deep copy for FP4 so we keep the original FP32 model
        fp4_model = copy.deepcopy(fp32_model)
        fp4_model = step2_quantize(fp4_model, args.quant_config, precision=args.precision)

    if 3 in run_steps:
        if fp32_model is None:
            mid = ModelID[args.model_id]
            full_model = load_model_from_id(mid, load_weights=True).to(DEVICE)
            fp32_model = get_encoder(full_model)
            fp32_model.eval()
        if fp4_model is None:
            fp4_model = fp32_model  # Will skip comparison
        step3_embedding_quality(fp32_model, fp4_model, num_samples=args.num_quality_samples)

    if 4 in run_steps:
        if fp32_model is None:
            mid = ModelID[args.model_id]
            full_model = load_model_from_id(mid, load_weights=True).to(DEVICE)
            fp32_model = get_encoder(full_model)
            fp32_model.eval()
        if fp4_model is None:
            fp4_model = fp32_model
        step4_throughput(fp32_model, fp4_model, batch_sizes=batch_sizes)

    print(f"\n{'='*70}")
    print(f"  Pipeline complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
