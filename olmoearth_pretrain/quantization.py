"""FP4/FP8 weight quantization utilities for OlmoEarth models.

Provides weight quantization using NVIDIA Model Optimizer (nvidia-modelopt)
for inference on Hopper (FP8) and Blackwell (FP4/FP8) GPUs.

Usage:
    from olmoearth_pretrain.quantization import quantize_model, check_modelopt_available

    if check_modelopt_available():
        model = quantize_model(model, calibration_fn, precision="fp8")  # or "fp4"
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def check_modelopt_available() -> bool:
    """Check if nvidia-modelopt is installed and importable.

    Returns:
        True if modelopt is available, False otherwise.
    """
    try:
        import modelopt.torch.quantization as mtq  # noqa: F401

        return True
    except (ImportError, AttributeError, RuntimeError):
        # AttributeError/RuntimeError can happen if torchvision is broken
        # (e.g., CPU-only torch with GPU torchvision). This is fine on a
        # properly configured CUDA environment.
        try:
            # Try direct submodule import as fallback
            from modelopt.torch.quantization import model_quant  # noqa: F401

            return True
        except Exception:
            return False


def get_modelopt_install_instructions() -> str:
    """Return install instructions for nvidia-modelopt."""
    return (
        "nvidia-modelopt is required for NVFP4 quantization.\n\n"
        "Install options (pick one):\n"
        "  1. pip install -U nvidia-modelopt[all]\n"
        "  2. From source (latest features):\n"
        "     git clone git@github.com:NVIDIA/Model-Optimizer.git\n"
        "     cd Model-Optimizer && pip install -e .[dev]\n"
        "  3. TensorRT-LLM container (ModelOpt pre-installed):\n"
        "     docker run --gpus all -it nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9\n"
        "     pip install -U nvidia-modelopt\n\n"
        "Requires: CUDA 12.8+, PyTorch 2.7+, Blackwell GPU (sm_100+) for real FP4 speedup"
    )


def count_quantizable_layers(model: nn.Module) -> dict[str, int]:
    """Count layers that can be quantized to FP4.

    Args:
        model: PyTorch model to inspect.

    Returns:
        Dict with counts per layer type.
    """
    counts: dict[str, int] = {}
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        if isinstance(module, nn.Linear):
            counts.setdefault("nn.Linear", 0)
            counts["nn.Linear"] += 1
        elif isinstance(module, nn.Conv2d):
            counts.setdefault("nn.Conv2d", 0)
            counts["nn.Conv2d"] += 1
        elif isinstance(module, nn.LayerNorm):
            counts.setdefault("nn.LayerNorm (skipped)", 0)
            counts["nn.LayerNorm (skipped)"] += 1
    return counts


def get_model_memory_mb(model: nn.Module) -> float:
    """Get approximate model memory footprint in MB.

    Args:
        model: PyTorch model.

    Returns:
        Memory in megabytes.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024 * 1024)


def _get_quant_config(precision: str, config: str):
    """Get modelopt quantization config for the given precision and config preset.

    Args:
        precision: "fp4" or "fp8".
        config: "default" (all Linear) or "mlp_only" (MLP layers only).

    Returns:
        modelopt quantization config dict.
    """
    import modelopt.torch.quantization as mtq

    if precision == "fp4":
        base_cfg = mtq.NVFP4_DEFAULT_CFG
        mlp_cfg_attr = "NVFP4_MLP_ONLY_CFG"
    elif precision == "fp8":
        base_cfg = mtq.FP8_DEFAULT_CFG
        mlp_cfg_attr = "FP8_MLP_ONLY_CFG"
    else:
        raise ValueError(f"Unknown precision: {precision}. Use 'fp4' or 'fp8'.")

    if config == "default":
        return base_cfg
    elif config == "mlp_only":
        if hasattr(mtq, mlp_cfg_attr):
            return getattr(mtq, mlp_cfg_attr)
        import copy

        quant_cfg = copy.deepcopy(base_cfg)
        quant_cfg["quant_cfg"]["*attn*weight_quantizer"] = {"enable": False}
        quant_cfg["quant_cfg"]["*attn*input_quantizer"] = {"enable": False}
        return quant_cfg
    else:
        raise ValueError(f"Unknown config: {config}. Use 'default' or 'mlp_only'.")


def quantize_model(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
    precision: str = "fp4",
    config: str = "default",
) -> nn.Module:
    """Quantize model weights using nvidia-modelopt.

    Supports FP4 (Blackwell) and FP8 (Hopper/Blackwell) precisions.

    Args:
        model: OlmoEarth model (encoder or full model).
        calibration_fn: Callable that runs calibration data through the model.
            Signature: calibration_fn(model) -> None.
            Should run ~32-256 representative samples.
        precision: "fp4" (NVFP4, Blackwell only) or "fp8" (FP8 E4M3, Hopper+).
        config: "default" (all Linear) or "mlp_only" (skip attention layers).

    Returns:
        Model with quantizer nodes inserted (fake-quantized).
    """
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        raise ImportError(get_modelopt_install_instructions())

    quant_cfg = _get_quant_config(precision, config)
    logger.info(f"Applying {precision.upper()} quantization with config={config}")
    model = mtq.quantize(model, quant_cfg, forward_loop=calibration_fn)
    logger.info(f"{precision.upper()} quantization complete")
    return model


def quantize_model_nvfp4(
    model: nn.Module,
    calibration_fn: Callable[[nn.Module], None],
    config: str = "default",
) -> nn.Module:
    """Quantize model weights to NVFP4. Convenience alias for quantize_model(precision='fp4')."""
    return quantize_model(model, calibration_fn, precision="fp4", config=config)


def count_quantizer_nodes(model: nn.Module) -> int:
    """Count the number of TensorQuantizer nodes inserted by modelopt.

    Args:
        model: Quantized model.

    Returns:
        Number of quantizer nodes found.
    """
    count = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Quantizer" in cls_name or "quantizer" in cls_name.lower():
            count += 1
    return count


def load_and_quantize_model(
    model_id: str = "OLMOEARTH_V1_BASE",
    config: str = "default",
    num_calib_batches: int = 32,
) -> nn.Module:
    """One-call convenience: load an OlmoEarth model and quantize to NVFP4.

    Args:
        model_id: ModelID name (e.g., "OLMOEARTH_V1_NANO", "OLMOEARTH_V1_BASE").
        config: Quantization config ("default" or "mlp_only").
        num_calib_batches: Number of synthetic calibration batches.

    Returns:
        Quantized encoder model on CUDA.
    """
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

    mid = ModelID[model_id]
    model = load_model_from_id(mid, load_weights=True).cuda().eval()
    encoder = model.encoder

    def calibration_fn(m: nn.Module) -> None:
        m.eval()
        for _ in range(num_calib_batches):
            # Synthetic Sentinel-2 L2A input
            from olmoearth_pretrain.data.constants import Modality
            from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

            s2_spec = Modality.SENTINEL2_L2A
            s2 = torch.randn(4, 64, 64, 1, s2_spec.num_bands, device="cuda")
            mask = torch.full(
                (4, 64, 64, 1, s2_spec.num_band_sets),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.long,
                device="cuda",
            )
            ts = torch.zeros(4, 1, 3, dtype=torch.long, device="cuda")
            sample = MaskedOlmoEarthSample(
                sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=ts
            )
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    m(sample, patch_size=2)

    return quantize_model_nvfp4(encoder, calibration_fn, config=config)
