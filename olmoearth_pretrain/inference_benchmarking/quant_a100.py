"""INT8 quantization utilities for A100 inference using TorchAO."""

import re
from collections.abc import Callable
from logging import getLogger

import torch

try:
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
        smooth_fq_linear_to_inference,
        swap_linear_with_smooth_fq_linear,
    )

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

logger = getLogger(__name__)


def _is_transformer_linear(mod: torch.nn.Module, name: str) -> bool:
    """Match common names for Q/K/V projections, out_proj, and MLP fc1/fc2.

    Adjust patterns if your modules differ.
    """
    if not isinstance(mod, torch.nn.Linear):
        return False
    pats = [
        r"\bq(_proj)?\b",
        r"\bk(_proj)?\b",
        r"\bv(_proj)?\b",
        r"\bout(_proj)?\b",
        r"\bqkv\b",
        r"\bmlp\.(fc1|fc2)\b",
        r"\b(attn\.)?(q_proj|k_proj|v_proj|out_proj)\b",
        # OlmoEarth specific patterns
        r"\b(att_in|att_out)\b",
        r"\bw_(1|2|3)\b",
        r"\b(q|k|v)_norm\b",
    ]
    return any(re.search(p, name, re.IGNORECASE) for p in pats)


def apply_int8_quant(
    model: torch.nn.Module,
    *,
    mode: str = "w8a8",
    filter_fn: Callable[[torch.nn.Module, str], bool] | None = None,
    smoothquant: bool = False,
    inductor_preset: bool = True,
) -> torch.nn.Module:
    """In-place quantization of transformer linears on CUDA.

    Returns the same model reference (modified).

    Args:
        model: The model to quantize (should be in eval mode on CUDA)
        mode: Quantization mode - "w8a8" (weight+activation INT8) or "w8" (weight-only)
        filter_fn: Optional function to determine which modules to quantize
        smoothquant: Apply outlier smoothing before quantization
        inductor_preset: Set TorchAO's inductor-friendly config flags
    """
    if not TORCHAO_AVAILABLE:
        raise ImportError(
            "TorchAO is not available. Install it with: pip install torchao"
        )

    model.eval().to("cuda")

    if filter_fn is None:

        def filter_fn(m, n):
            return _is_transformer_linear(m, n)

    if smoothquant:
        logger.info("Applying SmoothQuant calibration...")
        # Swap fake-quant linears -> bake scales -> restore inference linears
        swap_linear_with_smooth_fq_linear(model, filter_fn=filter_fn)
        smooth_fq_linear_to_inference(model)

    if mode.lower() == "w8a8":
        logger.info("Applying W8A8 (dynamic activation + weight INT8) quantization...")
        cfg = Int8DynamicActivationInt8WeightConfig(set_inductor_config=inductor_preset)
    elif mode.lower() in ("w8", "weight-only", "wo"):
        logger.info("Applying W8 (weight-only INT8) quantization...")
        cfg = Int8WeightOnlyConfig(set_inductor_config=inductor_preset)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    quantize_(model, cfg, filter_fn=filter_fn)  # in-place
    logger.info(f"Successfully applied {mode.upper()} quantization")
    return model


def compile_for_a100(
    model: torch.nn.Module,
    *,
    fullgraph: bool = True,
    mode: str = "max-autotune",
    allow_tf32: bool = True,
) -> torch.nn.Module:
    """torch.compile with Inductor for CUDA graphs + autotuned kernels on A100.

    Args:
        model: The model to compile
        fullgraph: Whether to compile the entire graph (True for best performance)
        mode: Compilation mode - "default", "reduce-overhead", or "max-autotune"
        allow_tf32: Enable TF32 for FP32 matmul operations
    """
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable TF32 on Ampere for FP32 matmul fallback
        torch.set_float32_matmul_precision("high")

    logger.info(f"Compiling model with mode={mode}, fullgraph={fullgraph}...")
    compiled = torch.compile(model, fullgraph=fullgraph, mode=mode)
    logger.info("Model compilation complete")
    return compiled


class FlashSDPA:
    """Context manager to force the Flash attention backend where available.

    Supports both the old and new PyTorch APIs.
    """

    def __enter__(self):
        """Enter the context manager."""
        # Newer API (PyTorch >= 2.5)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            self._ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            self._ctx.__enter__()
            self._which = "new"
            return self
        except Exception:
            # Older API (PyTorch 2.0–2.4)
            self._ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            )
            self._ctx.__enter__()
            self._which = "old"
            return self

    def __exit__(self, exc_type, exc, tb):
        """Exit the context manager."""
        self._ctx.__exit__(exc_type, exc, tb)
