# INT8 Inference Benchmarking on A100

This module provides **INT8 quantization** support for inference throughput benchmarking using **TorchAO + TorchInductor + Flash attention**. It keeps LayerNorm/Softmax in BF16/FP32 for stability and quantizes only the heavy Linear ops (Q/K/V/out-proj, MLP) to maximize throughput with minimal accuracy loss.

## Requirements

* **GPU:** A100 (SM80/Ampere)
* **PyTorch:** 2.2+ (2.4+ recommended)
* **CUDA:** 11.8+ (matching your PyTorch build)
* **Python:** 3.9–3.12
* **Packages:**

```bash
pip install --upgrade torchao
```

## Quick Start

### Basic INT8 Benchmarking

```python
from olmoearth_pretrain.inference_benchmarking.data_models import RunParams
from olmoearth_pretrain.inference_benchmarking.run_throughput_benchmark import (
    ThroughputBenchmarkRunnerConfig,
)

# Create config with INT8 enabled
config = ThroughputBenchmarkRunnerConfig(
    default_run_params=RunParams(
        model_size="base",
        batch_size=128,
        int8_enabled=True,           # Enable INT8 quantization
        int8_mode="w8a8",             # W8A8 (weight+activation) or "w8" (weight-only)
        int8_smoothquant=False,       # Set True if you see >1% metric drop
        compile_mode="max-autotune",  # TorchInductor compile mode
        compile_fullgraph=True,       # Compile entire graph for best perf
        bf16=True,                    # Keep non-quantized ops in BF16
    ),
    sweep_keys=["batch"],  # Sweep batch sizes
    sweep_group_name="int8_baseline",
)

runner = config.build()
runner.run()
```

### Sweeping INT8 Parameters

```python
# Sweep over INT8 modes (W8A8 vs W8 weight-only)
config = ThroughputBenchmarkRunnerConfig(
    default_run_params=RunParams(
        model_size="base",
        batch_size=128,
        int8_enabled=True,
    ),
    sweep_keys=["int8_mode"],  # Sweeps over ["w8a8", "w8"]
    sweep_group_name="int8_mode_comparison",
)

# Or sweep over compile modes
config = ThroughputBenchmarkRunnerConfig(
    default_run_params=RunParams(
        model_size="base",
        batch_size=128,
        int8_enabled=True,
        int8_mode="w8a8",
    ),
    sweep_keys=["compile_mode"],  # Sweeps over compile strategies
    sweep_group_name="compile_mode_sweep",
)
```

## Configuration Parameters

### INT8 Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `int8_enabled` | `bool` | `False` | Enable INT8 quantization |
| `int8_mode` | `str` | `"w8a8"` | Quantization mode: `"w8a8"` (weight+activation INT8) or `"w8"` (weight-only) |
| `int8_smoothquant` | `bool` | `False` | Apply SmoothQuant outlier smoothing (use if accuracy drops >1%) |
| `compile_mode` | `str` | `"max-autotune"` | TorchInductor mode: `"default"`, `"reduce-overhead"`, or `"max-autotune"` |
| `compile_fullgraph` | `bool` | `True` | Compile entire graph as single unit (best for performance) |

### Quantization Modes Explained

* **W8A8 (dynamic activations):** Best raw compute speed on A100. Quantizes both weights and activations to INT8.
* **W8 (weight-only):** Big memory-bandwidth win, often zero accuracy loss. Best if you're memory-bound or want larger batches.

## Available Sweep Keys

All standard sweeps plus INT8-specific ones:

```python
# Standard sweeps
"batch"           # batch_size: [1, 2, 4, 8, 16, 32, 64, 128, ...]
"image"           # image_size: [1, 2, 4, 8, 16, 32, 64, 128]
"patch"           # patch_size: [1, 2, 4, 8]
"time"            # num_timesteps: [1, 2, 4, 6, 8, 12]
"model_size"      # model_size: ["nano", "tiny", "base", "large"]
"bf16"            # bf16: [True, False]

# INT8-specific sweeps
"int8_enabled"    # int8_enabled: [True, False]
"int8_mode"       # int8_mode: ["w8a8", "w8"]
"int8_smoothquant" # int8_smoothquant: [True, False]
"compile_mode"    # compile_mode: ["default", "reduce-overhead", "max-autotune"]
```

## Run Name Format

When INT8 is enabled, the run name includes quantization info:

```
base_cuda_bf16_int8_w8a8_max-autotune_s2_is64_ps4_ts12_bs128
                └─────┬─────┘  └─────┬─────┘
                 INT8 mode    Compile mode
```

With SmoothQuant:
```
base_cuda_bf16_int8_w8a8_smoothq_max-autotune_s2_is64_ps4_ts12_bs128
                             └──┬──┘
                          SmoothQuant
```

## What Gets Quantized

✅ **Quantized to INT8:**
- Attention Linear layers (Q, K, V projections, out_proj)
- MLP Linear layers (fc1, fc2, w_1, w_2, w_3)

✅ **Kept in BF16/FP32:**
- LayerNorm (cheap + numerically sensitive)
- Softmax (sensitive to quantization)
- Residual connections
- Embedding layers (can be excluded via custom filter)

## Performance Tips

1. **Start with W8A8** for maximum compute throughput
2. **Try W8 weight-only** if:
   - Memory-bound workloads
   - Accuracy is sensitive
   - Want to increase batch size
3. **Use `compile_mode="max-autotune"`** for best throughput (longer compile time)
4. **Use `compile_mode="reduce-overhead"`** for faster iteration during development
5. **Increase batch size** after INT8 to saturate Tensor Cores
6. **Static shapes** are preferred - dynamic shapes reduce specialization

## Accuracy Guardrails

1. **Drop ≤ ~0.5–1.0%?** ✅ Ship it
2. **Drop > ~1%?**
   - Set `int8_smoothquant=True` and re-run
   - Try `int8_mode="w8"` (weight-only)
   - Increase batch size to reclaim throughput
3. **Still sensitive?** Custom filter to exclude first/last layers

### Custom Filtering Example

```python
from olmoearth_pretrain.inference_benchmarking.quant_a100 import apply_int8_quant

def custom_filter(module: torch.nn.Module, name: str) -> bool:
    """Custom filter to exclude specific layers."""
    # Skip first embedding layer
    if "patch_embed" in name:
        return False
    # Skip final classifier
    if "head" in name or "classifier" in name:
        return False
    # Default filtering for attention/MLP linears
    from olmoearth_pretrain.inference_benchmarking.quant_a100 import _is_transformer_linear
    return _is_transformer_linear(module, name)

# Apply quantization with custom filter
model = apply_int8_quant(
    model,
    mode="w8a8",
    filter_fn=custom_filter,
    smoothquant=False,
)
```

## Throughput Optimization Checklist

✅ **INT8 quantization enabled** (`int8_enabled=True`)
✅ **Flash attention** (automatically enabled with INT8)
✅ **BF16 autocast** for non-quantized ops (`bf16=True`)
✅ **TF32 allowed** (automatically set by `compile_for_a100`)
✅ **torch.compile** with `mode="max-autotune"`
✅ **Larger batch sizes** after INT8
✅ **Static shapes** (fixed batch/seq/patch sizes)
✅ **DataLoader tuning** (pin_memory, non_blocking, persistent_workers)

## Environment Variables (Optional)

```bash
# Improve allocator behavior for long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# See Inductor compile logs (debugging)
# export TORCH_LOGS="+dynamo,graph_breaks"
```

## Example Sweep Configurations

### Basic INT8 vs FP16 Comparison

```python
config = ThroughputBenchmarkRunnerConfig(
    default_run_params=RunParams(model_size="base", batch_size=128),
    sweep_keys=["int8_enabled"],
    sweep_group_name="int8_vs_fp16",
)
```

### INT8 Mode Comparison (W8A8 vs W8)

```python
config = ThroughputBenchmarkRunnerConfig(
    default_run_params=RunParams(
        model_size="base",
        batch_size=128,
        int8_enabled=True,
    ),
    sweep_keys=["int8_mode"],
    sweep_group_name="w8a8_vs_w8",
)
```

### Batch Size Sweep with INT8

```python
config = ThroughputBenchmarkRunnerConfig(
    default_run_params=RunParams(
        model_size="base",
        int8_enabled=True,
        int8_mode="w8a8",
    ),
    sweep_keys=["batch"],
    sweep_group_name="int8_batch_sweep",
)
```

### Cross-Product Sweep (Multiple Dims)

```python
config = ThroughputBenchmarkRunnerConfig(
    default_run_params=RunParams(model_size="base"),
    sweep_dict={
        "int8_enabled": [True, False],
        "batch_size": [64, 128, 256],
    },
    cross_product_sweep=True,  # All combinations
    sweep_group_name="int8_batch_cross",
)
```

## Common Pitfalls

❌ **Mismatched preprocessing** vs training → unstable activations → accuracy drop
❌ **Quantizing LayerNorm/Softmax** → avoided by default filter
❌ **Dynamic shapes** → poorer specialization; prefer fixed sizes
❌ **Under-utilized GPU** → increase batch size, ensure I/O overlaps
❌ **Compilation time too high** → switch to `compile_mode="reduce-overhead"`

## Architecture Details

### Module Structure

```
olmoearth_pretrain/inference_benchmarking/
├── quant_a100.py              # INT8 quantization utilities
│   ├── apply_int8_quant()     # Apply TorchAO quantization
│   ├── compile_for_a100()     # TorchInductor compilation
│   └── FlashSDPA              # Flash attention context manager
├── data_models.py             # RunParams with INT8 config
├── run_throughput_benchmark.py # Main benchmarking runner
├── constants.py               # Sweep configurations
└── README_INT8.md            # This file
```

### Key Integration Points

1. **Model building** (`build_model`): Applies INT8 quantization and compilation when enabled
2. **Forward pass** (`run_forward`): Uses FlashSDPA context manager for INT8 runs
3. **Run naming**: Includes INT8 mode and compile settings in W&B run names
4. **Sweep support**: All INT8 parameters are swappable via standard sweep mechanism

## FAQ

**Q: Do I need calibration data?**
A: Not for W8A8 dynamic activations. Activations are quantized dynamically during inference.

**Q: Why keep LayerNorm/Softmax in BF16?**
A: They're cheap ops and numerically sensitive. Quantizing them often costs more accuracy than compute savings.

**Q: Can I use this on non-A100 GPUs?**
A: Yes, but it's optimized for A100 Tensor Cores. H100 would be even faster. Older GPUs may not see the same speedups.

**Q: What about 2:4 sparsity?**
A: That's an advanced optimization requiring retraining to 2:4 semi-structured sparsity patterns. Not covered here but can double INT8 Tensor Core throughput on supported kernels.

**Q: Why is compile time so long?**
A: `mode="max-autotune"` autotuning searches for best kernels. Use `mode="reduce-overhead"` for faster iteration during development, then switch to `max-autotune` for final benchmarks.

## Support

For issues or questions:
1. Check that TorchAO is installed: `pip list | grep torchao`
2. Verify PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Check CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
4. Review logs for compilation or quantization errors

## References

- [TorchAO Documentation](https://github.com/pytorch/ao)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [TorchInductor](https://pytorch.org/docs/stable/torch.compiler.html)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
