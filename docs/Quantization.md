# Embedding Quantization for Evaluations

## Goal

Evaluate the impact of quantizing output embeddings to int8 on KNN and Linear Probe evaluation performance. This is motivated by storage efficiency considerations - int8 embeddings require 4x less storage than float32, which is important for large-scale deployment scenarios like AlphaEarth.

## Quantization Scheme

We use a **power-based quantization scheme** (matching AlphaEarth):

```python
# Quantization (float32 → int8)
POWER = 2.0
SCALE = 127.5
sat = embeddings.abs().pow(1/POWER) * embeddings.sign()  # Apply sqrt, preserve sign
quantized = (sat * SCALE).clamp(-127, 127).round().to(int8)

# Dequantization (int8 → float32)
rescaled = quantized.float() / SCALE
dequantized = rescaled.abs().pow(POWER) * rescaled.sign()  # Apply square, preserve sign
```

The power function (square root for quantization, square for dequantization) helps preserve information for non-uniform embedding distributions.

## Usage

### Running Full Sweep with Quantization

Add `--quantize_embeddings` flag to enable int8 quantization:

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --project_name=01_14_quantize_evals \
  --select_best_val \
  --trainer.callbacks.downstream_evaluator.run_on_test=True \
  --trainer.max_duration.value=700000 \
  --trainer.max_duration.unit=steps \
  --quantize_embeddings
```

### Running Full Sweep Baseline (No Quantization)

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --project_name=01_14_quantize_evals \
  --select_best_val \
  --trainer.callbacks.downstream_evaluator.run_on_test=True \
  --trainer.max_duration.value=700000 \
  --trainer.max_duration.unit=steps
```

## Implementation Details

- **Quantization happens in**: `olmoearth_pretrain/evals/embeddings.py` (`get_embeddings` function)
- **Dequantization happens in**: `olmoearth_pretrain/train/callbacks/evaluator_callback.py` (`_val_embed_probe` method)
- **Config option**: `DownstreamTaskConfig.quantize_embeddings` (default: `False`)
- **Run naming**: Quantized runs have `_qt` suffix in WandB

## Expected Outcome

Compare KNN accuracy between quantized and non-quantized embeddings to determine if int8 quantization is viable for production use without significant performance degradation.
