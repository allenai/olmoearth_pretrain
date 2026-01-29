# DDP vs FSDP Comparison Experiments

This directory contains scripts to compare DDP (DistributedDataParallel) vs FSDP (FullyShardedDataParallel) training to diagnose why results differ between the two strategies.

## Quick Start

Launch both comparison experiments on Beaker:

```bash
./scripts/ddp_fsdp_comparison/launch_comparison.sh
```

Or launch individually:

```bash
# DDP run
python scripts/ddp_fsdp_comparison/compare_ddp_fsdp.py launch ddp_test ai2/saturn-cirrascale \
    --dp_type=ddp \
    --launch.num_gpus=4

# FSDP run
python scripts/ddp_fsdp_comparison/compare_ddp_fsdp.py launch fsdp_test ai2/saturn-cirrascale \
    --dp_type=fsdp \
    --launch.num_gpus=4
```

## Configuration

Both runs use identical settings except for the parallelism strategy:

| Setting | Value |
|---------|-------|
| Model | Tiny (192 embedding, 12 depth, 3 heads) |
| GPUs | 4 |
| Training Steps | 500 |
| Global Batch Size | 64 (16 per GPU) |
| Modalities | sentinel2_l2a, sentinel1 |
| Masking | Random (50% encode, 50% decode) |
| EMA | Disabled (1.0, 1.0) |
| Seed | 42 (data), 12536 (init) |

## Differences Being Tested

| Aspect | DDP | FSDP |
|--------|-----|------|
| `dp_config.name` | `ddp` | `fsdp` |
| `dp_config.param_dtype` | None | `bfloat16` |
| Model sharding | None | Per-layer |

## What to Look For

After both runs complete, compare in W&B (`ddp_vs_fsdp_comparison` project):

1. **Loss curves** (`train/PatchDisc`) - Should match closely
2. **Gradient norms** (`optim/total grad norm`) - Key diagnostic
3. **Downstream accuracy** (`m-eurosat/linear_probe_acc`) - Final validation

## Potential Sources of Divergence

1. **Target encoder initialization order** - `deepcopy` before FSDP sharding
2. **EMA updates with DTensor** - Different handling for sharded vs replicated params
3. **Mixed precision** - FSDP uses explicit `MixedPrecisionPolicy`
4. **Gradient reduction semantics** - `set_is_last_backward` vs `no_sync()`

## Debugging

Dry run to inspect configs:

```bash
python scripts/ddp_fsdp_comparison/compare_ddp_fsdp.py dry_run test local --dp_type=ddp
python scripts/ddp_fsdp_comparison/compare_ddp_fsdp.py dry_run test local --dp_type=fsdp
```
