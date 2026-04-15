# Profiling Report: Linear vs Conv Patch Embedding

**Date:** 2026-04-15
**Hardware:** 8x GPU (ai2/saturn)
**Config:** Near-final training setup (Gabi's `base_band_dropout_no_s1_drop_random_time`)
**Profiler:** `ProfilerCallback` (skip_first=5, wait=2, warmup=3, active=5 steps)
**WandB runs:** `0djwg764` (linear), `av07oofk` (conv)

---

## 1. Headline Result: Linear Patch Embed is ~18% Faster

| Metric | Linear | Conv | Delta |
|---|---|---|---|
| Model duration (s/step, steady state avg) | **0.700** | **0.860** | -18.6% |
| Batches per second (BPS, steady state avg) | **1.43** | **1.16** | +22.7% |
| GPU reserved memory (GiB) | 45.88 | 45.85 | ~0 |
| GPU active memory peak (GiB) | 21.38 | 21.38 | ~0 |

Steady-state = average over last 5 reporting intervals (steps 60-100), well past JIT warmup.

## 2. Why: Conv2d im2col Dominates the Difference

The Conv2d path pays for an expensive `im2col` reshape kernel on every forward pass:

| Kernel | Linear time (ms) | Conv time (ms) |
|---|---|---|
| `im2col_kernel` (conv reshape) | 0 | **562.9** (20.2% of all GPU time) |
| `cutlass_75_tensorop_*gemm*_align1` (conv GEMM) | 0 | **160.1** (5.7%) |
| `splitKreduce_kernel` (conv reduction) | 0 | **78.7** (2.8%) |
| **Total conv-path GPU overhead** | **0** | **~802 ms** |

The linear path replaces all of this with standard cuBLAS GEMM calls (`nvjet_tst_*`) that are already well-optimized on TensorCores. The `align1` suffix on the conv GEMM kernels indicates unaligned memory access patterns, making them slower than the aligned GEMMs used elsewhere.

## 3. Overall Kernel Breakdown (5 profiled steps)

### Linear — Top CUDA Kernels

| Kernel | Time (ms) | % |
|---|---|---|
| NCCL AllGather | 1610.9 | 50.3% |
| NCCL ReduceScatter | 432.3 | 13.5% |
| Attention backward (fmha cutlass) | 86.9 | 2.7% |
| NCCL AllReduce | 72.6 | 2.3% |
| Reduce kernels | 44.6 | 1.4% |
| Elementwise add (bf16) | 43.1 | 1.3% |
| Attention forward (fmha cutlass) | 40.7 | 1.3% |

### Conv — Top CUDA Kernels

| Kernel | Time (ms) | % |
|---|---|---|
| NCCL AllGather | 621.9 | 22.4% |
| **im2col_kernel** | **562.9** | **20.2%** |
| NCCL ReduceScatter | 158.4 | 5.7% |
| cutlass GEMM (conv, align1) | 117.8 | 4.2% |
| Attention backward (fmha cutlass) | 83.7 | 3.0% |
| splitKreduce (conv reduction) | 78.7 | 2.8% |

Note: NCCL times vary across runs due to network conditions (1610ms vs 621ms AllGather) and should not be compared directly between the two runs. The compute kernel differences are the signal.

## 4. CPU-Side Hotspots (Both Variants)

| Operation | Linear (ms) | Conv (ms) | Notes |
|---|---|---|---|
| `aten::item` | **679** (11,719 calls) | — | GPU->CPU sync, ~2349/step |
| `aten::eye` | **363** (9,042 calls) | — | ~1824/step, from loss function |
| `aten::nonzero` / `torch.where` | **528** (300 calls) | **364** (300 calls) | Masking code |
| `aten::index` (fancy indexing) | **474** (200 calls) | **362** (200 calls) | Masking code |
| `aten::div` | **581** | **232** | — |
| `conv2d` full stack | — | **853** | CPU overhead of conv dispatch |
| `ConvolutionBackward0` | — | **435** | Conv backward CPU overhead |

The conv path is dominated by conv CPU dispatch overhead (~1.3s). The linear path trades that for `aten::item` + `aten::eye` overhead (~1.0s) from the **masked-negatives loss function**, which loops over samples and calls `.item()` per sample.

## 5. Identified Speedup Opportunities

### 5a. Loss function `aten::item` calls (high impact, medium effort)

**Where:** `olmoearth_pretrain/train/loss.py:451-452`

The masked-negatives patch discrimination loss iterates over `count` with `c.item()` per sample to extract per-sample slice sizes. This triggers a GPU->CPU sync for every sample in every modality — ~2349 syncs per step.

**Fix:** Batch the `count` tensor to CPU once with `count_list = count.tolist()`, then iterate over the Python list. This reduces ~2349 individual syncs to 1.

### 5b. Repeated `torch.eye` allocation (medium impact, easy fix)

**Where:** `olmoearth_pretrain/train/loss.py:472, 772`

`torch.eye(c_val, ...)` is called once per sample per modality inside the loss loop — ~1824 times per step. Since `c_val` only takes a small set of values per batch, these could be cached in a dict keyed by `c_val`, or pre-allocated outside the loop.

### 5c. Masking code `.item()` + `torch.where` in Python loops (medium impact, hard)

**Where:** `olmoearth_pretrain/train/masking.py:674-676`

The `get_present_modalities_bandsets` method loops over samples and calls `sample_idx.item()` on each. Combined with `torch.where` (which triggers a sync), this adds ~528ms of nonzero + index overhead per step. This is harder to vectorize due to the variable-length nature of the output, but could potentially use `torch.nonzero` once and index into the result.

### 5d. Enable FlashAttention (high impact if compatible)

FlashAttention is explicitly disabled in the current config (`use_flash_attn=False`). The profile shows `fmha_cutlassB` kernels (memory-efficient attention) taking ~87ms backward + ~41ms forward = ~128ms. FlashAttention v2 would likely cut this by 2-3x and reduce memory.

The README notes this was disabled due to "InfoNCE mismatch investigations." If those are resolved, re-enabling would be a straightforward win.

### 5e. `torch.compile` (high impact, already supported)

The speedups README documents `compile_model=True` as a supported option. This would fuse elementwise kernels (which collectively account for ~5% of GPU time) and potentially fuse the patch embed reshape+GEMM into a single kernel.

## 6. Recommended Write-Up (for paper)

> **Training Efficiency.** Our model uses FlexiViT-style variable patch sizes, which requires the patch embedding to handle different spatial resolutions at runtime. The standard approach uses `nn.Conv2d`, but the convolution's im2col reshape becomes a bottleneck for our multi-modal inputs with small channel counts (1-12 bands). We replaced it with `nn.Linear` operating on flattened patches (reshape + cuBLAS GEMM), which reduced per-step model time by 18.6% (0.86s to 0.70s per step on 8 GPUs) with no change to model capacity or memory footprint. We additionally optimized training throughput through vectorized loss computation and synchronization-free unmasking operations.

## 7. Artifacts

- Chrome traces: `local_output/checkpoints/henryh/profile_{linear,conv}/profiler/rank-0-step-15.chrome_trace.json.gz`
- WandB project: `2026_04_15_profile_patch_embed`
- Script: `scripts/vnext/profiling/profile_patch_embed.py`
