# v1.2 training speedup program (July 2026)

Scripts from the v1.2 base-model speedup investigation (branch
`gabi/fuse-adamw-speedup`). Outcome: **1.39× end-to-end training speedup**
(full production run 104.4h vs ~147h on 8×H100) with matched-step loss
identical to the fourth decimal and downstream evals within seed noise at
every rung. The validated changes (fused AdamW default in `base.py`;
projection-only target + replicated DP) graduated to the production configs;
torch.compile and flash-attn were investigated and rejected.

`base.py` here is a frozen snapshot of `scripts/official/v1_2/base.py` at
archive time so the scripts remain runnable as-is (they `from base import`).

## Ladder results (full 667k-step runs, W&B `2026_07_08_fused_adamw_e2e`)

| config                        | median step | wall clock (ratio) |
|-------------------------------|-------------|--------------------|
| unfused FSDP (original)       | 0.636s      | ~147h (1.00)       |
| + fused AdamW                 | 0.534s      | ~130h (0.85)       |
| + replicated DP               | 0.479s      | 107.2h (0.74)      |
| + projection-only target      | 0.465s      | 104.4h (0.72)      |

## Contents

Phase 1 — attribution + flag ladder (500-step benchmarks, W&B
`2026_07_08_speed_benchmark`): `speed_benchmark.py` (profiler + padding
metric; found the run 99.7% compute-bound, padding only ~6% so flash-attn
lost, compile/fused won) + `launch_speed_benchmark.sh`.

Phase 2 — compile+fused validation A/B (W&B `2026_07_09_compile_fused_ab`):
`compile_fused_ab.py` + `launch_compile_fused_ab.sh`. Found torch.compile
degrades training (loss gap opens at the ~3k-step phase transition, grows
monotonically); fused AdamW tracks the baseline at seed-noise level.

Phase 3 — compile debugging: `compile_fix_emulate_casts.py`
(`torch._inductor.config.emulate_precision_casts`), `compile_fix_eager_rope.py`
(RoPE excluded from compile via `torch.compiler.disable`) +
`launch_compile_debug.sh`, later `base_proj_target_ddp_compile_e2e.py` +
`launch_compile_envelope.sh`. Neither numerics fix changed the trajectory
(±0.0003) — the failure is frontend-level (Dynamo/AOTAutograd/functionalized
RNG), not Inductor codegen. Compile rejected.

Phase 4 — end-to-end production arms (W&B `2026_07_08_fused_adamw_e2e`):
`base_fused_e2e.py` + `launch_fused_e2e.sh` (fused vs unfused),
`base_proj_target_e2e.py`, `base_ddp_e2e.py`, `base_proj_target_ddp_e2e.py` +
`launch_target_ddp_e2e.sh` (single-change and combined arms, incl. a
seed-change pair used as the eval noise floor).

Pitfall preserved for posterity: never compute the patch-discrimination loss
inside `torch.autocast` — bmm silently re-casts to bf16 despite internal
`.float()` calls (+1.3 Disc loss, visible from step 2). Both train modules
now wrap loss computation in `torch.autocast(..., enabled=False)`.
