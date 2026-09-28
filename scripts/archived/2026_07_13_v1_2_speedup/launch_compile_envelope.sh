#!/bin/bash
# torch.compile re-test on the new runtime stack (proj_target + replicated DP +
# fused AdamW), judged against a SEED-NOISE ENVELOPE instead of point-wise
# loss-curve tracking (the old criterion rejects any change that alters fp
# execution order -- seed changes diverge at the ~3000-step phase transition too).
#
# Arms (all in W&B project 2026_07_08_fused_adamw_e2e):
#   envelope: v1_2_base_proj_target_ddp_v2 (dnkcy2c2, seed 12536) is already
#             running; two more seeds below bound the seed-noise band.
#             --init_seed changes model init + dropout streams; data order
#             (data_loader.seed) stays fixed, matching what compile perturbs.
#   compile:  same config + compile_model=true (TORCH_LOGS=recompiles on the
#             job for the recompile audit -- check rank-0 stderr).
#
# Verdict after ~8k steps: accept compile iff its loss level and the 4k/8k
# eval-job metrics fall INSIDE the envelope spanned by the three seeds, and
# the recompile count is bounded. If it fails cleanly, drop compile for good
# and invest in manual op-batching instead.
set -e

PROJECT="2026_07_08_fused_adamw_e2e"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[$CLUSTER]"
COMMON_OVERRIDES="--trainer.callbacks.wandb.project=$PROJECT"

# Seed-noise envelope (config identical to v1_2_base_proj_target_ddp_v2)
# python scripts/archived/2026_07_13_v1_2_speedup/base_proj_target_ddp_e2e.py launch v1_2_base_proj_target_ddp_seed8956 $CLUSTER \
#     $LAUNCH_ARGS $COMMON_OVERRIDES --init_seed=8956

# python scripts/archived/2026_07_13_v1_2_speedup/base_proj_target_ddp_e2e.py launch v1_2_base_proj_target_ddp_seed30991 $CLUSTER \
#     $LAUNCH_ARGS $COMMON_OVERRIDES --init_seed=30991

# Compile arm (default seed 12536, directly comparable to dnkcy2c2)
python scripts/archived/2026_07_13_v1_2_speedup/base_proj_target_ddp_compile_e2e.py launch v1_2_base_proj_target_ddp_compile $CLUSTER \
    $LAUNCH_ARGS $COMMON_OVERRIDES
