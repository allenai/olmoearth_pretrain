#!/bin/bash
# v1.2 spatial register-bottleneck sweep, built on the _il frontier.
#
# Fixed architecture (baked into regbtl_v1_2_common.build_regbtl_model_config):
#   gdyn   -- dynamic single-latent register grid (register_grid_size=0)
#   il     -- interleaved reads ([read -> self] x4)
#   pdproj -- per-depth read projections
#   register_dim=768
# The encoder self-attention keeps v1.2's 3D mixed RoPE; the register bottleneck reads and
# the decoder cross-attention run in 2D (the register grid is spatial-only).
#
# 2x2 sweep:
#   ic / noic   -- InfoNCE instance contrastive loss on (from base) / off
#   lsa / nolsa -- bottleneck latent self-attention on / off
#
# In-loop evals run as separate (non-blocking) Beaker jobs and add the fifty_cities
# random-split S2 + S1+S2 probes. The architecture is baked into the scripts (not CLI
# overrides) so those eval jobs reconstruct the matching model.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

IC_LSA="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_ic_lsa.py"
IC_NOLSA="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_ic_nolsa.py"
NOIC_LSA="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa.py"
NOIC_NOLSA="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_nolsa.py"
# Single-forward-pass twin of NOIC_LSA: identical config, but the plain (non-contrastive)
# train module runs one forward pass per batch instead of two (the contrastive loss is 0).
NOIC_LSA_1FWD="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd.py"
# Same single-forward-pass run, but with the fused AdamW kernel (fused=True) for extra speed.
NOIC_LSA_1FWD_FUSEDADAMW="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd_fusedadamw.py"
# Register-width sweep on the noic_lsa recipe with ALL validated speedups (base_faster.py):
# 1fwd + fused AdamW + projection-only target + replicated DP (ddp) + bf16 autocast.
D128_FASTER="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_faster.py"
D256_FASTER="scripts/official/v1_2/regbtl_v1_2_gdyn_d256_il_pdproj_noic_lsa_faster.py"
D512_FASTER="scripts/official/v1_2/regbtl_v1_2_gdyn_d512_il_pdproj_noic_lsa_faster.py"
# Width sweep with the bottleneck attention DECOUPLED to encoder width (register_attn_dim
# =768, 12x64 heads, reads consume full-width K/V): register_dim is purely the storage
# bottleneck. Supersedes the _faster head-count variants (2/4-head runs were 2x slow;
# 16/32-dim-head runs degraded on spatial evals).
D128_WIDEREAD="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread.py"
D256_WIDEREAD="scripts/official/v1_2/regbtl_v1_2_gdyn_d256_il_pdproj_noic_lsa_wideread.py"
D512_WIDEREAD="scripts/official/v1_2/regbtl_v1_2_gdyn_d512_il_pdproj_noic_lsa_wideread.py"

# python "$IC_LSA" launch "regbtl_v1_2_gdyn_d768_il_pdproj_ic_lsa" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$IC_NOLSA" launch "regbtl_v1_2_gdyn_d768_il_pdproj_ic_nolsa" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$NOIC_LSA" launch "regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$NOIC_NOLSA" launch "regbtl_v1_2_gdyn_d768_il_pdproj_noic_nolsa" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$NOIC_LSA_1FWD" launch "regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$NOIC_LSA_1FWD_FUSEDADAMW" launch "regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd_fusedadamw" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$D128_FASTER" launch "regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_faster_v2" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$D256_FASTER" launch "regbtl_v1_2_gdyn_d256_il_pdproj_noic_lsa_faster_v2" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

# python "$D512_FASTER" launch "regbtl_v1_2_gdyn_d512_il_pdproj_noic_lsa_faster_v2" "$CLUSTER" \
#     $LAUNCH_ARGS \
#     --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_WIDEREAD" launch "regbtl_v1_2_gdyn_d128_wideread" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D256_WIDEREAD" launch "regbtl_v1_2_gdyn_d256_wideread" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D512_WIDEREAD" launch "regbtl_v1_2_gdyn_d512_wideread" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
