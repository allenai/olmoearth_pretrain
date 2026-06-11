#!/bin/bash
# Spatial register bottleneck sweep (Perceiver-style) — INDIVIDUAL launch commands.
#
# This is a MENU, not a run-top-to-bottom script. Run the variable block below ONCE
# (paste it / `source` the header), then run any individual launch command you still
# need — so you can skip runs that have already started.
#
# 24 runs: register_grid_size in {8, 16, 32} x register_dim in {192, 288, 528, 768}, each:
#   - low-weight register supervision:  regbtl_base10k_scale0.25_g{G}_d{D}
#   - no supervision (pure JEPA):       regbtl_base10k_scale0.25_g{G}_d{D}_nosup
# All use RoPE base10k + coordinate scale 0.25; head_dim = dim/12 in {16, 24, 44, 64}.
# Registers per grid: g8 -> 64, g16 -> 256, g32 -> 1024.
# register_read_depth=1, register_latent_depth=4, register_num_heads=12 (set in scripts).
#
# + 2 dynamic-grid runs (register_grid_size=0): a SINGLE learned latent cloned across
#   a grid that MATCHES THE PATCH GRID at forward time (translation-invariant prior, no
#   fixed grid size). d768, sup + nosup. Tagged "gdyn".
# + 4 interleaved-read runs (register_interleave=true): [read -> self] x register_latent_depth
#   instead of one up-front read (Perceiver/DETR/Flamingo). d768, g16 + gdyn, sup + nosup.
#   Tagged "il". See the final two sections below.

SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_supervision_register_bottleneck.py"
NOSUP_SCRIPT="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/hidden1_register_bottleneck_no_supervision.py"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"
WANDB_PROJECT="--trainer.callbacks.wandb.project=2026_04_22_add_hidden_layer_to_initial_projection"
ROPE="--model.encoder_config.rope_coordinate_scale=0.25 --model.decoder_config.rope_coordinate_scale=0.25"

# ============================ supervised (12) ===========================

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d192" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d288" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d528" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g8_d768" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d192" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d288" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d528" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d192" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d288" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d528" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g32_d768" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# # ========================= no supervision (12) =========================

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d192_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d288_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d528_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g8_d768_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=8 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d192_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d288_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d528_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d192_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=192 --model.decoder_config.register_dim=192

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d288_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=288 --model.decoder_config.register_dim=288

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d528_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=528 --model.decoder_config.register_dim=528

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g32_d768_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=32 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# # ==================== dynamic grid: single cloned latent (2) ====================
# # register_grid_size=0 -> one shared latent cloned across the patch grid (no fixed grid).
# # 0 (not null) is the dynamic sentinel so it survives config serialization; the decoder
# # has no grid field.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768

# # =================== interleaved reads: [read -> self] x4 (4) ===================
# # register_interleave=true interleaves each cross-attention read with a latent self-
# # attention block (Perceiver/DETR/Flamingo) so the registers re-query after each refine,
# # instead of the default single up-front read. register_latent_depth=4 -> 4 reads + 4 self.
# # d768, on the g16 fixed grid and the gdyn dynamic grid, sup + nosup. Tagged "il".

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768_il" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_g16_d768_il_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=16 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true

# # ============ multi-depth reads: read from intermediate encoder layers (2) ============
# # register_read_layers gives the 1-indexed encoder depths the bottleneck reads from -- one
# # [read -> self-attend] step per entry, each reading the patch tokens AT THAT DEPTH instead
# # of re-reading the final layer. Motivation: the alignment/bottleneck objective drives the
# # final layer to drop modality-unique info; reading earlier layers recovers it (Lee et al.,
# # CVPR 2026, "Beyond What's Shared"). Encoder is ViT-base (12 layers); stride 3 ->
# # [3,6,9,12] (4 reads). Forces the interleaved schedule and overrides
# # register_read_depth/register_latent_depth. Dynamic grid, sup + nosup. Tagged "mdr3".
# # Brackets are quoted for the shell.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]'

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]'

# # ========= contrastive projection from the register tokens + read-norm cleanup (2) =========
# # Re-baselines the mdr3 frontier (dynamic grid, d768, read_layers=[3,6,9,12]) under two
# # code changes that are now always-on for register-bottleneck models:
# #   1. project_and_aggregate (the contrastive/pooled projection) now pools the REGISTER
# #      tokens only and is sized to register_dim, instead of mean-pooling the patch tokens.
# #      The contrastive view now sees exactly what the JEPA decoder + frozen evals see.
# #   2. the multi-depth read drops the redundant encoder LayerNorm in _finalize_read_tokens
# #      (the bottleneck's input_norm already normalizes every K/V source, and reusing the
# #      encoder output-norm both double-normed and coupled its affine across read depths).
# # Config is identical to the mdr3 runs above; the new tag just tracks the new behavior so
# # it A/Bs cleanly against them. Dynamic grid, sup + nosup. Tagged "creg".
# # NOTE: checkpoints from older register-bottleneck runs will NOT load project_and_aggregate
# # (its Linear is now register_dim-wide), so these start fresh.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_creg" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]'

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_creg_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]'

# # ============ no instance contrastive loss: InfoNCE weight = 0 (2) ============
# # Same mdr3 frontier config as the creg runs above, but the instance (InfoNCE) contrastive
# # loss is zeroed (--train_module.contrastive_config.loss_config.weight=0). InfoNCELoss.compute
# # returns weight * cross_entropy, so weight=0 removes its gradient contribution entirely.
# # Because the register-contrastive change only routes the project_and_aggregate pooled
# # vector into this loss, weight=0 effectively ablates that pathway -- isolating whether the
# # instance contrastive signal helps the bottleneck at all. Dynamic grid, sup + nosup.
# # Tagged "noic" (no instance contrastive).

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_noic" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --train_module.contrastive_config.loss_config.weight=0

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_noic_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --train_module.contrastive_config.loss_config.weight=0

# # === instance contrastive on patch tokens + per-depth read norms/projections (2) ===
# # Same mdr3 frontier (dynamic grid, d768, read_layers=[3,6,9,12]) with TWO opt-in changes,
# # both enabled together:
# #   1. register_contrastive_source=encoder_tokens: the instance (InfoNCE) contrastive loss
# #      pools the ENCODER PATCH TOKENS (final-embedding-size, masked-mean) instead of the
# #      register latents -- the pre-bottleneck behavior. project_and_aggregate is sized to
# #      the encoder embedding size again (NOT register_dim), so this reverses the "creg"
# #      change #1; the contrastive view sees the patch tokens, not what the decoder/evals see.
# #   2. register_per_depth_read_proj=true: each multi-depth read layer ([3,6,9,12]) gets its
# #      OWN input LayerNorm and K/V down-projection instead of a single shared pair, since
# #      different encoder depths have different per-channel statistics.
# # Tagged "ictok_pdproj". Dynamic grid, sup + nosup.
# # NOTE: both changes alter the parameter set (project_and_aggregate width; per-depth
# # norms/projs), so these start fresh -- older checkpoints will not load.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true

# # === per-depth read norms/projections, no instance contrastive loss (2) ===
# # The ictok_pdproj config above with the instance (InfoNCE) contrastive loss ZEROED
# # (--train_module.contrastive_config.loss_config.weight=0). With the contrastive loss off,
# # register_contrastive_source is moot (the project_and_aggregate vector it routes is no
# # longer in the loss), so it is dropped here -- the only live change vs the plain mdr3
# # frontier is register_per_depth_read_proj=true. A/Bs against:
# #   - mdr3_noic        -> isolates per-depth read norms/projections (both have no IC loss)
# #   - mdr3_ictok_pdproj -> isolates the instance contrastive loss (both have per-depth proj)
# # Tagged "pdproj_noic". Dynamic grid, sup + nosup.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_pdproj_noic" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_per_depth_read_proj=true \
#     --train_module.contrastive_config.loss_config.weight=0

# python "$NOSUP_SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_pdproj_noic_nosup" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_per_depth_read_proj=true \
#     --train_module.contrastive_config.loss_config.weight=0
# # ============ no instance contrastive loss: gdyn_d768_il, supervised (1) ============
# # Identical to regbtl_base10k_scale0.25_gdyn_d768_il above, but drops the instance-level
# # InfoNCE contrastive loss (train_module.contrastive_config -> null). The patch
# # discrimination (JEPA) loss and low-weight register supervision are unchanged. No code
# # change needed: contrastive_config is Optional and the train module no-ops when it's None.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_noic" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true \
#     --train_module.contrastive_config=null

# # ===== interleaved reads + per-depth read norms/projections: il_pdproj (2) =====
# # Final-layer-only reads (register_interleave=true, NO register_read_layers), but each of
# # the register_latent_depth read blocks gets its OWN input LayerNorm + K/V down-projection
# # (register_per_depth_read_proj=true) instead of sharing one pair. The reads all re-query
# # the SAME final encoder layer, so per-block projections let successive reads extract a
# # different view of it through their own lens, rather than being forced through one shared
# # projection. Generalizes per_depth_read_proj (previously multi-depth only) to the il
# # schedule. Dynamic grid, d768. Tagged "il_pdproj".
# #   - with instance (InfoNCE) contrastive loss   -> il_pdproj
# #   - without it (contrastive_config -> null)     -> il_pdproj_noic
# # A/Bs against regbtl_base10k_scale0.25_gdyn_d768_il (shared proj) and ..._il_noic above.
# # NOTE: per-depth norms/projs change the parameter set (input_norm/kv_proj ->
# # input_norms.i/kv_projs.i), so these start fresh; existing il checkpoints (shared pair)
# # are unaffected and load unchanged.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_pdproj" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true \
#     --model.encoder_config.register_per_depth_read_proj=true

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_pdproj_noic" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true \
#     --model.encoder_config.register_per_depth_read_proj=true \
#     --train_module.contrastive_config=null

# ============ learned per-read residual gates on the mdr3 frontier (2) ============
# The mdr3_ictok_pdproj frontier (dynamic grid, d768, read_layers=[3,6,9,12],
# register_contrastive_source=encoder_tokens, register_per_depth_read_proj=true) plus
# register_learned_read_weighting=true: each of the 4 reads gets a learnable scalar gate on
# its residual contribution to the latent (z + g_d * (read_d(z) - z)), ELMo/LayerScale-style.
# Gates init to 1.0 so it is a strict no-op at init (reproduces mdr3_ictok_pdproj exactly)
# and the model can LEARN to down-weight the early mid-level reads ([3,6,9]) that dilute the
# pretext-aligned final-layer read ([12]) -- the leading explanation for mdr3 losing to il on
# in-domain probes/regression. The learned gates are exposed as register_bottleneck.read_gates
# (log them: collapse toward [12] -> mid-level reads weren't pulling their weight). Tagged "lrw".
#   - lrw          -> mdr3_ictok_pdproj + gates, instance contrastive (InfoNCE) ON
#   - pdproj_lrw_noic -> same with the IC loss zeroed; register_contrastive_source is then moot
#     (its project_and_aggregate vector is no longer in the loss) so it is dropped, matching
#     the pdproj_noic block. A/Bs against mdr3_pdproj_noic (isolates the gates) and against
#     mdr3_ictok_pdproj_lrw (isolates the IC loss).
# NOTE: read_gates is a new parameter, so these start fresh; older checkpoints will not load.

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj_lrw" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    '--model.encoder_config.register_read_layers=[3,6,9,12]' \
    --model.encoder_config.register_contrastive_source=encoder_tokens \
    --model.encoder_config.register_per_depth_read_proj=true \
    --model.encoder_config.register_learned_read_weighting=true

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_pdproj_lrw_noic" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    '--model.encoder_config.register_read_layers=[3,6,9,12]' \
    --model.encoder_config.register_per_depth_read_proj=true \
    --model.encoder_config.register_learned_read_weighting=true \
    --train_module.contrastive_config.loss_config.weight=0

# ============ read-layer schedule sweep on the mdr3 frontier (3) ============
# The mdr3_ictok_pdproj frontier (dynamic grid, d768, encoder_tokens contrastive,
# per_depth_read_proj) with everything fixed EXCEPT register_read_layers. This is the
# hard-pruned complement to the learned-gate (lrw) runs: instead of letting the model
# down-weight the early mid-level reads, we just remove them. Tests whether dropping the
# earliest read(s) recovers the in-domain probe/regression fidelity that the full
# [3,6,9,12] blend dilutes -- and, as a bonus, fewer reads mean less of the cross-attention
# dilution over large token fields that drives mdr3's large-grid penalty. A/Bs directly
# against regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj ([3,6,9,12]). Tagged by the
# layers read. Supervised only (nosup omitted -- it consistently underperforms here).
#   - mdr_6_9_12   -> drop the earliest (layer-3) read
#   - mdr_9_12     -> only the two latest reads
#   - mdr_8_10_12  -> three late, evenly-spaced reads

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr_6_9_12_ictok_pdproj" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    '--model.encoder_config.register_read_layers=[6,9,12]' \
    --model.encoder_config.register_contrastive_source=encoder_tokens \
    --model.encoder_config.register_per_depth_read_proj=true

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr_9_12_ictok_pdproj" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    '--model.encoder_config.register_read_layers=[9,12]' \
    --model.encoder_config.register_contrastive_source=encoder_tokens \
    --model.encoder_config.register_per_depth_read_proj=true

python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr_8_10_12_ictok_pdproj" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
    --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
    '--model.encoder_config.register_read_layers=[8,10,12]' \
    --model.encoder_config.register_contrastive_source=encoder_tokens \
    --model.encoder_config.register_per_depth_read_proj=true
