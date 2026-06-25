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

# # # ===== interleaved reads + per-depth read norms/projections: il_pdproj (2) =====
# # # Final-layer-only reads (register_interleave=true, NO register_read_layers), but each of
# # # the register_latent_depth read blocks gets its OWN input LayerNorm + K/V down-projection
# # # (register_per_depth_read_proj=true) instead of sharing one pair. The reads all re-query
# # # the SAME final encoder layer, so per-block projections let successive reads extract a
# # # different view of it through their own lens, rather than being forced through one shared
# # # projection. Generalizes per_depth_read_proj (previously multi-depth only) to the il
# # # schedule. Dynamic grid, d768. Tagged "il_pdproj".
# # #   - with instance (InfoNCE) contrastive loss   -> il_pdproj
# # #   - without it (contrastive_config -> null)     -> il_pdproj_noic
# # # A/Bs against regbtl_base10k_scale0.25_gdyn_d768_il (shared proj) and ..._il_noic above.
# # # NOTE: per-depth norms/projs change the parameter set (input_norm/kv_proj ->
# # # input_norms.i/kv_projs.i), so these start fresh; existing il checkpoints (shared pair)
# # # are unaffected and load unchanged.

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

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj_lrw" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true \
#     --model.encoder_config.register_learned_read_weighting=true

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_pdproj_lrw_noic" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_per_depth_read_proj=true \
#     --model.encoder_config.register_learned_read_weighting=true \
#     --train_module.contrastive_config.loss_config.weight=0

# # ============ read-layer schedule sweep on the mdr3 frontier (3) ============
# # The mdr3_ictok_pdproj frontier (dynamic grid, d768, encoder_tokens contrastive,
# # per_depth_read_proj) with everything fixed EXCEPT register_read_layers. This is the
# # hard-pruned complement to the learned-gate (lrw) runs: instead of letting the model
# # down-weight the early mid-level reads, we just remove them. Tests whether dropping the
# # earliest read(s) recovers the in-domain probe/regression fidelity that the full
# # [3,6,9,12] blend dilutes -- and, as a bonus, fewer reads mean less of the cross-attention
# # dilution over large token fields that drives mdr3's large-grid penalty. A/Bs directly
# # against regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj ([3,6,9,12]). Tagged by the
# # layers read. Supervised only (nosup omitted -- it consistently underperforms here).
# #   - mdr_6_9_12   -> drop the earliest (layer-3) read
# #   - mdr_9_12     -> only the two latest reads
# #   - mdr_8_10_12  -> three late, evenly-spaced reads

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr_6_9_12_ictok_pdproj" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[6,9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr_9_12_ictok_pdproj" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr_8_10_12_ictok_pdproj" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[8,10,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true

# ============ fused multi-depth read source on the il schedule: il_fsum (2) ============
# RAEv2-style (arXiv 2605.18324, sec 2.1 "MLS") multi-layer fusion of the K/V source: the
# encoder taps at [3,6,9,12] are combined into ONE fused source, and the bottleneck runs
# the EXACT il schedule ([read -> self] x4) reading that source -- so this is a single
# config delta vs regbtl_base10k_scale0.25_gdyn_d768_il (final-layer source) and tests
# whether mid-depth content (mdr3's external-transfer edge) survives when delivered
# order-free, without mdr3's sequential-injection dilution or its 4-separate-reads
# large-grid penalty. Win condition is two-dimensional: match il on the in-domain
# probes/regression AND match mdr3 on external transfer.
#   - il_fsum     -> register_fused_read=uniform: parameter-free standardize-and-average.
#     The combination has NO learnable weights, so the pretext loss cannot collapse it
#     onto the pretext-aligned layer 12 (the RAEv2 "training-free" property) -- mid-level
#     features are preserved even though the training objective would discard them.
#     Parameter set is IDENTICAL to il (the fusion adds no parameters).
#   - il_fsum_lrn -> register_fused_read=learned: per-depth norm + projection to
#     register_dim, mean-combined (replaces the shared input_norm/kv_proj, so this starts
#     fresh). The projections CAN re-weight depths; per-depth contribution norms are
#     logged as register_read_source_norms/{3,6,9,12} -- if they collapse onto 12 and the
#     run loses mdr3's transfer gains while il_fsum keeps them, that demonstrates that
#     learned depth-weighting under the pretext objective optimizes the wrong target
#     (the same mechanism predicted to cap the lrw gate runs).
# Dynamic grid, d768, supervised only (nosup consistently underperforms here). The il
# baseline keeps the default contrastive source/IC loss, so these do too.

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_fsum" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_fused_read=uniform

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_fsum_lrn" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_fused_read=learned

# ============ NEW MASKING: sampled 1/2/3 encode modalities (3) ============
# Re-run of the anchor set under the masking change in commit 551ad3e1b
# (RandomTimeWithDecodeMaskingStrategy now samples num_encode ~ Uniform{1..k} instead of
# the deterministic ceil(n*encode_ratio), so the encoder sees 1, 2, OR 3 of {S2,S1,Landsat}
# ~uniformly instead of always exactly 2). The masking is shared data-side code, so these
# launch from current HEAD with NO config change vs their old-masking counterparts -- only
# the run name differs (tag "m123"). Purpose: cleanly A/B the masking change against the
# existing old-masking knn/lp baselines, across the architecture axis (no bottleneck /
# single-read / multi-depth). Watch the SRTM S1 / S2 / S1+S2 split (does S1+S2 - S2 go
# positive?) and the unimodal S1 evals.
#   1. rope baseline (NO bottleneck) -- control: does the masking help fusion on its own?
#      Uses the rope.py launcher, same project/args; $ROPE already sets base10k + scale0.25.
#   2. gdyn_d768_il               -- in-domain frontier (single final-layer read).
#   3. gdyn_d768_mdr3_ictok_pdproj -- external-transfer frontier (multi-depth read).

# python "scripts/official/v1_1/rope.py" launch "rope_base10k_scale0.25_m12" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_il_m12" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     --model.encoder_config.register_interleave=true

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj_m12" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true

# ============ WINDOWED (local) SPATIAL ATTENTION: window in {4, 8, 16} (6) ============
# Restricts every encoder + register attention block (encoder self-attention, the register
# read, and the latent self-attention; the decoder is untouched) to a square sliding window
# of side attn_window_size PATCH CELLS centred on each token. When the input patch grid is
# no larger than the window in both dims the model falls back to full attention, so the
# effect only kicks in once the grid exceeds the window. The window is a pure attention mask
# computed from the existing 2D-RoPE coordinates -- it adds NO parameters, so it is
# checkpoint-compatible and only changes which tokens attend to which.
# Requires spatial_pos_encoding="rope" (set in the scripts) and use_flash_attn=false (the
# flash varlen path cannot express a 2D spatial mask; all these runs are already non-flash).
# Two runs per window size, tagged "w{N}":
#   1. rope baseline (NO bottleneck) -- does local attention help fusion on its own?
#   2. gdyn_d768_mdr3_ictok_pdproj   -- the external-transfer frontier (multi-depth read).
# These launch from current HEAD (same sampled-1/2/3-encode masking as the m123 runs above).
# Windows {4, 8} only -- the training patch grid maxes out at ~13, so a 16-cell window would
# always fall back to full attention (no-op).

# python "scripts/official/v1_1/rope.py" launch "rope_base10k_scale0.25_w4" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.attn_window_size=4

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj_w4" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true \
#     --model.encoder_config.attn_window_size=4

# # python "scripts/official/v1_1/rope.py" launch "rope_base10k_scale0.25_w8" "$CLUSTER" \
# #     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
# #     --model.encoder_config.attn_window_size=8

# python "$SCRIPT" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj_w8" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --model.encoder_config.register_grid_size=0 --model.encoder_config.register_dim=768 --model.decoder_config.register_dim=768 \
#     '--model.encoder_config.register_read_layers=[3,6,9,12]' \
#     --model.encoder_config.register_contrastive_source=encoder_tokens \
#     --model.encoder_config.register_per_depth_read_proj=true \
#     --model.encoder_config.attn_window_size=8

# ============ ORIGINAL MASKING: rope baseline, no instance contrastive loss (1) ============
# Plain rope baseline (NO bottleneck) under the ORIGINAL masking strategy and with the
# instance-level InfoNCE contrastive loss removed (--train_module.contrastive_config=null).
# RandomTimeWithDecodeMaskingStrategy now exposes bandset_split_strategy; the default
# "original" restores the pre-change deterministic split (num_encode = ceil(n*encode_ratio),
# i.e. always ~2 of {S2,S1,Landsat}), reverting the sampled-1/2/3 (m123) and capped (m12)
# behaviors. base.py's _masking_config omits the field, so it picks up the "original" default
# -- no masking override is needed here (and adding a new strategy_config key via CLI is not
# reliably supported), so the original split comes from the data-side default. Tagged
# "origmask_noic". A/Bs against rope_base10k_scale0.25_m12 (isolates the masking) and against
# the IC-on rope baselines (isolates the contrastive loss).

# python "scripts/official/v1_1/rope.py" launch "rope_base10k_scale0.25_origmask_noic" "$CLUSTER" \
#     $LAUNCH_ARGS $WANDB_PROJECT $ROPE \
#     --train_module.contrastive_config=null

# ============ no latent self-attention (nolsa): drop the latent self-attention (3) ============
# Re-runs the il / il_pdproj_noic / mdr3_ictok_pdproj frontiers WITHOUT the bottleneck's
# latent self-attention blocks (register_latent_self_attn=false): the registers are produced
# by the cross-attention read(s) alone, with no register-to-register mixing (the read count
# is unchanged). Isolates the latent transformer's contribution. Each is the nolsa
# counterpart of the like-named run above. Tagged "nolsa".
#
# Unlike the runs above, the architecture is NOT set via CLI overrides -- it is baked into
# dedicated scripts. This is required because these runs also set run_as_beaker_job=true: the
# in-loop evals are launched as a SEPARATE (non-blocking) Beaker job that reconstructs the
# model from the script's build_model_config, so the no-latent-self-attn architecture (and
# the per-run read schedule) must live in the script, not on the train launch's CLI. Those
# scripts also add the fifty_cities random-split S2 + S1+S2 segmentation probes to the
# in-loop evals (now affordable since the evals no longer block training).

NOLSA_IL="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/regbtl_gdyn_d768_il_nolsa.py"
NOLSA_IL_PDPROJ_NOIC="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/regbtl_gdyn_d768_il_pdproj_noic_nolsa.py"
NOLSA_MDR3="scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/regbtl_gdyn_d768_mdr3_ictok_pdproj_nolsa.py"

python "$NOLSA_IL" launch "regbtl_base10k_scale0.25_gdyn_d768_il_nolsa" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE

python "$NOLSA_IL_PDPROJ_NOIC" launch "regbtl_base10k_scale0.25_gdyn_d768_il_pdproj_noic_nolsa" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE

python "$NOLSA_MDR3" launch "regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj_nolsa" "$CLUSTER" \
    $LAUNCH_ARGS $WANDB_PROJECT $ROPE
