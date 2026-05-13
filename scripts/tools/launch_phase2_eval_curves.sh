#!/usr/bin/env bash
#
# Launch checkpoint-sweep evals for the phase2.0 base + large pretrains.
#
# Cadence: every 50k steps (50k .. 650k).
# Two task groups per model:
#   1. NEW aux probes (12 tasks: random + geographic splits over
#      worldcover / osm / srtm / canopy / cdl / worldcereal) — full label budget.
#   2. Typical downstream evals (pastis s1/s2, m-eurosat, m-bigearthnet,
#      m-so2sat, yemen_crop, mados, geo_ecosystem_annual_test) plus embedding
#      diagnostics on m-eurosat and pastis-s2 — at 10% labels.
#
# Total: 4 beaker jobs (2 models × 2 task groups), all logging into one
# wandb project so the curves are easy to compare.
#
# Run from the repo root with the project venv active:
#   bash scripts/tools/launch_phase2_eval_curves.sh
#
# Skip lists are (EVAL_TASKS minus the desired include set). If you add or
# rename tasks in olmoearth_pretrain/internal/all_evals.py, update them below.

set -euo pipefail

CLUSTER="ai2/saturn-cirrascale"
WANDB_PROJECT="2026_05_phase2_eval_curves"
STEPS="50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000"

BASE_CKPT="/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/"
BASE_MODULE="scripts/official/base.py"
LARGE_CKPT="/weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/"
LARGE_MODULE="scripts/official/large.py"

# Tasks to KEEP in the new-aux sweep (16):
#   pretrain_{worldcover,osm,srtm,canopy,cdl,worldcereal}{,_geo}  (12 aux probes)
#   plus 4 standard evals at FULL labels for comparison vs 10% in typical sweep:
#   m_so2sat, mados, pastis_sentinel2, yemen_crop
NEW_AUX_SKIP="awf_lulc_map,burnrisk_8d_nbac,forest_loss_driver,geo_ecosystem_annual_test,m_bigearthnet,m_brick_kiln,m_cashew_plant,m_eurosat,m_eurosat_embed_diag,m_forestnet,m_sa_crop_type,nandi_crop_map,nigeria_settlement,pastis128_sentinel1,pastis128_sentinel1_sentinel2,pastis128_sentinel2,pastis_sentinel1,pastis_sentinel1_sentinel2,pastis_sentinel2_embed_diag,sen1floods11,tolbi_crop"

# Tasks to KEEP in the typical sweep (10):
#   pastis_sentinel{1,2}, m_{eurosat,bigearthnet,so2sat}, yemen_crop, mados,
#   geo_ecosystem_annual_test, m_eurosat_embed_diag, pastis_sentinel2_embed_diag
TYPICAL_SKIP="awf_lulc_map,burnrisk_8d_nbac,forest_loss_driver,m_brick_kiln,m_cashew_plant,m_forestnet,m_sa_crop_type,nandi_crop_map,nigeria_settlement,pastis128_sentinel1,pastis128_sentinel1_sentinel2,pastis128_sentinel2,pastis_sentinel1_sentinel2,sen1floods11,tolbi_crop,pretrain_canopy_regression,pretrain_canopy_regression_geo,pretrain_cdl_probe,pretrain_cdl_probe_geo,pretrain_osm_probe,pretrain_osm_probe_geo,pretrain_srtm_regression,pretrain_srtm_regression_geo,pretrain_worldcereal_probe,pretrain_worldcereal_probe_geo,pretrain_worldcover_probe,pretrain_worldcover_probe_geo"

launch_sweep() {
  local ckpt="$1"
  local module="$2"
  local model_name="$3"
  local skip="$4"
  shift 4
  python -m olmoearth_pretrain.internal.full_eval_sweep \
    --cluster="$CLUSTER" \
    --checkpoint_dir="$ckpt" \
    --module_path="$module" \
    --defaults_only \
    --steps="$STEPS" \
    --project_name="$WANDB_PROJECT" \
    --model_name="$model_name" \
    --task-skip-names="$skip" \
    "$@"
}

launch_sweep "$BASE_CKPT"  "$BASE_MODULE"  "phase2_base_aux_sweep"        "$NEW_AUX_SKIP"
launch_sweep "$BASE_CKPT"  "$BASE_MODULE"  "phase2_base_typical10_sweep"  "$TYPICAL_SKIP" --label_percentages=0.10
launch_sweep "$LARGE_CKPT" "$LARGE_MODULE" "phase2_large_aux_sweep"       "$NEW_AUX_SKIP"
launch_sweep "$LARGE_CKPT" "$LARGE_MODULE" "phase2_large_typical10_sweep" "$TYPICAL_SKIP" --label_percentages=0.10
