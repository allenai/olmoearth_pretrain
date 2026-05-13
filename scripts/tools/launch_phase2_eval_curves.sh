#!/usr/bin/env bash
#
# Launch checkpoint-sweep evals for the phase2.0 base + large pretrains.
#
# Cadence: every 50k steps (50k .. 650k).
# One task group per model:
#   - New aux probes (12 tasks: random + geographic splits over
#     worldcover / osm / srtm / canopy / cdl / worldcereal).
#   - Typical downstream evals (pastis s1/s2, m-eurosat, m-bigearthnet,
#     m-so2sat, yemen_crop, mados, geo_ecosystem_annual_test) plus embedding
#     diagnostics on m-eurosat and pastis-s2.
#
# Total: 2 beaker jobs (base + large), all logging into one wandb project so
# the curves are easy to compare.
#
# Run from the repo root with the project venv active:
#   bash scripts/tools/launch_phase2_eval_curves.sh
#
# Skip lists are (EVAL_TASKS minus the desired include set). If you add or
# rename tasks in olmoearth_pretrain/internal/all_evals.py, update them below.

set -euo pipefail

CLUSTER="ai2/saturn"
WANDB_PROJECT="2026_05_phase2_eval_curves"
STEPS="50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000"
LAUNCH_PRIORITY="urgent"

BASE_CKPT="/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/"
BASE_MODULE="scripts/official/base.py"
LARGE_CKPT="/weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/"
LARGE_MODULE="scripts/official/large.py"

# Tasks to KEEP in the unified sweep (22):
#   pretrain_{worldcover,osm,srtm,canopy,cdl,worldcereal}{,_geo}  (12 aux probes)
#   pastis_sentinel{1,2}, m_{eurosat,bigearthnet,so2sat}, yemen_crop, mados,
#   geo_ecosystem_annual_test, m_eurosat_embed_diag, pastis_sentinel2_embed_diag.
TASK_SKIP="awf_lulc_map,burnrisk_8d_nbac,forest_loss_driver,m_brick_kiln,m_cashew_plant,m_forestnet,m_sa_crop_type,nandi_crop_map,nigeria_settlement,pastis128_sentinel1,pastis128_sentinel1_sentinel2,pastis128_sentinel2,pastis_sentinel1_sentinel2,sen1floods11,tolbi_crop"

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
    --launch.priority="$LAUNCH_PRIORITY" \
    "$@"
}

launch_sweep "$BASE_CKPT"  "$BASE_MODULE"  "phase2_base_eval_sweep"  "$TASK_SKIP" --label_percentages=0.10
launch_sweep "$LARGE_CKPT" "$LARGE_MODULE" "phase2_large_eval_sweep" "$TASK_SKIP" --label_percentages=0.10
