#!/bin/bash
# Run embedding diagnostics on EMA vs no-EMA checkpoints across training.
# Each model gets one wandb run with checkpoint_step as x-axis.
# Results go to wandb project: 2026_ema_embedding_analysis

set -e

TASKS='["m_eurosat","pastis_sentinel2","mados"]'
PROJECT="2026_ema_embedding_analysis"
MODULE="scripts/official/base.py"
CLUSTER="ai2/saturn-cirrascale"
EXTRA_CLUSTERS='--launch.clusters=["ai2/saturn-cirrascale","ai2/jupiter-cirrascale-2","ai2/titan","ai2/ceres"]'

# --- henryh EMA full (no maps, s2/s1/landsat only) ---
echo "=== Launching EMA full sweep ==="
python -m olmoearth_pretrain.internal.full_eval_sweep \
    --checkpoint_dir=/weka/dfive-default/helios/checkpoints/henryh/bugfix_base_random_s1s2landsat_random_patchdisc_nocon_emafull \
    --module_path="${MODULE}" \
    --cluster="${CLUSTER}" \
    --steps=20000,45000,65000,86500 \
    --embedding_diagnostics_only \
    --model_name="henryh_emafull_sweep" \
    "--trainer.callbacks.downstream_evaluator.tasks_to_run=${TASKS}" \
    "--trainer.callbacks.wandb.project=${PROJECT}" \
    '--common.training_modalities=["sentinel2_l2a","sentinel1","landsat"]' \
    "${EXTRA_CLUSTERS}"

# --- joer base (no EMA, all modalities) ---
echo "=== Launching Base no-EMA sweep ==="
python -m olmoearth_pretrain.internal.full_eval_sweep \
    --checkpoint_dir=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02 \
    --module_path="${MODULE}" \
    --cluster="${CLUSTER}" \
    --steps=150000,300000,450000,600000 \
    --embedding_diagnostics_only \
    --model_name="joer_base_noema_sweep" \
    "--trainer.callbacks.downstream_evaluator.tasks_to_run=${TASKS}" \
    "--trainer.callbacks.wandb.project=${PROJECT}" \
    "${EXTRA_CLUSTERS}"

echo "Both jobs launched."
