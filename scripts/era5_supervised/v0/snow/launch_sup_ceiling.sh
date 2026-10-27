#!/usr/bin/env bash
#
# v1.1.5 supervised ceiling for the snow evals: era5enc_1105_sup_snow.
#
# Objective A only (reconstruction off): the 1601 encoder recipe (script
# defaults + day-of-year fix) trained from scratch on the snow melt-timing
# targets of the full 'all_<family>' groups (one regression head each, targets
# z-scored with the registry train stats), LR 1e-4 as for the v1.1.x, FLUXNET
# (1103) and USDM (1104) ceilings. The in-loop probes (every 1000 steps, frozen
# encoder) read the snow evals plus LFMC; the best value over steps is the
# ceiling estimate. A frozen linear probe can understate what direct
# supervision reaches, so read it next to the gbm baselines (baselines.py).
#
# Usage (repo root; commit pushed):
#   bash scripts/era5_supervised/v0/snow/launch_sup_ceiling.sh
#   SUBCMD=dry_run bash ...   # build + print the config locally

set -euo pipefail

SUBCMD="${SUBCMD:-launch}"
RUN="${RUN:-era5enc_1105_sup_snow}"
CLUSTER="${CLUSTER:-ai2/saturn}"
TASKS="${TASKS:-snow_sdd_anom_l30}"
EVALS="${EVALS:-snow_sdd_anom_l30_eval,snow_sdd_anom_l60_eval,snow_sdd_anom_post_eval}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" scripts/era5_supervised/v0/base.py "${SUBCMD}" "${RUN}" "${CLUSTER}" \
  common.enable_supervised=True \
  common.enable_reconstruction=False \
  "common.tasks=[${TASKS}]" \
  common.encoder_patch_day_of_year=center \
  common.learning_rate=1e-4 \
  common.max_steps="${MAX_STEPS:-30000}" \
  common.eval_interval=1000 \
  "common.eval_tasks=[${EVALS},lfmc_woody_eval]" \
  "common.eval_probe_lr_grid=[1e-3,1e-2,3e-2,1e-1]" \
  common.eval_run_on_test=True \
  "common.eval_results_jsonl=/weka/dfive-default/rslearn-eai/datasets/snow/probe_results/${RUN}.jsonl" \
  --trainer.callbacks.wandb.project=era5_snow_eval
