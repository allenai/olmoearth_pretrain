#!/usr/bin/env bash
#
# v1.1.3 supervised ceiling for the FLUXNET evals: era5enc_1103_sup_flux.
#
# Objective A only (reconstruction off): the 1601 encoder recipe (script
# defaults + day-of-year fix) trained from scratch on three FLUXNET targets of
# the full 'all' group (8-day GPP, LE and GPP anomaly; one regression head
# each, targets z-scored with the registry train stats), LR 1e-4 as for the
# v1.1.x ceilings. The in-loop probes (every 1000 steps, frozen encoder) read
# the four FLUXNET evals plus LFMC; the best value over steps is the ceiling
# estimate. As in v1.1.x, a frozen linear probe can understate what direct
# supervision reaches, so read it next to the offline gbm baseline.
#
# Usage (repo root; commit pushed):
#   bash scripts/era5_supervised/v0/fluxnet/launch_sup_ceiling.sh
#   SUBCMD=dry_run bash ...   # build + print the config locally

set -euo pipefail

SUBCMD="${SUBCMD:-launch}"
RUN="${RUN:-era5enc_1103_sup_flux}"
CLUSTER="${CLUSTER:-ai2/saturn}"
FLUX="fluxnet_gpp_eval,fluxnet_le_eval,fluxnet_nee_eval,fluxnet_gpp_anom_eval"

.venv/bin/python scripts/era5_supervised/v0/base.py "${SUBCMD}" "${RUN}" "${CLUSTER}" \
  common.enable_supervised=True \
  common.enable_reconstruction=False \
  "common.tasks=[fluxnet_gpp,fluxnet_le,fluxnet_gpp_anom]" \
  common.encoder_patch_day_of_year=center \
  common.learning_rate=1e-4 \
  common.max_steps="${MAX_STEPS:-30000}" \
  common.eval_interval=1000 \
  "common.eval_tasks=[${FLUX},lfmc_woody_eval]" \
  "common.eval_probe_lr_grid=[1e-3,1e-2,3e-2,1e-1]" \
  common.eval_run_on_test=True \
  "common.eval_results_jsonl=/weka/dfive-default/rslearn-eai/datasets/fluxnet/probe_results/${RUN}.jsonl" \
  --trainer.callbacks.wandb.project=era5_fluxnet_eval
