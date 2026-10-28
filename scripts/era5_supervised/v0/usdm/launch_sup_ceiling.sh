#!/usr/bin/env bash
#
# v1.1.4 supervised ceiling for the USDM evals: era5enc_1104_sup_usdm_r3
# (r1, era5enc_1104_sup_usdm, trained a dfut8 head and probed without
# probe_pooling; stopped at step ~50. r2 ran to step 29,250 with its heads
# frozen at random init, since they were created after the optimizer, and
# could not resume after a preemption; fixed in 3c2d35621. r2 also predates
# the evalu group, so its probes read the first eval group).
#
# Objective A only (reconstruction off): the 1601 encoder recipe (script
# defaults + day-of-year fix) trained from scratch on three USDM targets of the
# full 'all' group (current severity, 8- and 26-week past change; one
# regression head each, targets z-scored with the registry train stats; the
# coming-change targets are not learnable from the history, so no head),
# LR 1e-4 as for the v1.1.x ceilings and the FLUXNET ceiling (1103). The in-loop
# probes (every 1000 steps, frozen encoder; the change evals read the last 4
# tokens, registry probe_pooling) read the USDM evals plus LFMC; the
# best value over steps is the ceiling estimate. A frozen linear probe can
# understate what direct supervision reaches, so read it next to the gbm
# baselines (baselines.py).
#
# Usage (repo root; commit pushed):
#   bash scripts/era5_supervised/v0/usdm/launch_sup_ceiling.sh
#   SUBCMD=dry_run bash ...   # build + print the config locally

set -euo pipefail

SUBCMD="${SUBCMD:-launch}"
RUN="${RUN:-era5enc_1104_sup_usdm_r3}"
CLUSTER="${CLUSTER:-ai2/saturn}"
EVALS="${EVALS:-usdm_dpast26_eval,usdm_dpast13_eval,usdm_sev_eval}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" scripts/era5_supervised/v0/base.py "${SUBCMD}" "${RUN}" "${CLUSTER}" \
  common.enable_supervised=True \
  common.enable_reconstruction=False \
  "common.tasks=[usdm_sev,usdm_dpast8,usdm_dpast26]" \
  common.encoder_patch_day_of_year=center \
  common.learning_rate=1e-4 \
  common.max_steps="${MAX_STEPS:-30000}" \
  common.eval_interval=1000 \
  "common.eval_tasks=[${EVALS},lfmc_woody_eval]" \
  "common.eval_probe_lr_grid=[1e-3,1e-2,3e-2,1e-1]" \
  common.eval_run_on_test=True \
  "common.eval_results_jsonl=/weka/dfive-default/rslearn-eai/datasets/usdm/probe_results/${RUN}.jsonl" \
  --trainer.callbacks.wandb.project=era5_usdm_eval \
  --launch.min_runtime=5h
