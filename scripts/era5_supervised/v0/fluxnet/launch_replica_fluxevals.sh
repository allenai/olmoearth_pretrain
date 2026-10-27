#!/usr/bin/env bash
#
# v1.6.8: era5enc_1608_doyc_fluxevals, a replica of era5enc_1601_doyc (default
# seed, same code path) with the five FLUXNET evals added to the in-loop probes.
#
# The probes fork the RNG, but materializing the extra eval datasets creates
# DataLoader iterators that draw from the global torch RNG, so training is not
# bit-identical to 1601 after the startup eval: read 1608 as a further seed of
# the 1601 recipe. The FLUXNET curves are read at the same 1000-step cadence
# and probe seed as every other eval, from the step-0 random-projection floor
# to step 50k, next to the six existing evals of the same run.
#
# Usage (repo root; commit pushed):
#   bash scripts/era5_supervised/v0/fluxnet/launch_replica_fluxevals.sh
#   SUBCMD=dry_run bash ...

set -euo pipefail

SUBCMD="${SUBCMD:-launch}"
RUN="${RUN:-era5enc_1608_doyc_fluxevals}"
CLUSTER="${CLUSTER:-ai2/saturn}"
FLUX="fluxnet_rgpp_eval,fluxnet_gpp_eval,fluxnet_le_eval,fluxnet_nee_eval,fluxnet_gpp_anom_eval"
OLD="dfmc1000h_grav_eval,lfmc_woody_eval,cybench_wheat_eval,cybench_maize_eval,burnrisk_canada_nbac_eval,landslide_era5_eval"

.venv/bin/python scripts/era5_supervised/v0/base.py "${SUBCMD}" "${RUN}" "${CLUSTER}" \
  common.encoder_patch_day_of_year=center \
  "common.eval_tasks=[${OLD},${FLUX}]" \
  "common.eval_results_jsonl=/weka/dfive-default/rslearn-eai/datasets/fluxnet/probe_results/${RUN}.jsonl" \
  --trainer.callbacks.wandb.project=era5_fluxnet_eval
