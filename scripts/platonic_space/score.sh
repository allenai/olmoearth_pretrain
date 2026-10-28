#!/usr/bin/env bash
#
# Score exported embedding dirs with favyen's Strata scorer (his venv and code, read-only;
# writes only under ROOT). Names are dirs under ROOT/eval_embeddings, e.g.
#   bash score.sh raw_aef/step0 paired_l128_zaef/step20000
#   PASTIS=1 bash score.sh ...      # also the PASTIS segmentation LP (slow; one probe seed)
#
# Quantization follows what each vector is: raw_aef / raw_oe already sit on the int8 power
# grid and concat is float (none), raw_tess gets TESSERA's own round trip as in the harness,
# and the unit-norm latents and concat_pca* get the AEF power scheme.
# --allow_missing because export.py keeps only the windows where all three products exist.

set -euo pipefail

ROOT=/weka/dfive-default/hadriens/platonic_space
STRATA=/weka/dfive-default/favyen/strata
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${STRATA}/olmoearth_strata:${STRATA}/olmoearth_pretrain"
PY="${STRATA}/venvs/oe_strata/bin/python"
WORK="${ROOT}/eval_work"
mkdir -p "${WORK}" "${ROOT}/eval_results"

for d in "$@"; do
  name="${d%%/*}"
  case "${name}" in
    raw_aef | raw_oe | concat) q=none ;;
    raw_tess) q=tessera ;;
    *) q=aef_power ;;
  esac
  out="${ROOT}/eval_results/${d//\//_}${PASTIS:+_pastis}.json"
  if [[ -e "${out}" ]]; then echo "skip ${out}"; continue; fi
  datasets=ethiopia_crops,africa_crop_mask,canada_crops_fine,canada_crops_coarse,descals,lcmap_lu,glance,us_trees
  echo "== ${d} (quantization ${q})"
  (cd "${WORK}" && nice "${PY}" -m olmoearth_strata.evals.score --emb_dir "${ROOT}/eval_embeddings/${d}" \
    --quantization "${q}" --allow_missing --datasets "${datasets}" --seeds 0,1,2,3,4 \
    ${PASTIS:+--pastis --pastis_seeds 0} --work_dir "${WORK}" --workers "${WORKERS:-48}" --out "${out}" 2>&1 | tail -14)
done
