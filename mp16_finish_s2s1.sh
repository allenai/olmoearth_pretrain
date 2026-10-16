#!/usr/bin/env bash
# Finish the smoke test on S2+S1: meta summaries -> H5 conversion.
set -uo pipefail
WT=/root/repos/oe_pretrain_corpus_v2
BASE=/weka/dfive-default/olmoearth_pretrain/dataset/mp16/pretrain_test_1000
OLM=$BASE/olmoearth_dataset
PY=/root/dev/.venv/bin/python
export PYTHONPATH=$WT
cd "$WT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "===== meta summaries (S2 + S1) ====="
for mod in sentinel2_l2a sentinel1; do
  for ts in two_week year; do
    log "meta: $mod $ts"
    $PY -m olmoearth_pretrain.dataset_creation.make_meta_summary \
      --olmoearth_path "$OLM" --modality $mod --time_span $ts
  done
done

log "===== h5 conversion (S2 + S1) ====="
$PY -m olmoearth_pretrain.internal.run_h5_conversion \
  --tile_path="$OLM" \
  --supported_modality_names='[sentinel1,sentinel2_l2a]' \
  --compression=zstd --compression_opts=3 --tile_size=128

log "===== DONE ====="
echo "h5 files: $(find "$OLM" -name '*.h5' 2>/dev/null | wc -l)"
find "$OLM" -name "*.h5" | head -3
