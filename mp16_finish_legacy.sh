#!/usr/bin/env bash
# Re-convert S2+S1 using legacy 12x _moNN layers (matches the configs we materialized),
# then meta summaries + H5. Reads already-materialized rslearn data (no re-download).
set -uo pipefail
WT=/root/repos/oe_pretrain_corpus_v2
BASE=/weka/dfive-default/olmoearth_pretrain/dataset/mp16/pretrain_test_1000
DS=$BASE/rslearn_dataset
OLM=$BASE/olmoearth_dataset
PY=/root/dev/.venv/bin/python
W=32
export PYTHONPATH=$WT
cd "$WT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

for mod in sentinel2_l2a sentinel1; do
  log "===== convert $mod (legacy monthly) ====="
  $PY -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.$mod \
    --ds_path "$DS" --olmoearth_path "$OLM" --workers $W --legacy-monthly
done

log "===== meta summaries ====="
for mod in sentinel2_l2a sentinel1; do
  for ts in two_week year; do
    log "meta: $mod $ts"
    $PY -m olmoearth_pretrain.dataset_creation.make_meta_summary \
      --olmoearth_path "$OLM" --modality $mod --time_span $ts
  done
done

log "===== H5 conversion ====="
$PY -m olmoearth_pretrain.internal.run_h5_conversion \
  --tile_path="$OLM" \
  --supported_modality_names='[sentinel1,sentinel2_l2a]' \
  --compression=zstd --compression_opts=3 --tile_size=128

log "===== DONE ====="
echo "olmoearth dirs:"; ls "$OLM"
echo "h5 files: $(find "$OLM" -name '*.h5' 2>/dev/null | wc -l)"
find "$OLM" -name "*.h5" | head -3
