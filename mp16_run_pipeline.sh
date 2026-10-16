#!/usr/bin/env bash
# MP-16 1000-point pretraining dataset: materialize S2+S1+Landsat -> convert -> meta -> h5.
set -uo pipefail
WT=/root/repos/oe_pretrain_corpus_v2
BASE=/weka/dfive-default/olmoearth_pretrain/dataset/mp16/pretrain_test_1000
DS=$BASE/rslearn_dataset
OLM=$BASE/olmoearth_dataset
CFG=$WT/data/rslearn_dataset_configs
PY=/root/dev/.venv/bin/python
RSLEARN=/root/dev/.venv/bin/rslearn
W=32
export PYTHONPATH=$WT
cd "$WT"
mkdir -p "$OLM"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

for mod in sentinel2_l2a sentinel1 landsat; do
  log "===== $mod : prepare ====="
  cp "$CFG/config_$mod.json" "$DS/config.json"
  $RSLEARN dataset prepare --root "$DS" --group res_10 --workers $W \
    --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 30 --ignore-errors
  log "===== $mod : materialize ====="
  $RSLEARN dataset materialize --root "$DS" --group res_10 --workers $W \
    --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 30 --ignore-errors
  log "===== $mod : convert to olmoearth ====="
  $PY -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.$mod \
    --ds_path "$DS" --olmoearth_path "$OLM" --workers $W
done

log "===== meta summaries ====="
for mod in sentinel2_l2a sentinel1 landsat; do
  for ts in two_week year; do
    $PY -m olmoearth_pretrain.dataset_creation.make_meta_summary \
      --olmoearth_path "$OLM" --modality $mod --time_span $ts
  done
done

log "===== h5 conversion ====="
$PY -m olmoearth_pretrain.internal.run_h5_conversion \
  --tile_path="$OLM" \
  --supported_modality_names='[landsat,sentinel1,sentinel2_l2a]' \
  --compression=zstd --compression_opts=3 --tile_size=128

log "===== PIPELINE DONE ====="
find "$OLM" -name "*.h5" | head -3
echo "h5 count: $(find "$OLM" -name '*.h5' 2>/dev/null | wc -l)"
