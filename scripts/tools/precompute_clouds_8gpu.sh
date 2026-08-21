#!/usr/bin/env bash
# Precompute OmniCloudMask cloud-class maps for the OSM-sampling pretrain set across the
# 8 GPUs in this Beaker job (one sharded process per GPU). Idempotent + resumable: rerun
# (or launch more jobs) to finish/continue. Reads code from the weka-mounted repo and OCM
# weights pre-staged on weka, so it needs no git clone and no HuggingFace download.
#
# Env knobs (all optional):
#   NUM_SHARDS  total shards across ALL jobs (default 8 = a single 8-GPU job covers all)
#   SHARD_BASE  first shard index this job owns (default 0; use 8,16,... for extra jobs)
#   GPUS        GPUs (= processes) in THIS job (default 8)
#   H5_DIR      h5 build to compute clouds FOR (default: cloud_mask_cache.DEFAULT_H5_DIR,
#               the old cdl_gse_... map set). The cache is written to the sibling
#               cloud_masks_omnicloudmask path of whatever this points at, which is also
#               where default_cache_dir() will look for it at training time. A cache is
#               NOT transferable between h5 builds: sidecars are keyed by raw h5 sample
#               id, and the builds are not index-aligned (measured 2026-08-21: 0/150
#               probed indices agreed between the old map set and the new-maps
#               reflectance build, see scripts/tools/check_h5_sample_alignment.py). Set
#               this whenever training on a build that has no cache of its own.
set -euo pipefail

REPO=/weka/dfive-default/yawenz/olmoearth_pretrain
WEIGHTS=/weka/dfive-default/helios/dataset/osm_sampling/cloud_masks_omnicloudmask/_ocm_weights
NUM_SHARDS=${NUM_SHARDS:-8}
SHARD_BASE=${SHARD_BASE:-0}
GPUS=${GPUS:-8}

python -m pip install -q omnicloudmask h5py hdf5plugin einops numpy || true
cd "$REPO"

echo "launching $GPUS shard processes: NUM_SHARDS=$NUM_SHARDS SHARD_BASE=$SHARD_BASE"
echo "h5_dir: ${H5_DIR:-<cloud_mask_cache.DEFAULT_H5_DIR>}"
pids=()
for g in $(seq 0 $((GPUS - 1))); do
  shard=$((SHARD_BASE + g))
  CUDA_VISIBLE_DEVICES="$g" OCM_MODEL_DIR="$WEIGHTS" PYTHONPATH="$REPO" \
    python -m olmoearth_pretrain.data.cloud_mask_cache \
      ${H5_DIR:+--h5_dir "$H5_DIR"} \
      --num_shards "$NUM_SHARDS" --shard "$shard" &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL SHARDS DONE (rc=$rc)"
exit $rc
