#!/bin/bash
# Run eval sweep for percentile-based quantization at different bit levels.
#
# This script runs evaluations with 8-bit, 4-bit, 2-bit, and 1-bit quantization
# using precomputed percentile boundaries.
#
# Prerequisites:
#   1. Compute quantiles first (run from repo root):
#      python scripts/tools/compute_embedding_quantiles.py \
#          --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
#          --h5py_dir=/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828 \
#          --output=scripts/archived/2026-01-28_percentile_quant_experiment/quantiles.h5 \
#          --num_samples=10000 \
#          --batch_size=32
#
# Usage:
#   ./scripts/archived/2026-01-28_percentile_quant_experiment/run_percentile_quant_sweep.sh

set -e

PROJECT="percentile-quant-sweep"
CHECKPOINT=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200
MODULE_PATH=scripts/official/base.py
QUANTILE_CONFIG=scripts/archived/2026-01-28_percentile_quant_experiment/quantiles.h5

# Run baseline without quantization first
echo "Running baseline (no quantization)..."
python -m olmoearth_pretrain.internal.full_eval_sweep \
    --cluster=ai2/saturn \
    --checkpoint_path=$CHECKPOINT \
    --module_path=$MODULE_PATH \
    --trainer.max_duration.value=700000 \
    --trainer.max_duration.unit=steps \
    --trainer.callbacks.wandb.project="$PROJECT" \
    --task-skip-names=pastis128_sentinel2,pastis128_sentinel1,pastis128_sentinel1_sentinel2 \
    --select_best_val

# Run with each bit level
for BITS in 8 4 2 1; do
    echo "Running ${BITS}-bit quantization..."
    python -m olmoearth_pretrain.internal.full_eval_sweep \
        --cluster=ai2/saturn \
        --checkpoint_path=$CHECKPOINT \
        --module_path=$MODULE_PATH \
        --trainer.max_duration.value=700000 \
        --trainer.max_duration.unit=steps \
        --trainer.callbacks.wandb.project="$PROJECT" \
        --task-skip-names=pastis128_sentinel2,pastis128_sentinel1,pastis128_sentinel1_sentinel2 \
        --select_best_val \
        --quantize_embeddings \
        --quantize_bits=$BITS \
        --quantile_config_path=$QUANTILE_CONFIG
done

echo "Done! Check results in wandb project: $PROJECT"
