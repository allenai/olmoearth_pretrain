#!/bin/bash
# Maps ablation experiments: Removing one decode-only modality at a time
# This script ablates each of the 6 decode-only modalities to understand their individual contribution
#
# Full training modalities: sentinel2_l2a, sentinel1, landsat, worldcover, srtm, openstreetmap_raster, wri_canopy_height_map, cdl, worldcereal
# Decode-only modalities: worldcover, srtm, openstreetmap_raster, wri_canopy_height_map, cdl, worldcereal

# Cluster settings
CLUSTER="ai2/ceres-cirrascale"
CLUSTERS="[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]"
PRIORITY="high"
WANDB_PROJECT="2026_01_27_maps_ablation"
NUM_GPUS=8

# Ablation 1: Remove WORLDCOVER
python scripts/official/base.py launch ablation_no_worldcover $CLUSTER \
    --common.dataloader_side_masking=True \
    --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,srtm,openstreetmap_raster,wri_canopy_height_map,cdl,worldcereal]' \
    --data_loader.masking_config.strategy_config.only_decode_modalities='[srtm,openstreetmap_raster,wri_canopy_height_map,cdl,worldcereal]' \
    --launch.clusters="$CLUSTERS" \
    --launch.priority=$PRIORITY \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Ablation 2: Remove SRTM
python scripts/official/base.py launch ablation_no_srtm $CLUSTER \
    --common.dataloader_side_masking=True \
    --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,worldcover,openstreetmap_raster,wri_canopy_height_map,cdl,worldcereal]' \
    --data_loader.masking_config.strategy_config.only_decode_modalities='[worldcover,openstreetmap_raster,wri_canopy_height_map,cdl,worldcereal]' \
    --launch.clusters="$CLUSTERS" \
    --launch.priority=$PRIORITY \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Ablation 3: Remove OPENSTREETMAP_RASTER
python scripts/official/base.py launch ablation_no_openstreetmap_raster $CLUSTER \
    --common.dataloader_side_masking=True \
    --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,worldcover,srtm,wri_canopy_height_map,cdl,worldcereal]' \
    --data_loader.masking_config.strategy_config.only_decode_modalities='[worldcover,srtm,wri_canopy_height_map,cdl,worldcereal]' \
    --launch.clusters="$CLUSTERS" \
    --launch.priority=$PRIORITY \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Ablation 4: Remove WRI_CANOPY_HEIGHT_MAP
python scripts/official/base.py launch ablation_no_wri_canopy_height_map $CLUSTER \
    --common.dataloader_side_masking=True \
    --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,worldcover,srtm,openstreetmap_raster,cdl,worldcereal]' \
    --data_loader.masking_config.strategy_config.only_decode_modalities='[worldcover,srtm,openstreetmap_raster,cdl,worldcereal]' \
    --launch.clusters="$CLUSTERS" \
    --launch.priority=$PRIORITY \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Ablation 5: Remove CDL
python scripts/official/base.py launch ablation_no_cdl $CLUSTER \
    --common.dataloader_side_masking=True \
    --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,worldcover,srtm,openstreetmap_raster,wri_canopy_height_map,worldcereal]' \
    --data_loader.masking_config.strategy_config.only_decode_modalities='[worldcover,srtm,openstreetmap_raster,wri_canopy_height_map,worldcereal]' \
    --launch.clusters="$CLUSTERS" \
    --launch.priority=$PRIORITY \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT

# Ablation 6: Remove WORLDCEREAL
python scripts/official/base.py launch ablation_no_worldcereal $CLUSTER \
    --common.dataloader_side_masking=True \
    --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,worldcover,srtm,openstreetmap_raster,wri_canopy_height_map,cdl]' \
    --data_loader.masking_config.strategy_config.only_decode_modalities='[worldcover,srtm,openstreetmap_raster,wri_canopy_height_map,cdl]' \
    --launch.clusters="$CLUSTERS" \
    --launch.priority=$PRIORITY \
    --launch.num_gpus=$NUM_GPUS \
    --trainer.callbacks.wandb.project=$WANDB_PROJECT
