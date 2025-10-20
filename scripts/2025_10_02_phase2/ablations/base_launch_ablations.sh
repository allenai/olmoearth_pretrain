# no contrastive loss ablation
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_no_contrastive ai2/ceres-cirrascale  --train_module.contrastive_config.loss_config.weight=0.0 --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=high --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# random masking
python scripts/2025_10_02_phase2/ablations/base_random_masking.py launch phase2.0_base_random_masking ai2/ceres-cirrascale --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=high --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# MAE
python scripts/2025_10_02_phase2/ablations/base_mae.py launch phase2.0_base_mae ai2/ceres-cirrascale  --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=normal --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# random init the target projections
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_random_target ai2/ceres-cirrascale  --train_module.reinit_targets=True --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=normal --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# original patch disc loss
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_patchdisc_loss ai2/ceres-cirrascale  --train_module.loss_config.loss_config.type="patch_discrimination_new" --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=normal --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
###### Modality abations below ######
# I have successively removed modalities here, so that
# each run has fewer modalities than a previous one
# no ag maps (removing worldcereal and CDL)
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_no_ag_maps ai2/ceres-cirrascale  --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=high --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,worldcover,srtm,openstreetmap_raster,wri_canopy_height_map]' --train_module.masking_config.strategy_config.only_decode_modalities='[worldcover,srtm,openstreetmap_raster,wri_canopy_height_map]' --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# no maps (removing ag maps + worldcover, openstreetmap, canopy height)
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_no_maps ai2/ceres-cirrascale  --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=high --common.training_modalities='[sentinel2_l2a,sentinel1,landsat,srtm]' --train_module.masking_config.strategy_config.only_decode_modalities='[srtm]' --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# no decode modalities (removing maps + srtm)
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_no_maps_srtm ai2/ceres-cirrascale  --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=normal --common.training_modalities='[sentinel2_l2a,sentinel1,landsat]' --train_module.masking_config.strategy_config.only_decode_modalities='[]' --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# no landsat
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_no_maps_srtm_landsat ai2/ceres-cirrascale  --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=normal --common.training_modalities='[sentinel2_l2a,sentinel1]' --train_module.masking_config.strategy_config.only_decode_modalities='[]' --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
# s2 only (no s1)
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_no_maps_srtm_landsat_s1_v3 ai2/ceres-cirrascale  --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=urgent --common.training_modalities='[sentinel2_l2a]' --train_module.masking_config.strategy_config.only_decode_modalities='[]' --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
### extra ablations ###
# ema active again
python scripts/2025_10_02_phase2/base.py launch phase2.0_base_ema ai2/ceres-cirrascale --launch.clusters='[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]' --launch.priority=normal --train_module.ema_decay='[0.996,1.0]' --trainer.callbacks.wandb.project=2025_10_08_phase2_ablations
