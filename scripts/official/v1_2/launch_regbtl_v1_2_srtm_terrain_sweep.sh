#!/bin/bash
# v1.2 register-bottleneck SRTM-TERRAIN test, at the best-known supervision weight (w0p1).
#
# As of the terrain refactor, the `srtm` modality is a single 4-band terrain modality
# [elevation, slope, aspect_sin, aspect_cos]: only elevation is stored on disk, slope and
# aspect are derived from it at load time (see compute_srtm_bands). Every run using srtm
# now carries slope+aspect automatically -- there is no separate flag. So this sweep is
# just the winning w0p1 configs re-launched on the terrain code; the no-terrain baseline
# is the existing elevation-only w0p1 runs (regbtl_v1_2_gdyn_*_regsup_w0p1) in the SAME
# project, launched from launch_regbtl_v1_2_fullweight_sweep.sh before the refactor.
#
# w0p1 was the best supervision weight at both d128 and d768 in the 2026_07_02_perceiver
# sweep (mean eval 0.556 / 0.576, reached by 380k steps), which is why terrain is tested
# only at w0p1 rather than re-sweeping the weight axis. Trains at the base.py default 300
# epochs, matching the elevation-only w0p1 runs so the comparison is at a fixed horizon.
#
# COMPARISON NOTE: the terrain runs share srtm's h5 data with the baselines; the only
# difference is the code version (1-band vs 4-band srtm). Compare terrain-vs-baseline at
# MATCHED checkpoint steps -- do not compare a fresh terrain run against a further-along
# baseline.
set -e

PROJECT="2026_07_02_perceiver"
CLUSTER="ai2/jupiter"
LAUNCH_ARGS="--launch.num_gpus=8 --launch.priority=urgent --launch.clusters=[ai2/jupiter,ai2/ceres]"

# The winning w0p1 configs. srtm is 4-band terrain on the current code, so no config
# changes are needed -- only distinct run names (suffix _srtmterrain) vs the baselines.
D768_REGSUP_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_w0p1.py"
D768_REGSUP_LATLON_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup_latlon_w0p1.py"
D128_REGSUP_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_w0p1.py"
D128_REGSUP_LATLON_W0P1="scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_latlon_w0p1.py"

python "$D768_REGSUP_W0P1" launch "regbtl_v1_2_gdyn_d768_regsup_w0p1_srtmterrain_v2" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D768_REGSUP_LATLON_W0P1" launch "regbtl_v1_2_gdyn_d768_regsup_latlon_w0p1_srtmterrain_v2" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP_W0P1" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_srtmterrain_v2" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"

python "$D128_REGSUP_LATLON_W0P1" launch "regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1_srtmterrain_v2" "$CLUSTER" \
    $LAUNCH_ARGS \
    --trainer.callbacks.wandb.project="$PROJECT"
