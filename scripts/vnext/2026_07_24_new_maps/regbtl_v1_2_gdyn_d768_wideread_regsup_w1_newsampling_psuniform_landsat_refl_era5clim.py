"""d768 wideread + regsup w1 (newmaps) + a POOLED ERA5 climate-prediction task.

Twin of ``regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl``
(same d768 bottleneck, 1fwd + fused AdamW, newsampling/psuniform, reflectance norm,
six decode-only map supervision at ``base_weight=1.0``) with ONE addition:

* ``era5_10`` is added as a DECODE-ONLY target (never encoded) AND given a register
  supervision head, so the model must PREDICT the per-scene climate signature from the
  registers. Because ERA5 is non-spatial (a 1x1, 12-month, 6-var pixel), the head runs
  through the supervision head's non-spatial path: it MEAN-POOLS the register grid
  ``[B, n_h, n_w, D] -> [B, D]`` and regresses the climate signature. The signature is
  the full 12-month TRAJECTORY (``temporal_reduction="flatten"`` -> a flat 72-dim
  ``[12 months x 6 vars]`` vector), i.e. the intra-year PATTERN / seasonality, not just
  the annual level -- seasonality is the axis that actually separates climate zones
  (Mediterranean vs monsoon vs continental), so it is the more useful climate target.

Why pooled, not per-cell: predicting a scene-level target from EVERY register cell
(the default spatial-map path) would force all 2.5 km cells toward one climate vector
-- exactly the homogenisation risk that could cost the dense PASTIS ps=1 probes. The
non-spatial path pools first, so the gradient reaches each cell only through the mean:
it nudges the AGGREGATE representation to be climate-aware (which is what a climate-zone
clustering eval reads) while leaving per-cell spatial detail intact. See
``scripts/tools/era5_climate_zone_eval.py`` for the climate-zone comparison harness and
``PASTIS ps=1`` as the guardrail metric.

Knobs this arm exists to explore:
* ``ERA5_SUPERVISION_WEIGHT`` -- the pull of the climate task. Kept LOW by default
  (0.1) because ERA5 is coarse/low-entropy: too strong and the registers collapse
  toward climate-zone-level content. Sweep this to trade climate-awareness against
  spatial detail.

``era5_10`` is regressed in NORMALISED space (the norm config carries an era5 entry),
so the wildly different band scales (surface-pressure ~9.3e4 vs precip ~3e-3) are
comparable, and the loss is L1 (precip is heavy-tailed). The 72-dim target is the
12-month x 6-var climate normal in calendar order, so no observation-timestamp
alignment is needed (month index == calendar month); the head just reproduces the
scene's annual weather shape. Stored ERA5 is a complete 12-month stack (the converter
drops the modality on any NaN), so ``flatten`` always sees T=12.
"""

import logging

from base import build_common_components as _base_build_common_components
from base import build_trainer_config as _base_build_trainer_config
from base import build_visualize_config
from perceiver_common import (
    add_register_supervision,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
    build_1fwd_dataloader_config,
    build_faster_train_module_config,
    build_wideread_regbtl_model_config,
    route_loop_evals_to_beaker,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
SUPERVISION_BASE_WEIGHT = 1.0
# Independent weight for the ERA5 climate task (NOT scaled by the map task-type
# weights). Low on purpose -- the homogenisation risk scales with this knob. Sweep.
ERA5_SUPERVISION_WEIGHT = 0.1
# ERA5 band count (2m-temp, 2m-dewpoint, sfc-pressure, u-wind, v-wind, precip) and the
# monthly stack depth. The non-spatial head regresses the flattened 12x6 trajectory,
# so its output width is ERA5_NUM_MONTHS * ERA5_NUM_BANDS = 72.
ERA5_NUM_BANDS = 6
ERA5_NUM_MONTHS = 12
ERA5_SIGNATURE_DIM = ERA5_NUM_MONTHS * ERA5_NUM_BANDS
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl_era5clim.py"
)

# ERA5-inclusive new-maps h5 (see the era5in arm for the build note). Confirm the
# trailing sample-count segment against run_h5_conversion's final ``h5py_dir``.
ERA5_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_era5_10_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)
LANDSAT_REFL_NORM_CONFIG = "computed_landsat_reflectance.json"


def build_common_components(
    script: str, cmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """New-maps common components with ``era5_10`` appended to ``training_modalities``.

    Being in ``training_modalities`` makes ERA5 loadable/normalised; the masking
    override below (``only_decode_modalities``) is what keeps it OUT of the encoder,
    so the registers must infer climate from the imagery rather than copy an input.
    """
    common = _base_build_common_components(script, cmd, run_name, cluster, overrides)
    if Modality.ERA5_10.name not in common.training_modalities:
        common.training_modalities = [
            *common.training_modalities,
            Modality.ERA5_10.name,
        ]
    return common


def _mark_era5_decode_only(strategy_config: dict) -> None:
    """Append ``era5_10`` to a masking strategy's ``only_decode_modalities`` in place.

    Copies the list first so the shared base ``ONLY_DECODE_MODALITIES`` constant is
    never mutated (other arms in the same process rely on it).
    """
    key = "only_decode_modalities"
    current = list(strategy_config.get(key, []))
    if Modality.ERA5_10.name not in current:
        strategy_config[key] = [*current, Modality.ERA5_10.name]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 wideread + map supervision at w1 + a pooled ERA5 climate-prediction head."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config = add_register_supervision(config, base_weight=SUPERVISION_BASE_WEIGHT)
    # ERA5 is non-spatial -> the supervision head routes it through the mean-pooled
    # register path automatically. ``temporal_reduction="flatten"`` maps the
    # [B, 12, 6] target to the per-scene [B, 72] climate-normal signature that the
    # pooled [B, 72] head emits, so the head predicts the full monthly PATTERN.
    config.supervision_head_config.modality_configs[Modality.ERA5_10.name] = (
        SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=ERA5_SIGNATURE_DIM,
            weight=ERA5_SUPERVISION_WEIGHT,
            regression_loss_type="l1",
            temporal_reduction="flatten",
        )
    )
    return config


def build_dataloader_config(common: CommonComponents):
    """Single-view newsampling dataloader; ERA5 forced decode-only in the mask."""
    config = apply_uniform_patch_sizes(
        apply_new_sampling(build_1fwd_dataloader_config(common))
    )
    _mark_era5_decode_only(config.masking_config.strategy_config)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW + ddp/bf16 train module; ERA5 forced decode-only in the mask."""
    config = apply_microbatch(build_faster_train_module_config(common))
    _mark_era5_decode_only(config.masking_config.strategy_config)
    # Keep the patch-discrimination negatives consistent with the maps: decode-only
    # modalities are excluded from the negative set.
    loss_cfg = config.loss_config.loss_config
    if "mask_negatives_for_modalities" in loss_cfg:
        current = list(loss_cfg["mask_negatives_for_modalities"])
        if Modality.ERA5_10.name not in current:
            loss_cfg["mask_negatives_for_modalities"] = [
                *current,
                Modality.ERA5_10.name,
            ]
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """New-maps dataset on the ERA5-inclusive reflectance h5 + matching norm."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ERA5_H5_DIR,
        training_modalities=common.training_modalities,
        computed_norm_config=LANDSAT_REFL_NORM_CONFIG,
    )


def build_trainer_config(common: CommonComponents):
    """New-maps base trainer, with in-loop evals routed through Beaker jobs."""
    return route_loop_evals_to_beaker(_base_build_trainer_config(common), MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
