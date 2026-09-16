"""proj128lin_sup768 (detached 128/64 student) + a POOLED ERA5 climate-prediction task.

Twin of ``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl``
-- same d768 teacher, detached ``Linear(768, 128)`` student + 64 Matryoshka prefix, w1
register supervision, distillation, and proj/ps=1 PASTIS evals at 768 / 128 / 64 -- with
ONE addition:

* ``era5_10`` is added as a DECODE-ONLY target (never encoded) AND given a register
  supervision head, so the model must PREDICT the per-scene climate signature from the
  d768 registers. ERA5 is non-spatial (a 1x1, 12-month, 6-var pixel), so the head runs
  through the supervision head's non-spatial path: it MEAN-POOLS the register grid
  ``[B, n_h, n_w, D] -> [B, D]`` and regresses the signature. The signature is the full
  12-month TRAJECTORY (``temporal_reduction="flatten"`` -> a flat 72-dim
  ``[12 months x 6 vars]`` vector), i.e. the intra-year PATTERN / seasonality -- the
  axis that actually separates climate zones.

Because the supervision heads stay on the d768 registers (``sup768``), the ERA5 head
also reads d768; the DETACHED student is trained purely by distillation and only
inherits climate-awareness through the teacher it distills. Pooling before predicting
keeps the per-cell spatial detail (the ps=1 PASTIS readout) intact: the gradient reaches
each cell only through the mean, nudging the AGGREGATE representation toward climate.
``ERA5_SUPERVISION_WEIGHT`` (0.1) is the knob for that pull; sweep it against the ps=1
probes. Regressed in NORMALISED space (the norm config carries an era5 entry) with L1
(precip is heavy-tailed). See ``scripts/tools/era5_climate_zone_eval.py`` for the
climate-zone comparison harness.
"""

import logging

from base import build_common_components as _base_build_common_components
from base import build_trainer_config as _base_build_trainer_config
from base import build_visualize_config
from perceiver_common import route_loop_evals_to_beaker
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl import (
    LANDSAT_REFL_NORM_CONFIG,
    add_pastis_ps1_eval_tasks,
    add_projected_eval_tasks,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl import (
    build_dataloader_config as _proj_build_dataloader_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl import (
    build_model_config as _proj_build_model_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl import (
    build_train_module_config as _proj_build_train_module_config,
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

# Independent weight for the ERA5 climate task (NOT scaled by the map task-type
# weights). Low on purpose -- the homogenisation risk scales with this knob. Sweep.
ERA5_SUPERVISION_WEIGHT = 0.1
# ERA5 band count and monthly stack depth. The non-spatial head regresses the flattened
# 12x6 trajectory, so its output width is ERA5_NUM_MONTHS * ERA5_NUM_BANDS = 72.
ERA5_NUM_BANDS = 6
ERA5_NUM_MONTHS = 12
ERA5_SIGNATURE_DIM = ERA5_NUM_MONTHS * ERA5_NUM_BANDS

# Must point at THIS script so the Beaker eval jobs rebuild the matching model
# (era5_10 supervision head + decode-only masking).
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl_era5clim.py"
)

# ERA5-inclusive new-maps reflectance h5 (era5_10 added to run_h5_conversion's modality
# list). era5_10 sorts right after cdl in the auto-derived folder name; the trailing
# sample-count segment is unchanged at 1138828 (confirmed against the final h5py_dir).
ERA5_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_era5_10_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)


def build_common_components(
    script: str, cmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """proj128lin common components with ``era5_10`` appended to ``training_modalities``.

    Being in ``training_modalities`` makes ERA5 loadable/normalised; the masking
    override below (``only_decode_modalities``) keeps it OUT of the encoder, so the
    registers must infer climate from the imagery rather than copy an input.
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
    """proj128lin model (d768 teacher + detached 128/64 student) + a pooled ERA5 head."""
    config = _proj_build_model_config(common)
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
    """proj128lin dataloader; ERA5 forced decode-only in the mask."""
    config = _proj_build_dataloader_config(common)
    _mark_era5_decode_only(config.masking_config.strategy_config)
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """proj128lin train module; ERA5 forced decode-only in the mask + negatives."""
    config = _proj_build_train_module_config(common)
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
    """proj128lin dataset on the ERA5-inclusive reflectance h5 + matching norm."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ERA5_H5_DIR,
        training_modalities=common.training_modalities,
        computed_norm_config=LANDSAT_REFL_NORM_CONFIG,
    )


def build_trainer_config(common: CommonComponents):
    """proj128lin trainer: student-head evals + ps=1 PASTIS, routed through Beaker.

    Re-derived here (rather than imported from the anchor) so ``MODULE_PATH`` points at
    this ERA5 script; otherwise the eval jobs would rebuild the non-ERA5 anchor model.
    """
    return route_loop_evals_to_beaker(
        add_pastis_ps1_eval_tasks(
            add_projected_eval_tasks(_base_build_trainer_config(common))
        ),
        MODULE_PATH,
    )


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
