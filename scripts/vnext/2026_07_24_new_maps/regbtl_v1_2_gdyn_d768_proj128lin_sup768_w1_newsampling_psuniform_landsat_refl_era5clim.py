"""proj128lin_sup768 (detached 128/64 student) + a POOLED ERA5 climate-prediction task.

Twin of ``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl``
-- same d768 teacher, detached ``Linear(768, 128)`` student + 64 Matryoshka prefix, w1
register supervision, distillation, and proj/ps=1 PASTIS evals at 768 / 128 / 64 -- with
ONE addition:

* ``era5_10`` is added as a SUPERVISION-ONLY target (loaded + normalised, but never
  encoded OR decoded by the ViT) AND given a register supervision head, so the model
  must PREDICT the per-scene climate signature from the d768 registers. ERA5 is
  non-spatial (a 1x1, 12-month, 6-var pixel), so the head runs through the supervision
  head's non-spatial path: it MEAN-POOLS the register grid ``[B, n_h, n_w, D] ->
  [B, D]`` and regresses the signature. The signature is the full 12-month TRAJECTORY
  (``temporal_reduction="flatten"`` -> a flat 72-dim ``[12 months x 6 vars]`` vector),
  i.e. the intra-year PATTERN / seasonality -- the axis that actually separates
  climate zones. ERA5 is kept out of ``supported_modality_names`` (encoder + decoder)
  and out of the token budget / temporal subsetting
  (``extra_budget_exclude_modalities``) so its full T=12 stack reaches the flatten head
  (T=12 -> 72) without the ViT temporal encodings, which assume modality-T ==
  len(timestamps), tripping over the longer ERA5 sequence. See the wideread era5clim
  twin for the full rationale.

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
from dataclasses import replace

from base import build_common_components as _base_build_common_components
from base import build_trainer_config as _base_build_trainer_config
from base import build_visualize_config
from olmo_core.train.common import Duration
from perceiver_common import route_loop_evals_to_beaker
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl import (
    LANDSAT_REFL_NORM_CONFIG,
    PROJ_EVAL_INTERVAL_STEPS,
    PROJECTION_DIMS,
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
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
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

# Offline K=16 ERA5 climate-zone labels (KMeans over the 72-dim signature), built by
# ``scripts/tools/era5_climate_zone_eval.py build-zones`` over ERA5_H5_DIR. Keep K in
# sync with the ``era5_climate_zone`` EvalDatasetConfig.num_classes (=16).
CLIMATE_ZONE_ZONES_NPZ = (
    "/weka/dfive-default/yawenz/eval_sets/era5_climate_zones/era5_zones_k16.npz"
)
# Imagery inputs for the climate-zone probe. ERA5 is deliberately EXCLUDED: the probe
# asks whether the IMAGERY embedding (not a copied ERA5 input) separates climate zones.
CLIMATE_ZONE_INPUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
]


def build_climate_zone_eval_task() -> DownstreamTaskConfig:
    """Pooled linear-probe of the K=16 ERA5 climate zones from the d768 registers.

    In-loop counterpart to ``era5_climate_zone_eval.py score``'s linear-probe accuracy /
    macro-F1: mean-pool the scene embedding and fit a linear classifier over the offline
    KMeans zones. Regressed on the SAME interval as the projected catalog so all three
    widths land in one eval job.
    """
    return DownstreamTaskConfig(
        dataset="era5_climate_zone",
        embedding_batch_size=16,
        probe_batch_size=256,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        eval_interval=Duration.steps(PROJ_EVAL_INTERVAL_STEPS),
        input_modalities=CLIMATE_ZONE_INPUT_MODALITIES,
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.01,
        primary_metric=EvalMetric.MACRO_F1,
        h5py_dir=ERA5_H5_DIR,
        climate_zone_npz_path=CLIMATE_ZONE_ZONES_NPZ,
        pretrain_train_samples=4096,
        pretrain_valid_samples=1024,
        pretrain_test_samples=1024,
    )


def add_climate_zone_eval_tasks(trainer_config):
    """Add the ERA5 climate-zone probe on d768 + both student widths (proj128 / 64).

    The base task reads the d768 registers; each ``_proj{dim}`` twin reads the DETACHED
    student via ``eval_on_projected_registers`` so one eval job scores climate-zone
    separability at 768 / 128 / 64 and tracks whether distillation carries the teacher's
    climate-awareness down to the shipped widths. Placed FIRST for the same preemption
    reason as the other projected catalogs (tail tasks lose their metrics first).
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    base = build_climate_zone_eval_task()
    proj = {
        f"era5_climate_zone_proj{dim}": replace(
            base, eval_on_projected_registers=True, eval_projection_dim=dim
        )
        for dim in PROJECTION_DIMS
    }
    evaluator.tasks = {**proj, "era5_climate_zone": base, **evaluator.tasks}
    return trainer_config


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


def _mark_era5_supervision_only(config: LatentMIMConfig) -> None:
    """Drop ``era5_10`` from the encoder + decoder ``supported_modality_names`` in place.

    ERA5 stays in ``training_modalities`` (loaded/normalised/collated as
    ``batch.era5_10``) but is removed from both ViT modality lists, so it is never
    patch-embedded, encoded, or decoded -- it exists purely as the register-supervision
    target. This is what lets ERA5 keep its full T=12 stack (see
    ``extra_budget_exclude_modalities``) without the ViT's temporal encodings, which
    assume modality-T == len(timestamps), tripping over the longer ERA5 sequence.
    Copies each list first so no shared base list is mutated.
    """
    era5 = Modality.ERA5_10.name
    for sub_config in (config.encoder_config, config.decoder_config):
        sub_config.supported_modality_names = [
            m for m in sub_config.supported_modality_names if m != era5
        ]


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
    # Keep ERA5 out of the ViT entirely (supervision-only); it is read from the batch
    # by the register-supervision head, not encoded/decoded.
    _mark_era5_supervision_only(config)
    return config


def build_dataloader_config(common: CommonComponents):
    """proj128lin dataloader; ERA5 kept full-length & out of the budget."""
    config = _proj_build_dataloader_config(common)
    # Supervision-only ERA5: exclude from the token budget AND from temporal subsetting
    # so its 12-month climate signature reaches the flatten head intact (T=12 -> 72).
    config.extra_budget_exclude_modalities = [Modality.ERA5_10.name]
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """proj128lin train module (ERA5 is supervision-only)."""
    return _proj_build_train_module_config(common)


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
        add_climate_zone_eval_tasks(
            add_pastis_ps1_eval_tasks(
                add_projected_eval_tasks(_base_build_trainer_config(common))
            )
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
