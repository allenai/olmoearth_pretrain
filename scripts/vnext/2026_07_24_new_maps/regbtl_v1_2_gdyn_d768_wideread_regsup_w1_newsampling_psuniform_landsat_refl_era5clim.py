"""d768 wideread + regsup w1 (newmaps) + a POOLED ERA5 climate-prediction task.

Twin of ``regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl``
(same d768 bottleneck, 1fwd + fused AdamW, newsampling/psuniform, reflectance norm,
six decode-only map supervision at ``base_weight=1.0``) with ONE addition:

* ``era5_10`` is added as a SUPERVISION-ONLY target (loaded + normalised, but never
  encoded OR decoded by the ViT) AND given a register supervision head, so the model
  must PREDICT the per-scene climate signature from the registers. Because ERA5 is
  non-spatial (a 1x1, 12-month, 6-var pixel), the head runs through the supervision
  head's non-spatial path: it MEAN-POOLS the register grid ``[B, n_h, n_w, D] ->
  [B, D]`` and regresses the climate signature. The signature is the full 12-month
  TRAJECTORY (``temporal_reduction="flatten"`` -> a flat 72-dim ``[12 months x 6 vars]``
  vector), i.e. the intra-year PATTERN / seasonality, not just the annual level --
  seasonality is the axis that actually separates climate zones (Mediterranean vs
  monsoon vs continental), so it is the more useful climate target.

Why supervision-only rather than a ViT decode-only modality: the flatten head needs the
FULL 12-month stack as its target, but the newsampling shape sampler temporally
subsamples every encoded/decoded modality to a shared ``max_t`` (often < 12), and the
ViT's temporal encodings (month embedding + 3D RoPE) assume each modality's timestep
count matches the shared ``timestamps`` window. Routing ERA5 through the ViT at T=12
therefore either shrinks the target (flatten emits T*6 < 72, a shape mismatch against
the 72-wide head) or crashes the temporal encodings. Since the register-supervision
head reads ``batch.era5_10`` and the pooled registers directly -- it never touches the
ERA5 decoder tokens or ``timestamps`` -- ERA5 does not need to be a ViT modality at all.
So it is kept out of ``supported_modality_names`` (encoder + decoder) and out of the
token budget / temporal subsetting (``extra_budget_exclude_modalities``), which keeps
its full T=12 stack. ``build_common_components`` still lists it in
``training_modalities`` so it is loaded, normalised and collated as ``batch.era5_10``.

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
drops the modality on any NaN), and because ERA5 is exempt from temporal subsetting
(``extra_budget_exclude_modalities``) the supervision target always sees T=12 -> 72.
"""

import logging

from base import build_common_components as _base_build_common_components
from base import build_trainer_config as _base_build_trainer_config
from base import build_visualize_config
from olmo_core.train.common import Duration
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
# Multiple of the checkpointer save_interval (5000); this teacher-only arm has no
# projected student, so the climate-zone probe is a single d768 task.
CLIMATE_ZONE_EVAL_INTERVAL_STEPS = 40000


def build_climate_zone_eval_task() -> DownstreamTaskConfig:
    """Pooled linear-probe of the K=16 ERA5 climate zones from the d768 registers.

    In-loop counterpart to ``era5_climate_zone_eval.py score``'s linear-probe accuracy /
    macro-F1: mean-pool the scene embedding and fit a linear classifier over the offline
    KMeans zones. This wideread arm ships only the d768 teacher, so there is a single
    task (no projected student widths).
    """
    return DownstreamTaskConfig(
        dataset="era5_climate_zone",
        embedding_batch_size=16,
        probe_batch_size=256,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        eval_interval=Duration.steps(CLIMATE_ZONE_EVAL_INTERVAL_STEPS),
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
    """Add the ERA5 climate-zone probe (single d768 task) to the evaluator, placed FIRST.

    Placed FIRST because eval jobs at urgent priority can be preempted mid-run and it is
    the tail tasks that systematically lose their metrics.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks = {
        "era5_climate_zone": build_climate_zone_eval_task(),
        **evaluator.tasks,
    }
    return trainer_config


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
    # Keep ERA5 out of the ViT entirely (supervision-only); it is read from the batch
    # by the register-supervision head, not encoded/decoded.
    _mark_era5_supervision_only(config)
    return config


def build_dataloader_config(common: CommonComponents):
    """Single-view newsampling dataloader; ERA5 kept full-length & out of the budget."""
    config = apply_uniform_patch_sizes(
        apply_new_sampling(build_1fwd_dataloader_config(common))
    )
    # Supervision-only ERA5: exclude from the token budget AND from temporal subsetting
    # so its 12-month climate signature reaches the flatten head intact (T=12 -> 72).
    config.extra_budget_exclude_modalities = [Modality.ERA5_10.name]
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW + ddp/bf16 train module (ERA5 is supervision-only)."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """New-maps dataset on the ERA5-inclusive reflectance h5 + matching norm."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ERA5_H5_DIR,
        training_modalities=common.training_modalities,
        computed_norm_config=LANDSAT_REFL_NORM_CONFIG,
    )


def build_trainer_config(common: CommonComponents):
    """New-maps base trainer + ERA5 climate-zone probe, evals routed through Beaker."""
    return route_loop_evals_to_beaker(
        add_climate_zone_eval_tasks(_base_build_trainer_config(common)),
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
