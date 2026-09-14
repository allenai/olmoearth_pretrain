"""d768 teacher + detached linear 128d student (newmaps, w1) on LANDSAT REFLECTANCE h5.

Same experiment slot as ``regbtl_v1_2_gdyn_d128_wideread_regsup_w1_newsampling_
psuniform_landsat_refl`` -- same new-maps recipe, same w1 register supervision, same
reflectance h5 and reflectance-scale Landsat norm stats -- with ONE change: the 128
is a PROJECTION, not the bottleneck.

* the d128 twin stores the register grid at 128 dims and trains that width NATIVELY
  under the pretext loss;
* this run keeps the bottleneck at ``register_dim=768`` (the teacher) and hangs a
  DETACHED per-cell ``Linear(768, 128)`` student off it, trained only by the
  distillation terms (cosine through a learned back-projection + Gram), with its
  first 64 dims a self-sufficient Matryoshka prefix -- so one checkpoint ships 768 /
  128 / 64.

This is the ``proj128lin_sup768`` variant of the official v1.2 detached-student
program (see ``scripts/official/v1_2/regbtl_v1_2_proj_common.py`` for the full
motivation and variant matrix), ported onto the new maps + Landsat reflectance. Its
premise, from the 2026-07-30 embedding-evals synthesis: d768 registers beat every
native-d128 arm by ~6 mIoU on the frozen ps=1 PASTIS probes while matching them on
center-pixel classification, i.e. training the narrow width natively under the
pretext loss is the bottleneck, not the 128-dim budget -- so distill the narrow
width from a wide teacher instead.

``sup768``: the supervision heads stay on the d768 registers only, so the encoder
and the primary bottleneck train EXACTLY as the d768 w1 arm does (the student's
input is detached). That makes this run encoder-identical to
``regbtl_v1_2_gdyn_d768_wideread_regsup_w1_newsampling_psuniform_landsat_refl`` --
its d768 evals double as a sanity anchor against that run, and the two 128d arms
(native vs distilled) are what this script exists to compare.

The projection knobs are set here rather than in ``perceiver_common`` because they
are what this script changes; nothing else in the folder trains a student.
"""

import logging
from dataclasses import replace

from base import build_common_components, build_visualize_config
from base import build_trainer_config as _base_build_trainer_config
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

from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# Teacher (primary bottleneck) width: full encoder width, no compression.
REGISTER_DIM = 768
# Shipped student widths. 128 is the product width; 64 is trained as a self-sufficient
# Matryoshka PREFIX of it (own back-projection and Gram term), so a single stored
# artifact serves both by truncation.
PROJECTION_DIMS = [128, 64]
# ``lin``: per-cell Linear(768, 128) on the DETACHED registers -- is the teacher's
# information linearly readable per cell at the low width? (The ``perceiver`` student,
# a second wideread bottleneck at d128, is the other arm of that question and is not
# run here.)
PROJECTION_TYPE = "linear"
# ``sup768``: supervision heads on the d768 registers only. The student trains purely
# by distillation, which keeps this run encoder-identical to the d768 w1 arm.
SUPERVISION_SOURCE = "registers"
# The w1 arm, matching the d128 twin this is read against: 10x the perceiver_common
# default (0.1). Set here because supervision strength is held fixed on purpose.
SUPERVISION_BASE_WEIGHT = 1.0
MODULE_PATH = (
    "scripts/vnext/2026_07_24_new_maps/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_landsat_refl.py"
)

# New-maps h5 whose Landsat modality is TOA reflectance / brightness temperature.
LANDSAT_REFL_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_glo30_landsat_meta_canopy_height_openstreetmap_raster_"
    "sentinel1_sentinel2_l2a_worldcereal_worldcover/1138828"
)
# Reflectance-scale Landsat norm stats (computed.json with only the landsat entry
# replaced); a resource under olmoearth_pretrain/data/norm_configs.
LANDSAT_REFL_NORM_CONFIG = "computed_landsat_reflectance.json"

# Base eval tasks duplicated onto the student head. The three dense-segmentation
# probes are where the native-d128 arms lost to d768, so they are the readout this
# experiment is judged on; the pooled classification tasks (m-eurosat, so2sat,
# yemen_crop) are not duplicated -- d128 already matched d768 there, and every
# duplicate costs eval-job runtime. A name missing from the base catalog raises
# KeyError at import rather than silently dropping a task.
PROJ_EVAL_TASK_NAMES = (
    "pastis",
    "fifty_cities_sentinel2",
    "fifty_cities_sentinel1_sentinel2",
)
# 40k, not the base catalog's 20k, on EVERY task in this run. The duplicates make
# this a 14-task eval job, and 14-task jobs are exactly the ones that outran a 20k
# window in the official proj runs: consecutive jobs then overlap while sharing one
# resumed W&B run, and the second writer's rows are silently dropped (observed as
# missing projected metrics). Must stay a multiple of the checkpointer's
# save_interval (5000).
PROJ_EVAL_INTERVAL_STEPS = 40000

# The frozen per-pixel (patch_size=1, window_size=16) PASTIS embedding probes -- the
# deployment-scenario readout (frozen ps=1 embeddings probed for segmentation, the way
# Tessera/AEF are compared) and the metric on which d768 beat every native-d128 arm.
# The new-maps base catalog only carries the POOLED ``pastis`` linear probe, so these
# ps=1 exports are added explicitly from the canonical registry. They read the
# ``pastis_rslearn`` pretrain-mirror export, which must be materialized where the eval
# Beaker job can read it. A name missing from the registry raises KeyError at import.
PASTIS_PS1_TASK_NAMES = (
    "pastis_ws16_ps1_sentinel2_pretrain_export",
    "pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export",
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 wideread regbtl + w1 register supervision + a detached [128, 64] student."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config = add_register_supervision(config, base_weight=SUPERVISION_BASE_WEIGHT)
    config.encoder_config.register_projection_dims = list(PROJECTION_DIMS)
    config.encoder_config.register_projection_type = PROJECTION_TYPE
    config.supervision_source = SUPERVISION_SOURCE
    return config


def build_dataloader_config(common: CommonComponents):
    """Single-view newsampling dataloader with patch-size sampling forced to uniform."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(build_1fwd_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW + ddp/bf16 train module at the newsampling microbatch size.

    The distillation terms need no configuration: ``LatentMIMTrainModuleConfig``
    defaults to cosine weight 1.0 + flat Gram 1.0, which fire as soon as the encoder
    carries a projection student. The student sits in the encoder's param group and
    so inherits its LR schedule (the flat-LR variant is a separate official arm).
    """
    return apply_microbatch(build_faster_train_module_config(common))


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """New-maps dataset on the Landsat-reflectance h5 + matching reflectance norm."""
    return OlmoEarthDatasetConfig(
        h5py_dir=LANDSAT_REFL_H5_DIR,
        training_modalities=common.training_modalities,
        computed_norm_config=LANDSAT_REFL_NORM_CONFIG,
    )


def add_projected_eval_tasks(trainer_config):
    """Duplicate the segmentation probes onto the student head, at both widths.

    Each ``_proj{d}`` task is its base twin with ``eval_on_projected_registers``, so
    one eval job scores the same checkpoint at 768 (base tasks) / 128 / 64 and the
    distillation quality is tracked per checkpoint without a separate launch. The
    duplicates are placed FIRST: eval jobs at urgent priority can be preempted
    mid-run and it is the tail tasks that systematically lose their metrics, so the
    student -- the product this run exists to produce -- runs before the catalog.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    base_tasks = {
        name: replace(task, eval_interval=Duration.steps(PROJ_EVAL_INTERVAL_STEPS))
        for name, task in evaluator.tasks.items()
    }
    proj_tasks = {
        f"{name}_proj{dim}": replace(
            base_tasks[name],
            eval_on_projected_registers=True,
            eval_projection_dim=dim,
        )
        for name in PROJ_EVAL_TASK_NAMES
        for dim in PROJECTION_DIMS
    }
    evaluator.tasks = {**proj_tasks, **base_tasks}
    return trainer_config


def add_pastis_ps1_eval_tasks(trainer_config):
    """Add the frozen ps=1 (ws16) PASTIS embedding probes, on d768 + both student widths.

    The base new-maps catalog only has the POOLED ``pastis`` linear probe; these tasks
    are the per-pixel (patch_size=1) exports this program is judged on. Each base task is
    duplicated onto the detached student at every Matryoshka width via
    ``eval_on_projected_registers`` / ``eval_projection_dim``, so one eval job scores the
    same checkpoint at 768 / 128 / 64. Placed FIRST for the same reason as the projected
    catalog duplicates: eval jobs at urgent priority can be preempted mid-run and it is
    the tail tasks that systematically lose their metrics.
    """
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    base_ps1 = {
        name: replace(
            EMBEDDING_EVAL_TASKS[name],
            eval_interval=Duration.steps(PROJ_EVAL_INTERVAL_STEPS),
        )
        for name in PASTIS_PS1_TASK_NAMES
    }
    proj_ps1 = {
        f"{name}_proj{dim}": replace(
            task, eval_on_projected_registers=True, eval_projection_dim=dim
        )
        for name, task in base_ps1.items()
        for dim in PROJECTION_DIMS
    }
    evaluator.tasks = {**proj_ps1, **base_ps1, **evaluator.tasks}
    return trainer_config


def build_trainer_config(common: CommonComponents):
    """New-maps base trainer + student-head evals + ps=1 PASTIS, all routed through Beaker."""
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
