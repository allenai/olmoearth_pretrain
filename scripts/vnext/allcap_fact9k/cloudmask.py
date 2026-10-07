"""allcap_fact9k fork: cloudy S2 tokens are not decoder targets.

Everything is ``base.py`` except that the dataset also reads the S2 scene
classification, and the collator turns S2 decoder targets whose patch is mostly
cloud (SCL cloud shadow / cloud / cirrus) into MISSING, so the loss never asks
the model to predict a cloud. Cloudy S2 tokens stay encoder inputs; S1 and
Landsat targets are unchanged (the corpus has no Landsat QA band).

The fork started from base_allcap_fact9k_1's step 75000 checkpoint (model,
optimizer, and data-loader position), so from there both runs see the same
batches and masks apart from the dropped targets. joer 2026-10-06. That
checkpoint no longer exists: olmo-core's CheckpointerCallback keeps only the
last ``max_checkpoints`` (default 3) permanent checkpoints of a job.

Launch (8 GPUs):
    PYTHONPATH=. python scripts/vnext/allcap_fact9k/cloudmask.py launch \
        base_allcap_fact9k_cloudmask_1 ai2/jupiter --launch.num_gpus=8
"""

import importlib.util
from pathlib import Path

from upath import UPath

from olmoearth_pretrain.internal.experiment import CommonComponents, main

# base.py imports the v1.2 script as module "base", so load it under another name.
_spec = importlib.util.spec_from_file_location(
    "allcap_fact9k_base", Path(__file__).with_name("base.py")
)
assert _spec is not None and _spec.loader is not None
allcap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(allcap)

FORK_FROM = (
    "/weka/dfive-default/olmoearth_pretrain/checkpoints/joer/"
    "base_allcap_fact9k_1/step75000"
)


def build_dataset_config(common: CommonComponents):
    """base.py's dataset plus per-pixel S2 cloud flags."""
    config = allcap.build_dataset_config(common)
    config.load_s2_cloud_mask = True
    return config


def build_trainer_config(common: CommonComponents):
    """base.py's trainer, initialized from the baseline's step 75000 checkpoint.

    Only a first start (no checkpoint in this run's save folder yet) gets the load
    path; restarts resume from this run's own checkpoints. The load path must not
    stay set: the train module reads ``<load_path>/config.json`` on every
    checkpoint save and load, so it crashed once the baseline's step 75000 was
    deleted.
    """
    config = allcap.build_trainer_config(common)
    save_folder = UPath(common.save_folder)
    if not (save_folder.exists() and any(save_folder.glob("step*"))):
        config.load_path = FORK_FROM
    return config


if __name__ == "__main__":
    main(
        common_components_builder=allcap.build_common_components,
        model_config_builder=allcap.build_model_config,
        train_module_config_builder=allcap.build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=allcap.build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=allcap.v1_2_base.build_visualize_config,
    )
