"""v1.2 register bottleneck: gdyn + il + pdproj, no instance contrastive, latent self-attn ON, single forward pass.

Identical to ``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa`` in every architectural and
optimization respect, but runs a SINGLE forward pass per batch instead of two.

The ``noic`` arm sets ``contrastive_config=None``, so the InfoNCE instance contrastive
loss is 0. Yet the base ``ContrastiveLatentMIMTrainModule`` still runs two forward passes
per batch (on masked views ``a`` and ``b``) and averages the JEPA loss over them -- the
second forward pass only ever existed to feed the (now-absent) contrastive loss. With the
contrastive loss gone that second pass is pure wasted compute.

This script swaps in the plain :class:`LatentMIMTrainModule` (one forward pass, one masked
view) and drops the dataloader to ``num_masked_views=1`` so it yields the single-view batch
tuple that module expects. All model, loss, masking, optimizer, scheduler, EMA, and trainer
settings are copied verbatim from the base builders so the only difference from
``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa`` is forward-passes-per-batch.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
"""

import logging

from base import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from base import (
    build_dataloader_config as _base_build_dataloader_config,
)
from base import (
    build_train_module_config as _base_build_train_module_config,
)
from base import (
    build_trainer_config as _base_build_trainer_config,
)
from regbtl_v1_2_common import add_loop_eval_beaker_job, build_regbtl_model_config

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Gdyn + il + pdproj register bottleneck, latent self-attention ON."""
    return build_regbtl_model_config(common, latent_self_attn=True)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """Plain single-forward-pass train module, mirroring the base (contrastive) config.

    The base builder returns a ``ContrastiveLatentMIMTrainModuleConfig`` that runs two
    forward passes per batch. Since the ``noic`` arm zeroes the contrastive loss, we copy
    its fields into a :class:`LatentMIMTrainModuleConfig`, which runs a single forward pass.
    Copying (rather than hard-coding) keeps this in lockstep with ``base.py``; only the
    contrastive-specific fields (``contrastive_config``, ``reinit_targets``) are dropped.
    """
    base = _base_build_train_module_config(common)
    return LatentMIMTrainModuleConfig(
        optim_config=base.optim_config,
        rank_microbatch_size=base.rank_microbatch_size,
        transform_config=base.transform_config,
        masking_config=base.masking_config,
        loss_config=base.loss_config,
        mae_loss_config=base.mae_loss_config,
        token_exit_cfg=base.token_exit_cfg,
        max_grad_norm=base.max_grad_norm,
        scheduler=base.scheduler,
        ema_decay=base.ema_decay,
        dp_config=base.dp_config,
        regularizer_config=base.regularizer_config,
        autocast_precision=base.autocast_precision,
        compile_model=base.compile_model,
        compile_loss=base.compile_loss,
        find_unused_parameters=base.find_unused_parameters,
        state_dict_save_opts=base.state_dict_save_opts,
        state_dict_load_opts=base.state_dict_load_opts,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Base dataloader, but a single masked view so it yields the single-view batch tuple.

    The plain :class:`LatentMIMTrainModule` expects ``(patch_size, MaskedOlmoEarthSample)``;
    the base config's ``num_masked_views=2`` would instead yield the ``(a, b)`` two-view
    tuple used by the contrastive module.
    """
    config = _base_build_dataloader_config(common)
    config.num_masked_views = 1
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer config + fifty_cities evals routed through a Beaker job."""
    return add_loop_eval_beaker_job(_base_build_trainer_config(common), MODULE_PATH)


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
