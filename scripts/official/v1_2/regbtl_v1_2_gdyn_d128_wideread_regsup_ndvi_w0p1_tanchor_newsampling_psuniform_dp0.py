"""The d128 NDVI tanchor frontier with stochastic depth OFF (``drop_path=0.0``).

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform`` with one
line changed. Its A/B partner is that run directly -- same eval set
(``add_loop_eval_beaker_job``), so the existing curves are the control and nothing needs
re-running.

WHY: ``drop_path=0.1`` has been in every run of this program since v1.2 base without ever
being measured, and three things suggest it is doing nothing here.

* It is a supervised-ViT default (DeiT/timm), where it guards against overfitting a deep
  stack on limited labels. That is not this regime: SSL on a global corpus for 667k steps.
  MAE uses drop_path=0 for ViT-B pretraining and only introduces it at fine-tuning.
* It is not even following the recipe it came from. ``Encoder`` passes a FLAT rate to
  every block (``flexi_vit.py``, the ``self.blocks`` construction) rather than the
  depth-linear ramp ``linspace(0, drop_path, depth)`` that DeiT/timm use.
* The pretext already carries substantial stochasticity: the mask itself,
  ``band_dropout_rate=0.2`` on S2 and Landsat, patch-size sampling, and the decorrelated
  sampler.

SCOPE: ``drop_path`` reaches ONLY the encoder trunk blocks. ``SpatialRegisterBottleneck``
builds its read and latent blocks without passing it (so they get the ``Block`` default of
0.0 -> ``nn.Identity()``), and ``PredictorConfig.drop_path`` already defaults to 0.0 and is
never overridden in ``base.py``. So this run turns off the model's only stochastic depth,
and at 12 trunk blocks it is the largest such effect the program has.

HOW TO READ IT: a NULL is the expected and useful outcome -- it retires a hyperparameter
and removes a (small) training cost. Frame this as "can we delete this knob", not "does it
help". Against the measured eval noise floor (LP 0.88 pts mean, 5.71 max) a small
regularization effect is not detectable in one seed anyway; only a clear regression would
be informative in the other direction.

RELEVANCE TO THE EARLY-READ ARMS: they inherit this same trunk-only scoping, so
``..._earlyread_e3_l35_...`` regularizes 3 blocks and leaves 35 bottleneck blocks with no
stochastic depth at all. If this ablation shows drop_path matters, that asymmetry is worth
revisiting there; if it is a null, the early-read arms need no adjustment.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_model_config as _base_build_model_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

DROP_PATH = 0.0
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_dp0.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The frontier model config with the encoder trunk's stochastic depth disabled."""
    config = _base_build_model_config(common)
    config.encoder_config.drop_path = DROP_PATH
    # The decoder is already at 0.0 by default; assert so this run cannot silently become
    # a two-variable change if base.py starts setting it.
    assert config.decoder_config.drop_path == 0.0, (
        "decoder drop_path is expected to be 0.0; this ablation targets the encoder only"
    )
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer + the frontier's own eval set, so its existing curves are the control."""
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
