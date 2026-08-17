"""d128 wideread + regsup + NDVI (psuniform, tanchor) + GPS + simple temporal encoding.

The COMBINED metadata-conditioning arm: both single-change arms stacked on
``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``.

* GPS TOKEN ENCODING (as in ``..._psuniform_gps``): the sample's (lat, lon) as
  unit-sphere (x, y, z) through a learned 2-layer MLP into the RoPE-idle spatial
  encoding slot, broadcast to every token; latlon rides along decode-only
  (input-only, no supervision head); per-sample GPS dropout 0.5 zeroes the xyz
  BEFORE the MLP so the no-GPS eval path is trained. rslearn evals attach
  window-center latlon, so GPS is used at eval where available.
* SIMPLE TEMPORAL ENCODING (as in ``..._psuniform_simpletemporal``): the frozen
  month table replaced by a learned 2-layer MLP of [frac_year (years since 2020),
  sin/cos annual phase, year_valid] in the same slot on BOTH encoder and decoder;
  per-sample year dropout 0.5 (frac_year + year_valid zeroed, phase kept).

The two encodings occupy different quarter-slots ([3n:4n] and [2n:3n]) and are
otherwise independent; sampler, anchor, NDVI arm, supervision weight, and 3D RoPE
are unchanged. A/B partners: the base psuniform ndvi tanchor run and the two
single-change arms, giving the full 2x2 over {GPS, simple-temporal}.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_regsup_common import (
    add_register_supervision,
    build_extra_decode_dataloader_config,
    build_extra_decode_dataset_config,
    build_extra_decode_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
REGISTER_TEMPORAL_ANCHOR = "year_start"
# NDVI feeds the time-conditioned supervision arm; latlon feeds the GPS token
# encoding (input-only -- no supervision head, include_latlon stays False).
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name, Modality.LATLON.name]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform"
    "_gps_simpletemporal.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup + NDVI, anchored read, GPS + simple temporal encodings."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    config.encoder_config.use_gps_encoding = True
    # Simple temporal replacement on both sides (year dropout keeps the 0.5 default).
    config.encoder_config.use_simple_temporal_encoding = True
    config.decoder_config.use_simple_temporal_encoding = True
    return add_register_supervision(
        config,
        include_latlon=False,
        include_ndvi=True,
        base_weight=SUPERVISION_BASE_WEIGHT,
    )


def build_dataset_config(common: CommonComponents):
    """Base dataset config, additionally deriving ndvi from the raw S2 bands."""
    return build_extra_decode_dataset_config(common, EXTRA_DECODE_MODALITIES)


def build_dataloader_config(common: CommonComponents):
    """ndvi-aware newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(
            build_extra_decode_dataloader_config(common, EXTRA_DECODE_MODALITIES)
        )
    )


def build_train_module_config(common: CommonComponents):
    """ndvi-aware faster train module with a halved rank microbatch size."""
    return apply_microbatch(
        build_extra_decode_train_module_config(common, EXTRA_DECODE_MODALITIES)
    )


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
