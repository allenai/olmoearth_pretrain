"""d128 wideread + regsup + NDVI (w0p1, newsampling, UNIFORM ps, tanchor) + GPS encoding.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform`` plus
a GPS TOKEN ENCODING: the sample's (lat, lon) is mapped to unit-sphere (x, y, z) and
passed through a learned 2-layer MLP into the additive encoding slot that absolute
spatial encodings would occupy (idle under the v1.2 RoPE modes), broadcast to every
token. Everything else -- sampler, anchor, NDVI arm, supervision weight -- is
unchanged, so the A/B partner is the psuniform ndvi tanchor run itself.

GPS reaches the encoder via the regsup_latlon ride-along pattern: latlon is appended
to the dataset's training modalities and the masking strategy's only_decode
modalities, so it rides along in the batch without ever being tokenized (and, under
newsampling, without consuming token budget). Unlike the regsup_latlon arm, latlon
here is an INPUT, not a supervision target (``include_latlon=False``).

Half of the samples have their (x, y, z) zeroed at train time
(``gps_dropout_rate=0.5``, the ``EncoderConfig`` default) BEFORE the MLP, so the
"no GPS" input gets its own learned embedding -- eval datasets carry no latlon and
take exactly that path, as do samples whose stored latlon is MISSING_VALUE.
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
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_gps.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup incl. time-conditioned NDVI, anchored read, GPS encoding."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    config.encoder_config.use_gps_encoding = True
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
