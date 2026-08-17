"""d128 wideread + regsup + NDVI (w0p1, newsampling, UNIFORM ps, tanchor) + simple temporal.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform`` with
the additive temporal encoding REPLACED: instead of the frozen sinusoidal month table,
each timestep's minimal 4-number signal -- [frac_year (years since 2020), sin/cos of
the annual phase, year_valid] (``get_simple_temporal_encoding``, ported from the
trope_simple_temporal line) -- is passed through a learned 2-layer MLP into the same
encoding slot, on both the encoder and the decoder (mirroring how the original
trope_simple_temporal run applied it to both). Everything else -- sampler, anchor,
NDVI arm, supervision weight, 3D RoPE's relative-time coordinate -- is unchanged, so
the A/B partner is the psuniform ndvi tanchor run itself.

WHY: the month table carries only the annual phase; the model has no absolute-year
signal anywhere (RoPE time is relative). The frac_year channel adds one, and the MLP
can shape the seasonal encoding instead of using fixed sinusoids. Year dropout at 0.5
(per sample, model-side: frac_year + year_valid zeroed, annual phase KEPT) trains the
no-trustworthy-year path that eval datasets with synthesized dates exercise.
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
EXTRA_DECODE_MODALITIES = [Modality.NDVI.name]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform"
    "_simpletemporal.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup + NDVI, anchored read, simple temporal encoding."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    # Drop-in temporal-encoding replacement on both sides (year dropout keeps
    # the EncoderConfig/PredictorConfig default of 0.5).
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
