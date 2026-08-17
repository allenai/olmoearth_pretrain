"""cand_ndvi + per-modality encoder trunk layers (QKV/output/MLP routed by modality).

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``
(cand_ndvi) with ``per_modality_layers=True`` on the encoder: every trunk block gives
each supported modality its own fused QKV, attention output projection and MLP,
routed by per-token modality IDs. The register bottleneck (whose latents have no
modality) and the decoder keep shared parameters, so the perceiver read/LSA stack and
the 2D-decoder path are byte-identical to cand_ndvi. Everything else -- sampler,
anchor, NDVI arm, token budget, LR schedule, epochs(300), loss set and in-loop eval
catalog -- matches cand_ndvi knob for knob, so the comparison is single-variable.

WHY: a re-test of the 2026-06 per-modality-capacity experiment
(favyen/20260608-per-modality-layers), which was null on the pre-perceiver stack. The
architecture has since changed substantially (register bottleneck, 3D mixed RoPE,
newsampling); per-modality capacity in the trunk may land differently when the trunk's
job is to feed a shared spatial register grid rather than to be the embedding itself.
A/B partner: cand_ndvi
(``..._w0p1_tanchor_newsamp_psuniform``, same everything, shared trunk params).

NOTE: with 10 supported modalities at d768 x depth 12 this adds roughly 850M params
(mostly per-modality MLPs). Empty routes are still executed every forward so FSDP
collectives stay in sync, but expect a noticeable step-time and optimizer-memory cost;
if the launch OOMs, halve ``rank_microbatch_size`` again via a CLI override.
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
    "_permod.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """cand_ndvi model with per-modality encoder trunk layers."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    # The single variable under test: per-modality QKV/output/MLP in the encoder
    # trunk, routed by token modality. Bottleneck and decoder stay shared.
    config.encoder_config.per_modality_layers = True
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
