"""Gram OFF, 2-layer MLP back-projection head (arm: mlpgram0).

One cell of the 2x2 {Gram weight} x {back-projection head} matrix on top of
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm``, which IS the
(gram=1, linear head) cell and is not re-run. The other three cells are this
module and its two siblings (``_gram0``, ``_mlpgram1``, ``_mlpgram0``).
Everything else -- teacher, sampler, supervision, data, in-loop evals -- is
byte-identical to that base, so the four runs are a clean factorial.

THE TWO AXES.

* **Gram weight** (``projection_distill_gram_weight``, 1.0 -> 0.0). Gram is the
  only distillation term that touches the SERVED embedding directly: it is an MSE
  between the student's and the teacher's token-token cosine-similarity matrices,
  taken on the raw prefix. Cosine, by contrast, constrains the student only
  *through* the back-projection head. Note TESSERA v2 DOES NOT USE GRAM -- its
  distillation loss is per-prefix cosine alone -- so gram=0 is the arm that
  actually matches the recipe this family was built to follow. Sixteen
  gram-*scope* arms were previously a null (all inside the 0.88 pt LP noise
  floor), but gram *presence* has never been tested.

* **Back-projection head** (``register_back_projection_hidden``, None -> 256).
  A single ``Linear(d, 768)`` demands the student be a linear image of the
  teacher, which is close to demanding PCA of a 768->128 compression. SimReg's
  ablation of exactly this module put a deeper head 3.7 pts (1-NN) / 10.2 pts
  (linear probe) ahead of the bare Linear, ~94% of it from the first hidden layer.
  The head is discarded at inference, so this is free at serving time and the
  shipped d128 architecture is unchanged.

WHY A FACTORIAL AND NOT THE CORNER. The two axes are expected to INTERACT, in the
direction that makes the corner cell the risky one rather than the best one. With
a bare Linear head, cosine already pins the student's geometry fairly tightly, so
Gram is largely redundant -- consistent with the scope sweep's null. Give the head
a hidden layer and it absorbs more of the discrepancy, which loosens cosine's grip
on the raw prefix and leaves Gram as the only term holding the served embedding's
geometry. So the prediction under test is that Gram becomes load-bearing exactly
when it never was before, i.e. (mlp, gram=1) > (mlp, gram=0) while
(linear, gram=0) == (linear, gram=1). Running only the corner could not tell a
head-depth win from a Gram loss.

READOUT. ``_proj128`` / ``_proj64`` in-loop probes are the outcome;
``projection/distill_cosine_d{128,64}`` is the diagnostic. A head that overfits
shows as the cosine loss falling while the probes do not move. Watch the
d64/d128 cosine ratio too: it ran ~1.5x and widening on the parent, and prefix
terms are summed UNWEIGHTED, so a deeper head that fits the narrow prefix better
also silently re-weights the Matryoshka objective.

THIS CELL: gram=0, 2-layer MLP head at H=256. Both changes -- the hypothesised best cell, and the one where nothing but the head constrains the served embedding.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm import (
    build_dataloader_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm import (
    build_model_config as _base_build_model_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm import (
    build_train_module_config as _base_build_train_module_config,
)
from regbtl_v1_2_proj_common import (
    BACK_PROJECTION_HIDDEN,
    set_proj_aeftrial_loop_evals,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm_mlpgram0.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The stunorm base's model config, with this cell's head architecture."""
    config = _base_build_model_config(common)
    config.encoder_config.register_back_projection_hidden = BACK_PROJECTION_HIDDEN
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The stunorm base's train module, with this cell's Gram weight."""
    config = _base_build_train_module_config(common)
    config.projection_distill_gram_weight = 0.0
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer + AEF trials and PASTIS on both student widths."""
    return set_proj_aeftrial_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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
