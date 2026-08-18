"""WIDE-AND-SHALLOW BOTTLENECK: 6-layer trunk, 8 blocks at 768, output 128.

The read/latent stack runs at ``register_dim=768`` and a single in-graph
``Linear(768, 128)`` projects its output, so the decoder, the supervision heads and the
frozen evals all still consume a 128-wide grid. Shared K/V, budget 3072, micro 64.

WHY THIS SHAPE. The first early-read sweep measured wall-clock against the cand_ndvi
frontier (0.66 s/step) and found MACs to be the wrong proxy entirely:

    arm                    MACs    s/step
    dp0 (12 trunk + 4)     1.00x     0.66     <- the default
    e6_l8  (6 + 8)         0.51x     0.44
    e3_l12 (3 + 12)        0.28x     0.79
    e6_l16 (6 + 16)        0.64x     1.12
    e3_l35 (3 + 35)        0.30x     1.27

Time tracks the number of sequential bottleneck blocks, not FLOPs: each block does
arithmetically trivial work on an ``n_h*n_w x 128`` grid, so the GPU is launch-bound, and
deep stacks lose on the clock even as they win on MACs. Depth is therefore the expensive
axis and width is the cheap one -- at an unchanged kernel count, wider blocks are close to
free if the launch-bound reading is right.

That argues for spending on width and economizing on depth, which is what these arms do.
The quality case for width is already on record: the d768 register runs beat every d128
arm by ~6 mIoU on the frozen ps=1 PASTIS probes (0.6447 vs 0.5800 at ws16 S1+S2, Tessera
v2 large 0.5938), and ``regbtl_v1_2_proj_common`` concluded the narrow width's NATIVE
training was the bottleneck rather than the 128-dim budget.

WHAT THIS COSTS. At 768 a block's projections are 6x the FLOPs and its MLP 36x. This is
NOT a speedup and should not be read as one -- the hypothesis is FLAT s/step, bought with
a much more capable bottleneck. The measurement is self-interpreting: land near 8-block
d128 timings and the stack is launch-bound (width is free, and that generalizes well past
this sweep); land far slower and it is bandwidth-bound, which would also retire the
widening idea.

At 768 the ``attn_dim`` decoupling is a no-op, so these are plain tied-width blocks -- the
most hardware-friendly shape available here.

OUTPUT PROJECTION vs THE DISTILLATION STUDENT: ``register_projection_dims`` trains a
DETACHED student alongside a d768 primary. This is different -- one linear map in the
gradient path, trained by the pretext loss like the rest of the model, with 128 as the
model's actual output rather than a side artifact.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_earlyread_common import (
    build_earlyread_model_config,
    set_earlyread_loop_evals,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

TRUNK_DEPTH = 6
LATENT_DEPTH = 8
REGISTER_DIM = 768
OUTPUT_DIM = 128
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_earlyread_e6_l8_r768o128_ndvi_tanchor_newsampling_psuniform_sharedkv.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """6-layer trunk + 8 blocks at 768, projected to 128 on output."""
    return build_earlyread_model_config(
        common,
        trunk_depth=TRUNK_DEPTH,
        latent_depth=LATENT_DEPTH,
        shared_read_kv=True,
        register_dim=REGISTER_DIM,
        output_dim=OUTPUT_DIM,
    )


def build_trainer_config(common: CommonComponents):
    """Base trainer + ONLY the S1+S2+Landsat embedding evals (pastis/ethiopia/descals)."""
    return set_earlyread_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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
