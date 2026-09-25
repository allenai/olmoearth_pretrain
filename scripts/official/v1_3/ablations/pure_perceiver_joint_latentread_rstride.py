"""Joint latentread trained at RANDOM latent strides, from one latent per pixel to one per patch.

The latent STRIDE (pixels per latent along each side) is drawn per training forward pass
(``random_latent_stride=True``): uniformly among the divisors of the batch's patch size
whose latent count fits a budget of 512 latents per sample, with the patch stride
always allowed. Stride 1 is one latent per pixel, stride ``patch_size`` one per patch,
so one run trains every output resolution in between, and large grids get a coarser
stride instead of being excluded (no pixel-side cap).

Shapes are the NORMAL v1.3 pretraining ones: patch embed base 8, patch sizes 1..8,
grids up to 32 patches, token budget 3,072. (A first version at patch sizes 1..4 on
Favyen's pixel-register shapes was stopped at step ~1.3k to move here; at 2,048 latents
and microbatch 32 it peaked at 55 GiB with no OOM.) Under this sampler about 23% of
batches get sub-patch per-pixel latents, concentrated at small patch sizes: the budget
rules out stride 1 for 71% of patch-size-8 batches.

Evaluation and inference use stride 1 (``eval_latent_stride``): per-pixel embeddings.
The supervision heads keep the default unfold of the maximum patch size with the
existing bilinear resize, which fits any stride. Evals: the student at ps1 plus d128
at ps4, as in ``pure_perceiver_joint_latentread_pixlat.py``.

Budget and microbatch: 512 latents per sample at v1.3's microbatch of 64 (one pass per
step). Earlier versions ran at 2,048 latents / microbatch 32 as
``v1_3_vit0_rstride_ps8_latentread_joint12`` (stopped at step 5.6k; 1.5x slower than
the patch-latent arms) and at patch sizes 1..4 before that.

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_vit0_rstride_ps8_lb512_latentread_joint12``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import SAMPLED_HW_P_LIST as V1_3_SAMPLED_HW_P_LIST  # noqa: E402
from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_train_module_config,  # noqa: E402
    build_visualize_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_dataloader_config as _pixlat_dataloader_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_model_config as _pixlat_model_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_trainer_config as _pixlat_trainer_config,
)

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.joint_latent import JointLatentConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_3/ablations/pure_perceiver_joint_latentread_rstride.py"
)

# Per-sample latent budget. 4,096 OOMed at microbatch 32; 2,048 fit at 32 (55-68 GiB peak)
# but two passes per step made it ~1.5x slower than the patch-latent arms (profiled:
# launch-bound idle GPU, not FLOPs). 512 keeps the per-sample ceiling at the patch
# stride's 1,024 latents -- the patch-latent latentread arm's, which fits at 64 -- so
# this arm runs at v1.3's rank microbatch of 64 (``base.build_train_module_config``).
MAX_LATENTS = 512
# The sampler's default tile extent, i.e. no pixel-side cap beyond the stored tiles.
TILE_SIZE = OlmoEarthDataLoaderConfig.__dataclass_fields__["tile_size"].default
# Normal v1.3 pretraining shapes: patch embed base 8 (also the dataloader's max patch
# size) and grids up to 32 patches.
MAX_PATCH_SIZE = 8
SAMPLED_HW_P_LIST = list(V1_3_SAMPLED_HW_P_LIST)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Pixel-latent latentread with random strides under a latent budget."""
    config = _pixlat_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig) and perceiver.pixel_latents
    perceiver.random_latent_stride = True
    perceiver.max_latents = MAX_LATENTS
    perceiver.eval_latent_stride = 1
    config.encoder_config.max_patch_size = MAX_PATCH_SIZE
    assert config.supervision_head_config is not None
    # Default unfold (max_patch_size) + bilinear resize fits every stride.
    config.supervision_head_config.spatial_unfold = None
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.3 pretraining shapes (patch sizes 1..8, grids <= 32), no pixel-side cap."""
    config = _pixlat_dataloader_config(common)
    config.tile_size = TILE_SIZE
    config.max_patch_size = MAX_PATCH_SIZE
    config.sampled_hw_p_list = list(SAMPLED_HW_P_LIST)
    return config


def build_trainer_config(common: CommonComponents, module_path: str = MODULE_PATH):
    """The pixel-latent arm's evals; sibling arms pass their own ``module_path``."""
    return _pixlat_trainer_config(common, module_path=module_path)


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
