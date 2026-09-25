"""Joint latentread trained at RANDOM latent strides, from one latent per pixel to one per patch.

``pure_perceiver_joint_latentread_pixlat.py`` always lays one latent per pixel, so its
latent count grows with the sample's pixel area and its sampler has to cap samples at
64 pixels a side. This arm instead draws the latent STRIDE per training forward pass
(``random_latent_stride=True``): uniformly among the divisors of the batch's patch
size whose latent count fits a budget of 4,096 latents per sample, with the patch
stride always allowed. Stride 1 is one latent per pixel, stride ``patch_size`` one per
patch, so one run trains every output resolution in between, and large grids get a
coarser stride instead of being excluded -- the 64-pixel cap is gone and the sampler
is back to Favyen's grids of up to 24 patches (96 pixels a side at patch size 4).

Evaluation and inference use stride 1 (``eval_latent_stride``): per-pixel embeddings,
as in the pixel-latent arm. The supervision heads keep the default unfold of the
maximum patch size (4) with the existing bilinear resize, which fits any stride.

Everything else is ``pure_perceiver_joint_latentread_pixlat.py``: point latents,
latentread, patch embed base 4, patch sizes 1..4, microbatch 32, and its evals (the
student at ps1 plus d128 at ps4).

W&B project ``20260921_perceiver_shapes``; trained as
``v1_3_vit0_rstride_latentread_joint12``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_dataloader_config as _pixlat_dataloader_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_model_config as _pixlat_model_config,
)
from pure_perceiver_joint_latentread_pixlat import (  # noqa: E402
    build_train_module_config,
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

# Per-sample latent budget: the pixel-latent arm's worst case (64 x 64 pixels).
MAX_LATENTS = 4096
# The sampler's default tile extent, i.e. no pixel-side cap beyond the stored tiles.
TILE_SIZE = OlmoEarthDataLoaderConfig.__dataclass_fields__["tile_size"].default


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Pixel-latent latentread with random strides under a latent budget."""
    config = _pixlat_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, JointLatentConfig) and perceiver.pixel_latents
    perceiver.random_latent_stride = True
    perceiver.max_latents = MAX_LATENTS
    perceiver.eval_latent_stride = 1
    assert config.supervision_head_config is not None
    # Default unfold (max_patch_size) + bilinear resize fits every stride.
    config.supervision_head_config.spatial_unfold = None
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The pixel-latent sampler WITHOUT its 64-pixel cap; the latent budget bounds cost."""
    config = _pixlat_dataloader_config(common)
    config.tile_size = TILE_SIZE
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
