"""Shared builders for resized copies of the released v1.3 Base (``rc_ld1_pixtgt_pix512``).

The released OlmoEarth-v1.3-Base is ``v1_3_rc_ld1_pixtgt_pix512`` step 667200, trained
from ``rc_ld1_pixtgt_pix512.py`` at 9f5a25b02. The resized runs start from that exact
model config, built by the release chain, and change ONLY the size fields:

* encoder ``embedding_size`` / ``depth`` / ``num_heads``;
* decoder ``encoder_embedding_size`` / ``decoder_embedding_size`` / ``depth`` /
  ``num_heads``;
* the Perceiver's ``register_dim`` (teacher width, also the supervision and
  distillation heads' input width, set from it at build time) and ``attn_dim``
  (attention at encoder width, as in Base), and the decoder's matching ``register_dim``.

Everything else -- the student ``[128, 64]``, per-pixel random-stride latents, pixel
targets, the train module, data, sampler, schedule, loop evals -- comes from the release
builders unchanged. ``tests/unit/test_v1_3_sized_configs.py`` diffs the configs to keep
it that way.

Each run also drops olmo_core's async-save duration metric, whose rank-dependent
recording hangs the gloo bookkeeping all-reduce (see ``olmo_core_patches``).
"""

import logging
import sys
from collections.abc import Callable
from typing import Any
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_ld1_pixtgt_pix512 import (  # noqa: E402
    build_model_config as build_release_model_config,
)
from rc_pixtgt_pix512 import build_train_module_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.olmo_core_patches import (  # noqa: E402
    drop_async_checkpoint_duration_metric,
)

logger = logging.getLogger(__name__)

# The async-save duration metric is recorded at a rank-dependent step and hangs the gloo
# bookkeeping all-reduce; see olmo_core_patches.
drop_async_checkpoint_duration_metric()


def build_sized_model_config(
    common: CommonComponents,
    *,
    width: int,
    encoder_depth: int,
    num_heads: int,
    decoder_depth: int,
    register_dim: int,
) -> LatentMIMConfig:
    """The released Base model config with only its size fields replaced."""
    config = build_release_model_config(common)
    encoder = config.encoder_config
    decoder = config.decoder_config
    perceiver = encoder.perceiver_config
    assert perceiver is not None
    # Base runs the Perceiver's attention at encoder width; keep that invariant.
    assert perceiver.attn_dim == encoder.embedding_size
    config.encoder_config = replace(
        encoder,
        embedding_size=width,
        depth=encoder_depth,
        num_heads=num_heads,
        perceiver_config=replace(perceiver, register_dim=register_dim, attn_dim=width),
    )
    config.decoder_config = replace(
        decoder,
        encoder_embedding_size=width,
        decoder_embedding_size=width,
        depth=decoder_depth,
        num_heads=num_heads,
        register_dim=register_dim,
    )
    return config


def build_sized_trainer_config(common: CommonComponents, module_path: str):
    """The release run's trainer (loop evals incl. ps4/ps2 student) for ``module_path``."""
    return build_mix_trainer_config(
        common, module_path, ps4_student_evals=True, ps2_student_evals=True
    )


def run_sized(
    model_config_builder: Callable[[CommonComponents], LatentMIMConfig],
    trainer_config_builder: Callable[[CommonComponents], Any],
) -> None:
    """Run a resized release config with the release data/train-module builders.

    The run script must also expose ``build_common_components``,
    ``build_train_module_config``, ``build_model_config`` and ``build_trainer_config``
    at module level: in-loop eval jobs (``checkpoint_sweep_evals``) import the script
    and look those names up, falling back to defaults (the wrong tokenization) if absent.
    """
    main(
        common_components_builder=build_common_components,
        model_config_builder=model_config_builder,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=trainer_config_builder,
        visualize_config_builder=build_visualize_config,
    )
