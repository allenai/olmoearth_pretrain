"""Compile-debug arm 2: torch.compile with the RoPE application excluded (eager).

Companion to ``compile_fix_emulate_casts.py`` (see its docstring for the full
context). This arm removes the suspect subgraph entirely: the four
``apply_*_rope`` functions are wrapped with ``torch.compiler.disable`` so
dynamo graph-breaks around them and they run eagerly -- exactly the eager
forward AND backward for the learnable-frequency trig chain -- while the rest
of each block (layernorms, SDPA, MLP, drop-path) stays compiled.

The wrapping monkeypatches the *attention module's* imported bindings (it
imports the functions by name), plus the encodings module for completeness.
Done in this script rather than in the library so the other debug arm on the
same commit is not affected; if this arm fixes training, the permanent fix is
moving the ``@torch.compiler.disable`` decorators into ``encodings.py``.

Otherwise identical to the failed ``ab_compile_only`` arm: compile ON
(hardcoded), fused OFF, 8000 steps (all warmup), same W&B project.

Interpretation vs ab_compile_only and the production run q3x9lvww:
* tracks production -> the compiled RoPE subgraph (forward or backward) is
  confirmed guilty, and this configuration is itself shippable (keeps the
  layernorm/MLP compile wins).
* still diverges -> the problem lives elsewhere in the compiled block
  (SDPA/mask pattern, drop-path RNG, MLP fusion); next probe would be
  ``backend="aot_eager"`` to separate tracing from Inductor codegen.
"""

import logging

import torch

import olmoearth_pretrain.nn.attention as _attention
import olmoearth_pretrain.nn.encodings as _encodings

for _name in (
    "apply_2d_axial_rope",
    "apply_2d_mixed_rope",
    "apply_3d_axial_rope",
    "apply_3d_mixed_rope",
):
    _eager_fn = torch.compiler.disable(getattr(_encodings, _name))
    setattr(_encodings, _name, _eager_fn)
    setattr(_attention, _name, _eager_fn)

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from base import (  # noqa: E402
    build_train_module_config as _base_build_train_module_config,
)
from base import build_trainer_config as _base_build_trainer_config  # noqa: E402
from compile_fused_ab import build_model_config  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

AB_STEPS = 8000
WANDB_PROJECT = "2026_07_09_compile_fused_ab"


def build_train_module_config(common: CommonComponents):
    """Production train module with compile ON (the flag under test)."""
    config = _base_build_train_module_config(common)
    config.compile_model = True
    return config


def build_trainer_config(common: CommonComponents):
    """Production trainer config, 8k steps, no in-loop evals."""
    trainer_config = _base_build_trainer_config(common)
    trainer_config.max_duration = Duration.steps(AB_STEPS)
    trainer_config.callbacks.pop("downstream_evaluator", None)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
