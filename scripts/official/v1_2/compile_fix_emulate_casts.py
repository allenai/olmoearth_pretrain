"""Compile-debug arm 1: torch.compile with Inductor emulating eager precision casts.

The compile_model=true A/B arm diverged from the (matched-seed) production run
q3x9lvww starting at the ~3000-step learning phase transition, with depressed
grad norms -- systematically degraded gradients somewhere in the compiled
blocks. The leading suspect is Inductor fusing/reassociating the deliberate
fp32 upcasts in the learnable-RoPE trig chain (``encodings.apply_3d_mixed_rope``
upcasts to fp32 for angles/sin/cos and casts back to bf16; the learnable
per-layer frequency parameters receive tiny precision-sensitive gradients
through that chain).

This arm turns on ``torch._inductor.config.emulate_precision_casts`` (which
makes Inductor round intermediates exactly where eager casts do) and otherwise
matches the failed ``ab_compile_only`` arm: compile ON (hardcoded below, no
CLI flag needed), fused OFF, 8000 steps (all inside LR warmup, so directly
comparable to the production baseline), same W&B project.

Interpretation vs ab_compile_only and the production run q3x9lvww:
* tracks production -> Inductor precision-cast handling was the bug; this
  flag (small perf cost) or eager-RoPE is the fix.
* still diverges -> precision casts are innocent; see the eager-RoPE arm
  (``compile_fix_eager_rope.py``) to isolate the RoPE subgraph entirely.

The flag is set at import time so every rank sets it before any compilation
happens (env vars set at launch time do NOT propagate to the Beaker job).
"""

import logging

import torch._inductor.config as inductor_config

inductor_config.emulate_precision_casts = True

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
