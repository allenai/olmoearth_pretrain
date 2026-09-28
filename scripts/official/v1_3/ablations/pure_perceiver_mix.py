"""Shared builders for the token-mixing pure-Perceiver arms (``pure_perceiver_mix*.py``).

``pure_perceiver.py`` (``v1_3_vit0_trope_ld12``) reads the raw patch embeddings: a
read is a weighted average of tokens, so nothing computes on a token before it is
pooled, and on a single-timestep input no two pixels interact before the latent
blocks. These arms keep that model and add TOKEN-MIXING blocks
(``PerceiverConfig.token_mix_layout``): self-attention over the patch tokens
restricted to a spatial neighbourhood of cells -- every timestep and modality of the
cells within ``token_mix_radius`` (1 = 3x3) -- interleaved with the reads, so each read
pools tokens already refined by the mixing before it. On a single timestep a 3x3
mixing block is local spatial attention (a content-dependent 3x3 conv); on a
12-timestep, 3-modality input it mixes space, time and modality in one hop.

No latent-only blocks: every read keeps just its paired latent block, so latent depth
equals the read count (the RC showed 2 and 4 latent layers score the same, and on
multi-timestep inputs the latent blocks are a few percent of the compute), and the
budget goes to the token mixing instead. (A first launch topped every arm up to 12
latent blocks with latent-only blocks; it was stopped before training, 2026-09-28.) Reads stay
point reads (window-centre time anchor), as in ``trope_ld12``. The mixing runs on a
Perceiver-internal copy of the tokens: the encoder output and the latent-MIM target
(projection-only patch embeddings) are unchanged.

IN-LOOP EVALS: ``trope_ld12``'s student evals (AEF trials + year-aligned PASTIS at
d128 / d64) plus two catalog tasks on the d768 register grid (the non-distilled
output): ``m-eurosat`` (single-timestep S2 kNN) and ``pastis`` (the S2 linear-probe
segmentation task, not the year-aligned embedding variant).

W&B project ``20260921_perceiver_shapes``.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    STUDENT_LOOP_EVAL_INTERVAL_STEPS,
    set_student_loop_evals,
)
from olmo_core.train.common import Duration  # noqa: E402
from pure_perceiver import WANDB_PROJECT  # noqa: E402
from pure_perceiver import (  # noqa: E402
    build_model_config as _pure_perceiver_model_config,
)
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import PerceiverConfig  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

# v1.2 catalog tasks scored on the d768 register grid (not the student).
REGISTER_EVAL_TASKS = ("m-eurosat", "pastis")


def interleaved_layout(n_mix: int, n_read: int) -> str:
    """``n_mix`` M and ``n_read`` R spread evenly, ending on a read.

    The mixing is distributed so every read sees one more round of mixing than the
    previous one.
    """
    layout = ""
    for i in range(n_read):
        # Mixing blocks due before read i: an even share of n_mix, front-loaded.
        target = -(-n_mix * (i + 1) // n_read)
        layout += "M" * (target - layout.count("M")) + "R"
    return layout


def build_mix_model_config(
    common: CommonComponents,
    *,
    n_mix: int = 0,
    n_read: int = 0,
    radius: int = 1,
    mix_dim: int | None = None,
    mix_heads: int | None = None,
    layout: str | None = None,
) -> LatentMIMConfig:
    """``trope_ld12`` with ``n_mix`` neighbourhood-mixing blocks and ``n_read`` reads.

    ``layout`` overrides the interleaved default with an explicit schedule (its ``R``
    count sets the number of reads, and the total latent depth is whatever it holds).
    """
    config = _pure_perceiver_model_config(common)
    perceiver = config.encoder_config.perceiver_config
    assert isinstance(perceiver, PerceiverConfig) and perceiver.read_time_rope
    assert not perceiver.read_time_range  # point reads, as in trope_ld12
    if layout is None:
        layout = interleaved_layout(n_mix, n_read)
    perceiver.latent_depth = layout.count("R")
    perceiver.token_mix_layout = layout
    perceiver.token_mix_radius = radius
    perceiver.token_mix_dim = mix_dim
    perceiver.token_mix_num_heads = mix_heads
    return config


def build_mix_trainer_config(common: CommonComponents, module_path: str):
    """``trope_ld12``'s student evals + m-eurosat / pastis on the d768 registers."""
    v1_2_trainer = _v1_2_build_trainer_config(common)
    catalog = v1_2_trainer.callbacks["downstream_evaluator"].tasks
    register_tasks = {
        name: replace(
            catalog[name],
            eval_interval=Duration.steps(STUDENT_LOOP_EVAL_INTERVAL_STEPS),
        )
        for name in REGISTER_EVAL_TASKS
    }
    trainer_config = set_student_loop_evals(v1_2_trainer, module_path)
    trainer_config.callbacks["downstream_evaluator"].tasks.update(register_tasks)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config
