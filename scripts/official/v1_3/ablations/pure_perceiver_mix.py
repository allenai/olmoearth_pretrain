"""Shared builders for the token-mixing pure-Perceiver arms (``pure_perceiver_mix*.py``).

``pure_perceiver.py`` (``v1_3_vit0_trope_ld12``) reads the raw patch embeddings: a
read is a weighted average of tokens, so nothing computes on a token before it is
pooled, and on a single-timestep input no two pixels interact before the latent
blocks. These arms keep that model and add TOKEN-MIXING blocks
(``PerceiverConfig.token_mix_layout``): self-attention over the patch tokens
restricted to a spatial neighbourhood of cells -- every timestep and modality of the
cells within ``token_mix_radius`` (2 = 5x5, the default) -- interleaved with the reads,
so each read pools tokens already refined by the mixing before it. On a single timestep
a 5x5 mixing block is local spatial attention (a content-dependent 5x5 conv); on a
12-timestep, 3-modality input it mixes space, time and modality in one hop.

No latent-only blocks: every read keeps just its paired latent block, so latent depth
equals the read count (the RC showed 2 and 4 latent layers score the same, and on
multi-timestep inputs the latent blocks are a few percent of the compute), and the
budget goes to the token mixing instead. (A first launch topped every arm up to 12
latent blocks with latent-only blocks; it was stopped before training, 2026-09-28.)

Window: 5x5 cells. The arms first trained ~1.5k steps at 3x3 (``*_rl*`` run names without
``w5``) and were replaced: a 1-GPU profile put 5x5 at +5-10% step time for the 768-dim
arm (FlexAttention 26% -> 30% of GPU time) and +1-4% at 128 dims, for twice the one-hop
reach. Reads stay
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
    aeftrial_loop_eval_tasks,
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
# Datasets also scored at patch size 2 (``ps2_student_evals``): PASTIS plus the two
# fastest AEF-trial tasks (~3.5 min each per checkpoint, vs 15-50 min for the rest).
PS2_STUDENT_EVAL_DATASETS = (
    "pastis_year_aligned",
    "africa_crop_mask_year_aligned",
    "ethiopia_crops_year_aligned",
)


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
    radius: int = 2,
    mix_dim: int | None = None,
    mix_heads: int | None = None,
    layout: str | None = None,
    pixel_latents: bool = False,
    max_latents: int | None = None,
    latent_stride_bias: float | None = None,
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
    if pixel_latents:
        # Sub-patch latents with a random stride under the budget in training, one
        # latent per pixel at eval (the joint random-stride arms' convention).
        assert max_latents is not None
        perceiver.pixel_latents = True
        perceiver.random_latent_stride = True
        perceiver.max_latents = max_latents
        perceiver.eval_latent_stride = 1
        perceiver.latent_stride_bias = latent_stride_bias
    return config


def build_mix_trainer_config(
    common: CommonComponents,
    module_path: str,
    *,
    ps4_student_evals: bool = False,
    ps2_student_evals: bool = False,
):
    """``trope_ld12``'s student evals + m-eurosat / pastis on the d768 registers.

    ``ps4_student_evals`` adds the d128 student at patch size 4 on the AEF + PASTIS
    tasks (named ``*_ws16_ps4_*_proj128``), for arms whose latents stay per pixel at a
    coarse patch size. ``ps2_student_evals`` adds the same at patch size 2 on
    ``PS2_STUDENT_EVAL_DATASETS`` only (named ``*_ws16_ps2_*_proj128``).
    """
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
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.tasks.update(register_tasks)
    if ps4_student_evals:
        for name, task in aeftrial_loop_eval_tasks(
            STUDENT_LOOP_EVAL_INTERVAL_STEPS
        ).items():
            assert "_ps1_" in name, name
            evaluator.tasks[name.replace("_ps1_", "_ps4_") + "_proj128"] = replace(
                task, patch_size=4, eval_on_student_registers=True, eval_student_dim=128
            )
    if ps2_student_evals:
        for name, task in aeftrial_loop_eval_tasks(
            STUDENT_LOOP_EVAL_INTERVAL_STEPS
        ).items():
            if not name.startswith(PS2_STUDENT_EVAL_DATASETS):
                continue
            evaluator.tasks[name.replace("_ps1_", "_ps2_") + "_proj128"] = replace(
                task, patch_size=2, eval_on_student_registers=True, eval_student_dim=128
            )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config
