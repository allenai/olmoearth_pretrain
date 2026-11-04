r"""Window retrieval on v1.3: fine-tune the released model with a window-retrieval loss.

Goal: from the stored d128 student embedding of a 12-month range, retrieve the
embedding fresh inference on any contiguous sub-range would give, with no encoder
pass at query time. A frozen check (``scripts/tools/window_retrieval_*``) measures
how far the released store gets with a retriever alone; these runs let the model
itself learn to keep window information, starting from the v1.3 release weights.

Every step runs the release recipe (``../base.py``: LatentMIM loss, supervision
heads, student distillation) plus ``olmoearth_pretrain.train.window_retrieval``:

* one window length per microbatch, a start per sample inside its valid steps;
* a no-grad, UNMASKED encoder pass on the input sliced to the window gives the
  target ``z_t`` (the student embedding fresh inference on the window returns);
* the retriever (``olmoearth_pretrain.nn.window_retriever``, FiLM, width 2048,
  depth 4: the frozen check's best; low-rank saturated at ~25% of the gap) maps
  ``z_T`` = student(location tokens of the main, masked pass, NOT detached)
  plus the window to ``ẑ_t``; MSE to ``z_t`` at weight 1.0, plus 3x the MSE after
  removing each sample's mean over location tokens. In the frozen check ~40% of
  what plain MSE recovers is one constant offset per window, which no classifier
  can use; the centred term puts the gradient on within-tile structure.

Three arms, one module each so in-loop eval jobs rebuild the right model:

* ``pooler_only.py``: every encoder parameter outside the AttentionPooler frozen;
  the retrieval loss reaches the retriever, student and pooler. The release losses
  still train the pooler (and decoder, heads), which keeps the d768 grid anchored.
* ``end_to_end.py``: everything trainable; the retrieval loss also reaches the
  encoder.
* ``from_scratch.py``: the full v1.3 run with the retrieval loss from step 0: no
  initialization, v1.3's own optimizer, LR schedule (cosine over its 300-epoch
  ``max_duration``) and in-loop evals, hard-stopped at step 667,200 (where the
  release checkpoint was taken).

The two fine-tunes start from the converted release checkpoint
(``RELEASE_WEIGHTS``: encoder, pooler, student, decoder, heads; the retriever starts
at the identity), run 30k steps with cosine from a 1k warmup, and score the
release run's AEF-trial kNN + PASTIS set on the d128 student every 10k steps as
the annual-quality non-regression check. Window retrieval itself is scored
offline with the PASTIS / PLANTEUR window-retrieval evals.

Launch (8 GPUs per node)::

    python scripts/official/v1_3/window_retrieval/pooler_only.py launch \
        wr_pooler_only ai2/jupiter --launch.num_gpus=8 --launch.priority=urgent
    python scripts/official/v1_3/window_retrieval/end_to_end.py launch \
        wr_end_to_end ai2/jupiter --launch.num_nodes=4 --launch.num_gpus=8 \
        --launch.priority=urgent
    python scripts/official/v1_3/window_retrieval/from_scratch.py launch \
        wr_from_scratch ai2/jupiter --launch.num_nodes=<as the v1.3 run> \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import sys
from pathlib import Path

# The recipe builds on v1.3's base.py one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
    set_student_loop_evals,
)
from base import build_model_config as _base_build_model_config  # noqa: E402
from base import (
    build_train_module_config as _base_build_train_module_config,  # noqa: E402
)
from base import build_trainer_config as _base_build_trainer_config  # noqa: E402
from olmo_core.optim.scheduler import CosWithWarmup  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.window_retriever import WindowRetrieverConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

# The released v1.3 weights in the current parameter layout, written by
# scripts/official/v1_3/convert_legacy_checkpoint.py from .../helios/v1_3_release.
RELEASE_WEIGHTS = "/weka/dfive-default/gabrielt/helios/v1_3_release_current/weights.pth"

WINDOW_RETRIEVAL_WEIGHT = 1.0
WINDOW_RETRIEVAL_CENTERED_WEIGHT = 3.0
WINDOW_LENGTHS = (2, 11)
FINE_TUNES = ("pooler_only", "end_to_end")
FINE_TUNE_STEPS = 30000
FINE_TUNE_WARMUP_STEPS = 1000
FINE_TUNE_LR = {"pooler_only": 1e-4, "end_to_end": 5e-5}
FINE_TUNE_LOOP_EVAL_INTERVAL_STEPS = 10000
# The step the v1.3 release checkpoint was taken at.
FROM_SCRATCH_STEPS = 667200
WANDB_PROJECT = "2026_10_30_window_retrieval"


def build_model_config(common: CommonComponents, mode: str) -> LatentMIMConfig:
    """The v1.3 model plus a low-rank retriever; fine-tunes start from the release."""
    config = _base_build_model_config(common)
    config.window_retriever_config = WindowRetrieverConfig(
        dim=128, arch="film", width=2048, depth=4
    )
    if mode in FINE_TUNES:
        config.init_weights_path = RELEASE_WEIGHTS
    return config


def build_train_module_config(
    common: CommonComponents, mode: str
) -> LatentMIMTrainModuleConfig:
    """The release train module plus the retrieval loss; ``mode`` sets what trains."""
    config = _base_build_train_module_config(common)
    config.window_retrieval_weight = WINDOW_RETRIEVAL_WEIGHT
    config.window_retrieval_lengths = WINDOW_LENGTHS
    config.window_retrieval_centered_weight = WINDOW_RETRIEVAL_CENTERED_WEIGHT
    config.freeze_encoder_except_pooler = mode == "pooler_only"
    if mode in FINE_TUNES:
        config.optim_config.lr = FINE_TUNE_LR[mode]
        config.scheduler = CosWithWarmup(warmup_steps=FINE_TUNE_WARMUP_STEPS)
    return config


def build_trainer_config(common: CommonComponents, mode: str, module_path: str):
    """Fine-tunes: 30k steps, d128 evals every 10k. From scratch: the release run's."""
    config = _base_build_trainer_config(common)
    config.callbacks["wandb"].project = WANDB_PROJECT
    if mode not in FINE_TUNES:
        # max_duration stays v1.3's, so the LR schedule is the release run's.
        config.hard_stop = Duration.steps(FROM_SCRATCH_STEPS)
        return set_student_loop_evals(config, module_path)
    config.max_duration = Duration.steps(FINE_TUNE_STEPS)
    return set_student_loop_evals(
        config,
        module_path,
        interval_steps=FINE_TUNE_LOOP_EVAL_INTERVAL_STEPS,
        projection_dims=(128,),
    )


def run(mode: str, module_path: str) -> None:
    """Run one arm."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=lambda common: build_model_config(common, mode),
        train_module_config_builder=lambda common: build_train_module_config(
            common, mode
        ),
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=lambda common: build_trainer_config(
            common, mode, module_path
        ),
        visualize_config_builder=build_visualize_config,
    )
