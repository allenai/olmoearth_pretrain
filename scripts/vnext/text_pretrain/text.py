r"""v1.3 + a text head: location tokens aligned with GPT-5.5 grid captions.

Everything is ``v1_3.py`` except the text head and the caption data it needs:

* every pretraining window has a caption per 320 m cell of a 4 x 4 grid, written by
  GPT-5.5 from the window's label layers (CaptionStore, ``CAPTION_DIR``);
* the captions are embedded once with google/embeddinggemma-300m, document prompt
  (``CAPTION_EMBEDDINGS_DIR``);
* the dataset adds ``caption_cell``, which cropping, CutMix and flips carry along with
  the imagery. It is a dataset modality only: the encoder, decoder, masking and token
  budget never see it;
* the TextAlignmentHead averages the teacher location tokens over each (sample, cell)
  and runs a symmetric InfoNCE against those cells' caption embeddings.

The student learns the text only through distillation from the teacher grid.

    python scripts/vnext/text_pretrain/text.py launch <run_name> ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v1_3 import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from v1_3 import build_dataset_config as _v1_3_build_dataset_config  # noqa: E402
from v1_3 import build_model_config as _v1_3_build_model_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.text_alignment import (  # noqa: E402
    TextAlignmentHeadConfig,
)

logger = logging.getLogger(__name__)

_CAPTIONS = "/weka/dfive-default/agent/piperbots/2026-11-04-piperw-text-pretrain/loader"
CAPTION_DIR = f"{_CAPTIONS}/store"
CAPTION_EMBEDDINGS_DIR = f"{_CAPTIONS}/emb_gemma"
# The InfoNCE starts near log(cells per microbatch), ~7, against supervision terms
# around 1: 0.1 keeps it from dominating at the start. Untuned.
TEXT_WEIGHT = 0.1
TEXT_DIM = 768


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.3 with the text head on the teacher location tokens."""
    config = _v1_3_build_model_config(common)
    config.text_head_config = TextAlignmentHeadConfig(
        embeddings_dir=CAPTION_EMBEDDINGS_DIR,
        text_dim=TEXT_DIM,
        weight=TEXT_WEIGHT,
    )
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """v1.3's dataset plus caption_cell, which only the dataset knows about."""
    config = _v1_3_build_dataset_config(common)
    config.training_modalities = [
        *config.training_modalities,
        Modality.CAPTION_CELL.name,
    ]
    config.caption_dir = CAPTION_DIR
    return config


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
