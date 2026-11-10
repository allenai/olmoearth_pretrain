r"""v1.3 + a text head on cell captions and per-pixel text labels.

Everything in ``text.py`` (v1.3 plus the cell-caption InfoNCE), and:

* per-pixel labels for every pretraining window, read at load time from its WorldCover,
  CDL, WorldCereal and OSM rasters as class names;
* ~100k windows from the open-set segmentation bank (up to 500 per source dataset, from
  210 datasets), converted to the pretraining format with their class masks as labels,
  and concatenated with the pretraining set. The v1.3 eval datasets and their sources,
  and ClipEarth's unseen datasets, are left out (see the session's mix/selection.json);
* one label vocabulary (3,067 names, generic and absence classes dropped), embedded with
  embeddinggemma-300m and the captions' document prompt;
* a label term in the text head: each (sample, label)'s mean location token is
  classified among the microbatch's labels plus random vocabulary distractors, through
  the caption projection, so captions and labels share one text space.

Open-set windows have no captions or label layers: their caption_cell and map
modalities are missing, so only the label term and the image losses see them.
Training runs the published v1.3 run's 667,200 steps (300 epochs of the pretraining set alone),
so the schedule matches v1.3 although the data is larger.

    python scripts/vnext/text_pretrain/text_mix.py launch <run_name> ai2/jupiter \
        --launch.num_gpus=8 --launch.priority=urgent
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from olmo_core.train.common import Duration  # noqa: E402
from olmo_core.train.config import TrainerConfig  # noqa: E402
from text import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_train_module_config,
    build_visualize_config,
)
from text import build_dataset_config as _text_build_dataset_config  # noqa: E402
from text import build_model_config as _text_build_model_config  # noqa: E402
from text import build_trainer_config as _text_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

_MIX = "/weka/dfive-default/agent/piperbots/2026-11-04-piperw-text-pretrain/mix"
LABEL_VOCAB_DIR = f"{_MIX}/vocab"
OPEN_SET_H5PY_DIR = f"{_MIX}/h5/103112"
# Same reasoning as the caption weight: the cross-entropy starts near log(candidates).
LABEL_WEIGHT = 0.1
NUM_LABEL_DISTRACTORS = 1024
# The published v1.3 run's length: 300 epochs of the 1,138,828-window pretraining set at 512.
MAX_STEPS = 667_200


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """text.py's model with the label term on the text head."""
    config = _text_build_model_config(common)
    head = config.text_head_config
    assert head is not None
    head.label_vocab_dir = LABEL_VOCAB_DIR
    head.label_weight = LABEL_WEIGHT
    head.num_label_distractors = NUM_LABEL_DISTRACTORS
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """The captioned pretraining set and the open-set windows, both with text_label."""
    pretrain = _text_build_dataset_config(common)
    pretrain.training_modalities = [
        *pretrain.training_modalities,
        Modality.TEXT_LABEL.name,
    ]
    pretrain.label_vocab_dir = LABEL_VOCAB_DIR
    open_set = OlmoEarthDatasetConfig(
        h5py_dir=OPEN_SET_H5PY_DIR,
        training_modalities=list(pretrain.training_modalities),
        label_vocab_dir=LABEL_VOCAB_DIR,
    )
    return OlmoEarthConcatDatasetConfig(dataset_configs=[pretrain, open_set])


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """text.py's trainer, run for the published v1.3 run's number of steps."""
    config = _text_build_trainer_config(common)
    config.max_duration = Duration.steps(MAX_STEPS)
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
