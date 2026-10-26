"""Data diagnostic: v1.3's own encoder on the AllCaptures corpus, monthly samples only.

The released v1.3 model (``rc_ld1_pixtgt_pix512``: 12 x d768 encoder, no narrow stage,
no slot merge) trained on AllCaptures with the D' sampler (``allcap_slot_n4d192_fe``:
v1.3's real-token budget and microbatch) restricted to 30-day composites (one clear
capture per bin). If it lags v1.3 like the narrow-stage arms (~3 PASTIS points at
480k), the gap is the corpus or its sampling, not the narrow stage and merge.

Evals: the allcap set without the every-capture tasks (a full-year every-capture
tile is ~10k tokens per sample through the full trunk without a merge).

W&B project ``20261016_allcap_slot_narrow``; trained as ``v1_3_allcap_v13enc_monthly``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allcap_narrow import (  # noqa: E402
    build_allcap_trainer_config,
    build_common_components,
    build_dataset_config,
)
from allcap_slot_n4d192_fe import (  # noqa: E402
    build_dataloader_config as _fe_dataloader,
)
from allcap_slot_n4d192_fe import (  # noqa: E402
    build_train_module_config,
)
from base import build_visualize_config  # noqa: E402
from rc_ld1_pixtgt_pix512 import (  # noqa: E402
    build_model_config as _rc_ld1_model_config,
)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.encodings import use_compiled_mixed_rope  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/ablations/allcap_v13enc_monthly.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.3's model (no narrow stage, no merge) for the corpus's Landsat L2."""
    use_compiled_mixed_rope(True)
    config = _rc_ld1_model_config(common)
    config.encoder_config.band_dropout_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT_L2.name,
    ]
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """D''s sampler with monthly composites only."""
    config = _fe_dataloader(common)
    config.composite_days_choices = [30.0]
    return config


def build_trainer_config(common: CommonComponents):
    """The allcap eval set without the every-capture tasks, re-importing THIS module."""
    config = build_allcap_trainer_config(common, MODULE_PATH)
    evaluator = config.callbacks["downstream_evaluator"]
    evaluator.tasks = {
        name: task for name, task in evaluator.tasks.items() if not task.every_capture
    }
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
