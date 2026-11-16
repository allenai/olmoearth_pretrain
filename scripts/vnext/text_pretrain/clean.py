"""Held-out classes and locations for the text arms (the *_clean scripts).

The split spec holds out 100 classes (168 names with their aliases) and 10% of 2-degree
regions with a 0.5-degree buffer, shared with ClipEarth's clean split. Text targets drop
the held classes everywhere and everything in held and buffer locations; captions that
name a held class come from a filtered store with those cells blanked (same window ids,
so the caption embeddings still line up).
"""

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

_SPLITS = "/weka/dfive-default/agent/piperbots/2026-11-16-piperw-clean-splits"
TEXT_HOLDOUT_SPEC = f"{_SPLITS}/split_spec.json"
CLEAN_CAPTION_DIR = f"{_SPLITS}/caption_store"


def clean_model(config: LatentMIMConfig) -> LatentMIMConfig:
    """Keep held-out labels out of the label distractors."""
    assert config.text_head_config is not None
    config.text_head_config.text_holdout_spec = TEXT_HOLDOUT_SPEC
    return config


def clean_dataset(
    config: OlmoEarthDatasetConfig | OlmoEarthConcatDatasetConfig,
) -> OlmoEarthDatasetConfig | OlmoEarthConcatDatasetConfig:
    """Mask held-out text targets in every dataset; captions from the filtered store."""
    datasets = (
        config.dataset_configs
        if isinstance(config, OlmoEarthConcatDatasetConfig)
        else [config]
    )
    for dataset in datasets:
        dataset.text_holdout_spec = TEXT_HOLDOUT_SPEC
        if dataset.caption_dir is not None:
            dataset.caption_dir = CLEAN_CAPTION_DIR
    return config
