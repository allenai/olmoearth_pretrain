"""Shared functions across evaluation datasets."""

import json
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files

import torch
from torch.utils.data import default_collate

from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample


def eval_collate_fn(
    batch: Sequence[tuple[MaskedOlmoEarthSample, torch.Tensor]],
) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
    """Collate function for DataLoaders."""
    samples, targets = zip(*batch)
    # we assume that the same values are consistently None
    collated_sample = default_collate([s.as_dict(return_none=False) for s in samples])
    collated_target = default_collate([t for t in targets])
    return MaskedOlmoEarthSample(**collated_sample), collated_target


@lru_cache(maxsize=1)
def load_min_max_stats() -> dict:
    """Load the min/max stats for a given dataset."""
    with (
        files("olmoearth_pretrain.evals.datasets.config") / "minmax_stats.json"
    ).open() as f:
        return json.load(f)
