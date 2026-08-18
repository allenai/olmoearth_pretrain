"""End-to-end plumbing tests for registry-dataset loader flags.

Every loader flag has to survive three hops to have any effect:

    DownstreamTaskConfig -> DownstreamEvaluator._get_data_loader (extra_kwargs)
                         -> get_eval_dataset (registry branch)
                         -> from_registry_entry -> RslearnToOlmoEarthDataset

``get_eval_dataset`` takes ``**kwargs``, so a flag dropped at the second hop is
swallowed in silence: the loader keeps its default (masking off), so the
dataset's own "input missing" warning never fires either, and the eval runs as
if the flag had never been set. That is exactly how ``l8_pixel_cloud_mask``
shipped: it was threaded through the task config, the evaluator and the loader,
but never through the registry branch, and every ``_l8pixmask`` task silently
duplicated its unmasked sibling.

These tests derive the flag list by introspection rather than listing it, so a
newly added flag is covered the moment it exists: any DownstreamTaskConfig field
that is also a RslearnToOlmoEarthDataset.__init__ parameter must arrive at
from_registry_entry with the value the task config set. Adding such a field
without a SENTINELS entry fails the coverage test rather than passing quietly.
"""

import dataclasses
import inspect
from typing import Any

import pytest
from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets as datasets_pkg
import olmoearth_pretrain.train.callbacks.evaluator_callback as callback_mod
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.datasets.rslearn_dataset import (
    RslearnToOlmoEarthDataset,
    from_registry_entry,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamEvaluator,
    DownstreamTaskConfig,
)

# Flags that from_registry_entry deliberately accepts under a different name.
RENAMED = {"input_modalities": "input_modalities_override"}

# A non-default value per flag, used to prove the value itself is forwarded
# rather than just the keyword. Keep in sync with the task config; the coverage
# test below fails if a flag has no sentinel.
SENTINELS: dict[str, Any] = {
    "window_size": 8,
    "tile_samples": True,
    "label_at_center_pixel": True,
    "scl_cloud_mask": True,
    "scl_cloud_classes": (8, 9),
    "landsat_cloud_cover_max": 42.0,
    "l8_pixel_cloud_mask": True,
    "l8_pixel_cloud_bits": 0b01000,
    "landsat_reflectance": True,
    "computed_norm_config": "computed_landsat_reflectance.json",
    "input_modalities": [Modality.LANDSAT.name.lower()],
    "norm_method": NormMethod.NORM_YES_CLIP_3_STD,
    "norm_stats_from_pretrained": True,
}

REGISTRY_DATASET = "some_registry_dataset"


def loader_flags() -> set[str]:
    """Task-config fields that the rslearn loader itself accepts."""
    config_fields = {f.name for f in dataclasses.fields(DownstreamTaskConfig)}
    loader_params = set(
        inspect.signature(RslearnToOlmoEarthDataset.__init__).parameters
    ) - {"self"}
    return config_fields & loader_params


class _StubDataset(Dataset):
    """Minimal stand-in for the dataset get_eval_dataset would build."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Any:
        raise IndexError(idx)


def test_sentinels_cover_every_loader_flag() -> None:
    """A new loader flag must be given a sentinel, not silently skipped."""
    assert loader_flags() <= set(SENTINELS), (
        f"loader flags without a SENTINELS entry: {sorted(loader_flags() - set(SENTINELS))}"
    )


@pytest.mark.parametrize("flag", sorted(loader_flags()))
def test_from_registry_entry_accepts_loader_flag(flag: str) -> None:
    """from_registry_entry must expose every flag the loader understands."""
    params = inspect.signature(from_registry_entry).parameters
    expected = RENAMED.get(flag, flag)
    assert expected in params, (
        f"from_registry_entry has no '{expected}' parameter, so the "
        f"'{flag}' task-config flag can never reach the dataset."
    )


def test_get_eval_dataset_forwards_loader_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hop 2: the registry branch of get_eval_dataset forwards every flag."""
    received: dict[str, Any] = {}

    def fake_from_registry_entry(**kwargs: Any) -> Dataset:
        received.update(kwargs)
        return _StubDataset()

    monkeypatch.setattr(datasets_pkg, "get_dataset_entry", lambda name: object())
    monkeypatch.setattr(datasets_pkg, "from_registry_entry", fake_from_registry_entry)

    datasets_pkg.get_eval_dataset(
        eval_dataset=REGISTRY_DATASET,
        split="val",
        **{flag: SENTINELS[flag] for flag in loader_flags()},
    )

    for flag in sorted(loader_flags()):
        key = RENAMED.get(flag, flag)
        assert key in received, f"get_eval_dataset dropped '{flag}'"
        assert received[key] == SENTINELS[flag], (
            f"get_eval_dataset forwarded '{flag}' as {received[key]!r}, "
            f"expected {SENTINELS[flag]!r}"
        )


def test_evaluator_forwards_loader_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hop 1: the evaluator's extra_kwargs carry every flag it was configured with."""
    received: dict[str, Any] = {}

    def fake_get_eval_dataset(**kwargs: Any) -> Dataset:
        received.update(kwargs)
        return _StubDataset()

    monkeypatch.setattr(callback_mod, "get_eval_dataset", fake_get_eval_dataset)

    # DownstreamEvaluator.__init__ resolves the dataset against the live
    # registry, which a unit test has no access to; the flag plumbing under
    # test lives entirely in _get_data_loader, so populate the attributes it
    # reads directly.
    evaluator = object.__new__(DownstreamEvaluator)
    for flag in loader_flags():
        setattr(evaluator, flag, SENTINELS[flag])
    evaluator.dataset = REGISTRY_DATASET
    evaluator._is_registry_dataset = True
    evaluator.tile_size = None
    evaluator.num_workers = 0
    evaluator.label_fraction = 1.0
    evaluator.h5py_dir = None

    evaluator._get_data_loader(split="val", batch_size=1)

    for flag in sorted(loader_flags()):
        assert flag in received, (
            f"_get_data_loader did not pass '{flag}' to get_eval_dataset"
        )
        assert received[flag] == SENTINELS[flag]
