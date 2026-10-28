"""Tests for the per-rank-batch kept-modality cap (num_kept_modalities_choices)."""

import functools
from pathlib import Path

import numpy as np
import pytest

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoader,
    _IterableDatasetWrapper,
)
from olmoearth_pretrain.data.dataset import OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)
from olmoearth_pretrain.train.masking import MaskingConfig

BUDGETED = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
MAPS = [Modality.WORLDCOVER.name, Modality.OPENSTREETMAP_RASTER.name]


def _one_bandset_s2() -> TokenizationConfig:
    """S2 as one band set, like the v1.3 control."""
    return TokenizationConfig(
        overrides={
            Modality.SENTINEL2_L2A.name: ModalityTokenization(
                band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
            )
        }
    )


def _build(
    tmp_path: Path,
    setup_h5py_dir: Path,
    *,
    choices: list[int] | None,
    token_budget: int = 400,
    min_tokens: int = 0,
    tokenization_config: TokenizationConfig | None = None,
) -> OlmoEarthDataLoader:
    dataset = OlmoEarthDataset(
        h5py_dir=setup_h5py_dir,
        training_modalities=BUDGETED + MAPS,
        dtype=np.float32,
    )
    masking_strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "only_decode_modalities": MAPS,
            "num_visible": "uniform",
            "decode_visible_remainder": True,
        },
        tokenization_config=tokenization_config,
    ).build()
    collator = functools.partial(
        collate_single_masked_batched, transform=None, masking_strategy=masking_strategy
    )
    dataset.prepare()
    return OlmoEarthDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collator,
        target_device_type="cpu",
        token_budget=token_budget,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[2, 4, 6, 8, 12],
        time_priority_prob=0.5,
        min_tokens_per_instance=min_tokens,
        max_timesteps=12,
        tile_size=256,
        num_kept_modalities_choices=choices,
        masking_strategy=masking_strategy,
        num_masked_views=1,
        tokenization_config=tokenization_config,
    )


def test_drop_modalities_beyond_keeps_a_uniform_subset() -> None:
    """At most k budgeted modalities survive, maps and timestamps always do."""
    np.random.seed(0)
    counts = {m: 0 for m in ["a", "b", "c", "d"]}
    for _ in range(2000):
        sample = {m: 0 for m in ["a", "b", "c", "d", "map", "timestamps"]}
        masks = {m: 0 for m in ["a", "b", "c", "d"]}
        OlmoEarthDataset._drop_modalities_beyond(sample, masks, 2, frozenset({"map"}))
        kept = [m for m in ["a", "b", "c", "d"] if m in sample]
        assert len(kept) == 2 and "map" in sample and "timestamps" in sample
        assert set(masks) == set(kept)
        for m in kept:
            counts[m] += 1
    # Each modality kept with probability 1/2.
    assert all(800 < c < 1200 for c in counts.values()), counts
    sample = {"a": 0, "b": 0, "timestamps": 0}
    OlmoEarthDataset._drop_modalities_beyond(sample, {}, 3, frozenset())
    assert set(sample) == {"a", "b", "timestamps"}


def test_kept_count_is_per_rank_batch_and_sizes_the_crop(
    tmp_path: Path, setup_h5py_dir: Path
) -> None:
    """k is constant within a rank batch, both choices occur, and k * hw^2 * t fits."""
    budget, floor = 400, 30
    dl = _build(
        tmp_path,
        setup_h5py_dir,
        choices=[1, 2],
        token_budget=budget,
        min_tokens=floor,
        tokenization_config=_one_bandset_s2(),
    )
    dl.reshuffle()
    items = list(
        _IterableDatasetWrapper(dl)._get_batch_item_params_iterator(
            np.arange(800), dl.patch_sizes, dl.sampled_hw_p_list, rank_batch_size=4
        )
    )
    for start in range(0, len(items), 4):
        assert len({it[6] for it in items[start : start + 4]}) == 1
    largest: dict[int, int] = {}
    for _idx, _ps, hw, t, _lps, nv, k in items:
        assert nv is None and k in (1, 2)
        assert floor <= k * hw * hw * t <= budget, (k, hw, t)
        largest[k] = max(largest.get(k, 0), hw * hw * t)
    assert set(largest) == {1, 2}
    assert largest[1] > largest[2]


def test_kept_modalities_end_to_end(
    tmp_path: Path, setup_h5py_dir_100_samples: Path
) -> None:
    """With k = 1 each sample keeps one of S2/S1, both get kept across samples."""
    dl = _build(
        tmp_path,
        setup_h5py_dir_100_samples,
        choices=[1],
        token_budget=200,
        tokenization_config=_one_bandset_s2(),
    )
    dl.reshuffle()
    batches = list(_IterableDatasetWrapper(dl))
    assert batches
    kept_names: dict[str, int] = {}
    for _patch_size, masked, _lps in batches:
        for i in range(masked.sentinel2_l2a_mask.shape[0]):
            present = [
                name
                for name in BUDGETED
                if bool(
                    (
                        getattr(masked, f"{name}_mask")[i] != MaskValue.MISSING.value
                    ).any()
                )
            ]
            assert len(present) <= 1, present
            for name in present:
                kept_names[name] = kept_names.get(name, 0) + 1
    assert set(kept_names) == set(BUDGETED), kept_names


def test_kept_modalities_needs_one_bandset(
    tmp_path: Path, setup_h5py_dir: Path
) -> None:
    """S2's default three band sets are rejected."""
    with pytest.raises(ValueError, match="one band set"):
        _build(tmp_path, setup_h5py_dir, choices=[1, 2])


def test_kept_modalities_excludes_encoder_budget(
    tmp_path: Path, setup_h5py_dir: Path
) -> None:
    """The two shape modes cannot be combined."""
    dataset = OlmoEarthDataset(
        h5py_dir=setup_h5py_dir, training_modalities=BUDGETED, dtype=np.float32
    )
    with pytest.raises(ValueError, match="exclusive"):
        OlmoEarthDataLoader(
            dataset=dataset,
            work_dir=tmp_path,
            global_batch_size=1,
            collator=lambda x: x,
            target_device_type="cpu",
            min_patch_size=1,
            max_patch_size=1,
            sampled_hw_p_list=[2],
            token_budget=100,
            encoder_token_budget=100,
            max_num_visible=1,
            num_kept_modalities_choices=[1],
        )
