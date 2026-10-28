"""encoder_budget_half_t: e = 2 reproduces the all-token budget's shapes."""

import functools
from pathlib import Path

import numpy as np

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoader,
    _IterableDatasetWrapper,
)
from olmoearth_pretrain.data.dataset import OlmoEarthDataset
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)
from olmoearth_pretrain.train.masking import MaskingConfig

BUDGETED = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
MAPS = [Modality.WORLDCOVER.name]
HW_LIST = [2, 3, 4, 5, 6, 8, 10, 12, 16]


def _build(tmp_path: Path, h5_dir: Path, **kwargs) -> OlmoEarthDataLoader:
    tokenization = TokenizationConfig(
        overrides={
            Modality.SENTINEL2_L2A.name: ModalityTokenization(
                band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
            )
        }
    )
    dataset = OlmoEarthDataset(
        h5py_dir=h5_dir, training_modalities=BUDGETED + MAPS, dtype=np.float32
    )
    masking = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "only_decode_modalities": MAPS,
            "num_visible": "uniform",
            "decode_visible_remainder": True,
        },
        tokenization_config=tokenization,
    ).build()
    dataset.prepare()
    return OlmoEarthDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=functools.partial(
            collate_single_masked_batched, transform=None, masking_strategy=masking
        ),
        target_device_type="cpu",
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=HW_LIST,
        time_priority_prob=0.5,
        max_timesteps=12,
        tile_size=256,
        masking_strategy=masking,
        num_masked_views=1,
        tokenization_config=tokenization,
        **kwargs,
    )


def _shapes(dl: OlmoEarthDataLoader, n: int) -> list[tuple]:
    dl.reshuffle()
    return list(
        _IterableDatasetWrapper(dl)._get_batch_item_params_iterator(
            np.arange(n), dl.patch_sizes, dl.sampled_hw_p_list, rank_batch_size=1
        )
    )


def test_two_visible_matches_all_token_shapes(
    tmp_path: Path, setup_h5py_dir: Path
) -> None:
    """At e = 2, E = 128 / floor 10 gives exactly the shapes of hw^2 t in [10, 128]."""
    enc = _build(
        tmp_path / "enc",
        setup_h5py_dir,
        encoder_token_budget=128,
        min_encoder_tokens=10,
        max_num_visible=4,
        encoder_budget_half_t=True,
    )
    # Two budgeted band sets: the all-token budget 256 / floor 20 is hw^2 t in [10, 128].
    full = _build(
        tmp_path / "full",
        setup_h5py_dir,
        token_budget=256,
        min_tokens_per_instance=20,
    )
    enc_items = _shapes(enc, 6000)
    full_items = _shapes(full, 3000)
    by_e: dict[int, set] = {}
    for _idx, _ps, hw, t, _lps, e, _k in enc_items:
        assert e in (1, 2, 3, 4)
        assert 2 * 10 <= e * hw * hw * t <= 2 * 128, (e, hw, t)
        by_e.setdefault(e, set()).add((hw, t))
    full_shapes = {(hw, t) for _i, _p, hw, t, _l, _e, _k in full_items}
    assert all(10 <= hw * hw * t <= 128 for hw, t in full_shapes)
    assert by_e[2] == full_shapes
    # Fewer visible band sets allow larger crops, more visible smaller ones.
    assert max(hw * hw * t for hw, t in by_e[1]) > 128
    assert max(hw * hw * t for hw, t in by_e[4]) <= 64
