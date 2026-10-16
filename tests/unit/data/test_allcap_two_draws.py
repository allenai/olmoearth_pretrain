"""E5 arm C: two independent date-draws per allcap sample, their collator and train module.

Reuses the synthetic allcap H5 fixture of ``test_allcap.py`` (16x16 px, 40 S2 / 25 S1 /
10 Landsat captures, worldcover + srtm maps).
"""

from typing import Any
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig
from upath import UPath

from olmoearth_pretrain.data.allcap import (
    ALLCAP_RANDOM_CLEAR,
    read_allcap_sample,
    read_allcap_sample_two_draws,
    timestamps_to_epoch_days,
)
from olmoearth_pretrain.data.collate import collate_two_draw_masked_batched
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthSample,
    subset_sample_default,
)
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PerceiverConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.register_distillation_head import (
    RegisterDistillationHeadConfig,
)
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.two_view_latent_mim import (
    TwoViewLatentMIMTrainModuleConfig,
)

from . import test_allcap as _allcap

MODS = _allcap.MODS
allcap_dir = _allcap.allcap_dir  # Ilse's synthetic allcap H5 fixture

TAU = 0.6  # ~24 of the fixture's 40 S2 captures are clear: draws of 12 differ.
PATCH, HW_P = 2, 4  # an 8x8 px crop of the 16x16 tile


def _dataset(
    d: UPath, two_draws: bool, normalize: bool = False, **kwargs: Any
) -> OlmoEarthDataset:
    ds = OlmoEarthDataset(
        h5py_dir=d,
        training_modalities=MODS,
        dtype=np.float32,
        normalize=normalize,
        allcap_selection=ALLCAP_RANDOM_CLEAR,
        allcap_cloud_tau=TAU,
        allcap_two_draws=two_draws,
        **kwargs,
    )
    ds.prepare()
    return ds


def _args(target_t: int | None = None) -> GetItemArgs:
    return GetItemArgs(
        idx=0,
        patch_size=PATCH,
        sampled_hw_p=HW_P,
        token_budget=1000,
        target_t=target_t,
        budget_exclude_modalities=frozenset({"worldcover", "srtm"}),
    )


def _crop_offset(view: OlmoEarthSample, raw_srtm: np.ndarray) -> tuple[int, int]:
    """Locate the view's crop in the tile through the (random-valued) srtm map."""
    hw = view.srtm.shape[0]
    hits = [
        (h, w)
        for h in range(raw_srtm.shape[0] - hw + 1)
        for w in range(raw_srtm.shape[1] - hw + 1)
        if np.array_equal(raw_srtm[h : h + hw, w : w + hw], view.srtm)
    ]
    assert len(hits) == 1
    return hits[0]


def test_reader_two_draws_one_read_two_selections(
    allcap_dir: tuple[UPath, dict],
) -> None:
    """Each view is exactly read_allcap_sample under its own generator."""
    d, _ = allcap_dir
    with h5py.File(d / "sample_0.h5", "r") as f:
        views = read_allcap_sample_two_draws(
            f,
            MODS,
            ALLCAP_RANDOM_CLEAR,
            TAU,
            (np.random.default_rng(1), np.random.default_rng(2)),
        )
        singles = [
            read_allcap_sample(
                f, MODS, ALLCAP_RANDOM_CLEAR, TAU, np.random.default_rng(s)
            )
            for s in (1, 2)
        ]
    for (sample, masks), (ref, ref_masks) in zip(views, singles):
        assert sample.keys() == ref.keys()
        for k in sample:
            np.testing.assert_array_equal(sample[k], ref[k])
        for k in ref_masks:
            np.testing.assert_array_equal(masks[k], ref_masks[k])
    assert not np.array_equal(views[0][0]["timestamps"], views[1][0]["timestamps"])


def test_two_draws_share_crop_and_maps_differ_in_dates(
    allcap_dir: tuple[UPath, dict],
) -> None:
    """(a) Same crop, shape and static maps; different S2/S1 captures and dates."""
    d, arrays = allcap_dir
    ds = _dataset(d, two_draws=True)
    s2_days = timestamps_to_epoch_days(arrays["timestamps_sentinel2_l2a"])
    raw_s2 = arrays["sentinel2_l2a"].astype(np.float32)
    n_differ = 0
    for seed in range(8):
        np.random.seed(seed)
        patch_size, (a, b) = ds[_args()]
        assert patch_size == PATCH
        assert isinstance(a, OlmoEarthSample) and isinstance(b, OlmoEarthSample)
        # same shape: grid and number of timesteps
        for name in ("timestamps", *MODS):
            assert getattr(a, name).shape == getattr(b, name).shape, name
        assert a.sentinel2_l2a.shape[:2] == (PATCH * HW_P, PATCH * HW_P)
        # same static / decode-only maps, hence the same crop
        np.testing.assert_array_equal(a.worldcover, b.worldcover)
        np.testing.assert_array_equal(a.srtm, b.srtm)
        h0, w0 = _crop_offset(a, arrays["srtm"])
        assert _crop_offset(b, arrays["srtm"]) == (h0, w0)
        hw = PATCH * HW_P
        # the imagery of every present S2 slot is the raw capture of that date, at
        # that same crop
        for view in (a, b):
            days = timestamps_to_epoch_days(view.timestamps)
            s2 = view.sentinel2_l2a
            present = ~(s2 == MISSING_VALUE).all(axis=(0, 1, 3))
            assert present.any()
            for k in np.flatnonzero(present):
                (j,) = np.flatnonzero(s2_days == days[k])
                np.testing.assert_array_equal(
                    s2[:, :, k], raw_s2[h0 : h0 + hw, w0 : w0 + hw, j]
                )
        n_differ += not np.array_equal(a.timestamps, b.timestamps)
    assert n_differ >= 6


def test_two_draws_share_target_t_and_cutmix_crop(
    allcap_dir: tuple[UPath, dict],
) -> None:
    """The shape sampler's t is honoured by both views; cutmix shares its patch picks."""
    d, _ = allcap_dir
    for cutmix in (False, True):
        ds = _dataset(d, two_draws=True, apply_cutmix=cutmix)
        for seed in range(4):
            np.random.seed(seed)
            _, (a, b) = ds[_args(target_t=5)]
            assert a.timestamps.shape[0] == b.timestamps.shape[0] == 5
            np.testing.assert_array_equal(a.srtm, b.srtm)


def _getitem_before_arm_c(ds: OlmoEarthDataset, args: GetItemArgs) -> OlmoEarthSample:
    """OlmoEarthDataset.__getitem__ as it was before arm C (default subset, no NDVI)."""
    sample_dict, masks = ds.read_h5_file(ds._get_h5_file_path(args.idx))
    ts, masks = ds._crop_timestamps_and_masks(sample_dict["timestamps"], masks)
    sample_dict["timestamps"] = ts
    sample_dict, current_length = ds._pad_timestamps(sample_dict)
    sample, _ = ds.fill_sample_with_missing_values(sample_dict, masks)
    return subset_sample_default(
        sample,
        patch_size=args.patch_size,
        max_tokens_per_instance=args.token_budget,
        sampled_hw_p=args.sampled_hw_p,
        current_length=current_length,
        missing_timesteps_masks=masks,
        tokenization_config=args.tokenization_config,
        target_t=args.target_t,
        budget_exclude_modalities=args.budget_exclude_modalities,
    )


def test_default_single_draw_unchanged(allcap_dir: tuple[UPath, dict]) -> None:
    """(b) two_draws=False: one sample, same values and RNG use as the old code path."""
    d, _ = allcap_dir
    ds = _dataset(d, two_draws=False)
    assert ds.allcap_two_draws is False
    for seed in range(5):
        np.random.seed(seed)
        patch_size, sample = ds[_args()]
        after = np.random.random()
        assert patch_size == PATCH and isinstance(sample, OlmoEarthSample)
        np.random.seed(seed)
        ref = _getitem_before_arm_c(ds, _args())
        assert np.random.random() == after
        for name in sample._fields:
            x, y = getattr(sample, name), getattr(ref, name)
            assert (x is None) == (y is None), name
            if x is not None:
                np.testing.assert_array_equal(x, y)


def test_two_draws_needs_a_selection(allcap_dir: tuple[UPath, dict]) -> None:
    """allcap_two_draws without an allcap selection fails at construction."""
    d, _ = allcap_dir
    with pytest.raises(ValueError, match="allcap_selection"):
        OlmoEarthDataset(
            h5py_dir=d,
            training_modalities=MODS,
            dtype=np.float32,
            allcap_two_draws=True,
        )


def _masking_config() -> MaskingConfig:
    """v1.2/v1.3's masking with the maps decode-only."""
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ["worldcover", "srtm"],
        }
    )


def _pair_batch(d: UPath, n: int = 4) -> tuple[int, Any, Any]:
    # normalized, as in training (it also casts the int maps to float32)
    ds = _dataset(d, two_draws=True, normalize=True)
    np.random.seed(0)
    torch.manual_seed(0)
    items = [ds[_args()] for _ in range(n)]
    return collate_two_draw_masked_batched(
        items,
        transform=None,
        masking_strategy=_masking_config().build(),
        masking_strategy_b=None,
    )


def test_collator_two_masked_views(allcap_dir: tuple[UPath, dict]) -> None:
    """(c) Two masked batches with matching shapes, row i = same sample."""
    d, _ = allcap_dir
    patch_size, ma, mb = _pair_batch(d)
    assert patch_size == PATCH
    assert ma.batch_size == mb.batch_size == 4
    for name in ma._fields:
        x, y = getattr(ma, name), getattr(mb, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.shape == y.shape, name
    torch.testing.assert_close(ma.srtm, mb.srtm)
    assert not torch.equal(ma.timestamps, mb.timestamps)

    # a plain (patch_size, sample) item is used as both views
    ds = _dataset(d, two_draws=False)
    np.random.seed(0)
    _, pa, pb = collate_two_draw_masked_batched(
        [ds[_args()], ds[_args()]], None, _masking_config().build(), None
    )
    torch.testing.assert_close(pa.sentinel2_l2a, pb.sentinel2_l2a)


def _tiny_model() -> Any:
    """A tiny rc_pix512-shaped model: per-pixel Perceiver, student, supervision."""
    mods = [*MODS, Modality.LATLON.name]
    encoder = EncoderConfig(
        supported_modality_names=mods,
        embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=1.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        perceiver_config=PerceiverConfig(
            register_dim=16,
            latent_depth=1,
            pixel_latents=True,
            random_latent_stride=True,
            max_latents=32,
            eval_latent_stride=1,
            student_dims=[8, 4],
            student_output_norm=True,
        ),
    )
    decoder = PredictorConfig(
        supported_modality_names=mods,
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=1.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_perceiver=True,
        register_dim=16,
    )
    return LatentMIMConfig(
        encoder_config=encoder,
        decoder_config=decoder,
        projection_only_target=True,
        supervision_head_config=SupervisionHeadConfig(
            modality_configs={
                "srtm": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=1,
                    weight=1.0,
                    regression_loss_type="l1",
                )
            }
        ),
        register_distillation_head_config=RegisterDistillationHeadConfig(
            back_projection_hidden=8
        ),
    ).build()


class _MockTrainer:
    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}
        self.global_step = 0
        self.max_steps = 100

    def record_metric(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        self.metrics[name] = value


@pytest.mark.parametrize("with_contrastive", [True, False])
def test_two_view_train_module_forward_backward(
    allcap_dir: tuple[UPath, dict], with_contrastive: bool
) -> None:
    """(d) Tiny-model CPU train_batch: finite losses, gradients reach the encoder."""
    d, _ = allcap_dir
    batch = _pair_batch(d)
    torch.manual_seed(0)
    model = _tiny_model()
    config = TwoViewLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=2,  # 2 microbatches of 2
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ["worldcover", "srtm"],
            }
        ),
        masking_config=_masking_config(),
        token_exit_cfg={m: 0 for m in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
        contrastive_config=(
            LossConfig(loss_config={"type": "InfoNCE", "weight": 0.05})
            if with_contrastive
            else None
        ),
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = config.build(model, device=torch.device("cpu"))
        train_module.on_attach = MagicMock(return_value=None)  # type: ignore
        trainer = _MockTrainer()
        train_module._attach_trainer(trainer)
        train_module.train_batch(batch)

    m = trainer.metrics
    view_loss = m[f"train/{train_module.total_loss_name}"]
    assert torch.isfinite(view_loss) and torch.isfinite(m["train/total"])
    assert float(m["train/nonfinite_loss_microbatches"]) == 0
    if with_contrastive:
        con = m["train/contrastive"]
        assert torch.isfinite(con) and float(con) > 0
        torch.testing.assert_close(m["train/total"], view_loss + con)
        head = model.encoder.project_and_aggregate
        assert all(p.grad is not None for p in head.parameters())
    else:
        assert "train/contrastive" not in m
        torch.testing.assert_close(m["train/total"], view_loss)
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_choose_start_t_hint_shared_and_nearest() -> None:
    """_choose_start_t: historical draw without a hint, shared start with one.

    Without a hint it is one np.random.choice (historical). With a shared hint view a
    stores its start; view b reuses it, or the nearest valid start if it is invalid.
    """
    from olmoearth_pretrain.data.dataset import _choose_start_t

    np.random.seed(0)
    ref = int(np.random.choice([0, 1, 2, 3]))
    np.random.seed(0)
    assert _choose_start_t([0, 1, 2, 3], None) == ref
    hint: list = [None]
    np.random.seed(1)
    a = _choose_start_t([0, 1, 2, 3, 4], hint)
    assert hint[0] == a
    assert _choose_start_t([0, 1, 2, 3, 4], hint) == a
    hint = [3]
    assert _choose_start_t([0, 1, 5], hint) == 1  # |1-3| = 2 ties |5-3|; lower wins
    assert _choose_start_t([4, 7], hint) == 4


def test_two_draws_share_window_start(
    allcap_dir: tuple[UPath, dict], monkeypatch: Any
) -> None:
    """With t < 12 view b starts its window at view a's slot index whenever valid."""
    import olmoearth_pretrain.data.dataset as dsmod

    calls: list[tuple[list[int], int]] = []
    orig = dsmod._choose_start_t

    def spy(valid: list[int], hint: list | None) -> int:
        out = orig(valid, hint)
        calls.append((list(valid), out))
        return out

    monkeypatch.setattr(dsmod, "_choose_start_t", spy)
    d, _ = allcap_dir
    ds = _dataset(d, two_draws=True)
    n_checked = 0
    for seed in range(20):
        calls.clear()
        np.random.seed(seed)
        _, (a, b) = ds[_args(target_t=5)]
        assert len(calls) == 2
        (_, start_a), (valid_b, start_b) = calls
        if start_a in valid_b:
            assert start_b == start_a
            n_checked += 1
        else:
            assert start_b == min(valid_b, key=lambda s: (abs(s - start_a), s))
    assert n_checked >= 1
