"""CPU unit tests for GeoBench v2 band mapping and the opt-in naip_10 path.

All tests use fake samples (no GeoBench v2 data needed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from einops import rearrange

import olmoearth_pretrain.evals.datasets.geobench_v2_dataset as gb2
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.evals.datasets import get_eval_dataset
from olmoearth_pretrain.evals.datasets.configs import (
    GB2_NAIP10_SLUGS,
    dataset_to_config,
)
from olmoearth_pretrain.evals.datasets.geobench_v2_dataset import (
    GeobenchV2Dataset,
    HighresModality,
    HighresResample,
    gb2_dataset_name,
    nominal_naip10_size,
    parse_gb2_dataset_name,
)
from olmoearth_pretrain.evals.datasets.geobench_v2_loaders import SLUG_TO_DATASET
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn_variable_time
from olmoearth_pretrain.evals.eval_wrapper import OlmoEarthEvalWrapper
from olmoearth_pretrain.evals.finetune.evaluate import _seg_logits_to_pixel
from olmoearth_pretrain.evals.task_types import TaskType
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

S2 = list(Modality.SENTINEL2_L2A.band_order)
NAIP = list(Modality.NAIP_10.band_order)
H = 32


def _const_bands(values: list[float], h: int = H, dtype: Any = torch.float32) -> Any:
    """(C, H, W) tensor whose channel i is the constant values[i]."""
    return torch.stack([torch.full((h, h), float(v)) for v in values]).to(dtype)


def _channel_values(hwtc: torch.Tensor, names: list[str]) -> dict[str, float]:
    """Map band name -> the (constant) value of that channel."""
    return {n: float(hwtc[..., i].mean()) for i, n in enumerate(names)}


# Distinct raw values per source band so a mis-mapping is detectable.
FLAIR2_RAW = {"red": 10, "green": 20, "blue": 30, "nir": 40, "elevation": 50}
SN7_RAW = {"red": 11, "green": 22, "blue": 33, "nir": 255}  # "nir" is alpha
SN2_RAW = {
    "coastal": 100,
    "blue": 200,
    "green": 300,
    "yellow": 400,
    "red": 500,
    "red_edge": 600,
    "nir1": 700,
    "nir2": 800,
}
SN2_PAN = 999


def _fake_raw(slug: str, h: int = H) -> dict[str, torch.Tensor]:
    band_order = SLUG_TO_DATASET[slug].band_order
    if slug == "flair2":
        assert list(band_order) == list(FLAIR2_RAW)
        image = _const_bands(list(FLAIR2_RAW.values()), h, torch.uint8)
        return {"image": image, "mask": torch.zeros(h, h, dtype=torch.long)}
    if slug == "spacenet7":
        assert band_order["s2"] == list(SN7_RAW)
        image = _const_bands(list(SN7_RAW.values()), h, torch.uint8)
        return {"image": image, "mask": torch.zeros(h, h, dtype=torch.long)}
    if slug == "spacenet2":
        assert band_order["worldview"] == list(SN2_RAW)
        return {
            "image_worldview": _const_bands(list(SN2_RAW.values()), h, torch.int32),
            "image_pan": _const_bands([SN2_PAN], h, torch.int32),
            "mask": torch.zeros(h, h, dtype=torch.long),
        }
    raise ValueError(slug)


# ---------------------------------------------------------------------------
# sentinel2_l2a path: band mapping fixes
# ---------------------------------------------------------------------------


def test_flair2_s2_maps_rgbn_by_name_and_drops_elevation() -> None:
    """FLAIR2 red lands in B04 (not B02) and elevation is not fed to any band."""
    olmo = gb2._sample_to_olmoearth(
        _fake_raw("flair2"), SLUG_TO_DATASET["flair2"].band_order, "flair2"
    )
    assert olmo.sentinel2_l2a is not None
    assert olmo.sentinel2_l2a.shape == (H, H, 1, len(S2))
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    k = 10000.0 / 255.0
    assert vals["B04"] == pytest.approx(FLAIR2_RAW["red"] * k, rel=1e-5)
    assert vals["B03"] == pytest.approx(FLAIR2_RAW["green"] * k, rel=1e-5)
    assert vals["B02"] == pytest.approx(FLAIR2_RAW["blue"] * k, rel=1e-5)
    assert vals["B08"] == pytest.approx(FLAIR2_RAW["nir"] * k, rel=1e-5)
    for band in set(S2) - {"B02", "B03", "B04", "B08"}:
        assert vals[band] == 0.0, band  # incl. B05, where elevation used to go


def test_spacenet2_s2_maps_worldview_by_name_and_rescales() -> None:
    """SpaceNet2 WV-3 bands go to their S2 counterparts in S2 DN scale; pan dropped."""
    olmo = gb2._sample_to_olmoearth(
        _fake_raw("spacenet2"), SLUG_TO_DATASET["spacenet2"].band_order, "spacenet2"
    )
    assert olmo.sentinel2_l2a is not None
    assert olmo.sentinel2_l2a.shape == (H, H, 1, len(S2))
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    k = 10000.0 / 2047.0
    expected = {
        "B01": SN2_RAW["coastal"],
        "B02": SN2_RAW["blue"],
        "B03": SN2_RAW["green"],
        "B04": SN2_RAW["red"],
        "B05": SN2_RAW["red_edge"],
        "B08": SN2_RAW["nir1"],
    }
    for band in S2:
        assert vals[band] == pytest.approx(expected.get(band, 0) * k, rel=1e-5), band
    assert not torch.isclose(olmo.sentinel2_l2a, torch.tensor(SN2_PAN * k)).any()


def test_spacenet7_s2_drops_alpha_channel() -> None:
    """SpaceNet7's 4th ("nir") channel is alpha: B08 stays zero."""
    olmo = gb2._sample_to_olmoearth(
        _fake_raw("spacenet7"), SLUG_TO_DATASET["spacenet7"].band_order, "spacenet7"
    )
    assert olmo.sentinel2_l2a is not None
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    k = 10000.0 / 255.0
    assert vals["B04"] == pytest.approx(SN7_RAW["red"] * k, rel=1e-5)
    assert vals["B03"] == pytest.approx(SN7_RAW["green"] * k, rel=1e-5)
    assert vals["B02"] == pytest.approx(SN7_RAW["blue"] * k, rel=1e-5)
    assert vals["B08"] == 0.0


def test_treesatai_uses_only_s2() -> None:
    """TreeSatAI feeds only image_s2 (not aerial or S1) through sentinel2_l2a."""
    band_order = SLUG_TO_DATASET["treesatai"].band_order
    s2_names = band_order["s2"]
    raw = {
        "image_s2": _const_bands(list(range(1, len(s2_names) + 1)), 6),
        "image_aerial": _const_bands([777] * 4, 6),
        "image_s1": _const_bands([888] * 2, 6),
        "label": torch.zeros(15),
    }
    olmo = gb2._sample_to_olmoearth(raw, band_order, "treesatai")
    assert olmo.modalities == ["sentinel2_l2a"]
    assert olmo.sentinel2_l2a is not None
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    for i, name in enumerate(s2_names):
        if name in S2:
            assert vals[name] == i + 1


def test_dataset_stats_are_keyed_to_new_layout() -> None:
    """geobench2_norm_stats.json no longer has elevation/alpha/pan in S2 slots."""
    stats = gb2._load_geobench_norm_stats()
    zero = {"mean": 0.0, "std": 1.0}
    assert stats["flair2"]["sentinel2_l2a"]["B05"] == zero  # was elevation
    assert stats["spacenet7"]["sentinel2_l2a"]["B08"] == zero  # was alpha
    assert stats["spacenet2"]["sentinel2_l2a"]["B11"] == zero  # was nir2
    for slug in GB2_NAIP10_SLUGS:
        s2 = stats[slug]["sentinel2_l2a"]
        assert set(s2) == set(S2)
        assert set(stats[slug]["naip_10"]) == set(NAIP)


# ---------------------------------------------------------------------------
# naip_10 path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slug,expected",
    [
        ("flair2", [10, 20, 30, 40]),
        ("spacenet7", [11, 22, 33, 0]),  # IR left empty (alpha)
        ("spacenet2", [v * 255 / 2047 for v in (500, 300, 200, 700)]),
    ],
)
def test_naip10_mapping(slug: str, expected: list[float]) -> None:
    """RGBN lands in naip_10 as (H, W, 1, 4) R, G, B, IR in uint8 scale."""
    olmo = gb2._sample_to_naip10(
        _fake_raw(slug), SLUG_TO_DATASET[slug].band_order, slug
    )
    assert olmo.modalities == ["naip_10"]
    assert olmo.naip_10 is not None
    assert olmo.naip_10.shape == (H, H, 1, 4)
    assert olmo.timestamps is not None and olmo.timestamps.shape == (1, 3)
    vals = _channel_values(olmo.naip_10, NAIP)
    for name, exp in zip(NAIP, expected):
        assert vals[name] == pytest.approx(exp, rel=1e-5), name


def test_naip10_unsupported_slug_raises() -> None:
    """Only the three sub-10 m slugs have a naip_10 mapping."""
    with pytest.raises(ValueError):
        gb2._sample_to_naip10({"image": torch.zeros(3, 4, 4)}, ["a"], "fotw")


def test_dataset_name_roundtrip() -> None:
    """Dataset-name suffixes encode the highres options."""
    cases = {
        "gb2-flair2": (HighresModality.SENTINEL2_L2A, HighresResample.NATIVE),
        "gb2-flair2-naip10": (HighresModality.NAIP_10, HighresResample.NATIVE),
        "gb2-flair2-naip10-nominal": (
            HighresModality.NAIP_10,
            HighresResample.NOMINAL,
        ),
        "gb2-burn_scars": (HighresModality.SENTINEL2_L2A, HighresResample.NATIVE),
    }
    for name, (mod, res) in cases.items():
        slug, m, r = parse_gb2_dataset_name(name)
        assert (m, r) == (mod, res)
        assert gb2_dataset_name(slug, m, r) == name
    with pytest.raises(ValueError):
        parse_gb2_dataset_name("gb2-flair2-naip")
    with pytest.raises(ValueError):
        gb2_dataset_name("flair2", "sentinel2_l2a", "nominal")


def test_naip10_configs() -> None:
    """Variants differ from the S2 config only in supported_modalities."""
    for slug in GB2_NAIP10_SLUGS:
        base = dataset_to_config(f"gb2-{slug}")
        for suffix in ("naip10", "naip10-nominal"):
            cfg = dataset_to_config(f"gb2-{slug}-{suffix}")
            assert cfg.supported_modalities == [Modality.NAIP_10.name]
            assert cfg.task_type == base.task_type
            assert cfg.num_classes == base.num_classes
            assert cfg.height_width == base.height_width
        assert base.supported_modalities == [Modality.SENTINEL2_L2A.name]


@pytest.mark.parametrize(
    "h,gsd,multiple,expected",
    [
        (512, 0.2, 4, 40),  # flair2, ps1
        (512, 0.3 * 650 / 512, 4, 80),  # spacenet2, ps1
        (512, 4.777, 16, 976),  # spacenet7, ps4 (upsampled)
        (512, 0.2, 16, 48),  # flair2, ps4
        (8, 0.2, 4, 4),  # never below one tile
    ],
)
def test_nominal_size(h: int, gsd: float, multiple: int, expected: int) -> None:
    """Nominal size ~= H * gsd / 2.5, rounded to the token multiple."""
    size = nominal_naip10_size(h, gsd, multiple)
    assert size == expected
    assert size % multiple == 0


class _FakeLoader:
    """Stands in for a GeoBench v2 tortilla loader."""

    def __init__(self, slug: str, h: int) -> None:
        self.slug = slug
        self.h = h
        self.band_order = SLUG_TO_DATASET[slug].band_order

    def __call__(self, root: str, split: str) -> _FakeLoader:
        return self

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _fake_raw(self.slug, self.h)


@pytest.fixture
def fake_loaders(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch SLUG_TO_DATASET with fake loaders of a given image size."""

    def install(h: int) -> None:
        for slug in GB2_NAIP10_SLUGS:
            monkeypatch.setitem(gb2.SLUG_TO_DATASET, slug, _FakeLoader(slug, h))

    return install


@pytest.mark.parametrize("slug", GB2_NAIP10_SLUGS)
@pytest.mark.parametrize("pretrained", [True, False])
def test_dataset_naip10_native(slug: str, pretrained: bool, fake_loaders: Any) -> None:
    """End-to-end naip_10 native sample: shape, normalization, re-zeroed IR."""
    fake_loaders(64)
    ds = get_eval_dataset(
        f"gb2-{slug}-naip10",
        "train",
        norm_stats_from_pretrained=pretrained,
        input_modalities=[Modality.NAIP_10.name],
        patch_size=4,
    )
    assert isinstance(ds, GeobenchV2Dataset)
    assert ds.config.supported_modalities == [Modality.NAIP_10.name]
    masked, label = ds[0]
    assert isinstance(masked, MaskedOlmoEarthSample)
    assert masked.modalities == ["naip_10"]
    assert masked.naip_10 is not None and masked.naip_10_mask is not None
    assert masked.naip_10.shape == (64, 64, 1, 4)
    assert masked.naip_10_mask.shape == (64, 64, 1, 4)
    assert label.shape == (64, 64)

    raw = gb2._sample_to_naip10(_fake_raw(slug, 64), ds._band_order, slug)
    assert raw.naip_10 is not None
    if pretrained:
        expected = Normalizer(Strategy.COMPUTED).normalize(
            Modality.NAIP_10, raw.naip_10.numpy()
        )
        if slug == "spacenet7":
            expected[..., 3] = 0.0  # empty IR is re-zeroed
        np.testing.assert_allclose(masked.naip_10.numpy(), expected, rtol=1e-5)
    if slug == "spacenet7":
        assert (masked.naip_10[..., 3] == 0).all()
    else:
        assert (masked.naip_10 != 0).all()


def test_dataset_naip10_nominal_resamples_image_not_label(fake_loaders: Any) -> None:
    """Nominal mode resizes the image to 2.5 m/px; the label keeps its size."""
    fake_loaders(512)
    ds = GeobenchV2Dataset(
        "gb2-flair2-naip10-nominal", "train", "", norm_stats_from_pretrained=True
    )
    masked, label = ds[0]
    assert masked.naip_10 is not None
    assert masked.naip_10.shape == (40, 40, 1, 4)
    assert label.shape == (512, 512)
    ds = GeobenchV2Dataset(
        "gb2-flair2", "train", "", highres_modality="naip_10",
        highres_resample="nominal", patch_size=4,
    )  # fmt: skip
    assert ds.config.supported_modalities == [Modality.NAIP_10.name]
    masked, _ = ds[0]
    assert masked.naip_10 is not None
    assert masked.naip_10.shape == (48, 48, 1, 4)


def test_dataset_native_requires_divisible_size(fake_loaders: Any) -> None:
    """Native naip_10 H, W must be divisible by 4 * patch_size."""
    fake_loaders(40)
    ds = GeobenchV2Dataset("gb2-flair2-naip10", "train", "", patch_size=4)
    with pytest.raises(ValueError, match="not divisible"):
        ds[0]


def test_dataset_option_validation(fake_loaders: Any) -> None:
    """Invalid or conflicting highres options are rejected."""
    fake_loaders(64)
    with pytest.raises(ValueError):
        GeobenchV2Dataset("gb2-flair2-naip10", "train", "", highres_resample="nominal")
    with pytest.raises(ValueError):
        GeobenchV2Dataset("gb2-flair2", "train", "", highres_resample="nominal")
    with pytest.raises(ValueError):
        GeobenchV2Dataset("gb2-fotw", "train", "", highres_modality="naip_10")


def test_dataset_s2_default_unchanged_and_rezeroed(fake_loaders: Any) -> None:
    """Default S2 path: missing S2 channels stay exactly zero after normalization."""
    fake_loaders(64)
    for pretrained in (True, False):
        ds = GeobenchV2Dataset(
            "gb2-flair2", "train", "", norm_stats_from_pretrained=pretrained
        )
        assert ds.config.supported_modalities == [Modality.SENTINEL2_L2A.name]
        masked, _ = ds[0]
        assert masked.modalities == ["sentinel2_l2a"]
        assert masked.sentinel2_l2a is not None
        x = masked.sentinel2_l2a
        for i, band in enumerate(S2):
            if band in ("B02", "B03", "B04", "B08"):
                assert (x[..., i] != 0).all(), band
            else:
                assert (x[..., i] == 0).all(), band


# ---------------------------------------------------------------------------
# Model path: naip_10-only batches through the encoder, probe and FT heads
# ---------------------------------------------------------------------------


def _naip10_encoder() -> torch.nn.Module:
    return EncoderConfig(
        supported_modality_names=[Modality.NAIP_10.name],
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=1.0,
        depth=1,
        drop_path=0.0,
        max_sequence_length=12,
    ).build()


@pytest.mark.parametrize("patch_size,h", [(4, 64), (1, 40)])
def test_naip10_batch_token_grid(patch_size: int, h: int) -> None:
    """A naip_10-only batch yields an (H / (4 * ps))^2 token grid; heads upsample."""
    ts = torch.tensor([[15, 6, 2020]])
    items = []
    for _ in range(2):
        sample = gb2.OlmoEarthSample(naip_10=torch.rand(h, h, 1, 4), timestamps=ts)
        items.append(
            (MaskedOlmoEarthSample.from_olmoearthsample(sample), torch.zeros(h, h))
        )
    batch, labels = eval_collate_fn_variable_time(items)
    wrapper = OlmoEarthEvalWrapper(
        model=_naip10_encoder(),
        task_type=TaskType.SEGMENTATION,
        patch_size=patch_size,
        pooling_type=PoolingType.MEAN,
    )
    wrapper.eval()
    with torch.no_grad():
        emb, _ = wrapper(batch, labels)
    grid = h // (Modality.NAIP_10.image_tile_size_factor * patch_size)
    assert emb.shape == (2, grid, grid, 16)

    # Finetune head: num_classes * ps^2 logits per token, pixel-shuffled then
    # resized to the label grid.
    num_classes = 3
    logits = torch.randn(2, grid, grid, num_classes * patch_size**2)
    label = torch.zeros(2, 512, 512)
    out = _seg_logits_to_pixel(logits, label, False, num_classes, patch_size)
    assert out.shape == (2, num_classes, 512, 512)


def test_linear_probe_small_token_grid() -> None:
    """Linear probe sizes its per-token output from height_width / token grid."""
    from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig
    from olmoearth_pretrain.evals.linear_probe import train_and_eval_probe

    # e.g. gb2-flair2-naip10-nominal at ps1: 40 px -> 10 x 10 tokens, 512 px label
    # (scaled down here: 4 x 4 tokens, 30 px label -> 7 px/token -> resized).
    n, grid, hw, d = 8, 4, 30, 8
    cfg = EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=hw,
        supported_modalities=[Modality.NAIP_10.name],
    )
    emb = torch.rand(n, grid, grid, d)
    lbl = torch.randint(0, 2, (n, hw, hw))
    result = train_and_eval_probe(
        config=cfg,
        train_embeddings=emb,
        train_labels=lbl,
        val_embeddings=emb,
        val_labels=lbl,
        test_embeddings=None,
        test_labels=None,
        device=torch.device("cpu"),
        batch_size=n,
        lr=0.1,
        epochs=1,
    )
    assert result.val_result is not None
    assert "miou" in result.val_result.metrics


def test_all_evals_has_naip10_tasks() -> None:
    """LP and FT task tables both carry the naip_10 variants."""
    pytest.importorskip("beaker")
    from olmoearth_pretrain.internal.all_evals import EVAL_TASKS, FT_EVAL_TASKS

    for tasks in (EVAL_TASKS, FT_EVAL_TASKS):
        for slug in GB2_NAIP10_SLUGS:
            for suffix, dataset_suffix in (
                ("naip10", "naip10"),
                ("naip10_nominal", "naip10-nominal"),
            ):
                task = tasks[f"gb2_{slug}_{suffix}"]
                assert task.dataset == f"gb2-{slug}-{dataset_suffix}"
                assert task.input_modalities == [Modality.NAIP_10.name]
                assert set(task.input_modalities) <= set(
                    dataset_to_config(task.dataset).supported_modalities
                )
            # Defaults of the S2 tasks are unchanged.
            assert tasks[f"gb2_{slug}"].input_modalities == [
                Modality.SENTINEL2_L2A.name
            ]
        assert tasks["gb2_flair2_naip10_nominal"].patch_size == 1


def test_rearrange_sanity() -> None:
    """_bchw_to_hwtc keeps channel order (guards the naip_10 band order)."""
    x = _const_bands([1, 2, 3, 4], 2)
    assert rearrange(x, "c h w -> h w 1 c")[0, 0, 0].tolist() == [1, 2, 3, 4]
    assert gb2._bchw_to_hwtc(x)[0, 0, 0].tolist() == [1, 2, 3, 4]
