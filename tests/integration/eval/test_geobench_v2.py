"""Integration tests for GeoBench v2 dataloaders, KNN, linear probe, and finetune."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.datasets.geobench_v2_dataset import GeobenchV2Dataset
from olmoearth_pretrain.evals.datasets.geobench_v2_loaders import SLUG_TO_DATASET
from olmoearth_pretrain.evals.datasets.paths import GEOBENCH2_DIR
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.eval_wrapper import OlmoEarthEvalWrapper
from olmoearth_pretrain.evals.finetune.train import run_finetune_eval
from olmoearth_pretrain.evals.knn import run_knn
from olmoearth_pretrain.evals.linear_probe import train_and_eval_probe
from olmoearth_pretrain.evals.metrics import EvalTaskResult
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_encoder(modality_names: list[str]) -> torch.nn.Module:
    cfg = EncoderConfig(
        supported_modality_names=modality_names,
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=12,
    )
    return cfg.build()


# ---------------------------------------------------------------------------
# Raw loader smoke tests — one sample from test split for every slug
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", list(SLUG_TO_DATASET.keys()))
def test_raw_loader_one_sample(slug: str) -> None:
    """Verify each raw loader can load exactly one sample from the test split."""
    root = str(GEOBENCH2_DIR / slug)
    loader_cls = SLUG_TO_DATASET[slug]
    ds = loader_cls(root=root, split="test")
    assert len(ds) > 0, f"{slug}: test split is empty"

    sample = ds[0]
    assert isinstance(sample, dict), f"{slug}: __getitem__ should return dict"
    # Every sample must have at least one tensor value
    tensors = [v for v in sample.values() if torch.is_tensor(v)]
    assert tensors, f"{slug}: no tensor values in sample"


# ---------------------------------------------------------------------------
# GeobenchV2Dataset wrapper — checks normalisation + masking pipeline
# ---------------------------------------------------------------------------

_WRAPPER_SLUGS = [
    "burn_scars",
    "kuro_siwo",
    "benv2",
    "biomassters",
    "treesatai",
    "caffe",
    "cloudsen12",
    "spacenet2",
    "spacenet7",
]


@pytest.mark.parametrize("slug", _WRAPPER_SLUGS)
def test_geobenchv2_dataset_wrapper(slug: str) -> None:
    """GeobenchV2Dataset returns (MaskedOlmoEarthSample, label) with sensible shapes."""
    dataset_name = f"gb2-{slug}"
    ds = GeobenchV2Dataset(dataset=dataset_name, split="test", partition="")
    assert len(ds) > 0, f"{slug}: wrapped test split is empty"

    masked, label = ds[0]
    assert isinstance(masked, MaskedOlmoEarthSample)
    assert isinstance(label, torch.Tensor)
    assert label.dtype in (torch.long, torch.float32, torch.float)


# ---------------------------------------------------------------------------
# KNN — uses GeobenchV2Dataset embeddings via a tiny encoder
# ---------------------------------------------------------------------------


def test_knn_with_real_data() -> None:
    """KNN runs end-to-end on a small slice of burn_scars (classification-like)."""
    device = torch.device("cpu")
    ds = GeobenchV2Dataset(dataset="gb2-burn_scars", split="train", partition="")
    # Take a tiny subset so the test is fast
    n = min(16, len(ds))
    loader = DataLoader(
        torch.utils.data.Subset(ds, list(range(n))),
        batch_size=4,
        collate_fn=eval_collate_fn,
    )

    encoder = _make_encoder([Modality.SENTINEL2_L2A.name])
    encoder.eval()

    wrapper = OlmoEarthEvalWrapper(
        model=encoder,
        task_type=TaskType.CLASSIFICATION,
        patch_size=4,
        pooling_type=PoolingType.MEAN,
        concat_features=False,
        use_pooled_tokens=False,
    )
    wrapper.eval()

    embeddings, labels = [], []
    with torch.no_grad():
        for batch, lbl in loader:
            emb, _ = wrapper(batch, lbl)  # emb: (B, D)
            embeddings.append(emb)
            # Flatten spatial labels to scalar via mode for this smoke test
            if lbl.dim() > 1:
                lbl = lbl.reshape(lbl.shape[0], -1).mode(dim=1).values
            labels.append(lbl.long())

    emb_all = torch.cat(embeddings)  # (N, D)
    lbl_all = torch.cat(labels)  # (N,)

    # Use half for train, half for val
    mid = max(1, n // 2)
    config = EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    )
    result = run_knn(
        config=config,
        train_embeddings=emb_all[:mid],
        train_labels=lbl_all[:mid],
        val_embeddings=emb_all[mid:],
        val_labels=lbl_all[mid:],
        test_embeddings=None,
        test_labels=None,
        device=device,
        k=min(5, mid),
    )
    assert isinstance(result, EvalTaskResult)
    assert result.val_result is not None


# ---------------------------------------------------------------------------
# Linear probe — uses synthetic embeddings (fast, no GPU needed)
# ---------------------------------------------------------------------------


def test_linear_probe_segmentation() -> None:
    """Linear probe for segmentation runs and returns EvalTaskResult."""
    device = torch.device("cpu")
    B, H, W, D, P = 32, 8, 8, 16, 4

    train_emb = torch.rand(B, H // P, W // P, D)
    val_emb = torch.rand(B, H // P, W // P, D)
    test_emb = torch.rand(B, H // P, W // P, D)
    train_lbl = torch.randint(0, 2, (B, H, W))
    val_lbl = torch.randint(0, 2, (B, H, W))
    test_lbl = torch.randint(0, 2, (B, H, W))

    config = EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=H,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    )
    result = train_and_eval_probe(
        config=config,
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        test_embeddings=test_emb,
        test_labels=test_lbl,
        device=device,
        batch_size=B,
        lr=0.1,
    )
    assert isinstance(result, EvalTaskResult)
    assert result.val_result is not None
    assert result.test_result is not None
    assert "miou" in result.val_result.metrics


def test_linear_probe_regression() -> None:
    """Linear probe for regression runs and returns EvalTaskResult with r2."""
    device = torch.device("cpu")
    B, H, W, D, P = 32, 8, 8, 16, 4

    train_emb = torch.rand(B, H // P, W // P, D)
    val_emb = torch.rand(B, H // P, W // P, D)
    test_emb = torch.rand(B, H // P, W // P, D)
    train_lbl = torch.rand(B, H, W)
    val_lbl = torch.rand(B, H, W)
    test_lbl = torch.rand(B, H, W)

    config = EvalDatasetConfig(
        task_type=TaskType.REGRESSION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        height_width=H,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    )
    result = train_and_eval_probe(
        config=config,
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        test_embeddings=test_emb,
        test_labels=test_lbl,
        device=device,
        batch_size=B,
        lr=0.1,
    )
    assert isinstance(result, EvalTaskResult)
    assert result.val_result is not None
    assert "r2" in result.val_result.metrics


# ---------------------------------------------------------------------------
# Finetune — one epoch on a tiny synthetic dataset
# ---------------------------------------------------------------------------


class _TinyGB2Dataset(torch.utils.data.Dataset):
    """Wraps the first N samples of a GeobenchV2Dataset for speed."""

    def __init__(self, slug: str, split: str, n: int = 4) -> None:
        self._inner = GeobenchV2Dataset(
            dataset=f"gb2-{slug}", split=split, partition=""
        )
        self._n = min(n, len(self._inner))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:  # type: ignore[override]
        return self._inner[idx]


def _make_loader(slug: str, split: str, batch_size: int = 2) -> DataLoader:
    return DataLoader(
        _TinyGB2Dataset(slug, split),
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
    )


@pytest.mark.parametrize(
    "slug,task_type,num_classes,modalities",
    [
        ("burn_scars", TaskType.SEGMENTATION, 2, [Modality.SENTINEL2_L2A.name]),
        (
            "kuro_siwo",
            TaskType.SEGMENTATION,
            3,
            [Modality.SENTINEL1.name, Modality.SRTM.name],
        ),
        ("benv2", TaskType.CLASSIFICATION, 19, [Modality.SENTINEL2_L2A.name]),
    ],
)
def test_finetune_one_epoch(
    slug: str,
    task_type: TaskType,
    num_classes: int,
    modalities: list[str],
    tmp_path: Path,
) -> None:
    """run_finetune_eval completes one epoch without error."""
    device = torch.device("cpu")
    encoder = _make_encoder(modalities)

    config = EvalDatasetConfig(
        task_type=task_type,
        imputes=[],
        num_classes=num_classes,
        is_multilabel=(slug == "benv2"),
        height_width=None if task_type == TaskType.CLASSIFICATION else 16,
        supported_modalities=modalities,
    )

    train_loader = _make_loader(slug, "train", batch_size=2)
    val_loader = _make_loader(slug, "test", batch_size=2)

    # Trainer stub — run_finetune_eval only uses trainer for wandb logging
    class _NoopTrainer:
        def log(self, *a: object, **kw: object) -> None:
            pass

        def _iter_callbacks(self) -> Iterator[object]:
            return iter([])

    result = run_finetune_eval(
        task_name=f"gb2_{slug}",
        task_config=config,
        trainer=_NoopTrainer(),
        model=encoder,
        device=device,
        lr=1e-3,
        epochs=1,
        patch_size=4,
        pooling_type=PoolingType.MEAN,
        use_pooled_tokens=False,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
    )
    assert isinstance(result, EvalTaskResult)
    assert result.val_result is not None
