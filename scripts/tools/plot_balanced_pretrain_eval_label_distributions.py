"""Cache and plot label distributions for balanced pretrain map-probe evals."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from olmoearth_pretrain.evals.datasets.pretrain_subset import (
    PretrainSplitStrategy,
    PretrainSubsetDataset,
)

OSM_SAMPLING_H5PY_DIR = Path(
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)
OSMBIG_H5PY_DIR = Path(
    "/weka/dfive-default/helios/dataset/osmbig/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1297928"
)
PRESTO_H5PY_DIR = Path(
    "/weka/dfive-default/helios/dataset/presto/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_wri_canopy_height_map/469728"
)

SPLIT_TO_SAMPLES = {
    "train": 6144,
    "valid": 3072,
    "test": 3072,
}

WORLDCOVER_CLASSES = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
CANOPY_EDGES = np.asarray([0.0, 1.0, 5.0, 10.0, 20.0, 40.0], dtype=np.float32)
SRTM_EDGES = np.asarray([0.0, 250.0, 500.0, 1000.0, 1500.0, 2500.0], dtype=np.float32)


@dataclass(frozen=True)
class DatasetSpec:
    """An H5 dataset root to summarize."""

    key: str
    h5py_dir: Path
    targets: tuple[str, ...]


@dataclass(frozen=True)
class EvalSpec:
    """A balanced eval label distribution to plot."""

    name: str
    dataset_key: str
    target: str
    kind: str


DATASET_SPECS = {
    "osmbig": DatasetSpec(
        key="osmbig",
        h5py_dir=OSMBIG_H5PY_DIR,
        targets=("worldcover", "openstreetmap_raster", "srtm"),
    ),
    "osm_sampling": DatasetSpec(
        key="osm_sampling",
        h5py_dir=OSM_SAMPLING_H5PY_DIR,
        targets=("wri_canopy_height_map", "cdl", "worldcereal"),
    ),
    "presto": DatasetSpec(
        key="presto",
        h5py_dir=PRESTO_H5PY_DIR,
        targets=(
            "worldcover",
            "openstreetmap_raster",
            "srtm",
            "wri_canopy_height_map",
            "cdl",
            "worldcereal",
        ),
    ),
}

EVAL_SPECS = [
    EvalSpec("pretrain_worldcover_probe", "osmbig", "worldcover", "class"),
    EvalSpec("pretrain_osm_probe", "osmbig", "openstreetmap_raster", "class"),
    EvalSpec("pretrain_srtm_regression", "osmbig", "srtm", "regression"),
    EvalSpec(
        "pretrain_canopy_regression",
        "osm_sampling",
        "wri_canopy_height_map",
        "regression",
    ),
    EvalSpec("pretrain_cdl_probe", "osm_sampling", "cdl", "class"),
    EvalSpec("pretrain_worldcereal_probe", "osm_sampling", "worldcereal", "class"),
    EvalSpec("presto_worldcover_probe", "presto", "worldcover", "class"),
    EvalSpec("presto_osm_probe", "presto", "openstreetmap_raster", "class"),
    EvalSpec("presto_srtm_regression", "presto", "srtm", "regression"),
    EvalSpec(
        "presto_canopy_regression",
        "presto",
        "wri_canopy_height_map",
        "regression",
    ),
    EvalSpec("presto_cdl_probe", "presto", "cdl", "class"),
    EvalSpec("presto_worldcereal_probe", "presto", "worldcereal", "class"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/balanced_pretrain_eval_label_distributions"),
    )
    parser.add_argument(
        "--sample-scale",
        type=float,
        default=1.0,
        help="Scale split sample counts for diagnostic plotting.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            PretrainSplitStrategy.BALANCED.value,
            PretrainSplitStrategy.BALANCED_GEOGRAPHIC.value,
        ],
        default=[
            PretrainSplitStrategy.BALANCED.value,
            PretrainSplitStrategy.BALANCED_GEOGRAPHIC.value,
        ],
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Parallel backend for the H5 cache scan.",
    )
    parser.add_argument(
        "--cache-max-samples",
        type=int,
        default=None,
        help="Optional deterministic cap per target cache for quick diagnostics.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only build per-sample label summary caches; skip plotting.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Regenerate caches even when cache files already exist.",
    )
    return parser.parse_args()


def split_sample_counts(sample_scale: float) -> dict[str, int]:
    """Return scaled split sample counts."""
    if sample_scale <= 0:
        raise ValueError("--sample-scale must be positive")
    return {
        split: max(1, int(samples * sample_scale))
        for split, samples in SPLIT_TO_SAMPLES.items()
    }


def cache_path(output_dir: Path, dataset_key: str, target: str) -> Path:
    """Return the cache path for one dataset target."""
    return output_dir / "cache" / f"{dataset_key}_{target}_label_summary.npz"


def h5_path(h5py_dir: Path, index: int) -> Path:
    """Return a sample H5 path."""
    return h5py_dir / f"sample_{index}.h5"


def read_raw_label(h5py_dir: Path, index: int, target: str) -> np.ndarray:
    """Read one raw target array from an H5 sample."""
    with h5py.File(h5_path(h5py_dir, index), "r") as h5_file:
        return np.asarray(h5_file[target])


def transform_label(raw: np.ndarray, target: str) -> np.ndarray:
    """Match pretrain-subset label transforms for plotting."""
    label = np.squeeze(raw)
    if target == "worldcover":
        if label.size and label.min() >= 0 and label.max() < len(WORLDCOVER_CLASSES):
            return label.astype(np.int64)
        mapped = np.full(label.shape, -1, dtype=np.int64)
        for class_idx, class_code in enumerate(WORLDCOVER_CLASSES):
            mapped[label == class_code] = class_idx
        return mapped
    if target == "openstreetmap_raster":
        channels_last = np.moveaxis(label, 0, -1) if label.shape[0] in (29, 30) else label
        valid = channels_last.sum(axis=-1) > 0
        classes = channels_last.argmax(axis=-1).astype(np.int64)
        classes[~valid] = -1
        return classes
    if target == "cdl":
        label = label.astype(np.int64)
        label[label == 0] = -1
        return label
    if target == "worldcereal":
        channels_last = np.moveaxis(label, 0, -1) if label.shape[0] == 8 else label
        valid = channels_last.sum(axis=-1) > 0
        classes = (channels_last[..., 0] > 0).astype(np.int64)
        classes[~valid] = -1
        return classes
    return label.astype(np.float32)


def label_distribution(label: np.ndarray, target: str) -> dict[int, int]:
    """Return class/bin counts for one transformed label tile."""
    if target == "srtm":
        values = np.digitize(label[np.isfinite(label)], SRTM_EDGES, right=True)
    elif target == "wri_canopy_height_map":
        values = np.digitize(label[np.isfinite(label)], CANOPY_EDGES, right=True)
    else:
        values = label.reshape(-1)
        values = values[values != -1]
    if values.size == 0:
        return {}
    labels, counts = np.unique(values.astype(np.int64), return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist(), strict=True))


def eligible_indices(dataset: DatasetSpec, target: str) -> np.ndarray:
    """Return H5 sample indices with sentinel2_l2a and target present."""
    metadata = pd.read_csv(dataset.h5py_dir / "sample_metadata.csv")
    mask = (metadata["sentinel2_l2a"].to_numpy() > 0) & (
        metadata[target].to_numpy() > 0
    )
    return metadata.loc[mask, "sample_index"].to_numpy(dtype=np.int64)


def cache_indices(indices: np.ndarray, max_samples: int | None, seed: int = 42) -> np.ndarray:
    """Optionally cap cache indices for quick diagnostic runs."""
    if max_samples is None or len(indices) <= max_samples:
        return indices
    return np.random.RandomState(seed).permutation(indices)[:max_samples]


def summarize_index(dataset: DatasetSpec, target: str, index: int) -> tuple[int, dict[int, int]]:
    """Read one sample and summarize its label/bin counts."""
    label = transform_label(read_raw_label(dataset.h5py_dir, index, target), target)
    return int(index), label_distribution(label, target)


def build_label_summary_cache(
    dataset: DatasetSpec,
    target: str,
    output_dir: Path,
    workers: int,
    max_samples: int | None,
    executor_name: Literal["process", "thread"],
    overwrite: bool,
) -> Path:
    """Cache sparse per-sample class/bin counts for one dataset target."""
    path = cache_path(output_dir, dataset.key, target)
    if path.exists() and not overwrite:
        print(f"using existing cache: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    indices = cache_indices(eligible_indices(dataset, target), max_samples=max_samples)
    executor_cls = ProcessPoolExecutor if executor_name == "process" else ThreadPoolExecutor

    sample_indices: list[int] = []
    label_ids: list[int] = []
    counts: list[int] = []
    row_offsets = [0]
    with executor_cls(max_workers=workers) as executor:
        futures = executor.map(
            summarize_index_for_pool,
            [(dataset, target, int(sample_index)) for sample_index in indices.tolist()],
            chunksize=16 if executor_name == "process" else 1,
        )
        for index, label_counts in tqdm(
            futures,
            total=len(indices),
            desc=f"cache {dataset.key}/{target}",
            unit="sample",
        ):
            sample_indices.append(index)
            for label_id, count in sorted(label_counts.items()):
                label_ids.append(label_id)
                counts.append(count)
            row_offsets.append(len(label_ids))

    np.savez_compressed(
        path,
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        row_offsets=np.asarray(row_offsets, dtype=np.int64),
        label_ids=np.asarray(label_ids, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        source_h5py_dir=str(dataset.h5py_dir),
        target=target,
    )
    print(f"wrote cache: {path}")
    return path


def summarize_index_for_pool(
    args: tuple[DatasetSpec, str, int],
) -> tuple[int, dict[int, int]]:
    """Pickle-friendly wrapper for parallel cache scans."""
    dataset, target, index = args
    return summarize_index(dataset, target, index)


class LabelSummaryCache:
    """Loaded sparse per-sample label/bin counts."""

    def __init__(self, path: Path) -> None:
        """Load cached summary arrays."""
        data = np.load(path, allow_pickle=False)
        self.sample_indices = data["sample_indices"]
        self.row_offsets = data["row_offsets"]
        self.label_ids = data["label_ids"]
        self.counts = data["counts"]
        self.index_to_row = {
            int(sample_index): row_idx
            for row_idx, sample_index in enumerate(self.sample_indices.tolist())
        }

    def labels_for_index(self, index: int) -> np.ndarray:
        """Return labels/bins with nonzero counts for one sample."""
        row_idx = self.index_to_row.get(int(index))
        if row_idx is None:
            return np.asarray([], dtype=np.int64)
        start = self.row_offsets[row_idx]
        end = self.row_offsets[row_idx + 1]
        return self.label_ids[start:end]

    def counts_for_indices(self, indices: np.ndarray) -> dict[int, int]:
        """Aggregate label/bin counts over selected samples."""
        total: dict[int, int] = {}
        for index in indices.tolist():
            row_idx = self.index_to_row.get(int(index))
            if row_idx is None:
                continue
            start = self.row_offsets[row_idx]
            end = self.row_offsets[row_idx + 1]
            for label_id, count in zip(
                self.label_ids[start:end].tolist(),
                self.counts[start:end].tolist(),
                strict=True,
            ):
                total[int(label_id)] = total.get(int(label_id), 0) + int(count)
        return total


def random_split_pool(indices: np.ndarray, split: str, seed: int = 42) -> np.ndarray:
    """Return uncapped random split pool."""
    shuffled = np.random.RandomState(seed).permutation(indices)
    train_end = int(len(shuffled) * 0.8)
    valid_end = train_end + int(len(shuffled) * 0.1)
    if split == "train":
        return shuffled[:train_end]
    if split in ("valid", "val"):
        return shuffled[train_end:valid_end]
    if split == "test":
        return shuffled[valid_end:]
    raise ValueError(f"Unsupported split: {split}")


def geographic_split_pool(
    indices: np.ndarray,
    latlons: np.ndarray,
    split: str,
    seed: int = 42,
    bin_size_deg: float = 5.0,
) -> np.ndarray:
    """Return uncapped geographic split pool."""
    sample_latlons = latlons[indices]
    lat_bin = np.floor(sample_latlons[:, 0] / bin_size_deg).astype(np.int64)
    lon_bin = np.floor(sample_latlons[:, 1] / bin_size_deg).astype(np.int64)
    unique_bins, inverse = np.unique(
        np.stack([lat_bin, lon_bin], axis=1),
        axis=0,
        return_inverse=True,
    )
    rng = np.random.RandomState(seed)
    bin_order = rng.permutation(len(unique_bins))
    train_end = int(len(unique_bins) * 0.8)
    valid_end = train_end + int(len(unique_bins) * 0.1)
    bucket_for_bin = np.full(len(unique_bins), "test", dtype=object)
    bucket_for_bin[bin_order[:train_end]] = "train"
    bucket_for_bin[bin_order[train_end:valid_end]] = "valid"
    normalized_split = "valid" if split == "val" else split
    return indices[bucket_for_bin[inverse] == normalized_split]


def split_pool(
    dataset: DatasetSpec,
    cache: LabelSummaryCache,
    strategy: PretrainSplitStrategy,
    split: str,
) -> np.ndarray:
    """Return split pool for cached sample indices."""
    if strategy == PretrainSplitStrategy.BALANCED_GEOGRAPHIC:
        latlons = np.load(dataset.h5py_dir / "latlon_distribution.npy")
        return geographic_split_pool(cache.sample_indices, latlons, split)
    return random_split_pool(cache.sample_indices, split)


def select_balanced_indices(
    dataset: DatasetSpec,
    cache: LabelSummaryCache,
    strategy: PretrainSplitStrategy,
    split: str,
    target_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Select balanced indices from cached label summaries."""
    pool = split_pool(dataset, cache, strategy, split)
    candidate_pool = PretrainSubsetDataset._cap_balance_candidates(
        candidate_positions=pool,
        target_size=target_size,
        seed=seed,
    )
    bins_by_index = {
        int(index): cache.labels_for_index(int(index))
        for index in candidate_pool.tolist()
    }
    return PretrainSubsetDataset._select_balanced_positions(
        candidate_positions=candidate_pool,
        balance_bins_by_position=bins_by_index,
        target_size=target_size,
        seed=seed,
    )


def plot_counts(
    spec: EvalSpec,
    strategy: PretrainSplitStrategy,
    split_counts: dict[str, dict[int, int]],
    output_path: Path,
    sample_scale: float,
) -> None:
    """Plot per-split label counts."""
    labels = sorted({label for counts in split_counts.values() for label in counts})
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.25), 5))
    for offset, split in zip([-width, 0, width], ["train", "valid", "test"], strict=True):
        values = [split_counts[split].get(label, 0) for label in labels]
        ax.bar(x + offset, values, width=width, label=split)

    ax.set_title(f"{spec.name} {strategy.value} label distribution")
    ax.set_xlabel("Class id" if spec.kind == "class" else "Regression bin id")
    ax.set_ylabel("Valid pixel count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels], rotation=90)
    ax.set_yscale("log")
    ax.legend()
    ax.text(
        0.01,
        0.99,
        f"Source: {spec.dataset_key} H5; sample_scale={sample_scale}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Cache per-sample summaries and generate balanced-eval plots."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_counts = split_sample_counts(args.sample_scale)
    requested_datasets = set(args.datasets)

    caches: dict[tuple[str, str], LabelSummaryCache] = {}
    for dataset_key in requested_datasets:
        dataset = DATASET_SPECS[dataset_key]
        for target in dataset.targets:
            path = build_label_summary_cache(
                dataset=dataset,
                target=target,
                output_dir=args.output_dir,
                workers=args.workers,
                max_samples=args.cache_max_samples,
                executor_name=args.executor,
                overwrite=args.overwrite_cache,
            )
            caches[(dataset_key, target)] = LabelSummaryCache(path)

    if args.cache_only:
        return

    summary = {
        "sample_scale": args.sample_scale,
        "cache_max_samples": args.cache_max_samples,
        "split_sample_counts": sample_counts,
        "plots": [],
    }
    for spec in EVAL_SPECS:
        if spec.dataset_key not in requested_datasets:
            continue
        dataset = DATASET_SPECS[spec.dataset_key]
        cache = caches[(spec.dataset_key, spec.target)]
        for strategy_name in args.strategies:
            strategy = PretrainSplitStrategy(strategy_name)
            split_counts: dict[str, dict[int, int]] = {}
            selected_counts: dict[str, int] = {}
            for split, target_size in sample_counts.items():
                selected = select_balanced_indices(
                    dataset=dataset,
                    cache=cache,
                    strategy=strategy,
                    split=split,
                    target_size=target_size,
                )
                split_counts[split] = cache.counts_for_indices(selected)
                selected_counts[split] = int(selected.size)

            output_path = args.output_dir / f"{spec.name}_{strategy.value}_labeldist.png"
            plot_counts(
                spec=spec,
                strategy=strategy,
                split_counts=split_counts,
                output_path=output_path,
                sample_scale=args.sample_scale,
            )
            summary["plots"].append(
                {
                    "eval": spec.name,
                    "dataset": spec.dataset_key,
                    "target": spec.target,
                    "strategy": strategy.value,
                    "path": str(output_path),
                    "selected_counts": selected_counts,
                    "labels": sorted(
                        {label for counts in split_counts.values() for label in counts}
                    ),
                }
            )
            print(output_path)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
