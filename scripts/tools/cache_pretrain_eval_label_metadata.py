"""Cache per-sample label metadata for balanced pretrain eval construction.

This scans H5 samples once and writes sparse per-sample label/bin counts that
can be reused later for balanced split selection and distribution plots.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # Registers HDF5 compression plugins in workers.
import numpy as np
import pandas as pd
from tqdm import tqdm

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

WORLDCOVER_CLASSES = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
CANOPY_EDGES = np.asarray([0.0, 1.0, 5.0, 10.0, 20.0, 40.0], dtype=np.float32)
SRTM_EDGES = np.asarray([0.0, 250.0, 500.0, 1000.0, 1500.0, 2500.0], dtype=np.float32)

REGRESSION_TARGETS = {"srtm", "wri_canopy_height_map"}


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset root and targets to cache."""

    key: str
    h5py_dir: Path
    targets: tuple[str, ...]


@dataclass(frozen=True)
class SampleSummary:
    """Sparse label metadata for one sample."""

    sample_index: int
    label_ids: tuple[int, ...]
    counts: tuple[int, ...]
    finite_count: int
    value_sum: float
    value_sumsq: float
    value_min: float
    value_max: float


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


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/pretrain_eval_label_metadata"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic cap per dataset/target for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_path(output_dir: Path, dataset_key: str, target: str) -> Path:
    """Return output cache path."""
    return output_dir / f"{dataset_key}_{target}_label_metadata.npz"


def eligible_indices(dataset: DatasetSpec, target: str) -> np.ndarray:
    """Return sample indices where sentinel2_l2a and target are present."""
    metadata = pd.read_csv(dataset.h5py_dir / "sample_metadata.csv")
    mask = (metadata["sentinel2_l2a"].to_numpy() > 0) & (
        metadata[target].to_numpy() > 0
    )
    return metadata.loc[mask, "sample_index"].to_numpy(dtype=np.int64)


def maybe_cap_indices(indices: np.ndarray, max_samples: int | None) -> np.ndarray:
    """Optionally cap sample indices for smoke tests."""
    if max_samples is None or len(indices) <= max_samples:
        return indices
    return np.random.RandomState(42).permutation(indices)[:max_samples]


def read_raw_label(h5py_dir: Path, sample_index: int, target: str) -> np.ndarray:
    """Read one raw target array from H5."""
    with h5py.File(h5py_dir / f"sample_{sample_index}.h5", "r") as h5_file:
        return np.asarray(h5_file[target])


def transform_label(raw: np.ndarray, target: str) -> np.ndarray:
    """Apply the same target transform used by pretrain eval labels."""
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


def summarize_regression(label: np.ndarray, target: str) -> SampleSummary:
    """Summarize regression target with fixed-bin counts and raw-value stats."""
    values = label[np.isfinite(label)].astype(np.float64)
    if values.size == 0:
        return SampleSummary(0, (), (), 0, 0.0, 0.0, float("nan"), float("nan"))
    edges = SRTM_EDGES if target == "srtm" else CANOPY_EDGES
    bins = np.digitize(values, edges, right=True).astype(np.int64)
    label_ids, counts = np.unique(bins, return_counts=True)
    return SampleSummary(
        sample_index=0,
        label_ids=tuple(label_ids.tolist()),
        counts=tuple(counts.tolist()),
        finite_count=int(values.size),
        value_sum=float(values.sum()),
        value_sumsq=float(np.square(values).sum()),
        value_min=float(values.min()),
        value_max=float(values.max()),
    )


def summarize_classification(label: np.ndarray) -> SampleSummary:
    """Summarize segmentation/class target with valid class counts."""
    values = label.reshape(-1).astype(np.int64)
    values = values[values != -1]
    if values.size == 0:
        return SampleSummary(0, (), (), 0, 0.0, 0.0, float("nan"), float("nan"))
    label_ids, counts = np.unique(values, return_counts=True)
    return SampleSummary(
        sample_index=0,
        label_ids=tuple(label_ids.tolist()),
        counts=tuple(counts.tolist()),
        finite_count=int(values.size),
        value_sum=0.0,
        value_sumsq=0.0,
        value_min=float("nan"),
        value_max=float("nan"),
    )


def summarize_sample(args: tuple[str, str, int]) -> SampleSummary:
    """Pickle-friendly worker for one sample."""
    h5py_dir_str, target, sample_index = args
    label = transform_label(read_raw_label(Path(h5py_dir_str), sample_index, target), target)
    if target in REGRESSION_TARGETS:
        summary = summarize_regression(label, target)
    else:
        summary = summarize_classification(label)
    return SampleSummary(
        sample_index=int(sample_index),
        label_ids=summary.label_ids,
        counts=summary.counts,
        finite_count=summary.finite_count,
        value_sum=summary.value_sum,
        value_sumsq=summary.value_sumsq,
        value_min=summary.value_min,
        value_max=summary.value_max,
    )


def write_cache(path: Path, dataset: DatasetSpec, target: str, summaries: list[SampleSummary]) -> None:
    """Write sparse sample summaries to compressed NPZ."""
    sample_indices: list[int] = []
    row_offsets = [0]
    label_ids: list[int] = []
    counts: list[int] = []
    finite_counts: list[int] = []
    value_sums: list[float] = []
    value_sumsqs: list[float] = []
    value_mins: list[float] = []
    value_maxs: list[float] = []

    for summary in summaries:
        sample_indices.append(summary.sample_index)
        label_ids.extend(summary.label_ids)
        counts.extend(summary.counts)
        row_offsets.append(len(label_ids))
        finite_counts.append(summary.finite_count)
        value_sums.append(summary.value_sum)
        value_sumsqs.append(summary.value_sumsq)
        value_mins.append(summary.value_min)
        value_maxs.append(summary.value_max)

    np.savez_compressed(
        path,
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        row_offsets=np.asarray(row_offsets, dtype=np.int64),
        label_ids=np.asarray(label_ids, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        finite_counts=np.asarray(finite_counts, dtype=np.int64),
        value_sums=np.asarray(value_sums, dtype=np.float64),
        value_sumsqs=np.asarray(value_sumsqs, dtype=np.float64),
        value_mins=np.asarray(value_mins, dtype=np.float64),
        value_maxs=np.asarray(value_maxs, dtype=np.float64),
        source_h5py_dir=str(dataset.h5py_dir),
        dataset_key=dataset.key,
        target=target,
        kind="regression" if target in REGRESSION_TARGETS else "classification",
        regression_bin_edges=SRTM_EDGES
        if target == "srtm"
        else CANOPY_EDGES
        if target == "wri_canopy_height_map"
        else np.asarray([], dtype=np.float32),
    )


def build_cache(
    dataset: DatasetSpec,
    target: str,
    output_dir: Path,
    workers: int,
    chunksize: int,
    max_samples: int | None,
    overwrite: bool,
) -> Path:
    """Build one target cache."""
    path = output_path(output_dir, dataset.key, target)
    if path.exists() and not overwrite:
        print(f"exists, skipping: {path}")
        return path

    indices = maybe_cap_indices(eligible_indices(dataset, target), max_samples)
    worker_args = [
        (str(dataset.h5py_dir), target, int(sample_index))
        for sample_index in indices.tolist()
    ]

    summaries: list[SampleSummary] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for summary in tqdm(
            executor.map(summarize_sample, worker_args, chunksize=chunksize),
            total=len(worker_args),
            desc=f"{dataset.key}/{target}",
            unit="sample",
        ):
            summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_cache(path, dataset, target, summaries)
    print(f"wrote {path}")
    return path


def main() -> None:
    """Build all requested metadata caches."""
    args = parse_args()
    manifest = {
        "output_dir": str(args.output_dir),
        "datasets": {},
        "args": {
            "workers": args.workers,
            "chunksize": args.chunksize,
            "max_samples": args.max_samples,
            "overwrite": args.overwrite,
        },
    }
    for dataset_key in args.datasets:
        dataset = DATASET_SPECS[dataset_key]
        manifest["datasets"][dataset_key] = {
            "spec": {**asdict(dataset), "h5py_dir": str(dataset.h5py_dir)},
            "targets": {},
        }
        for target in dataset.targets:
            path = build_cache(
                dataset=dataset,
                target=target,
                output_dir=args.output_dir,
                workers=args.workers,
                chunksize=args.chunksize,
                max_samples=args.max_samples,
                overwrite=args.overwrite,
            )
            manifest["datasets"][dataset_key]["targets"][target] = str(path)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
