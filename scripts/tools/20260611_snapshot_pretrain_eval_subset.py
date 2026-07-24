"""Materialize a frozen, fixed-size eval snapshot of a pretraining h5py dataset.

The pretrain_subset evals previously read directly from the live pretraining
datasets under /weka/dfive-default/helios/dataset. Those datasets are mutable
(periodically cleaned up, and not reproducible to the sample when recreated),
which breaks the evals in two ways: the h5py dir name encodes the exact sample
count, and the eval split selection permutes over the dataset length, so any
size change silently selects a different eval set.

This script copies a deterministic subset of a source dataset into the
protected eval folder (/weka/dfive-default/presto_eval_sets/pretrain_subset by
default), renumbered 0..N-1 with subsetted sidecar files, so the snapshot is a
self-contained h5py dataset that OlmoEarthDataset can read directly.

Selection guarantees a per-rule quota of samples satisfying each probe task's
(target modality + input modality) presence requirements, then fills the
remainder randomly up to --total. --total is the exact snapshot size and
becomes the directory name, so it must match the constants in
olmoearth_pretrain/internal/all_evals.py.

Expected invocations (run on a machine with weka mounted):

    python scripts/tools/20260611_snapshot_pretrain_eval_subset.py \
        --src /weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1291656 \
        --preset osmbig --total 65536 --dry-run

    python scripts/tools/20260611_snapshot_pretrain_eval_subset.py \
        --src /weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828 \
        --preset osm_sampling --total 98304 --dry-run

Drop --dry-run to copy. Copying is idempotent (existing files with matching
sizes are skipped), so an interrupted run can simply be rerun.
"""

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DST_ROOT = "/weka/dfive-default/presto_eval_sets/pretrain_subset"
SAMPLE_METADATA_FNAME = "sample_metadata.csv"
LATLON_FNAME = "latlon_distribution.npy"
COMPRESSION_SETTINGS_FNAME = "compression_settings.json"

# Presence-column requirements mirroring the pretrain_subset probe tasks in
# olmoearth_pretrain/internal/all_evals.py: each rule lists the columns that
# must all be present for a sample to be usable by that probe. The per-rule
# quota must comfortably exceed the probe's split sizes (6144/3072/3072), and
# for geographic splits valid/test only see ~10% of the eligible pool, so the
# quota should be >= ~10x the valid split size.
PRESETS: dict[str, list[list[str]]] = {
    "osmbig": [
        ["worldcover", "sentinel2_l2a"],
        ["openstreetmap_raster", "sentinel2_l2a"],
        ["srtm", "sentinel2_l2a"],
        ["srtm", "sentinel1"],
    ],
    "osm_sampling": [
        ["wri_canopy_height_map", "sentinel2_l2a"],
        ["cdl", "sentinel2_l2a"],
        ["worldcereal", "sentinel2_l2a"],
    ],
    # The regenerated osm_sampling h5 that carries glo30 (elevation/slope/
    # aspect) and meta_canopy_height in place of srtm/wri_canopy_height_map.
    # glo30 is present for ~99.6% of samples and meta_canopy_height for ~54%
    # (the converter drops tiles with nodata or <20% positive canopy), so the
    # canopy rule is the binding one for quota purposes.
    "dsm_canopy": [
        ["meta_canopy_height", "sentinel2_l2a"],
        ["glo30", "sentinel2_l2a"],
    ],
}


def select_indices(
    metadata_df: pd.DataFrame, rules: list[list[str]], quota: int, total: int, seed: int
) -> tuple[np.ndarray, list[dict]]:
    """Deterministically select `total` sample indices satisfying per-rule quotas."""
    num_samples = len(metadata_df)
    if total > num_samples:
        raise ValueError(f"--total {total} exceeds source size {num_samples}")
    rng = np.random.default_rng(seed)
    selected = np.zeros(num_samples, dtype=bool)
    rule_stats = []
    for columns in rules:
        eligible = (metadata_df[columns].to_numpy() > 0).all(axis=1)
        already = int((eligible & selected).sum())
        pool = np.flatnonzero(eligible & ~selected)
        need = min(quota - already, len(pool))
        if need > 0:
            selected[rng.choice(pool, size=need, replace=False)] = True
        satisfied = int((eligible & selected).sum())
        rule_stats.append(
            {
                "columns": columns,
                "eligible_in_source": int(eligible.sum()),
                "satisfied_in_snapshot": satisfied,
            }
        )
        if satisfied < quota:
            logger.warning(
                f"Rule {columns}: only {satisfied} eligible samples available "
                f"(quota {quota}); probe splits for this target will shrink."
            )
    num_selected = int(selected.sum())
    if num_selected > total:
        raise ValueError(
            f"Per-rule quotas required {num_selected} samples, which exceeds "
            f"--total {total}. Rerun with --total >= {num_selected} and update "
            f"the corresponding path constant in all_evals.py to match."
        )
    fill_pool = np.flatnonzero(~selected)
    selected[rng.choice(fill_pool, size=total - num_selected, replace=False)] = True
    return np.flatnonzero(selected), rule_stats


def copy_one(src_dir: Path, dst_dir: Path, old_index: int, new_index: int) -> None:
    """Copy one sample file, skipping if it already exists with the right size."""
    src = src_dir / f"sample_{old_index}.h5"
    dst = dst_dir / f"sample_{new_index}.h5"
    src_size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == src_size:
        return
    tmp = dst.with_suffix(".h5.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def estimate_size_bytes(src_dir: Path, indices: np.ndarray, seed: int) -> int:
    """Estimate total snapshot size by extrapolating from a sample of files."""
    rng = np.random.default_rng(seed)
    probe = rng.choice(indices, size=min(256, len(indices)), replace=False)
    sizes = [(src_dir / f"sample_{i}.h5").stat().st_size for i in probe]
    return int(np.mean(sizes) * len(indices))


def main() -> None:
    """Run the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source h5py dataset dir")
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--total", type=int, required=True, help="Exact snapshot size")
    parser.add_argument("--quota", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dst-root", default=DEFAULT_DST_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src)
    src_count = int(src_dir.name)
    metadata_df = pd.read_csv(src_dir / SAMPLE_METADATA_FNAME)
    latlons = np.load(src_dir / LATLON_FNAME)
    if not (src_count == len(metadata_df) == len(latlons)):
        raise ValueError(
            f"Source dir is inconsistent: dir name says {src_count} samples, "
            f"metadata has {len(metadata_df)}, latlons has {len(latlons)}"
        )

    indices, rule_stats = select_indices(
        metadata_df, PRESETS[args.preset], args.quota, args.total, args.seed
    )
    # corpus / format dir / modality dir, mirroring the source layout.
    dst_dir = (
        Path(args.dst_root)
        / src_dir.parents[2].name
        / src_dir.parents[1].name
        / src_dir.parent.name
        / str(args.total)
    )

    logger.info(f"Selected {len(indices)} / {src_count} samples")
    for stat in rule_stats:
        logger.info(
            f"  {'+'.join(stat['columns'])}: {stat['satisfied_in_snapshot']} in "
            f"snapshot ({stat['eligible_in_source']} eligible in source)"
        )
    size_gb = estimate_size_bytes(src_dir, indices, args.seed) / 1e9
    logger.info(f"Estimated snapshot size: {size_gb:.0f} GB")
    logger.info(f"Destination: {dst_dir}")
    if args.dry_run:
        logger.info("Dry run; nothing copied.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    sub_df = metadata_df.iloc[indices].copy()
    sub_df["sample_index"] = np.arange(len(indices))
    sub_df.to_csv(dst_dir / SAMPLE_METADATA_FNAME, index=False)
    with open(dst_dir / LATLON_FNAME, "wb") as f:
        np.save(f, latlons[indices])
    shutil.copyfile(
        src_dir / COMPRESSION_SETTINGS_FNAME, dst_dir / COMPRESSION_SETTINGS_FNAME
    )
    # new index -> source index, for provenance and debugging.
    with open(dst_dir / "source_indices.npy", "wb") as f:
        np.save(f, indices)
    with open(dst_dir / "provenance.json", "w") as f:
        json.dump(
            {
                "source": str(src_dir),
                "preset": args.preset,
                "quota": args.quota,
                "total": args.total,
                "seed": args.seed,
                "rule_stats": rule_stats,
                "script": "scripts/tools/20260611_snapshot_pretrain_eval_subset.py",
            },
            f,
            indent=2,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(copy_one, src_dir, dst_dir, int(old), new)
            for new, old in enumerate(indices)
        ]
        for future in tqdm(futures, desc="Copying samples"):
            future.result()

    num_copied = sum(
        1
        for p in dst_dir.iterdir()
        if p.name.startswith("sample_") and p.suffix == ".h5"
    )
    if num_copied != args.total:
        raise RuntimeError(f"Expected {args.total} sample files, found {num_copied}")
    logger.info(f"Done. Point the eval config at:\n{dst_dir}")


if __name__ == "__main__":
    main()
