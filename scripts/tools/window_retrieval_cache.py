"""Cache frozen-v1.3 d128 embeddings for the window-retrieval frozen check.

The retriever maps a stored annual embedding ``z_T`` plus a window ``t`` to the
embedding fresh inference on ``t`` would give, ``z_t``. With the encoder frozen
both are fixed, so they are computed once here and the retriever trains on the
cache (``window_retrieval_train.py``).

Two steps:

- ``scan`` (CPU): find pretraining samples whose 12 timesteps are 12
  consecutive months, and note each one's start month. Only the ``timestamps``
  dataset of each h5 file is read. Writes ``full_year_indices.npy``
  (index, year of the first step, start month) and ``calendar_year_indices.npy``
  (index, year) for the subset starting in January: the case the released store
  holds.
- ``cache`` (GPU): for each selected sample (calendar-year by default, or any
  other start month with ``--indices_file full_year_indices.npy
  --exclude_calendar``), run the encoder once on the full 12 months and once per
  window, each as its own forward pass on input sliced
  to that window (so no window embedding has attended to months outside it).
  Inference matches the store: Searchlight, patch size 2, one location query per
  pixel, unmasked. Keeps a random subset of interior pixels per sample, or with
  ``--patch_px`` one contiguous square of them (for retrievers that read
  neighbouring pixels).

Samples are 128 px tiles, so pixels near a tile edge see less context than in
the store (which embeds large domains); interior pixels are kept to limit that.

Usage::

    python scripts/tools/window_retrieval_cache.py scan --out_dir $OUT --workers 64
    python scripts/tools/window_retrieval_cache.py cache --out_dir $OUT \
        --num_shards 4 --shard_index 0 --num_samples 5000
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch
from upath import UPath

from olmoearth_pretrain.data.constants import (
    MAX_SEQUENCE_LENGTH,
    MISSING_VALUE,
    Modality,
)
from olmoearth_pretrain.data.dataset import OlmoEarthDataset, OlmoEarthDatasetConfig
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.model_loader import load_model_from_path
from olmoearth_pretrain.nn.searchlight import SearchlightSettings

logger = logging.getLogger(__name__)

# The h5 build v1.3 pretrained on (scripts/official/v1_2/base.py build_dataset_config).
DEFAULT_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_"
    "worldpop_wri_canopy_height_map/1138828"
)
# Converted v1.3 release checkpoint (config.json + weights.pth).
DEFAULT_CHECKPOINT = "/weka/dfive-default/gabrielt/helios/v1_3_release"
DEFAULT_MODALITIES = (
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
)

# Window lengths in months and their sampling weights: 2-11 months, favouring
# season-length windows. Full-year windows are not sampled (the retriever returns
# z_T unchanged there by construction).
WINDOW_LENGTHS = np.arange(2, 12)
WINDOW_WEIGHTS = np.array([1, 2, 2, 2, 2, 2, 2, 1, 1, 1], dtype=float)
WINDOW_WEIGHTS /= WINDOW_WEIGHTS.sum()


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def full_year_start(timestamps: np.ndarray) -> tuple[int, int] | None:
    """(year of the first step, start month) if timestamps are 12 consecutive months.

    Timestamps are ``[T, 3]`` as (day, month, year) with zero-indexed months. A
    calendar year is start month 0.
    """
    if timestamps.shape[0] != MAX_SEQUENCE_LENGTH:
        return None
    months = timestamps[:, 1].astype(int)
    years = timestamps[:, 2].astype(int)
    m0, y0 = int(months[0]), int(years[0])
    offsets = m0 + np.arange(12)
    if not np.array_equal(months, offsets % 12) or not np.array_equal(
        years, y0 + offsets // 12
    ):
        return None
    return y0, m0


def scan_one(h5_dir: UPath, index: int) -> tuple[int, tuple[int, int] | None]:
    """Read one sample's timestamps and test for 12 consecutive months."""
    path = h5_dir / ConvertToH5py.sample_file_pattern.format(index=index)
    try:
        with path.open("rb") as f, h5py.File(f, "r") as h5:
            return index, full_year_start(h5["timestamps"][()])
    except Exception:
        logger.exception(f"failed to read {path}")
        return index, None


def scan(args: argparse.Namespace) -> None:
    """Write the indices, years and start months of 12-consecutive-month samples."""
    h5_dir = UPath(args.h5py_dir)
    dataset = build_dataset(args)
    indices = np.asarray(dataset.sample_indices)
    if args.exclude_previous_limit:
        # Re-draw an earlier scan's random subset (same seed, same size) and skip it.
        rng = np.random.default_rng(args.seed)
        previous = rng.choice(
            indices, size=min(args.exclude_previous_limit, len(indices)), replace=False
        )
        indices = np.setdiff1d(indices, previous)
        logger.info(
            f"skipping {len(previous)} samples scanned before; {len(indices)} remain"
        )
    if args.scan_limit:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(
            indices, size=min(args.scan_limit, len(indices)), replace=False
        )
    logger.info(f"scanning {len(indices)} samples with {args.workers} threads")

    found: list[tuple[int, int, int]] = []
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for n, (index, ys) in enumerate(
            pool.map(lambda i: scan_one(h5_dir, int(i)), indices), 1
        ):
            if ys is not None:
                found.append((index, *ys))
            if n % 20000 == 0:
                rate = n / (time.time() - start)
                logger.info(
                    f"  {n}/{len(indices)} scanned ({rate:.0f}/s), {len(found)} full-year"
                )

    out = np.array(found, dtype=np.int64).reshape(-1, 3)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "full_year_indices.npy"), out)
    calendar = out[out[:, 2] == 0][:, :2]
    np.save(os.path.join(args.out_dir, "calendar_year_indices.npy"), calendar)
    months, counts = np.unique(out[:, 2], return_counts=True) if len(out) else ([], [])
    logger.info(
        f"{len(out)}/{len(indices)} samples are 12 consecutive months "
        f"({100 * len(out) / max(len(indices), 1):.1f}%), {len(calendar)} of them calendar years; "
        f"by start month {dict(zip(months.tolist(), counts.tolist()))}"
    )


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def build_dataset(args: argparse.Namespace) -> OlmoEarthDataset:
    """The pretraining dataset, prepared (sample indices filtered as in training)."""
    dataset = OlmoEarthDatasetConfig(
        h5py_dir=args.h5py_dir, training_modalities=list(args.modalities)
    ).build()
    dataset.prepare()
    return dataset


def load_sample(dataset: OlmoEarthDataset, index: int) -> MaskedOlmoEarthSample:
    """One full 12-step sample as an unmasked, normalized MaskedOlmoEarthSample (batch 1).

    Mirrors ``OlmoEarthDataset.__getitem__`` without the training crop: every
    timestep is kept, missing timesteps are marked MISSING per modality.
    """
    sample_dict, present = dataset.read_h5_file(dataset._get_h5_file_path(index))
    sample, missing_modalities = dataset.fill_sample_with_missing_values(
        sample_dict, present
    )
    raw = sample.as_dict()

    # Built unbatched (shape checks read H, W, T off the leading dims), batched at the end.
    fields: dict = {"timestamps": torch.as_tensor(raw["timestamps"]).long()}
    for name in dataset.training_modalities:
        data = raw[name]
        if name not in missing_modalities:
            # Absent timesteps are masked MISSING below. Searchlight needs missing
            # data to be whole timesteps, so stray nodata pixels inside a present
            # timestep become 0 (post-normalization) rather than a mask hole.
            missing = data == MISSING_VALUE
            data = np.where(
                missing, 0.0, dataset.normalize_image(Modality.get(name), data)
            )
        fields[name] = torch.as_tensor(data.astype(np.float32))
    masked = MaskedOlmoEarthSample.from_olmoearthsample(OlmoEarthSample(**fields))

    for name in dataset.training_modalities:
        mask = getattr(masked, MaskedOlmoEarthSample.get_masked_modality_name(name))
        if name in missing_modalities:
            mask[:] = MaskValue.MISSING.value
        elif name in present:
            absent = np.where(~np.asarray(present[name], dtype=bool))[0]
            mask[:, :, absent, :] = MaskValue.MISSING.value
    return masked._replace(
        **{k: v[None] for k, v in masked.as_dict(include_nones=False).items()}
    )


def slice_time(
    sample: MaskedOlmoEarthSample, start: int, end: int
) -> MaskedOlmoEarthSample:
    """Keep timesteps [start, end) of every spacetime field and the timestamps."""
    updates = {}
    for key, val in sample.as_dict(include_nones=False).items():
        if key == "timestamps":
            updates[key] = val[:, start:end]
            continue
        modality = key.removesuffix("_mask")
        if Modality.get(modality).is_spacetime_varying:
            updates[key] = val[:, :, :, start:end]
    return sample._replace(**updates)


def present_counts(sample: MaskedOlmoEarthSample, modalities: list[str]) -> np.ndarray:
    """Number of timesteps with data, per modality."""
    counts = []
    for name in modalities:
        mask = getattr(sample, MaskedOlmoEarthSample.get_masked_modality_name(name))
        per_t = (mask[0, 0, 0] != MaskValue.MISSING.value).any(dim=-1)
        counts.append(int(per_t.sum()))
    return np.array(counts, dtype=np.int8)


def sample_windows(rng: np.random.Generator, k: int) -> list[tuple[int, int]]:
    """K distinct contiguous (start, length) windows inside the 12 steps."""
    windows: set[tuple[int, int]] = set()
    while len(windows) < k:
        length = int(rng.choice(WINDOW_LENGTHS, p=WINDOW_WEIGHTS))
        start = int(rng.integers(0, 12 - length + 1))
        windows.add((start, length))
    return sorted(windows)


@torch.no_grad()
def embed(
    encoder: torch.nn.Module, sample: MaskedOlmoEarthSample, device: str
) -> torch.Tensor:
    """Per-pixel student embeddings [H, W, D], computed the way the store is."""
    sample = sample.to_device(device)
    with torch.autocast(device, dtype=torch.bfloat16):
        out = encoder(
            sample,
            patch_size=2,
            input_res=10,
            location_query_patch_size=1,
            searchlight=SearchlightSettings(),
        )
    return out["student_location_tokens"][0].float()


def cache(args: argparse.Namespace) -> None:
    """Embed the selected samples for this shard and write npz chunks."""
    device = "cuda"
    found = np.load(os.path.join(args.out_dir, args.indices_file))
    if found.shape[1] == 2:  # calendar_year_indices.npy: (index, year), start month 0
        found = np.concatenate(
            [found, np.zeros((len(found), 1), dtype=found.dtype)], axis=1
        )
    if args.exclude_calendar:
        found = found[found[:, 2] != 0]
    rng = np.random.default_rng(args.seed)
    # --offset skips samples an earlier cache took from the same shuffle.
    order = rng.permutation(len(found))[args.offset : args.offset + args.num_samples]
    mine = found[order[args.shard_index :: args.num_shards]]
    logger.info(f"shard {args.shard_index}/{args.num_shards}: {len(mine)} samples")

    dataset = build_dataset(args)
    model = load_model_from_path(UPath(args.checkpoint))
    encoder = model.encoder.to(device).eval()
    modalities = list(args.modalities)

    shard_rng = np.random.default_rng(args.seed + 1000 * args.shard_index)
    keys = ("z_T", "z_w", "win", "present", "index", "year", "start_month", "pix")
    chunk: dict[str, list] = {k: [] for k in keys}
    written = skipped = 0
    start = time.time()

    def flush() -> None:
        nonlocal chunk
        if not chunk["index"]:
            return
        name = f"shard{args.shard_index:02d}_{written:06d}.npz"
        np.savez(
            os.path.join(args.out_dir, name),
            **{k: np.stack(v) for k, v in chunk.items()},
        )
        logger.info(f"  wrote {name} ({len(chunk['index'])} samples)")
        chunk = {k: [] for k in chunk}

    for n, (index, year, start_month) in enumerate(mine, 1):
        try:
            sample = load_sample(dataset, int(index))
            z_T = embed(encoder, sample, device)
            H, W, _ = z_T.shape
            m = args.margin_px
            if args.patch_px:
                # One contiguous square, row-major, for retrievers that read neighbours.
                r0 = int(shard_rng.integers(m, H - m - args.patch_px + 1))
                c0 = int(shard_rng.integers(m, W - m - args.patch_px + 1))
                rr, cc = np.meshgrid(
                    np.arange(args.patch_px), np.arange(args.patch_px), indexing="ij"
                )
                rows, cols = (r0 + rr).ravel(), (c0 + cc).ravel()
            else:
                rows = shard_rng.integers(m, H - m, size=args.pixels_per_sample)
                cols = shard_rng.integers(m, W - m, size=args.pixels_per_sample)
            windows = sample_windows(shard_rng, args.windows_per_sample)
            z_w, present = [], []
            for s, length in windows:
                sub = slice_time(sample, s, s + length)
                present.append(present_counts(sub, modalities))
                z_w.append(embed(encoder, sub, device)[rows, cols].half().cpu().numpy())
        except Exception:
            logger.exception(f"sample {index} failed; skipping")
            skipped += 1
            continue

        chunk["z_T"].append(z_T[rows, cols].half().cpu().numpy())
        chunk["z_w"].append(np.stack(z_w))
        chunk["win"].append(np.array(windows, dtype=np.int8))
        chunk["present"].append(np.stack(present))
        chunk["index"].append(int(index))
        chunk["year"].append(int(year))
        chunk["start_month"].append(int(start_month))
        chunk["pix"].append(np.stack([rows, cols], axis=-1).astype(np.int16))
        written += 1
        if written % args.chunk_size == 0:
            flush()
        if n % 50 == 0:
            rate = n / (time.time() - start)
            logger.info(f"  {n}/{len(mine)} ({rate:.2f} samples/s), skipped {skipped}")
    flush()
    logger.info(
        f"shard {args.shard_index}: {written} samples cached, {skipped} skipped"
    )


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["scan", "cache"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--h5py_dir", default=DEFAULT_H5_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--seed", type=int, default=0)
    # scan
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--scan_limit", type=int, default=0, help="scan a random subset (0 = all)"
    )
    parser.add_argument(
        "--exclude_previous_limit",
        type=int,
        default=0,
        help="skip the samples an earlier --scan_limit N scan (same seed) covered",
    )
    # cache
    parser.add_argument(
        "--indices_file", default="calendar_year_indices.npy", help="under --out_dir"
    )
    parser.add_argument(
        "--exclude_calendar",
        action="store_true",
        help="skip samples starting in January",
    )
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip the first N of the shuffled selection",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--windows_per_sample", type=int, default=4)
    parser.add_argument("--pixels_per_sample", type=int, default=256)
    parser.add_argument(
        "--patch_px",
        type=int,
        default=0,
        help="store one contiguous patch_px x patch_px square (row-major) instead of random pixels",
    )
    parser.add_argument("--margin_px", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=200)
    args = parser.parse_args()

    if args.command == "scan":
        scan(args)
    else:
        cache(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
