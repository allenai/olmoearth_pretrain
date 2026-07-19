"""Prepare the "Similar But Different" (SBD) eval dataset into per-split tensors.

Source: HuggingFace dataset ``calebrob6/similar-but-different`` -- 30,927 Sentinel-2
L2A 32x32 patches (12 bands, GeoTIFF bytes in per-class parquet shards) across ten
ESA WorldCover classes, with a fixed scene-disjoint 80/10/10 split in ``splits.json``.
See https://huggingface.co/datasets/calebrob6/similar-but-different.

This mirrors ``fifty_cities_processor``/``process_mados``: it downloads the shards,
decodes each patch's embedded GeoTIFF, and writes, per split, an ``SBD_{split}.pt``
holding raw uint16 surface-reflectance images ``(N, 32, 32, 12)``, scalar class labels
``(N,)``, and the patch ids. It also writes ``norm_stats.json`` (train-split per-band
mean/std + robust min/max, on the raw-SR scale the framework expects) and copies the
pair/cluster eval artifacts.

Band order in the stored images is the source order, which equals
``EVAL_S2_L2A_BAND_NAMES``: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12.

Usage:
    python -m olmoearth_pretrain.evals.datasets.sbd_processor \
        --out_dir /weka/dfive-default/presto_eval_sets/similar_but_different
"""

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import tqdm
from huggingface_hub import snapshot_download

REPO_ID = "calebrob6/similar-but-different"
# Pinned dataset revision (commit SHA) for reproducibility. Passed as a literal to
# snapshot_download below so the download is version-locked (and static analysis can
# verify the pin). Bump this when intentionally moving to a newer dataset revision.
REVISION = "6104443755c3fab82ab148f7053f9f05a75677b1"
# The 12 L2A bands, in the order they appear in each patch GeoTIFF (== EVAL_S2_L2A_BAND_NAMES).
BAND_NAMES = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "11 - SWIR",
    "12 - SWIR",
]
SPLITS = ["train", "val", "test"]


def _decode_tif(tif_bytes: bytes) -> np.ndarray:
    """Decode a patch GeoTIFF -> (32, 32, 12) uint16 (H, W, C)."""
    with rasterio.open(io.BytesIO(tif_bytes)) as src:
        arr = src.read()  # (12, 32, 32)
    return np.transpose(arr, (1, 2, 0))  # (32, 32, 12)


def main() -> None:
    """Download SBD, decode shards, and write per-split tensors + norm stats."""
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out_dir",
        default="/weka/dfive-default/presto_eval_sets/similar_but_different",
    )
    ap.add_argument("--hf_cache", default=None, help="optional HF download cache dir")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading {REPO_ID}@{REVISION[:8]} ...")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="6104443755c3fab82ab148f7053f9f05a75677b1",  # == REVISION; literal for static pin check
        cache_dir=args.hf_cache,
    )
    local = Path(local)

    splits = json.loads((local / "splits.json").read_text())
    class_to_idx = splits["class_to_idx"]
    # patch_id -> split
    pid_to_split = {}
    for split in SPLITS:
        for _cls, pids in splits[split].items():
            for pid in pids:
                pid_to_split[pid] = split

    buffers: dict[str, dict[str, list]] = {
        s: {"images": [], "labels": [], "patch_ids": []} for s in SPLITS
    }

    shard_files = sorted((local / "shards").glob("*.parquet"))
    print(f"decoding {len(shard_files)} shards ...")
    for shard in shard_files:
        df = pd.read_parquet(shard)
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=shard.stem):
            pid = row["patch_id"]
            patch_split = pid_to_split.get(pid)
            if patch_split is None:
                continue  # patch not in any split (shouldn't happen)
            img = _decode_tif(row["tif"])  # (32, 32, 12) uint16
            buffers[patch_split]["images"].append(img)
            buffers[patch_split]["labels"].append(class_to_idx[row["wc_mode_label"]])
            buffers[patch_split]["patch_ids"].append(pid)

    for split in SPLITS:
        imgs = np.stack(buffers[split]["images"]).astype(np.uint16)  # (N,32,32,12)
        labels = np.array(buffers[split]["labels"], dtype=np.int64)
        torch.save(
            {
                "images": torch.from_numpy(imgs),
                "labels": torch.from_numpy(labels),
                "patch_ids": buffers[split]["patch_ids"],
            },
            out_dir / f"SBD_{split}.pt",
        )
        print(f"  {split}: {imgs.shape[0]} patches -> SBD_{split}.pt")

    # Train-split per-band norm stats on the raw-SR scale (mean/std + robust 2/98 pct).
    train_imgs = np.stack(buffers["train"]["images"]).astype(np.float64)
    flat = train_imgs.reshape(-1, len(BAND_NAMES))  # (N*32*32, 12)
    norm_stats = {}
    for i, band in enumerate(BAND_NAMES):
        col = flat[:, i]
        norm_stats[band] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(np.percentile(col, 2)),
            "max": float(np.percentile(col, 98)),
        }
    (out_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2))
    (out_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))

    # Copy the pair/cluster eval artifacts for future similar-but-different metrics.
    eval_dst = out_dir / "eval"
    eval_dst.mkdir(exist_ok=True)
    for name in ["test_pairs.parquet", "test_hard_clusters.parquet"]:
        src = local / "eval" / name
        if src.exists():
            shutil.copy(src, eval_dst / name)
    print(f"wrote norm_stats.json, class_to_idx.json, eval/ to {out_dir}")


if __name__ == "__main__":
    main()
