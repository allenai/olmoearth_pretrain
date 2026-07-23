"""Precompute + read OmniCloudMask cloud-class maps for the OSM-sampling pretrain h5 set.

One compressed ``.npz`` per sample, keyed by sample id and sharded ``id//SHARD_SIZE``:
    s2_cloud      : (H, W, T_s2)  uint8   OCM class per Sentinel-2 stored timestep
    landsat_cloud : (H, W, T_ls)  uint8   OCM class per Landsat stored timestep
Classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow, 255 no-data.

Each array MIRRORS the stored h5 modality's ``(H, W, T)`` (same timestep order, T =
number of stored/present timesteps), so it aligns 1:1 with the raw sample and can be
cropped / patchified in lockstep with the modality regardless of downstream transforms.

Validated OCM recipe (see local_output/osm_viz/HANDOFF.md):
  * Sentinel-2 : bands [B04, B03, B8A] = channel idx [2, 1, 7], native 128 px, raw DN.
  * Landsat    : bands [B4, B3, B5]   = channel idx [4, 3, 5], stride-sampled ``[::3]``
                 (~30 m) then nearest-upsampled back to the 128 px grid.

The reader (``cache_path`` / ``load_sample_clouds``) imports with no torch/OCM
dependency; ``predict_from_array`` is imported lazily only when computing. Compute is
idempotent (skips existing) and shardable (``--num-shards`` / ``--shard``) so it fans
out across many GPUs / Beaker jobs.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time

import numpy as np

logger = logging.getLogger(__name__)

# Default source h5 dir (full 1,138,828-sample OSM-sampling training set).
DEFAULT_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_"
    "worldcover_worldpop_wri_canopy_height_map/1138828"
)

# Cloud class codes.
CLEAR, THICK_CLOUD, THIN_CLOUD, SHADOW, NODATA = 0, 1, 2, 3, 255
SHARD_SIZE = 10000

# OCM band indices into the stored h5 channel axis.
S2_RGN_IDX = (2, 1, 7)  # B04, B03, B8A
LANDSAT_RGN_IDX = (4, 3, 5)  # B4, B3, B5
LANDSAT_STRIDE = 3  # ~30 m native sampling


def default_cache_dir(h5_dir: str = DEFAULT_H5_DIR) -> str:
    """Sibling cache dir: swap the ``h5py_data_*`` path component for the cache name."""
    parts = h5_dir.rstrip("/").split("/")
    for i, p in enumerate(parts):
        if p.startswith("h5py_data"):
            parts[i] = "cloud_masks_omnicloudmask"
            return "/".join(parts)
    return h5_dir.rstrip("/") + "_cloud_masks_omnicloudmask"


def cache_path(cache_dir: str, sample_id: int) -> str:
    """Sharded per-sample cache path."""
    return os.path.join(
        cache_dir, str(sample_id // SHARD_SIZE), f"sample_{sample_id}.npz"
    )


def load_sample_clouds(cache_dir: str, sample_id: int) -> dict[str, np.ndarray] | None:
    """Return {"s2_cloud": ..., "landsat_cloud": ...} (subset present), or None if uncached."""
    path = cache_path(cache_dir, sample_id)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _nearest_upsample_labels(labels: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour upsample a (h',w') integer label map to (h,w)."""
    yi = np.linspace(0, labels.shape[0] - 1, h).round().astype(int)
    xi = np.linspace(0, labels.shape[1] - 1, w).round().astype(int)
    return labels[yi][:, xi]


def compute_sample_clouds(
    h5_path: str, model_dir: str | None = None
) -> dict[str, np.ndarray]:
    """Run OCM on a sample's S2 and Landsat stored timesteps. Returns present sensors only.

    ``model_dir``: directory holding the OCM ``.safetensors`` weights. If provided (and
    the files are present) OCM loads from there instead of downloading from HuggingFace
    -- required for offline / multi-process Beaker runs. Defaults to OCM's own cache.
    """
    import h5py
    import hdf5plugin  # noqa: F401  (registers zstd)
    from omnicloudmask import predict_from_array

    def ocm(arr: np.ndarray) -> np.ndarray:  # arr: (3, H, W) float32
        return (
            np.asarray(predict_from_array(arr, destination_model_dir=model_dir))
            .squeeze()
            .astype(np.uint8)
        )

    out: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h:
        if "sentinel2_l2a" in h:
            s2 = h["sentinel2_l2a"][:].astype(np.float32)  # (H, W, T, C)
            hh, ww, tt, _ = s2.shape
            cl = np.full((hh, ww, tt), NODATA, np.uint8)
            r, g, n = S2_RGN_IDX
            for t in range(tt):
                if np.all(s2[:, :, t, :] == 0):
                    continue
                cl[:, :, t] = ocm(
                    np.stack([s2[:, :, t, r], s2[:, :, t, g], s2[:, :, t, n]], 0)
                )
            out["s2_cloud"] = cl
        if "landsat" in h:
            ls = h["landsat"][:].astype(np.float32)  # (H, W, T, C)
            hh, ww, tt, _ = ls.shape
            cl = np.full((hh, ww, tt), NODATA, np.uint8)
            r, g, n = LANDSAT_RGN_IDX
            s = LANDSAT_STRIDE
            for t in range(tt):
                if np.all(ls[:, :, t, :] == 0):
                    continue
                pred = ocm(
                    np.stack(
                        [ls[::s, ::s, t, r], ls[::s, ::s, t, g], ls[::s, ::s, t, n]], 0
                    )
                )
                cl[:, :, t] = _nearest_upsample_labels(pred, hh, ww)
            out["landsat_cloud"] = cl
    return out


def align_cloud_to_nominal(
    cloud_stored: np.ndarray, present_mask: np.ndarray, max_sequence_length: int
) -> np.ndarray:
    """De-compact a stored (H,W,T_stored) cloud map to nominal timesteps.

    Mirrors ``OlmoEarthDataset._fill_missing_timesteps`` EXACTLY (present timesteps
    placed at ``np.where(present_mask)`` positions), so the returned (H,W,T_nom)
    cloud map is time-aligned 1:1 with the corresponding modality after
    ``fill_sample_with_missing_values``. Missing timesteps are filled with NODATA.
    """
    h, w, t = cloud_stored.shape
    out = np.full((h, w, max_sequence_length), NODATA, dtype=np.uint8)
    present_indices = np.where(present_mask)[0]
    n = min(len(present_indices), t)
    if n > 0:
        out[:, :, present_indices[:n]] = cloud_stored[:, :, :n]
    return out


def save_atomic(path: str, arrays: dict[str, np.ndarray]) -> None:
    """Compressed, atomic (temp + rename) write so concurrent workers never see partials."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        try:
            # Group-writable on purpose: this is a shared cache on a multi-user
            # filesystem, so other workers/users must be able to manage the file.
            os.chmod(tmp, 0o664)  # nosec B103
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _num_samples(h5_dir: str) -> int:
    """The trailing dir component is the sample count (ids 0..N-1)."""
    leaf = os.path.basename(h5_dir.rstrip("/"))
    return int(leaf)


def precompute(
    h5_dir: str,
    cache_dir: str,
    num_shards: int,
    shard: int,
    limit: int | None = None,
    model_dir: str | None = None,
) -> None:
    """Compute-if-missing over this shard's sample ids (interleaved for balanced runtime)."""
    os.umask(0o002)  # group-writable so teammates can resume
    n = _num_samples(h5_dir)
    ids = [i for i in range(n) if i % num_shards == shard]
    if limit is not None:
        ids = ids[:limit]
    logger.info(
        "shard %d/%d: %d candidate samples -> %s",
        shard,
        num_shards,
        len(ids),
        cache_dir,
    )
    done = skipped = failed = 0
    t0 = time.time()
    for k, sid in enumerate(ids):
        path = cache_path(cache_dir, sid)
        if os.path.exists(path):
            skipped += 1
            continue
        h5_path = os.path.join(h5_dir, f"sample_{sid}.h5")
        if not os.path.exists(h5_path):
            continue
        try:
            save_atomic(path, compute_sample_clouds(h5_path, model_dir=model_dir))
            done += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            logger.warning("sample %d failed: %s", sid, e)
        if (k + 1) % 500 == 0:
            rate = (time.time() - t0) / max(done, 1)
            remain = (len(ids) - k - 1) * rate / 3600
            logger.info(
                "shard %d: %d/%d done=%d skip=%d fail=%d  %.2fs/sample ~%.1fh left",
                shard,
                k + 1,
                len(ids),
                done,
                skipped,
                failed,
                rate,
                remain,
            )
    logger.info(
        "shard %d COMPLETE: done=%d skipped=%d failed=%d in %.1f min",
        shard,
        done,
        skipped,
        failed,
        (time.time() - t0) / 60,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5_dir", default=DEFAULT_H5_DIR)
    ap.add_argument("--cache_dir", default=None, help="default: sibling of h5_dir")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument(
        "--limit", type=int, default=None, help="cap samples this shard (for testing)"
    )
    ap.add_argument(
        "--model_dir",
        default=os.environ.get("OCM_MODEL_DIR"),
        help="dir with OCM .safetensors weights (offline); default OCM HF cache",
    )
    args = ap.parse_args()
    cache = args.cache_dir or default_cache_dir(args.h5_dir)
    precompute(
        args.h5_dir, cache, args.num_shards, args.shard, args.limit, args.model_dir
    )
