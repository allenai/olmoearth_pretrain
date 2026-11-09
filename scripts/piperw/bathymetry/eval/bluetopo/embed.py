"""Token embeddings of the labelled samples from one pretraining checkpoint (S2 L2A only).

Each sample's 12 monthly S2 L2A images (missing months dropped), 128x128 px, patch 4 ->
a [32, 32, D] map: the AttentionPooler location tokens if the model has them (what the
in-loop evals probe), else encoder tokens mean-pooled over time and bandsets.

  python embed.py <checkpoint step dir> <out .npy> [labels file]
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.model_loader import load_pretrain_checkpoint
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
H5 = (
    "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)
PATCH = 4
S2 = Modality.SENTINEL2_L2A.name


@torch.no_grad()
def embed(model, s2: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Per-token encoder embeddings for one S2 time series."""
    keep = ~(s2 == MISSING_VALUE).all(axis=(0, 1, 3))
    x = torch.from_numpy(s2[:, :, keep])[None]
    nbs = model.encoder.tokenization_config.get_num_bandsets(S2)
    mask = torch.zeros(*x.shape[:4], nbs, dtype=torch.long)
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=x,
        sentinel2_l2a_mask=mask,
        timestamps=torch.from_numpy(timestamps[keep])[None].long(),
    )
    out = model.encoder(sample, patch_size=PATCH, fast_pass=True)
    if (
        getattr(model.encoder, "use_attention_pooler", False)
        and "location_tokens" in out
    ):
        grid = out["location_tokens"]
    else:
        grid = pool_unmasked_tokens(
            out["tokens_and_masks"], PoolingType.MEAN, spatial_pooling=True
        )
    return grid[0].float().numpy()


def main() -> None:
    """Embed the labelled samples with one checkpoint."""
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    labels = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "labels.npz"
    idx = np.load(labels)["idx"]
    torch.set_num_threads(96)
    model = load_pretrain_checkpoint(ckpt, device=torch.device("cpu")).eval()
    ds = OlmoEarthDataset(
        h5py_dir=UPath(H5), training_modalities=[S2], dtype=np.float32
    )

    def load(i):
        _, s = ds[GetItemArgs(idx=int(i), patch_size=PATCH, sampled_hw_p=128 // PATCH)]
        return s.sentinel2_l2a, s.timestamps

    feats = None
    with ThreadPoolExecutor(8) as pool:
        for k, (s2, ts) in enumerate(pool.map(load, idx)):
            f = embed(model, s2, ts)
            if feats is None:
                feats = np.lib.format.open_memmap(
                    out, mode="w+", dtype=np.float16, shape=(len(idx), *f.shape)
                )
            feats[k] = f
            if (k + 1) % 250 == 0:
                print(f"{k + 1}/{len(idx)}", flush=True)
    feats.flush()
    print(f"wrote {out}: {feats.shape}")


if __name__ == "__main__":
    main()
