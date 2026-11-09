"""Load a bundle via rslearn's OlmoEarth (seagrass_eval_venv) and compare to the reference.

Compares its S2 feature map with the reference saved by the olmoearth_pretrain checkout (same random input, same pooling).

  <seagrass venv python> check_bundle_rslearn.py <bundle dir> <ref.npz>
"""

import sys

import numpy as np
import olmoearth_pretrain_minimal
import torch
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage

bundle, ref = sys.argv[1], np.load(sys.argv[2])
print("minimal from", olmoearth_pretrain_minimal.__file__)
model = OlmoEarth(
    patch_size=4, model_path=bundle, embedding_size=192, autocast_dtype=None
).eval()
x = torch.from_numpy(ref["x"])  # C T H W, already normalised
with torch.no_grad():
    fmap = model(
        ModelContext(inputs=[{"sentinel2_l2a": RasterImage(x)}], metadatas=[None])
    ).feature_maps[0]
out = fmap[0].numpy()
diff = np.abs(out - ref["feat"]).max()
print(f"feature map {out.shape}, max abs diff vs reference {diff:.2e}")
assert diff < 1e-4
print("OK")
