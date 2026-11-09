"""Reference S2 feature map from a pretraining checkpoint.

Encoder patch tokens, mean over time and bandsets (what rslearn's OlmoEarth wrapper returns), for a fixed random input.

  .venv/bin/python ref_features.py <checkpoint step dir> <out.npz>
"""

import sys

import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.model_loader import load_pretrain_checkpoint

model = load_pretrain_checkpoint(sys.argv[1], device=torch.device("cpu")).eval()
g = torch.Generator().manual_seed(0)
c, t, hw = Modality.SENTINEL2_L2A.num_bands, 3, 32
x = torch.rand(c, t, hw, hw, generator=g)
nbs = model.encoder.tokenization_config.get_num_bandsets(Modality.SENTINEL2_L2A.name)
sample = MaskedOlmoEarthSample(
    sentinel2_l2a=x.permute(2, 3, 1, 0)[None],
    sentinel2_l2a_mask=torch.zeros(1, hw, hw, t, nbs, dtype=torch.long),
    timestamps=torch.tensor([[1, 0, 2024], [1, 1, 2024], [1, 2, 2024]])[None],
)
with torch.no_grad():
    tok = model.encoder(sample, patch_size=4, fast_pass=True)[
        "tokens_and_masks"
    ].sentinel2_l2a
feat = tok.mean(dim=[3, 4])[0].permute(2, 0, 1).numpy()
np.savez(sys.argv[2], x=x.numpy(), feat=feat)
print("saved", feat.shape)
