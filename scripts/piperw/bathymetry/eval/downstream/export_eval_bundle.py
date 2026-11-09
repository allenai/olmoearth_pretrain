"""Export a pretraining checkpoint as a model_path bundle without the sdb and bluetopo modalities.

The seagrass evals run rslearn with olmoearth_pretrain from git main, which does not know
Modality.SDB. sdb is decode-only, so the encoder never uses its parameters on S2 input:
drop them, drop sdb from the config, and check the stripped model loads strictly and gives
the same encoder output as the original. Both arms are exported the same way.

Usage (from the olmoearth_pretrain piperw/bathymetry checkout):
  .venv/bin/python export_eval_bundle.py <checkpoint step dir> <output dir>
"""

import copy
import json
import sys
from pathlib import Path

import torch

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.model_loader import (
    load_pretrain_checkpoint,
    patch_legacy_encoder_config,
)

DROP = {Modality.SDB.name, Modality.BLUETOPO.name}


def strip_config(node):
    """Remove DROP from every modality list and dict key in a config tree."""
    if isinstance(node, dict):
        return {k: strip_config(v) for k, v in node.items() if k not in DROP}
    if isinstance(node, list):
        return [strip_config(v) for v in node if not (isinstance(v, str) and v in DROP)]
    return node


def s2_only_input(
    num_bandsets: int, batch: int = 2, hw: int = 32, t: int = 3
) -> MaskedOlmoEarthSample:
    """Random S2-only input for checking an exported bundle."""
    g = torch.Generator().manual_seed(0)
    s2 = torch.rand(batch, hw, hw, t, Modality.SENTINEL2_L2A.num_bands, generator=g)
    # all zeros = MaskValue.ONLINE_ENCODER: every token is encoded
    mask = torch.zeros(batch, hw, hw, t, num_bandsets, dtype=torch.long)
    timestamps = torch.tensor([[15, m, 2019] for m in range(t)]).expand(batch, t, 3)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=timestamps
    )


def main() -> None:
    """Export a pretraining checkpoint as an rslearn OlmoEarth bundle."""
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    config = patch_legacy_encoder_config(json.load(open(ckpt / "config.json")))
    original = load_pretrain_checkpoint(ckpt, device=torch.device("cpu")).eval()

    # The whole config, not just the model: loaders read the encoder's input modalities
    # from masking_config.strategy_config.only_decode_modalities.
    stripped_config = strip_config(copy.deepcopy(config))
    state = {
        k: v
        for k, v in original.state_dict().items()
        if not any(f".{d}" in k or f"{d}_" in k for d in DROP)
    }
    dropped = len(original.state_dict()) - len(state)
    stripped = Config.from_dict(stripped_config["model"]).build().eval()
    stripped.load_state_dict(state, strict=True)

    tokenization = original.encoder.tokenization_config
    x = s2_only_input(tokenization.get_num_bandsets(Modality.SENTINEL2_L2A.name))
    with torch.no_grad():
        ta = original.encoder(x, patch_size=4)["tokens_and_masks"].sentinel2_l2a
        tb = stripped.encoder(x, patch_size=4)["tokens_and_masks"].sentinel2_l2a
    max_diff = float((ta - tb).abs().max())
    assert max_diff == 0.0, f"encoder outputs differ after stripping {DROP}: {max_diff}"

    out.mkdir(parents=True, exist_ok=True)
    json.dump(stripped_config, open(out / "config.json", "w"), indent=2)
    torch.save(state, out / "weights.pth")
    print(
        f"{ckpt} -> {out}: dropped {dropped} {DROP} tensors, S2 encoder output identical"
    )


if __name__ == "__main__":
    main()
