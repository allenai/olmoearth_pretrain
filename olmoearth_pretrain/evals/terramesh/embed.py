"""Embed the TerraMesh-Masks eval tiles with an OlmoEarth checkpoint.

  python -m olmoearth_pretrain.evals.terramesh.embed --checkpoint <dir> --inputs <dir> \
      --out <dir> [--readout head|location|student] [--tile 16]

``--checkpoint`` is a trainer checkpoint (config.json + model_and_optim/ or weights.pth)
or a released model dir. Each 256 px tile is cut into ``--tile`` px tiles (16 by
default, as ClipEarth embeds its windows; single date, so T = 1) and encoded with patch size 4 and a location query every 2 px. The location tokens are stitched back, repeated to 10 m pixels and
made unit length. ``head`` (the default when the checkpoint has a text head) projects the
encoder's location tokens through it first, so the vectors live in the text space and
``head.json`` records how to embed queries into it.

Writes <out>/<file_id>.npz with vectors (256, 256, D) float16 and valid (256, 256).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import types

import numpy as np
import torch
from torch import Tensor
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import (
    load_model_from_path,
    load_pretrain_checkpoint,
)

SIZE = 256
TILE = 16
MODALITIES = ("sentinel2_l2a", "sentinel1")
READOUT_KEYS = {
    "head": "location_tokens",
    "location": "location_tokens",
    "student": "student_location_tokens",
}


def _attn_mask_respecting_padding(
    self: torch.nn.Module, new_mask: Tensor, fast_pass: bool
) -> Tensor | None:
    """Encoder._maybe_get_attn_mask that keeps the mask in eval when padding exists.

    The stock method returns no mask outside training, so tiles with missing tokens
    would attend to padding.
    """
    if fast_pass or (not self.training and bool(new_mask.all())):
        return None
    return new_mask


def to_tiles(x: Tensor, tile: int = TILE) -> Tensor:
    """``[B, H, W, ...]`` -> ``[B * (H/tile) * (W/tile), tile, tile, ...]``."""
    b, h, w = x.shape[:3]
    rest = x.shape[3:]
    x = x.reshape(b, h // tile, tile, w // tile, tile, *rest).movedim(3, 2)
    return x.reshape(b * (h // tile) * (w // tile), tile, tile, *rest)


def from_tiles(tokens: Tensor, tile: int = TILE) -> Tensor:
    """``[n * n, g, g, D]`` tile tokens -> ``[SIZE, SIZE, D]``, repeated to pixels."""
    n, g = SIZE // tile, tokens.shape[1]
    grid = tokens.reshape(n, n, g, g, -1).movedim(2, 1).reshape(n * g, n * g, -1)
    repeat = SIZE // (n * g)
    return grid.repeat_interleave(repeat, 0).repeat_interleave(repeat, 1)


def load(path: str) -> torch.nn.Module:
    """A trainer checkpoint as its full model, else a released model."""
    p = UPath(path)
    if (p / "model_and_optim").exists():
        return load_pretrain_checkpoint(path)
    config = json.load((p / "config.json").open())
    if "train_module" in config:
        return load_pretrain_checkpoint(path)
    return load_model_from_path(path)


def sample_for(
    tile: dict,
    encoder: torch.nn.Module,
    normalizer: Normalizer,
    device: torch.device,
    size: int = TILE,
) -> MaskedOlmoEarthSample:
    """The tile as a batch of ``size`` px tiles, missing S2 pixels masked."""
    fields = {}
    for modality in MODALITIES:
        if modality not in tile:
            continue
        data = tile[modality].astype(np.float32)
        missing = np.zeros(data.shape[:3], dtype=bool)
        if modality == "sentinel2_l2a":
            missing |= tile["no_data"][:, :, None] | (data == 0).all(-1)
        data = normalizer.normalize(Modality.get(modality), data).astype(np.float32)
        data[missing] = 0
        num_band_sets = encoder.tokenization_config.get_num_bandsets(modality)
        mask = np.where(
            missing, MaskValue.MISSING.value, MaskValue.ONLINE_ENCODER.value
        )
        mask = np.repeat(mask[..., None], num_band_sets, axis=-1)
        fields[modality] = to_tiles(torch.from_numpy(data)[None], size).to(device)
        fields[f"{modality}_mask"] = to_tiles(torch.from_numpy(mask)[None], size).to(
            device
        )
    n = (SIZE // size) ** 2
    timestamps = torch.from_numpy(tile["timestamp"])[None, None].repeat(n, 1, 1)
    fields["timestamps"] = timestamps.to(device)
    return MaskedOlmoEarthSample(**fields)


def main() -> None:
    """Embed every input tile not already in ``--out``."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--readout", choices=sorted(READOUT_KEYS))
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--location_query_patch_size", type=int, default=2)
    p.add_argument("--tile", type=int, default=TILE, choices=(16, 32, 64, 128, 256))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load(a.checkpoint).to(device).eval()
    head = getattr(model, "text_head", None)
    readout = a.readout or ("head" if head is not None else "student")
    encoder = model.encoder
    encoder._maybe_get_attn_mask = types.MethodType(
        _attn_mask_respecting_padding, encoder
    )
    normalizer = Normalizer(Strategy.COMPUTED)
    os.makedirs(a.out, exist_ok=True)
    proj = None
    if readout == "head":
        if head is None:
            raise ValueError(f"{a.checkpoint} has no text head")
        proj = head.proj
        meta = head.embeddings.meta
        scale = float(head.log_scale.float().exp().clamp(max=100.0))
        info = {
            "checkpoint": a.checkpoint,
            "model": meta["model"],
            "prompt_name": meta["prompt_name"],
            "dim": head.embeddings.dim,
            "scale": scale,
        }
        json.dump(info, open(os.path.join(a.out, "head.json"), "w"), indent=1)
    for path in sorted(glob.glob(os.path.join(a.inputs, "*.npz"))):
        file_id = os.path.basename(path)[:-4]
        out = os.path.join(a.out, f"{file_id}.npz")
        if os.path.exists(out):
            continue
        tile = dict(np.load(path))
        sample = sample_for(tile, encoder, normalizer, device, a.tile)
        with (
            torch.no_grad(),
            torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            output = encoder(
                sample,
                patch_size=a.patch_size,
                input_res=10,
                fast_pass=False,
                location_query_patch_size=a.location_query_patch_size,
            )
        tokens = output[READOUT_KEYS[readout]].float()
        if proj is not None:
            with torch.no_grad():
                tokens = proj(tokens).float()
        grid = (
            torch.nn.functional.normalize(from_tiles(tokens, a.tile), dim=-1)
            .cpu()
            .numpy()
        )
        valid = ~tile["no_data"]
        grid[~valid] = 0
        np.savez(out + ".tmp.npz", vectors=grid.astype(np.float16), valid=valid)
        os.replace(out + ".tmp.npz", out)
    print("done", len(glob.glob(os.path.join(a.out, "*.npz"))), "tiles")


if __name__ == "__main__":
    main()
