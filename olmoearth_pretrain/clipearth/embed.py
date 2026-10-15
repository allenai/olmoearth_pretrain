"""Embed the open-set windows with a frozen OlmoEarth v1.3 encoder.

Each 128 px window is cut into 16 px tiles, the window size v1.3 was trained on, and
the tiles are encoded in batches. The AttentionPooler's location tokens (768-d) and the
student readout of them (128-d) are kept at ``--location_query_patch_size`` pixels per
token, stitched back into a grid per window, and reduced two ways:

* regions: for every labeled class in the window, the mean embedding over its pixels
  (weighted by the fraction of each location cell the class covers). These are the
  image side of the (image, text) pairs for CLIP training.
* tiles: the mean embedding of each 16 px tile, with the tile's centre lat/lon. These
  are the search index.

Output is one ``.npz`` per part of ``--part_size`` windows under ``--out_dir``; parts
that already exist are skipped, so a preempted job resumes where it stopped. Work is
split across Beaker replicas by ``BEAKER_REPLICA_RANK`` / ``BEAKER_REPLICA_COUNT``.

Run:
  python -m olmoearth_pretrain.clipearth.embed --checkpoint <dir> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import types

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from olmoearth_pretrain.clipearth.open_set import (
    MASK_NODATA,
    MODALITIES,
    WINDOW_SIZE,
    OpenSetWindows,
    load_records,
)
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import load_model_from_path

logger = logging.getLogger(__name__)

TILE = 16


def _attn_mask_respecting_padding(
    self: torch.nn.Module, new_mask: Tensor, fast_pass: bool
) -> Tensor | None:
    """Encoder._maybe_get_attn_mask that keeps the mask in eval when padding exists.

    The stock method returns no mask outside training, so when tiles in a batch have
    different numbers of missing tokens the shorter ones attend to padding and their
    embeddings depend on the rest of the batch.
    """
    if fast_pass:
        return None
    if not self.training and bool(new_mask.all()):
        return None
    return new_mask


def _collate(items: list[dict | None]) -> list[dict]:
    return [item for item in items if item is not None]


def _to_tiles(x: Tensor) -> Tensor:
    """``[B, H, W, ...]`` -> ``[B * (H/TILE) * (W/TILE), TILE, TILE, ...]``."""
    b, h, w = x.shape[:3]
    rest = x.shape[3:]
    x = x.reshape(b, h // TILE, TILE, w // TILE, TILE, *rest)
    x = x.movedim(3, 2)  # B, nh, nw, TILE, TILE, ...
    return x.reshape(b * (h // TILE) * (w // TILE), TILE, TILE, *rest)


def _from_tiles(x: Tensor, batch: int) -> Tensor:
    """``[B * n * n, g, g, D]`` -> ``[B, n * g, n * g, D]``."""
    n = WINDOW_SIZE // TILE
    g = x.shape[1]
    x = x.reshape(batch, n, n, g, g, x.shape[-1]).movedim(3, 2)
    return x.reshape(batch, n * g, n * g, x.shape[-1])


def build_sample(
    items: list[dict], encoder: torch.nn.Module, device: torch.device
) -> tuple[MaskedOlmoEarthSample, bool]:
    """Stack windows, cut them into tiles and build the encoder input.

    Returns the sample and whether every token is visible (so ``fast_pass`` is safe).
    """
    fields: dict[str, Tensor] = {}
    all_visible = True
    shape = items[0]["sentinel2_l2a"].shape
    for modality in MODALITIES:
        if not any(modality in item for item in items):
            continue
        num_bands = next(item[modality] for item in items if modality in item).shape[-1]
        data = torch.stack(
            [item.get(modality, torch.zeros(*shape[:3], num_bands)) for item in items]
        )
        missing = torch.stack(
            [
                item.get(f"{modality}_missing", torch.ones(shape[:3], dtype=torch.bool))
                for item in items
            ]
        )
        num_band_sets = encoder.tokenization_config.get_num_bandsets(modality)
        mask = torch.where(
            missing, MaskValue.MISSING.value, MaskValue.ONLINE_ENCODER.value
        )
        mask = mask.unsqueeze(-1).expand(*mask.shape, num_band_sets)
        all_visible &= not bool(missing.any())
        fields[modality] = _to_tiles(data).to(device, non_blocking=True)
        fields[f"{modality}_mask"] = _to_tiles(mask).to(device, non_blocking=True)
    tiles_per_window = (WINDOW_SIZE // TILE) ** 2
    timestamps = torch.stack([item["timestamps"] for item in items])
    fields["timestamps"] = timestamps.repeat_interleave(tiles_per_window, 0).to(device)
    return MaskedOlmoEarthSample(**fields), all_visible


def pool_regions(
    item: dict, location: Tensor, student: Tensor, min_pixels: int
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """Mean embedding per labeled class: ``(class_name, n_pixels, loc, student)``."""
    mask = item["mask"].to(location.device)
    cell = WINDOW_SIZE // location.shape[0]
    out = []
    for value, name in item["classes"].items():
        if value == MASK_NODATA:
            continue
        hit = (mask == value).float()
        n_pixels = int(hit.sum())
        if n_pixels < min_pixels:
            continue
        weight = torch.nn.functional.avg_pool2d(hit[None, None], cell)[0, 0]
        weight = weight / weight.sum()
        loc = (weight[..., None] * location).sum((0, 1))
        stu = (weight[..., None] * student).sum((0, 1))
        if torch.isfinite(loc).all() and torch.isfinite(stu).all():
            out.append((name, n_pixels, loc.cpu().numpy(), stu.cpu().numpy()))
    return out


def embed_part(
    items: list[dict],
    encoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, list]:
    """Encode a batch of windows and return its regions and tiles."""
    sample, all_visible = build_sample(items, encoder, device)
    with (
        torch.no_grad(),
        torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        output = encoder(
            sample,
            patch_size=args.patch_size,
            input_res=10,
            fast_pass=all_visible,
            location_query_patch_size=args.location_query_patch_size,
        )
    location = _from_tiles(output["location_tokens"].float(), len(items))
    student = _from_tiles(output["student_location_tokens"].float(), len(items))
    per_tile = (WINDOW_SIZE // TILE) ** 2
    g = location.shape[1] // (WINDOW_SIZE // TILE)
    out: dict[str, list] = {k: [] for k in _KEYS}
    for b, item in enumerate(items):
        for name, n_pixels, loc, stu in pool_regions(
            item, location[b], student[b], args.min_pixels
        ):
            out["region_window"].append(item["name"])
            out["region_class"].append(name)
            out["region_pixels"].append(n_pixels)
            out["region_loc"].append(loc)
            out["region_student"].append(stu)
        n = WINDOW_SIZE // TILE
        tile_loc = location[b].reshape(n, g, n, g, -1).mean((1, 3))
        tile_stu = student[b].reshape(n, g, n, g, -1).mean((1, 3))
        labeled = (item["mask"] != MASK_NODATA).float().reshape(n, TILE, n, TILE)
        out["tile_window"] += [item["name"]] * per_tile
        out["tile_rc"].append(
            np.stack(np.meshgrid(range(n), range(n), indexing="ij"), -1).reshape(-1, 2)
        )
        out["tile_lat_lon"].append(item["tile_lat_lon"].reshape(-1, 2))
        out["tile_labeled"].append(labeled.mean((1, 3)).reshape(-1).numpy())
        out["tile_loc"].append(tile_loc.reshape(per_tile, -1).cpu().numpy())
        out["tile_student"].append(tile_stu.reshape(per_tile, -1).cpu().numpy())
        out["window"].append(item["name"])
        out["window_split"].append(item["split"])
        out["window_slug"].append(item["slug"])
    return out


_KEYS = (
    "region_window",
    "region_class",
    "region_pixels",
    "region_loc",
    "region_student",
    "tile_window",
    "tile_rc",
    "tile_lat_lon",
    "tile_labeled",
    "tile_loc",
    "tile_student",
    "window",
    "window_split",
    "window_slug",
)


def _save(path: str, part: dict[str, list]) -> None:
    arrays = {}
    for key, values in part.items():
        if key in ("region_loc", "region_student"):
            arrays[key] = (
                np.stack(values).astype(np.float16)
                if values
                else np.zeros((0, 0), np.float16)
            )
        elif key in (
            "tile_rc",
            "tile_lat_lon",
            "tile_labeled",
            "tile_loc",
            "tile_student",
        ):
            dtype = np.float16 if key in ("tile_loc", "tile_student") else None
            arrays[key] = (
                np.concatenate(values).astype(dtype)
                if dtype
                else np.concatenate(values)
            )
        else:
            arrays[key] = np.asarray(values)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main() -> None:
    """Embed this replica's share of the windows."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--location_query_patch_size", type=int, default=2)
    parser.add_argument("--windows_per_batch", type=int, default=8)
    parser.add_argument("--part_size", type=int, default=512)
    parser.add_argument("--min_pixels", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="windows, for tests")
    parser.add_argument(
        "--rank", type=int, default=int(os.environ.get("BEAKER_REPLICA_RANK", 0))
    )
    parser.add_argument(
        "--world", type=int, default=int(os.environ.get("BEAKER_REPLICA_COUNT", 1))
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_model_from_path(args.checkpoint).encoder.to(device).eval()
    encoder._maybe_get_attn_mask = types.MethodType(
        _attn_mask_respecting_padding, encoder
    )

    records = sorted(load_records(), key=lambda r: r.name)[: args.limit]
    records = records[args.rank :: args.world]
    os.makedirs(args.out_dir, exist_ok=True)
    parts = [
        records[i : i + args.part_size] for i in range(0, len(records), args.part_size)
    ]
    logger.info(
        f"replica {args.rank}/{args.world}: {len(records)} windows in {len(parts)} parts"
    )
    for p, part_records in enumerate(parts):
        path = os.path.join(args.out_dir, f"r{args.rank:03d}_p{p:04d}.npz")
        if os.path.exists(path):
            continue
        start = time.time()
        loader = DataLoader(
            OpenSetWindows(part_records, tile_size=TILE),
            batch_size=args.windows_per_batch,
            num_workers=args.num_workers,
            collate_fn=_collate,
        )
        part: dict[str, list] = {k: [] for k in _KEYS}
        for items in loader:
            if not items:
                continue
            for key, values in embed_part(items, encoder, device, args).items():
                part[key] += values
        _save(path, part)
        logger.info(
            f"part {p + 1}/{len(parts)}: {len(part['window'])} windows, "
            f"{len(part['region_class'])} regions in {time.time() - start:.0f}s"
        )


if __name__ == "__main__":
    main()
