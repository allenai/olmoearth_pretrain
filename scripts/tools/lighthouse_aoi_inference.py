"""Embed AOI windows with a joint-latent checkpoint, tiled vs Lighthouse, and time it.

Reads whole windows of an rslearn dataset (S2 L2A + S1 + Landsat, twelve monthly
mosaics) through the eval pipeline's own conversion
(``RslearnToOlmoEarthDataset._transform_sample``: S1 to dB, pretraining
normalization, real mosaic timestamps), then runs any of four configurations:

* ``tiled_ps{P}``: 16 px crops at overlap 4, stitched exactly like rslearn's
  ``get_window_crop_options`` + ``RasterMerger`` (the geometry of the AOI pages);
* ``lh_ps{P}``: Lighthouse (``olmoearth_pretrain.nn.lighthouse``): one sliding
  16 px FOV per query, run over spatial chunks whose halo is the exact receptive
  field, so the stitched result equals one full-window forward.

``P`` is the token patch size; latents are always per pixel (``eval_latent_stride``
1), so every configuration outputs one 128-dim student embedding per 10 m pixel. The
output is L2-normalized and int8-quantized exactly like rslp's QuantizedEmbeddingHead
and written as a 128-band GeoTIFF on the window's grid.

Timing: every model forward is bracketed by ``torch.cuda.synchronize``; data loading,
normalization, quantization and writing are excluded. Each configuration is warmed up
on the first window (compile, FlexAttention kernels) before anything is timed.
Results go to ``timings.json`` (per window and per chunk/batch) next to the rasters.

Usage (on a GPU node with the dataset and checkpoint mounted):
    python scripts/tools/lighthouse_aoi_inference.py \
        --checkpoint /weka/.../v1_3_vit0_rstride_ps8_lb512_fast2_latentread_joint12/step667200 \
        --dataset /weka/dfive-default/gabrielt/aoi_embeddings_20260819b \
        --out_dir /weka/.../lighthouse_aoi_20261001 \
        --configs tiled_ps1 tiled_ps4 lh_ps1 lh_ps4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from rslearn.train.all_crops_dataset import get_window_crop_options

from olmoearth_pretrain.evals.datasets.rslearn_dataset import RslearnToOlmoEarthDataset
from olmoearth_pretrain.model_loader import load_pretrain_checkpoint
from olmoearth_pretrain.nn.joint_latent import JointLatentTransformer
from olmoearth_pretrain.nn.lighthouse import LighthouseSettings, lighthouse_reach_px
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger("lighthouse_aoi")

MODALITIES = ["sentinel2_l2a", "sentinel1", "landsat"]
# Band orders of data/large_scale_embeddings/s2_s1_landsat*.yaml (rslearn_projects),
# the configs that materialized and embedded these AOIs.
INPUTS: dict[str, Any] = {
    "sentinel2_l2a": {
        "data_type": "raster",
        "layers": ["sentinel2_l2a"],
        "bands": [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
        "passthrough": True,
        "dtype": "FLOAT32",
        "load_all_layers": True,
        "load_all_item_groups": True,
    },
    "sentinel1": {
        "data_type": "raster",
        "layers": ["sentinel1"],
        "bands": ["vv", "vh"],
        "passthrough": True,
        "dtype": "FLOAT32",
        "load_all_layers": True,
        "load_all_item_groups": True,
        "required": False,
    },
    "landsat": {
        "data_type": "raster",
        "layers": ["landsat"],
        "bands": ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
        "passthrough": True,
        "dtype": "FLOAT32",
        "load_all_layers": True,
        "load_all_item_groups": True,
        "required": False,
    },
}

QUANTIZE_POWER = 2.0
QUANTIZE_SCALE = 127.5
NODATA = -128


class _NoLabel(RslearnToOlmoEarthDataset):
    """The eval conversion for label-free windows (EmbeddingTask has no targets)."""

    def _parse_label(self, target: dict) -> torch.Tensor:  # noqa: D102
        return torch.zeros(1, 1)


def build_window_dataset(root: str, group: str, names: list[str] | None) -> Any:
    """Whole-window rslearn ModelDataset (no crops) over the AOI windows."""
    import jsonargparse
    from rslearn.train.data_module import RslearnDataModule

    val_config: dict[str, Any] = {"groups": [group], "skip_targets": True}
    if names:
        val_config["names"] = names
    data_config = {
        "class_path": "rslearn.train.data_module.RslearnDataModule",
        "init_args": {
            "path": root,
            "inputs": INPUTS,
            "task": {"class_path": "rslearn.train.tasks.embedding.EmbeddingTask"},
            "batch_size": 1,
            "num_workers": 0,
            "val_config": val_config,
        },
    }
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--data", type=RslearnDataModule)
    data = parser.instantiate_classes(parser.parse_object({"data": data_config})).data
    data.setup("validate")
    return data.datasets["val"]


class WindowSamples(torch.utils.data.Dataset):
    """(name, MaskedOlmoEarthSample, georeference) per whole window."""

    def __init__(self, model_dataset: Any) -> None:
        """Wrap a whole-window ModelDataset with the eval conversion."""
        self.model_dataset = model_dataset
        self.convert = _NoLabel(
            model_dataset=model_dataset, input_modalities=MODALITIES
        )

    def __len__(self) -> int:
        """Number of windows."""
        return len(self.model_dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[str, MaskedOlmoEarthSample, dict[str, Any]]:
        """One window: its name, converted sample and georeference."""
        inputs, target, meta = self.model_dataset[idx]
        sample, _ = self.convert._transform_sample(inputs, target)
        proj = meta.projection
        geo = {
            "crs": proj.crs.to_wkt(),
            "x_res": proj.x_resolution,
            "y_res": proj.y_resolution,
            "bounds": list(meta.window_bounds),
        }
        return meta.window_name, sample, geo


def _batched(
    sample: MaskedOlmoEarthSample, device: torch.device
) -> dict[str, torch.Tensor]:
    """The window's tensors with a batch axis, on ``device`` (kept for slicing)."""
    out = {}
    for k, v in sample.as_dict().items():
        if v is None:
            continue
        out[k] = v.unsqueeze(0).to(device)
    return out


def _crop(
    full: dict[str, torch.Tensor], rows: slice, cols: slice
) -> MaskedOlmoEarthSample:
    return MaskedOlmoEarthSample(
        **{k: (v if k == "timestamps" else v[:, rows, cols]) for k, v in full.items()}
    )


def _stack_crops(
    full: dict[str, torch.Tensor], boxes: list[tuple[int, int, int, int]]
) -> MaskedOlmoEarthSample:
    out = {}
    for k, v in full.items():
        if k == "timestamps":
            out[k] = v.expand(len(boxes), *v.shape[1:])
        else:
            out[k] = torch.cat([v[:, r0:r1, c0:c1] for (c0, r0, c1, r1) in boxes])
    return MaskedOlmoEarthSample(**out)


class Timer:
    """CUDA-synchronized wall clock."""

    def __init__(self, device: torch.device) -> None:
        """Time work on ``device``."""
        self.device = device

    def __enter__(self) -> Timer:
        """Start the clock after pending GPU work."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the clock after pending GPU work."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self.t0


def _student(
    encoder: torch.nn.Module, sample: MaskedOlmoEarthSample, ps: int, dim: int
) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = encoder(sample, patch_size=ps, input_res=10, fast_pass=True)
    return out["student_registers"][..., :dim].float()


def run_tiled(
    encoder: torch.nn.Module,
    full: dict[str, torch.Tensor],
    H: int,
    W: int,
    ps: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rslearn's crop grid + RasterMerger (trim overlap // 2 on non-boundary sides)."""
    crop, overlap = args.window_px, args.overlap_px
    boxes = get_window_crop_options((crop, crop), (overlap, overlap), (0, 0, W, H))
    trim = overlap // 2
    out = torch.zeros(H, W, args.dim, device=device)
    # RasterMerger sorts by (col, row) and lets later crops overwrite earlier ones.
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    times = []
    for i in range(0, len(boxes), args.tiled_batch):
        chunk = boxes[i : i + args.tiled_batch]
        batch = _stack_crops(full, chunk)
        with Timer(device) as t:
            emb = _student(encoder, batch, ps, args.dim)
        times.append(t.seconds)
        for (c0, r0, c1, r1), e in zip(chunk, emb):
            dr = trim if r0 != 0 else 0
            dc = trim if c0 != 0 else 0
            out[r0 + dr : min(r1, H), c0 + dc : min(c1, W)] = e[
                dr : min(r1, H) - r0, dc : min(c1, W) - c0
            ]
    processed = len(boxes) * crop * crop
    return out, {"forward_s": sum(times), "batches": times, "processed_px": processed}


def run_lighthouse(
    encoder: torch.nn.Module,
    full: dict[str, torch.Tensor],
    H: int,
    W: int,
    ps: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Core tiles + exact halo; each chunk is one Lighthouse forward."""
    perceiver: JointLatentTransformer = encoder.perceiver
    depth = len(perceiver.joint_blocks)
    halo = lighthouse_reach_px(args.window_px, ps, depth, encoder.max_patch_size)
    halo = math.ceil(halo / ps) * ps
    core = args.core_px[ps]
    out = torch.zeros(H, W, args.dim, device=device)
    chunks = []
    processed = 0
    for r in range(0, H, core):
        for c in range(0, W, core):
            r0, c0 = max(r - halo, 0), max(c - halo, 0)
            r1, c1 = min(r + core + halo, H), min(c + core + halo, W)
            perceiver.lighthouse = LighthouseSettings(
                fov_px=args.window_px,
                seq_chunk=args.seq_chunk,
                origin_px=(r0, c0),
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                with Timer(device) as t:
                    emb = _student(
                        encoder, _crop(full, slice(r0, r1), slice(c0, c1)), ps, args.dim
                    )
            finally:
                perceiver.lighthouse = None
            stats = dict(getattr(perceiver, "last_lighthouse_stats", {}))
            logger.info(
                "  chunk %dx%d px: %.2f s, peak %.1f GiB, %s",
                r1 - r0,
                c1 - c0,
                t.seconds,
                torch.cuda.max_memory_allocated(device) / 2**30,
                {k: round(v, 3) for k, v in stats.items()},
            )
            rc1, cc1 = min(r + core, H), min(c + core, W)
            out[r:rc1, c:cc1] = emb[0, r - r0 : rc1 - r0, c - c0 : cc1 - c0]
            processed += (r1 - r0) * (c1 - c0)
            chunks.append(
                {
                    "seconds": t.seconds,
                    "chunk_px": [r1 - r0, c1 - c0],
                    "core_px": [rc1 - r, cc1 - c],
                    "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0,
                    **stats,
                }
            )
    return out, {
        "forward_s": sum(ch["seconds"] for ch in chunks),
        "chunks": chunks,
        "processed_px": processed,
        "halo_px": halo,
        "core_px": core,
    }


def quantize(emb: torch.Tensor) -> np.ndarray:
    """L2 normalize + AlphaEarth-style int8 power quantization (rslp's head)."""
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sat = emb.abs().pow(1.0 / QUANTIZE_POWER) * emb.sign()
    q = (sat * QUANTIZE_SCALE).clamp(-127, 127).round().to(torch.int8)
    return q.permute(2, 0, 1).cpu().numpy()


def write_tif(path: Path, data: np.ndarray, geo: dict[str, Any]) -> None:
    """Write a 128-band int8 GeoTIFF on the window grid."""
    x_res, y_res = geo["x_res"], geo["y_res"]
    b = geo["bounds"]
    transform = Affine(x_res, 0, b[0] * x_res, 0, y_res, b[1] * y_res)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="int8",
        crs=rasterio.crs.CRS.from_wkt(geo["crs"]),
        transform=transform,
        nodata=NODATA,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data)


def parse_config(name: str) -> tuple[str, int]:
    """``tiled_ps4`` -> ("tiled", 4)."""
    mode, ps = name.split("_ps")
    assert mode in ("tiled", "lh"), name
    return mode, int(ps)


def main() -> None:
    """Run the configurations over every window."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--group", default="predict")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--configs", nargs="+", default=["tiled_ps1", "tiled_ps4", "lh_ps1", "lh_ps4"]
    )
    p.add_argument(
        "--window_px", type=int, default=16, help="tile size = Lighthouse FOV"
    )
    p.add_argument("--overlap_px", type=int, default=4)
    p.add_argument("--tiled_batch", type=int, default=64)
    p.add_argument("--core_px_ps1", type=int, default=96)
    p.add_argument("--core_px_ps4", type=int, default=512)
    p.add_argument("--seq_chunk", type=int, default=1 << 18)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument(
        "--max_px", type=int, default=None, help="crop windows (smoke tests)"
    )
    p.add_argument("--no_write", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--masked_attention",
        action="store_true",
        help="tiled configs: keep the masked FlexAttention path instead of the "
        "mask-free dense inference attention (nn/dense_joint_attention.py)",
    )
    args = p.parse_args()
    args.core_px = {1: args.core_px_ps1, 4: args.core_px_ps4}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = torch.device("cuda")
    model = load_pretrain_checkpoint(args.checkpoint, device=device)
    encoder = model.encoder
    assert isinstance(encoder.perceiver, JointLatentTransformer)
    assert encoder.perceiver.eval_latent_stride == 1
    if not args.masked_attention:
        # Not in this checkpoint's config.json (the flag postdates it); inference
        # only, exact up to reassociation. Lighthouse forwards dispatch before it.
        from olmoearth_pretrain.nn.dense_joint_attention import flash_attn

        if flash_attn is None:
            raise RuntimeError("dense inference attention needs flash-attn installed")
        encoder.perceiver.dense_inference_attention = True
    logger.info("loaded %s on %s", args.checkpoint, torch.cuda.get_device_name(device))

    windows = WindowSamples(build_window_dataset(args.dataset, args.group, args.names))
    loader = torch.utils.data.DataLoader(
        windows,
        batch_size=None,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        prefetch_factor=1 if args.num_workers else None,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runners = {"tiled": run_tiled, "lh": run_lighthouse}
    record: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "tiled_attention": "masked_flex" if args.masked_attention else "dense_flash",
        "args": {k: v for k, v in vars(args).items() if k != "core_px"},
        "windows": {},
    }
    warmed: set[str] = set()
    for name, sample, geo in loader:
        full = _batched(sample, device)
        if args.max_px:
            full = {
                k: (v if k == "timestamps" else v[:, : args.max_px, : args.max_px])
                for k, v in full.items()
            }
            geo = dict(geo)
        H, W = full["sentinel2_l2a"].shape[1:3]
        logger.info("window %s: %dx%d px", name, H, W)
        rec: dict[str, Any] = {"H": int(H), "W": int(W), "configs": {}}
        for cfg in args.configs:
            mode, ps = parse_config(cfg)
            if cfg not in warmed:
                logger.info("warm-up %s", cfg)
                runners[mode](encoder, full, H, W, ps, args, device)
                warmed.add(cfg)
            emb, info = runners[mode](encoder, full, H, W, ps, args, device)
            info["peak_gib_window"] = max(
                [c.get("peak_gib", 0.0) for c in info.get("chunks", [])] or [0.0]
            )
            info["s_per_km2"] = info["forward_s"] / (H * W / 1e4)
            info["overhead_x"] = info["processed_px"] / (H * W)
            logger.info(
                "%s %s: %.2f s (%.3f s/km2, %.2fx processed)",
                name,
                cfg,
                info["forward_s"],
                info["s_per_km2"],
                info["overhead_x"],
            )
            rec["configs"][cfg] = info
            if not args.no_write:
                write_tif(out_dir / cfg / f"{name}.tif", quantize(emb), geo)
            del emb
        record["windows"][name] = rec
        (out_dir / "timings.json").write_text(json.dumps(record, indent=1))
    logger.info("done: %s", out_dir)


if __name__ == "__main__":
    main()
