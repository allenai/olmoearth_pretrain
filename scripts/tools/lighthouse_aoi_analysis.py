"""Seams, agreement and panels for the rasters ``lighthouse_aoi_inference.py`` wrote.

Seam metrics are computed on the dequantized embeddings, so they do not depend on
any rendering basis. For each window and configuration, with
``d(x) = mean_{band,row} |E[:, :, x+1] - E[:, :, x]|`` (and the same down rows):

* ``tile12``: mean ``d`` across the merge seams of the tiled crop grid (rslearn's
  crop starts + the merger's overlap // 2 trim) over mean ``d`` at non-seam positions
  with the same phase mod 4; a seamless map gives ~1. Comparing within the phase
  mod 4 keeps a 4 px patch lattice from posing as a tile seam.
* ``lattice4``: max over median of the per-phase means of ``d`` modulo 4, with seam
  positions excluded: the ps4 patch grid (latents are per pixel but tokens are 4x4
  patches, so a patch lattice is the other artifact to look for).
* ``detected``: the period-free statistic of the AOI seam_metric.py (autocorrelation
  peak in lags 4..64, then peak-phase mean over off-phase median), for anything
  periodic the fixed periods miss -- e.g. Lighthouse chunk edges, which should not
  exist.

Agreement: per-pixel cosine between configurations on the same pixels.

Panels: PCA RGB in ONE frame for every configuration and window (all four
configurations are the same model, so their spaces coincide and no rotation is
needed): basis and 2/98 stretch fitted on a pooled sample. Plus S2 median true
colour, and native-resolution zoom crops for spotting seams.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

QUANTIZE_POWER = 2.0
QUANTIZE_SCALE = 127.5


def dequantize(q: np.ndarray) -> np.ndarray:
    """int8 (bands, H, W) -> float32 (H, W, bands)."""
    r = q.astype(np.float32) / QUANTIZE_SCALE
    return np.moveaxis(np.abs(r) ** QUANTIZE_POWER * np.sign(r), 0, -1)


def diff_profile(e: np.ndarray, axis: int) -> np.ndarray:
    """Mean |adjacent difference| along ``axis`` (0 = rows, 1 = cols) of (H, W, C)."""
    d = np.abs(np.diff(e, axis=axis))
    return d.mean(axis=(1 - axis, 2))


def phase_ratio(
    prof: np.ndarray, period: int, exclude: np.ndarray | None = None
) -> tuple[float, int]:
    """Max over median of the per-phase means, and the max phase.

    ``exclude`` masks positions out of the phase means (e.g. tile-seam differences,
    which would otherwise leak into a shorter period that divides the tile stride).
    """
    idx = np.arange(prof.size)
    keep = np.ones(prof.size, bool) if exclude is None else ~exclude
    phases = np.array([prof[(idx % period == k) & keep].mean() for k in range(period)])
    return float(phases.max() / np.median(phases)), int(phases.argmax())


def tile_seam_positions(n: int, crop: int = 16, overlap: int = 4) -> np.ndarray:
    """Bool mask over the n-1 adjacent differences that straddle a tiled-merge seam.

    rslearn crops start at 0, crop - overlap, ... and a last crop at n - crop; each
    non-first crop is kept from start + overlap // 2, so the seam is the difference
    between columns start + overlap // 2 - 1 and start + overlap // 2.
    """
    starts = [0, *range(crop - overlap, n - crop, crop - overlap)]
    if n - crop > 0:
        starts.append(n - crop)
    mask = np.zeros(n - 1, bool)
    for st in starts[1:]:
        mask[st + overlap // 2 - 1] = True
    return mask


def detected_seam(
    prof: np.ndarray, min_p: int = 4, max_p: int = 64
) -> tuple[float, int]:
    """Period-free seam ratio (as in the AOI seam_metric.py)."""
    x = prof - prof.mean()
    if x.size < 2 * max_p or not np.any(x):
        return float("nan"), 0
    ac = np.correlate(x, x, mode="full")[x.size - 1 :]
    ac /= ac[0]
    p = int(np.arange(min_p, max_p + 1)[np.argmax(ac[min_p : max_p + 1])])
    n = (prof.size // p) * p
    phases = prof[:n].reshape(-1, p)
    peak = int(phases.mean(0).argmax())
    off = np.delete(phases, peak, axis=1)
    return float(phases[:, peak].mean() / np.median(off)), p


def seam_stats(e: np.ndarray) -> dict[str, float]:
    """All seam metrics of one embedding raster, both axes."""
    out: dict[str, float] = {}
    for axis, tag in ((1, "x"), (0, "y")):
        prof = diff_profile(e, axis)
        # Tile seams vs. non-seam differences at the SAME phase mod 4, so a 4 px
        # patch lattice (4 divides the 12 px stride) cannot pose as a tile seam.
        seams = tile_seam_positions(prof.size + 1)
        idx = np.arange(prof.size)
        same4 = (idx % 4 == (np.nonzero(seams)[0][0] % 4)) & ~seams
        out[f"tile12_{tag}"] = float(prof[seams].mean() / prof[same4].mean())
        out[f"tile12_phase_{tag}"] = int(np.nonzero(seams)[0][0] % 12)
        # Tile seams share a phase mod 4 (12 is a multiple of 4), so they are
        # excluded before looking for the 4 px patch lattice.
        out[f"lattice4_{tag}"], out[f"lattice4_phase_{tag}"] = phase_ratio(
            prof, 4, exclude=seams
        )
        out[f"detected_{tag}"], out[f"detected_period_{tag}"] = detected_seam(prof)
        out[f"mean_adjdiff_{tag}"] = float(prof.mean())
    return out


def stretch(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-channel linear stretch to uint8."""
    return (np.clip((rgb - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def save(img: np.ndarray, path: Path, size: int | None, quality: int = 82) -> None:
    """Save RGB uint8; downsample (LANCZOS) to ``size`` px on the long side if given."""
    im = Image.fromarray(img)
    if size and max(im.size) > size:
        s = size / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    if path.suffix == ".png":
        im.save(path)
    else:
        im.save(path, quality=quality, progressive=True)


def s2_true_colour(dataset: str, name: str) -> np.ndarray:
    """Per-pixel median of the twelve S2 mosaics, B04/B03/B02, 2-98 stretched."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from lighthouse_aoi_inference import build_window_dataset

    ds = build_window_dataset(dataset, "predict", [name])
    inputs, _, _ = ds[0]
    img = inputs["sentinel2_l2a"].image  # (C, T, H, W); bands B02, B03, B04, ...
    img = img.numpy() if hasattr(img, "numpy") else img
    rgb = np.median(img[[2, 1, 0]], axis=1).transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(rgb, 2, axis=(0, 1)), np.percentile(rgb, 98, axis=(0, 1))
    return stretch(rgb, lo, hi)


def main() -> None:
    """Compute metrics and render panels for every window and configuration."""
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default=None, help="for S2 true colour panels")
    p.add_argument(
        "--configs", nargs="+", default=["tiled_ps1", "lh_ps1", "tiled_ps4", "lh_ps4"]
    )
    p.add_argument("--panel_px", type=int, default=640)
    p.add_argument("--zoom_px", type=int, default=160)
    p.add_argument("--sample_per_window", type=int, default=20000)
    a = p.parse_args()
    root = Path(a.out_dir)
    names = sorted(f.stem for f in (root / a.configs[0]).glob("*.tif"))
    rng = np.random.default_rng(0)

    # Pass 1: metrics + pooled PCA sample (same pixels for every configuration).
    metrics: dict[str, dict] = {}
    pool = []
    for name in names:
        embs = {}
        for cfg in a.configs:
            with rasterio.open(root / cfg / f"{name}.tif") as src:
                embs[cfg] = dequantize(src.read())
        H, W, _ = embs[a.configs[0]].shape
        idx = rng.choice(H * W, size=min(a.sample_per_window, H * W), replace=False)
        m: dict[str, dict] = {"seams": {}, "agreement": {}}
        for cfg, e in embs.items():
            m["seams"][cfg] = seam_stats(e)
            m["seams"][cfg]["norm_mean"] = float(np.linalg.norm(e, axis=-1).mean())
            pool.append(e.reshape(-1, e.shape[-1])[idx])
        for i, c1 in enumerate(a.configs):
            for c2 in a.configs[i + 1 :]:
                e1, e2 = embs[c1], embs[c2]
                cos = (e1 * e2).sum(-1) / (
                    np.linalg.norm(e1, axis=-1) * np.linalg.norm(e2, axis=-1) + 1e-8
                )
                m["agreement"][f"{c1}~{c2}"] = {
                    "cos_mean": float(cos.mean()),
                    "cos_p05": float(np.percentile(cos, 5)),
                }
        metrics[name] = m
        print(
            name,
            json.dumps({c: round(v["tile12_x"], 3) for c, v in m["seams"].items()}),
        )

    X = np.concatenate(pool)
    mu = X.mean(0)
    _, svals, vt = np.linalg.svd(
        X[rng.choice(len(X), min(len(X), 200000), replace=False)] - mu,
        full_matrices=False,
    )
    basis = vt[:3].T
    proj = (X - mu) @ basis
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    frame = {
        "explained_top3": float((svals[:3] ** 2).sum() / (svals**2).sum()),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
    }

    # Pass 2: panels.
    panels = root / "panels"
    panels.mkdir(exist_ok=True)
    for name in names:
        zoom = None
        for cfg in a.configs:
            with rasterio.open(root / cfg / f"{name}.tif") as src:
                e = dequantize(src.read())
            H, W, _ = e.shape
            rgb = stretch(
                ((e.reshape(-1, e.shape[-1]) - mu) @ basis).reshape(H, W, 3), lo, hi
            )
            save(rgb, panels / f"{name}__{cfg}.jpg", a.panel_px)
            if zoom is None:
                # Zoom where tiled seams are most visible: the crop with the largest
                # tile-phase excess in the first configuration.
                zr = (H // 2 - a.zoom_px // 2) // 12 * 12
                zc = (W // 2 - a.zoom_px // 2) // 12 * 12
                zoom = (zr, zc)
            zr, zc = zoom
            save(
                rgb[zr : zr + a.zoom_px, zc : zc + a.zoom_px],
                panels / f"{name}__{cfg}__zoom.png",
                None,
            )
        if a.dataset:
            s2 = s2_true_colour(a.dataset, name)
            save(s2, panels / f"{name}__s2.jpg", a.panel_px)
            zr, zc = zoom  # type: ignore[misc]
            save(
                s2[zr : zr + a.zoom_px, zc : zc + a.zoom_px],
                panels / f"{name}__s2__zoom.png",
                None,
            )
        metrics[name]["zoom_origin"] = list(zoom)  # type: ignore[arg-type]

    (root / "analysis.json").write_text(
        json.dumps({"frame": frame, "windows": metrics}, indent=1)
    )
    print("wrote", root / "analysis.json")


if __name__ == "__main__":
    main()
