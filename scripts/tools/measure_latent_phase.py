"""Measure the sub-token phase offset of per-pixel latents on full-input inference.

For each ``<run_dir>/step<N>`` checkpoint, encodes the same fixed set of pretraining
crops, unmasked, at token patch size 2 with one latent per pixel (the release's
inference setting), and pools every latent of every crop into one phase statistic per
checkpoint (see :mod:`olmoearth_pretrain.nn.latent_phase`), for both the d768 registers
and the shipped d128 student. Pooling all crops puts the noise floor at about
``(classes - 1) / num_latents``, far below a training microbatch's.

Usage:
    python scripts/tools/measure_latent_phase.py \
        --run_dirs /weka/.../v1_3_rc_ld1_pixtgt_pix512 /weka/.../phase1_pix512 \
        --steps 10000 20000 35000 667200 --out phase.json
"""

import argparse
import json
import logging
from typing import Any

import numpy as np
import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from upath import UPath

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

H5PY_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_"
    "worldcover_worldpop_wri_canopy_height_map/1138828"
)
MODALITIES = ["sentinel2_l2a", "sentinel1", "landsat"]


class PhaseAccumulator:
    """Running per-class sums for the between-class variance share over many crops."""

    def __init__(self, num_classes: int, dim: int) -> None:
        """Start empty for ``num_classes`` labels of ``dim``-wide latents."""
        self.sums = torch.zeros(num_classes, dim, dtype=torch.float64)
        self.counts = torch.zeros(num_classes, dtype=torch.float64)
        self.sq = 0.0

    def add(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        """Add latents ``z`` ``[N, D]`` with class ``labels`` ``[N]``."""
        z = z.double().cpu()
        labels = labels.cpu()
        self.sums.index_add_(0, labels, z)
        self.counts += torch.bincount(labels, minlength=len(self.counts)).double()
        self.sq += float(z.pow(2).sum())

    def result(self) -> dict[str, float]:
        """Between-class variance share, plus the class-mean gap relative to the RMS spread."""
        n = self.counts.sum()
        grand = self.sums.sum(0) / n
        means = self.sums / self.counts[:, None].clamp(min=1)
        between = float((self.counts[:, None] * (means - grand).pow(2)).sum())
        total = self.sq - float(n * grand.pow(2).sum())
        out = {"ratio": between / total, "n": float(n)}
        if len(self.counts) == 2:
            rms = (total / float(n)) ** 0.5
            out["gap_over_rms"] = float((means[1] - means[0]).norm()) / rms
        return out


def make_sample(
    sample_dict: dict, device: torch.device, tokenization: Any
) -> MaskedOlmoEarthSample:
    """Batch-of-one sample: every token visible except missing ones (as in evals)."""
    masked: dict = {}
    for key, val in sample_dict.items():
        if key == "timestamps":
            masked[key] = torch.from_numpy(val).unsqueeze(0).to(device)
            continue
        tensor = torch.from_numpy(val).unsqueeze(0).float().to(device)
        masked[key] = tensor
        missing = (tensor == MISSING_VALUE).any(dim=-1, keepdim=True)
        num_bandsets = tokenization.get_num_bandsets(key)
        mask = torch.where(
            missing,
            torch.tensor(MaskValue.MISSING.value, device=device),
            torch.tensor(MaskValue.ONLINE_ENCODER.value, device=device),
        )
        masked[f"{key}_mask"] = mask.expand(*tensor.shape[:-1], num_bandsets).long()
    return MaskedOlmoEarthSample(**masked)


def measure(
    model: torch.nn.Module,
    samples: list[dict],
    device: torch.device,
    patch_size: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Phase statistics for the registers and the student over all ``samples``."""
    accs: dict[str, dict[str, PhaseAccumulator]] = {}
    tokenization = model.encoder.patch_embeddings.tokenization_config
    for sample_dict in samples:
        batch = make_sample(sample_dict, device, tokenization)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            out = model.encoder(batch, patch_size=patch_size)
        for name in ("registers", "student_registers"):
            if out.get(name) is None:
                continue
            grid = out[name][0].float()  # [H, W, D], one latent per pixel
            height, width, dim = grid.shape
            r = patch_size
            if batch.sentinel2_l2a is not None:
                assert (height, width) == batch.sentinel2_l2a.shape[1:3], grid.shape
            rows = torch.arange(height)[:, None].expand(height, width)
            cols = torch.arange(width)[None, :].expand(height, width)
            labels = {
                "phase": ((rows % r) * r + (cols % r), r * r),
                "col": (cols % r, r),
                "row": (rows % r, r),
                "control_col": ((cols // r) % 2, 2),
                "control_row": ((rows // r) % 2, 2),
            }
            per = accs.setdefault(name, {})
            z = grid.reshape(-1, dim)
            for key, (lab, k) in labels.items():
                per.setdefault(key, PhaseAccumulator(k, dim)).add(z, lab.reshape(-1))
    return {
        name: {key: acc.result() for key, acc in per.items()}
        for name, per in accs.items()
    }


def load_model(step_dir: UPath, device: torch.device) -> torch.nn.Module:
    """Build the run's model from its config.json and load the step's weights."""
    with (step_dir / "config.json").open() as f:
        model_config = Config.from_dict(json.load(f)["model"])
    model = model_config.build()
    load_model_and_optim_state(str(step_dir / "model_and_optim"), model)
    return model.to(device).eval()


def main() -> None:
    """Measure every requested (run, step) that exists and write one JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--hw_p", type=int, default=24)
    parser.add_argument("--token_budget", type=int, default=20000)
    parser.add_argument("--patch_size", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda")

    dataset = OlmoEarthDataset(
        h5py_dir=UPath(H5PY_DIR),
        training_modalities=MODALITIES,
        dtype=np.float32,
        normalize=True,
    )
    dataset.prepare()
    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), size=args.num_samples, replace=False)
    samples = []
    for idx in indices:
        np.random.seed(int(idx))
        torch.manual_seed(int(idx))
        _, sample = dataset[
            GetItemArgs(
                idx=int(idx),
                patch_size=args.patch_size,
                sampled_hw_p=args.hw_p,
                token_budget=args.token_budget,
            )
        ]
        samples.append(sample.as_dict(ignore_nones=True))
    logger.info("Loaded %d crops of %d px", len(samples), args.hw_p * args.patch_size)

    results: dict = {}
    for run_dir in args.run_dirs:
        run = UPath(run_dir)
        for step in args.steps:
            step_dir = run / f"step{step}"
            if not (step_dir / "config.json").exists():
                logger.info("missing %s", step_dir)
                continue
            logger.info("measuring %s", step_dir)
            model = load_model(step_dir, device)
            results.setdefault(run.name, {})[str(step)] = measure(
                model, samples, device, args.patch_size
            )
            del model
            torch.cuda.empty_cache()
            print(json.dumps({run.name: {step: results[run.name][str(step)]}}))
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
