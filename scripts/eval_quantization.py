#!/usr/bin/env python3
"""Evaluate FP32 vs FP8 vs FP4 quantization on real satellite data.

Loads EuroSAT multispectral (13 Sentinel-2 bands) and runs KNN classification
with proper band mapping and normalization matching OlmoEarth's training pipeline.

Usage:
    python scripts/eval_quantization.py --model-id OLMOEARTH_V1_BASE
    python scripts/eval_quantization.py --model-id OLMOEARTH_V1_BASE --wandb
    python scripts/eval_quantization.py --model-id OLMOEARTH_V1_BASE --precision fp8

Requirements:
    pip install rasterio nvidia-modelopt
    EuroSAT_MS dataset: https://zenodo.org/records/7711810/files/EuroSAT_MS.zip
    Optional: pip install wandb (for logging)
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

# MSVC compatibility patch for ModelOpt CUDA extension
if sys.platform == "win32":
    import torch.utils.cpp_extension as torch_cpp_ext

    _orig = torch_cpp_ext.load

    def _patched(*a, **kw):
        f = list(kw.get("extra_cuda_cflags", []))
        f.append("-DUSE_CUDA")
        kw["extra_cuda_cflags"] = f
        return _orig(*a, **kw)

    torch_cpp_ext.load = _patched

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.quantization import (
    check_modelopt_available,
    count_quantizer_nodes,
    quantize_model,
)

# EuroSAT Sentinel-2 bands (13 bands in the .tif files, in this order):
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
EUROSAT_S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

# OlmoEarth expects Sentinel-2 L2A in this order (12 bands, no B10):
# B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09
OLMOEARTH_S2_BAND_ORDER = Modality.SENTINEL2_L2A.band_order
# Map: for each OlmoEarth band, find its index in EuroSAT's 13-band array
EUROSAT_TO_OLMOEARTH_INDICES = [
    EUROSAT_S2_BANDS.index(b) for b in OLMOEARTH_S2_BAND_ORDER
]


class EuroSATMultispectralDataset(Dataset):
    """EuroSAT multispectral dataset with proper Sentinel-2 band mapping.

    Loads .tif files with all 13 Sentinel-2 bands, selects the 12 bands
    used by OlmoEarth, reorders them, and normalizes using pretrained stats.
    """

    def __init__(self, data_dir: Path, split: str = "train", train_ratio: float = 0.8, seed: int = 42):
        self.data_dir = data_dir
        self.classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all samples
        all_samples = []
        for class_name in self.classes:
            class_dir = data_dir / class_name
            for tif_path in sorted(class_dir.glob("*.tif")):
                all_samples.append((tif_path, self.class_to_idx[class_name]))

        # Deterministic train/test split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))
        n_train = int(len(all_samples) * train_ratio)

        if split == "train":
            self.samples = [all_samples[i] for i in indices[:n_train]]
        else:
            self.samples = [all_samples[i] for i in indices[n_train:]]

        # Initialize pretrained normalizer
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        self.normalizer = Normalizer(Strategy.COMPUTED)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tif_path, label = self.samples[idx]

        # Read all 13 bands from .tif
        with rasterio.open(tif_path) as src:
            data = src.read()  # (13, H, W)

        # Transpose to (H, W, 13)
        data = data.transpose(1, 2, 0).astype(np.float32)

        # Select and reorder to OlmoEarth's 12-band order
        data_12 = data[:, :, EUROSAT_TO_OLMOEARTH_INDICES]  # (H, W, 12)

        # Add time dimension: (H, W, 1, 12)
        data_12 = np.expand_dims(data_12, axis=2)

        # Normalize using pretrained stats (same as training pipeline)
        data_12 = self.normalizer.normalize(Modality.SENTINEL2_L2A, data_12)

        # Create OlmoEarth sample
        s2_tensor = torch.tensor(data_12, dtype=torch.float32)
        timestamp = torch.tensor([[1, 6, 2020]], dtype=torch.long)  # June 1, 2020

        from olmoearth_pretrain.data.dataset import OlmoEarthSample
        from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample as TrainMasked
        sample = MaskedOlmoEarthSample.from_olmoearthsample(
            OlmoEarthSample(sentinel2_l2a=s2_tensor, timestamps=timestamp)
        )

        return sample, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Collate MaskedOlmoEarthSample batches."""
    from torch.utils.data.dataloader import default_collate
    samples, targets = zip(*batch)
    collated_sample = default_collate([s.as_dict() for s in samples])
    collated_target = default_collate(list(targets))
    return MaskedOlmoEarthSample(**collated_sample), collated_target


def find_eurosat_ms_dir() -> Path:
    """Find the EuroSAT_MS data directory."""
    candidates = [
        Path("/root/dataset/eurosat_ms/EuroSAT_MS"),
        Path("/root/dataset/EuroSAT_MS"),
        Path.home() / "dataset" / "eurosat_ms" / "EuroSAT_MS",
        Path.home() / "dataset" / "EuroSAT_MS",
        Path("data") / "EuroSAT_MS",
    ]
    env_dir = os.environ.get("EUROSAT_MS_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir))

    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c

    raise FileNotFoundError(
        "EuroSAT_MS not found. Download from:\n"
        "  wget https://zenodo.org/records/7711810/files/EuroSAT_MS.zip\n"
        "  unzip EuroSAT_MS.zip\n"
        f"Searched: {[str(c) for c in candidates]}"
    )


def extract_embeddings(encoder, dataset, device="cuda", max_samples=None, patch_size=16, batch_size=32):
    """Extract embeddings from dataset."""
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    all_embeddings = []
    all_labels = []
    n_processed = 0

    encoder.eval()
    for batch_sample, batch_label in loader:
        # Move to device
        sample_dict = batch_sample.as_dict()
        for key, val in sample_dict.items():
            sample_dict[key] = val.to(device=device)
        batch_sample = MaskedOlmoEarthSample.from_dict(sample_dict)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = encoder(batch_sample, patch_size=patch_size)

        emb = output["project_aggregated"].cpu()
        all_embeddings.append(emb)
        all_labels.append(batch_label)

        n_processed += emb.shape[0]
        if n_processed % 1000 < batch_size:
            print(f"    {n_processed}/{len(dataset)} samples processed")

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def knn_classify(train_emb, train_labels, test_emb, test_labels, k=20):
    """KNN classification using cosine similarity (matches official eval pipeline)."""
    train_emb = F.normalize(train_emb.float(), dim=1)
    test_emb = F.normalize(test_emb.float(), dim=1)
    k = min(k, len(train_emb))

    sim = test_emb @ train_emb.t()
    topk_sim, topk_idx = sim.topk(k, dim=1)

    # Weighted voting with temperature 0.07
    weights = torch.exp(topk_sim / 0.07)
    topk_labels = train_labels[topk_idx]

    n_classes = train_labels.max().item() + 1
    votes = torch.zeros(len(test_emb), n_classes)
    for c in range(n_classes):
        mask = (topk_labels == c).float()
        votes[:, c] = (weights * mask).sum(dim=1)

    preds = votes.argmax(dim=1)
    accuracy = (preds == test_labels).float().mean().item()

    per_class_acc = {}
    for c in range(n_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (preds[mask] == c).float().mean().item()
    macro_acc = sum(per_class_acc.values()) / len(per_class_acc)

    return {"accuracy": accuracy, "macro_accuracy": macro_acc, "n_test": len(test_labels), "n_train": len(train_labels), "k": k}


def make_calib_fn(device="cuda"):
    """Create calibration function for quantization."""
    def calib_fn(m):
        m.eval()
        s2_spec = Modality.SENTINEL2_L2A
        for _ in range(16):
            s2 = torch.randn(4, 64, 64, 1, s2_spec.num_bands, device=device)
            mask = torch.full(
                (4, 64, 64, 1, s2_spec.num_band_sets),
                MaskValue.ONLINE_ENCODER.value, dtype=torch.long, device=device,
            )
            ts = torch.zeros(4, 1, 3, dtype=torch.long, device=device)
            sample = MaskedOlmoEarthSample(sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=ts)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    m(sample, patch_size=2)
    return calib_fn


def benchmark_latency(encoder, device="cuda", patch_size=16, warmup=10, iters=50):
    """Measure inference latency per sample."""
    s2_spec = Modality.SENTINEL2_L2A
    s2 = torch.randn(1, 64, 64, 1, s2_spec.num_bands, device=device)
    mask = torch.full(
        (1, 64, 64, 1, s2_spec.num_band_sets),
        MaskValue.ONLINE_ENCODER.value, dtype=torch.long, device=device,
    )
    ts = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
    sample = MaskedOlmoEarthSample(sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=ts)

    encoder.eval()
    for _ in range(warmup):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                encoder(sample, patch_size=patch_size)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                encoder(sample, patch_size=patch_size)
    torch.cuda.synchronize()
    return ((time.time() - t0) / iters) * 1000


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantization on real Sentinel-2 data")
    parser.add_argument("--model-id", default="OLMOEARTH_V1_BASE", choices=[m.name for m in ModelID])
    parser.add_argument("--max-train", type=int, default=None, help="Max training samples for KNN (default: all)")
    parser.add_argument("--max-test", type=int, default=None, help="Max test samples (default: all)")
    parser.add_argument("--precision", type=str, nargs="+", default=["fp32", "fp8", "fp4"],
                        choices=["fp32", "fp8", "fp4"], help="Precisions to test")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for model")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--eurosat-dir", type=str, default=None, help="Path to EuroSAT_MS directory")
    args = parser.parse_args()

    # W&B setup
    if args.wandb:
        import wandb
        wandb.init(
            entity="2imi9-northeastern-university",
            project="OlmoEarth_Q",
            config={
                "model_id": args.model_id,
                "max_train": args.max_train,
                "max_test": args.max_test,
                "patch_size": args.patch_size,
                "task": "eurosat_ms_knn",
                "data": "EuroSAT_MS (13 Sentinel-2 bands)",
                "normalization": "pretrained_computed",
            },
        )

    print("=" * 60)
    print("  OlmoEarth Quantization Evaluation")
    print(f"  Model: {args.model_id}")
    print(f"  Task: EuroSAT 10-class KNN (Sentinel-2 multispectral)")
    print(f"  Patch size: {args.patch_size}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    # Load dataset
    print("\nStep 1: Loading EuroSAT multispectral...")
    eurosat_dir = Path(args.eurosat_dir) if args.eurosat_dir else find_eurosat_ms_dir()
    train_ds = EuroSATMultispectralDataset(eurosat_dir, split="train")
    test_ds = EuroSATMultispectralDataset(eurosat_dir, split="test")
    print(f"  {len(train_ds)} train, {len(test_ds)} test, {len(train_ds.classes)} classes")
    print(f"  Classes: {train_ds.classes}")
    print(f"  Band mapping: 13 S2 bands -> 12 OlmoEarth bands (no B10)")
    print(f"  Normalization: pretrained computed stats")

    # Load model
    print(f"\nStep 2: Loading {args.model_id}...")
    model = load_model_from_id(ModelID[args.model_id], load_weights=True).cuda().eval()
    encoder = model.encoder

    # Filter precisions
    precisions = list(args.precision)
    if not check_modelopt_available():
        precisions = [p for p in precisions if p == "fp32"]
        print("  WARNING: nvidia-modelopt not available, testing FP32 only")

    results = {}

    for precision in precisions:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating {precision.upper()}")
        print("=" * 60)

        if precision == "fp32":
            enc = encoder
        else:
            enc = copy.deepcopy(encoder)
            calib_fn = make_calib_fn()
            enc = quantize_model(enc, calib_fn, precision=precision)
            n_quantizers = count_quantizer_nodes(enc)
            print(f"  Quantizer nodes inserted: {n_quantizers}")

        # Measure latency
        print(f"  Measuring latency...")
        try:
            latency_ms = benchmark_latency(enc, patch_size=args.patch_size)
            print(f"  Latency: {latency_ms:.1f} ms/sample")
        except RuntimeError as e:
            if "CUDA_HOME" in str(e) or "MX" in str(e):
                print(f"  Latency: skipped (CUDA toolkit needed)")
                latency_ms = None
            else:
                raise

        # Extract embeddings
        print(f"  Extracting train embeddings...")
        t0 = time.time()
        try:
            train_emb, train_labels = extract_embeddings(
                enc, train_ds, max_samples=args.max_train, patch_size=args.patch_size
            )
            print(f"  Extracting test embeddings...")
            test_emb, test_labels = extract_embeddings(
                enc, test_ds, max_samples=args.max_test, patch_size=args.patch_size
            )
        except RuntimeError as e:
            if "CUDA_HOME" in str(e) or "MX" in str(e):
                print(f"  Embedding extraction failed (CUDA toolkit needed)")
                continue
            raise
        embed_time = time.time() - t0
        print(f"  Embedding extraction: {embed_time:.1f}s")
        print(f"  Embedding dim: {train_emb.shape[1]}")

        # KNN
        print(f"  Running KNN classification (k=20)...")
        knn_result = knn_classify(train_emb, train_labels, test_emb, test_labels)

        results[precision] = {**knn_result, "embed_time": embed_time, "embed_dim": train_emb.shape[1], "latency_ms": latency_ms}

        print(f"  Accuracy:       {knn_result['accuracy']:.4f}")
        print(f"  Macro Accuracy: {knn_result['macro_accuracy']:.4f}")

        if args.wandb:
            log_dict = {
                f"{precision}/accuracy": knn_result["accuracy"],
                f"{precision}/macro_accuracy": knn_result["macro_accuracy"],
                f"{precision}/embed_time_s": embed_time,
            }
            if latency_ms is not None:
                log_dict[f"{precision}/latency_ms"] = latency_ms
            wandb.log(log_dict)

        if precision != "fp32":
            del enc
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY — EuroSAT KNN Classification")
    print("=" * 60)
    print(f"  {'Precision':<10} {'Accuracy':>10} {'Macro Acc':>10} {'Latency(ms)':>12} {'Time (s)':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    for prec, r in results.items():
        lat_str = f"{r['latency_ms']:.1f}" if r['latency_ms'] is not None else "N/A"
        print(f"  {prec.upper():<10} {r['accuracy']:>10.4f} {r['macro_accuracy']:>10.4f} {lat_str:>12} {r['embed_time']:>10.1f}")

    if "fp32" in results:
        fp32_acc = results["fp32"]["accuracy"]
        print(f"\n  Accuracy drop vs FP32:")
        for prec, r in results.items():
            if prec != "fp32":
                drop = fp32_acc - r["accuracy"]
                print(f"    {prec.upper()}: {drop:+.4f} ({drop*100:+.2f}%)")

    if args.wandb:
        summary = {}
        for prec, r in results.items():
            summary[f"{prec}_accuracy"] = r["accuracy"]
            summary[f"{prec}_macro_accuracy"] = r["macro_accuracy"]
            if r["latency_ms"] is not None:
                summary[f"{prec}_latency_ms"] = r["latency_ms"]
        if "fp32" in results:
            for prec in ["fp8", "fp4"]:
                if prec in results:
                    summary[f"{prec}_accuracy_drop"] = results["fp32"]["accuracy"] - results[prec]["accuracy"]
        wandb.log(summary)
        wandb.finish()

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
