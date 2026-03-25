#!/usr/bin/env python3
"""Evaluate FP32 vs FP8 vs FP4 quantization on real satellite data.

Downloads EuroSAT (10-class land cover) via GeoBench and runs KNN classification
to measure actual downstream accuracy — not just cosine similarity.

Usage:
    python scripts/eval_quantization.py --model-id OLMOEARTH_V1_NANO
    python scripts/eval_quantization.py --model-id OLMOEARTH_V1_NANO --wandb

Requirements:
    pip install geobench nvidia-modelopt
    Optional: pip install wandb (for logging)
"""
from __future__ import annotations

import argparse
import copy
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.quantization import (
    check_modelopt_available,
    quantize_model,
)


def load_eurosat():
    """Load EuroSAT dataset from disk.

    EuroSAT RGB (3-channel) is used as a proxy for Sentinel-2 imagery.
    We expand 3 RGB channels to 12 bands (padding) to match OlmoEarth's input format.
    This is valid for relative comparison (FP32 vs FP8 vs FP4) since the same
    conversion is applied to all precisions.
    """
    from torchvision.datasets import ImageFolder

    # Search common locations
    candidates = [
        Path.home() / "dataset" / "eurosat" / "eurosat" / "EuroSAT_RGB",
        Path.home() / "dataset" / "eurosat" / "EuroSAT_RGB",
        Path.home() / "dataset" / "EuroSAT_RGB",
        Path("data") / "EuroSAT_RGB",
    ]
    data_dir = None
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            data_dir = c
            break

    if data_dir is None:
        raise FileNotFoundError(
            "EuroSAT_RGB not found. Download from:\n"
            "  https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip\n"
            f"Extract to: {candidates[0]}"
        )

    full_ds = ImageFolder(root=str(data_dir))
    print(f"  EuroSAT: {len(full_ds)} samples, {len(full_ds.classes)} classes")
    print(f"  Classes: {full_ds.classes}")

    # Deterministic 80/20 split
    n = len(full_ds)
    n_train = int(n * 0.8)
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=g).tolist()
    train_ds = torch.utils.data.Subset(full_ds, indices[:n_train])
    test_ds = torch.utils.data.Subset(full_ds, indices[n_train:])
    print(f"  Split: {len(train_ds)} train, {len(test_ds)} test")
    return train_ds, test_ds


def eurosat_to_olmoearth(img, label, device="cuda"):
    """Convert a torchvision EuroSAT sample to OlmoEarth format."""
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    import numpy as np

    s2_spec = Modality.SENTINEL2_L2A

    # img is a PIL Image (64x64 RGB)
    x = torch.tensor(np.array(img), dtype=torch.float32)  # [H, W, 3]
    x = (x - x.mean()) / (x.std() + 1e-6)

    # Expand 3 RGB channels → 12 bands (pad with zeros for non-RGB bands)
    h, w, _ = x.shape
    x_12 = torch.zeros(h, w, 12)
    x_12[:, :, :3] = x  # B02=Blue, B03=Green, B04=Red

    # OlmoEarth format: [1, H, W, T=1, C=12]
    x_12 = x_12.unsqueeze(0).unsqueeze(3)

    mask = torch.full(
        (1, h, w, 1, s2_spec.num_band_sets),
        MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )
    ts = torch.zeros(1, 1, 3, dtype=torch.long)

    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=x_12.to(device),
        sentinel2_l2a_mask=mask.to(device),
        timestamps=ts.to(device),
    )
    return sample, label


def extract_embeddings(encoder, dataset, device="cuda", max_samples=None):
    """Extract embeddings for all samples in a dataset."""
    embeddings = []
    labels = []
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    encoder.eval()
    for i in range(n):
        img, label = dataset[i]
        sample, label = eurosat_to_olmoearth(img, label, device=device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = encoder(sample, patch_size=2)
        emb = output["project_aggregated"].cpu()  # [1, D]
        embeddings.append(emb.squeeze(0))
        labels.append(label)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} samples processed")

    return torch.stack(embeddings), torch.tensor(labels)


def knn_classify(train_emb, train_labels, test_emb, test_labels, k=20):
    """KNN classification using cosine similarity."""
    # Normalize
    train_emb = F.normalize(train_emb.float(), dim=1)
    test_emb = F.normalize(test_emb.float(), dim=1)

    # Cosine similarity: [n_test, n_train]
    sim = test_emb @ train_emb.t()

    # Top-k
    topk_sim, topk_idx = sim.topk(k, dim=1)

    # Weighted voting
    weights = torch.exp(topk_sim / 0.07)  # temperature
    topk_labels = train_labels[topk_idx]

    n_classes = train_labels.max().item() + 1
    votes = torch.zeros(len(test_emb), n_classes)
    for c in range(n_classes):
        mask = (topk_labels == c).float()
        votes[:, c] = (weights * mask).sum(dim=1)

    preds = votes.argmax(dim=1)
    accuracy = (preds == test_labels).float().mean().item()

    # Per-class accuracy
    per_class_acc = {}
    for c in range(n_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (preds[mask] == c).float().mean().item()

    macro_acc = sum(per_class_acc.values()) / len(per_class_acc)

    return {
        "accuracy": accuracy,
        "macro_accuracy": macro_acc,
        "n_test": len(test_labels),
        "n_train": len(train_labels),
        "k": k,
    }


def make_calib_fn(device="cuda"):
    """Create calibration function for quantization."""
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    def calib_fn(m):
        m.eval()
        s2_spec = Modality.SENTINEL2_L2A
        for _ in range(16):
            s2 = torch.randn(4, 64, 64, 1, s2_spec.num_bands, device=device)
            mask = torch.full(
                (4, 64, 64, 1, s2_spec.num_band_sets),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.long,
                device=device,
            )
            ts = torch.zeros(4, 1, 3, dtype=torch.long, device=device)
            sample = MaskedOlmoEarthSample(
                sentinel2_l2a=s2, sentinel2_l2a_mask=mask, timestamps=ts
            )
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    m(sample, patch_size=2)

    return calib_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantization on real data")
    parser.add_argument("--model-id", default="OLMOEARTH_V1_NANO", choices=[m.name for m in ModelID])
    parser.add_argument("--max-train", type=int, default=2000, help="Max training samples for KNN")
    parser.add_argument("--max-test", type=int, default=1000, help="Max test samples")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    args = parser.parse_args()

    # W&B setup
    if args.wandb:
        import wandb
        run = wandb.init(
            entity="2imi9-northeastern-university",
            project="OlmoEarth_Q",
            config={
                "model_id": args.model_id,
                "max_train": args.max_train,
                "max_test": args.max_test,
                "task": "eurosat_knn",
            },
        )

    print("=" * 60)
    print("  OlmoEarth Quantization Evaluation on Real Data")
    print(f"  Model: {args.model_id}")
    print(f"  Task: EuroSAT 10-class KNN classification")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    # Load dataset
    print("\nStep 1: Loading EuroSAT...")
    train_ds, test_ds = load_eurosat()

    # Load model
    print(f"\nStep 2: Loading {args.model_id}...")
    model = load_model_from_id(ModelID[args.model_id], load_weights=True).cuda().eval()
    encoder = model.encoder

    # Test each precision
    precisions = ["fp32"]
    if check_modelopt_available():
        precisions.extend(["fp8", "fp4"])
    else:
        print("  WARNING: nvidia-modelopt not available, skipping FP8/FP4")

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

        # Extract embeddings
        print(f"  Extracting train embeddings...")
        t0 = time.time()
        train_emb, train_labels = extract_embeddings(enc, train_ds, max_samples=args.max_train)
        print(f"  Extracting test embeddings...")
        test_emb, test_labels = extract_embeddings(enc, test_ds, max_samples=args.max_test)
        embed_time = time.time() - t0
        print(f"  Embedding extraction: {embed_time:.1f}s")

        # KNN classification
        print(f"  Running KNN classification (k=20)...")
        knn_result = knn_classify(train_emb, train_labels, test_emb, test_labels)

        results[precision] = {
            **knn_result,
            "embed_time": embed_time,
            "embed_dim": train_emb.shape[1],
        }

        print(f"  Accuracy:       {knn_result['accuracy']:.4f}")
        print(f"  Macro Accuracy: {knn_result['macro_accuracy']:.4f}")

        if args.wandb:
            wandb.log({
                f"{precision}/accuracy": knn_result["accuracy"],
                f"{precision}/macro_accuracy": knn_result["macro_accuracy"],
                f"{precision}/embed_time_s": embed_time,
            })

        if precision != "fp32":
            del enc
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY — EuroSAT KNN Classification")
    print("=" * 60)
    print(f"  {'Precision':<10} {'Accuracy':>10} {'Macro Acc':>10} {'Time (s)':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for prec, r in results.items():
        print(f"  {prec.upper():<10} {r['accuracy']:>10.4f} {r['macro_accuracy']:>10.4f} {r['embed_time']:>10.1f}")

    if "fp32" in results:
        fp32_acc = results["fp32"]["accuracy"]
        print(f"\n  Accuracy drop vs FP32:")
        for prec, r in results.items():
            if prec != "fp32":
                drop = fp32_acc - r["accuracy"]
                print(f"    {prec.upper()}: {drop:+.4f} ({drop*100:+.2f}%)")

    if args.wandb:
        # Log summary metrics
        summary = {}
        for prec, r in results.items():
            summary[f"{prec}_accuracy"] = r["accuracy"]
            summary[f"{prec}_macro_accuracy"] = r["macro_accuracy"]
        if "fp32" in results:
            fp32_acc = results["fp32"]["accuracy"]
            for prec in ["fp8", "fp4"]:
                if prec in results:
                    summary[f"{prec}_accuracy_drop"] = fp32_acc - results[prec]["accuracy"]
        wandb.log(summary)
        wandb.finish()

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
