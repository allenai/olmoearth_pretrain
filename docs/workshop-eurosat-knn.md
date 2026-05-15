# Workshop: Running EuroSAT KNN Evaluation with OlmoEarth

Quick guide to running a KNN classification eval on EuroSAT using an OlmoEarth checkpoint.

EuroSAT is a 10-class land-use classification dataset (Sentinel-2, 64x64 patches). The KNN evaluator embeds all train/val/test images, then classifies by weighted vote over the 20 nearest neighbors (cosine similarity, temperature=0.07). No training is needed -- it's a pure probe of the pretrained representations.

---

## 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/allenai/olmoearth_pretrain.git
cd olmoearth_pretrain

# Install uv if you don't have it
pip install uv

# Create venv and install with eval dependencies
uv sync --locked --extra all-no-flash
source .venv/bin/activate
```

You need one GPU (any modern NVIDIA GPU with >=16GB VRAM is fine for KNN eval).

## 2. Download the Checkpoint

Pick a model size. Nano is the smallest and fastest:

```bash
pip install huggingface_hub

# Pick one:
huggingface-cli download allenai/OlmoEarth-v1-Nano  --local-dir ./OlmoEarth-v1-Nano   # fastest
huggingface-cli download allenai/OlmoEarth-v1-Tiny  --local-dir ./OlmoEarth-v1-Tiny
huggingface-cli download allenai/OlmoEarth-v1-Base  --local-dir ./OlmoEarth-v1-Base
huggingface-cli download allenai/OlmoEarth-v1-Large --local-dir ./OlmoEarth-v1-Large  # best accuracy
```

## 3. Download the EuroSAT Dataset

EuroSAT is part of the GeoBench benchmark suite. Download it from our public GCS bucket:

```bash
# Full benchmark suite (~15 GB)
gsutil -m rsync -r gs://ai2-olmoearth-projects-public-data/research_benchmarks /path/to/research_benchmarks

# Point the eval code at it
export GEOBENCH_DIR="/path/to/research_benchmarks/geobench"
```

If you only want EuroSAT, you can scope the sync:

```bash
gsutil -m rsync -r \
  gs://ai2-olmoearth-projects-public-data/research_benchmarks/geobench/classification_v1.0/m-eurosat \
  /path/to/research_benchmarks/geobench/classification_v1.0/m-eurosat

export GEOBENCH_DIR="/path/to/research_benchmarks/geobench"
```

## 4. Run the KNN Eval

> **Note:** The `full_eval_sweep` pipeline uses `--trainer.load_path` which expects
> olmo-core distributed checkpoints (the internal training format). The HuggingFace
> repos contain `weights.pth` in a different format, so `full_eval_sweep` will
> silently skip weight loading and give random-encoder accuracy (~67%).
>
> Use the standalone script below instead, which loads HF weights correctly.

Save this as `run_eurosat_knn.py`:

```python
"""Standalone EuroSAT KNN eval that works with HuggingFace checkpoints."""

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.geobench_dataset import GeobenchDataset
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn_variable_time
from olmoearth_pretrain.evals.embeddings import get_embeddings
from olmoearth_pretrain.evals.eval_wrapper import OlmoEarthEvalWrapper
from olmoearth_pretrain.evals.knn import run_knn
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.model_loader import load_model_from_path
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.evals.datasets import paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HF model dir (e.g. ./OlmoEarth-v1-Nano)")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {args.model_path}")
    model = load_model_from_path(args.model_path)
    encoder = model.encoder
    encoder = encoder.to(device)
    encoder.eval()

    config = dataset_to_config("m-eurosat")
    wrapper = OlmoEarthEvalWrapper(
        model=encoder,
        task_type=config.task_type,
        patch_size=args.patch_size,
        pooling_type=PoolingType.MEAN,
    )

    def make_loader(split):
        ds = GeobenchDataset(
            geobench_dir=paths.GEOBENCH_DIR,
            dataset="m-eurosat",
            split=split,
            partition="default",
            norm_stats_from_pretrained=True,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                         collate_fn=eval_collate_fn_variable_time)

    logger.info("Computing train embeddings...")
    train_emb, train_lab = get_embeddings(make_loader("train"), wrapper, is_train=True)
    logger.info("Computing val embeddings...")
    val_emb, val_lab = get_embeddings(make_loader("valid"), wrapper, is_train=False)
    logger.info("Computing test embeddings...")
    test_emb, test_lab = get_embeddings(make_loader("test"), wrapper, is_train=False)

    logger.info("Running KNN (k=20)...")
    result = run_knn(
        config=config,
        train_embeddings=train_emb,
        train_labels=train_lab,
        val_embeddings=val_emb,
        val_labels=val_lab,
        test_embeddings=test_emb,
        test_labels=test_lab,
        device=device,
        primary_metric=EvalMetric.ACCURACY,
    )

    print(f"\nVal  accuracy: {result.val_result.primary:.4f}")
    print(f"Val  metrics:  {result.val_result.metrics}")
    if result.test_result:
        print(f"Test accuracy: {result.test_result.primary:.4f}")
        print(f"Test metrics:  {result.test_result.metrics}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python run_eurosat_knn.py --model_path ./OlmoEarth-v1-Nano
```

This takes ~1-5 minutes depending on GPU. You should see 90%+ accuracy for all model
sizes. Expected approximate results:

| Model | EuroSAT Test Accuracy |
|-------|----------------------|
| Nano  | ~91%                 |
| Tiny  | ~93%                 |
| Base  | ~95%                 |
| Large | ~96%                 |

## 5. Visualize EuroSAT Samples

The GeoBench dataset loader has a built-in band visualizer. Here's a standalone script to browse EuroSAT images:

```python
"""Visualize EuroSAT samples from the GeoBench dataset."""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from geobench.task import load_task_specs

GEOBENCH_DIR = Path(os.environ.get(
    "GEOBENCH_DIR", "/path/to/research_benchmarks/geobench"
))

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

dataset_dir = GEOBENCH_DIR / "classification_v1.0" / "m-eurosat"
task = load_task_specs(dataset_dir)

# Patch get_dataset_dir (GeoBench quirk)
from types import MethodType
task.get_dataset_dir = MethodType(lambda self: dataset_dir, task)

dataset = task.get_dataset(split="test", partition_name="default")

# Plot a 4x4 grid of samples as RGB composites
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    sample = dataset[i * 50]  # spread across the dataset
    bands = [sample.bands[j].data for j in range(len(sample.bands))]
    x = np.stack(bands, axis=-1)

    # RGB = bands 3 (Red), 2 (Green), 1 (Blue) in GeoBench S2 order
    rgb = x[:, :, [3, 2, 1]].astype(float)
    # Simple percentile stretch for visualization
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)

    label = sample.label
    ax.imshow(rgb)
    ax.set_title(EUROSAT_CLASSES[label] if label < len(EUROSAT_CLASSES) else str(label),
                 fontsize=9)
    ax.axis("off")

plt.suptitle("EuroSAT Samples (Sentinel-2 RGB)", fontsize=14)
plt.tight_layout()
plt.savefig("eurosat_samples.png", dpi=150)
plt.show()
print("Saved to eurosat_samples.png")
```

Run it:

```bash
python visualize_eurosat.py
```

---

## How the KNN Eval Works Under the Hood

1. The model encodes every train/val/test EuroSAT image into an embedding vector (mean-pooled patch tokens).
2. For each val/test image, find the 20 nearest training embeddings by cosine similarity.
3. Predict the class via temperature-weighted vote: `weight = exp(similarity / 0.07)`.
4. Report accuracy (primary metric) and per-class F1.

Key code paths:
- Task registry: `olmoearth_pretrain/internal/all_evals.py` (`EVAL_TASKS["m_eurosat"]`)
- KNN logic: `olmoearth_pretrain/evals/knn.py`
- Dataset loader: `olmoearth_pretrain/evals/datasets/geobench_dataset.py`
- Dataset config: `olmoearth_pretrain/evals/datasets/configs.py`
