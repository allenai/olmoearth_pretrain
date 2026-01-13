<div align="center">
  <img src="https://raw.githubusercontent.com/allenai/olmoearth_pretrain/main/assets/OlmoEarth-logo.png" alt="OlmoEarth Logo" style="width: 600px; margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
</div>
<p align="center">
  <a href="https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/license-OlmoEarth-green">
  </a>
  <a href="https://huggingface.co/collections/allenai/olmoearth">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://allenai.org/papers/olmoearth">
    <img alt="Paper PDF" src="https://img.shields.io/badge/OlmoEarth-pdf-blue">
  </a>
</p>

The OlmoEarth models are a flexible, multi-modal, spatio-temporal family of foundation models for Earth Observations.

The OlmoEarth models exist as part of the [OlmoEarth platform](https://olmoearth.allenai.org/). The OlmoEarth Platform is an end-to-end solution for scalable planetary intelligence, providing everything needed to go from raw data through R&D, to fine-tuning and production deployment.

## Installation

We recommend Python 3.12, and recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).
To install dependencies with uv, run:

```bash
git clone git@github.com:allenai/olmoearth_pretrain.git
cd olmoearth_pretrain
uv sync --locked --all-groups --python 3.12
# only necessary for development
uv tool install pre-commit --with pre-commit-uv --force-reinstall
```

uv installs everything into a venv, so to keep using python commands you can activate uv's venv: `source .venv/bin/activate`. Otherwise, swap to `uv run python`.

### Inference-Only Installation

For inference and model loading without training dependencies:
```bash
uv sync --locked
```

OlmoEarth is built using [OLMo-core](https://github.com/allenai/OLMo-core.git). OLMo-core's published [Docker images](https://github.com/orgs/allenai/packages?repo_name=OLMo-core) contain all core and optional dependencies.


## Model Summary

<img src="https://raw.githubusercontent.com/allenai/olmoearth_pretrain/main/assets/model.png" alt="Model Architecture Diagram" style="width: 800px; margin-left:'auto' margin-right:'auto' display:'block'"/>

The OlmoEarth models are trained on three satellite modalities (Sentinel 2, Sentinel 1 and Landsat) and six derived maps (OpenStreetMap, WorldCover, USDA Cropland Data Layer, SRTM DEM, WRI Canopy Height Map, and WorldCereal).
| Model Size | Weights | Encoder Params | Decoder Params |
| --- | --- | --- | --- |
| Nano | [link](https://huggingface.co/allenai/OlmoEarth-v1-Nano) | 1.4M | 800K |
| Tiny | [link](https://huggingface.co/allenai/OlmoEarth-v1-Tiny) | 6.2M | 1.9M |
| Base | [link](https://huggingface.co/allenai/OlmoEarth-v1-Base) | 89M | 30M |
| Large | [link](https://huggingface.co/allenai/OlmoEarth-v1-Large) | 308M | 53M |

## Using OlmoEarth

[InferenceQuickstart](docs/Inference-Quickstart.md) shows how to initialize the
OlmoEarth model and apply it on a satellite image.

We also have several more in-depth tutorials for computing OlmoEarth embeddings and fine-tuning OlmoEarth on downstream tasks:

- [Fine-tuning OlmoEarth for Segmentation](https://github.com/allenai/olmoearth_projects/blob/main/docs/tutorials/FinetuneOlmoEarthSegmentation.md)
- [Computing Embeddings using OlmoEarth](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md)
- [Fine-tuning OlmoEarth in rslearn](https://github.com/allenai/rslearn/blob/master/docs/examples/FinetuneOlmoEarth.md)

Additionally, [`olmoearth_projects`](https://github.com/allenai/olmoearth_projects) has several examples of active OlmoEarth deployments.

## Data Summary

Our pretraining dataset contains 285,288 samples from around the world of 2.56km×2.56km regions, although many samples contain only a subset of the timesteps and modalities.

The distribution of the samples is available below:

<img src="https://raw.githubusercontent.com/allenai/olmoearth_pretrain/main/assets/datamap.png" alt="Training sample distribution" style="width: 500px; margin-left:'auto' margin-right:'auto' display:'block'"/>

The dataset can be downloaded [here](https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset).

Detailed instructions on how to make your own pretraining dataset are available in [the dataset README](docs/Dataset-Creation.md).

## Training scripts

Detailed instructions on how to pretrain your own OlmoEarth model are available in [Pretraining.md](docs/Pretraining.md).

## Evaluations

Detailed instructions on how to replicate our evaluations is available here:

- [Evaluations on Research Benchmarks](docs/Evaluation.md)
- [Evaluations on Partner Tasks](https://github.com/allenai/rslearn_projects/blob/master/rslp/olmoearth_evals/README.md)

## Running Tests

Tests can be run with different dependency configurations using `uv run`:

```bash
# Full test suite (all dependencies - flash attn including olmo-core)
uv run --all-groups --no-group flash-attn pytest tests/

# Model loading tests with full deps (with olmo-core)
uv run --all-groups --no-group flash-attn pytest tests_minimal_deps/

# Model loading tests with minimal deps only (no olmo-core)
uv run --group dev pytest tests_minimal_deps/
```

The `tests_minimal_deps/` directory contains tests that verify model loading works both with and without `olmo-core` installed. These run twice in CI to ensure compatibility.

## License

This code is licensed under the [OlmoEarth Artifact License](LICENSE).
