# OlmoEarth Pretrain

Allen Institute for AI's OlmoEarth Pretrain project

Earth system foundation model: data, training, and evaluation

launching training runs on beaker
## General Setup

**Requirements:** Python 3.11 or higher (Python 3.12 recommended)

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (other ways to do it are documented [here](https://docs.astral.sh/uv/getting-started/installation/))
2. Navigate to root directory of this repo and run `uv sync --locked --all-groups --python 3.12`
3. Install the pre-commit tool `uv tool install pre-commit --with pre-commit-uv --force-reinstall`
4. uv installs everything into a venv, so to keep using `python` commands you can activate uv's venv: `source .venv/bin/activate`. Otherwise, swap to `uv run python`.


## OlmoEarth Pretrain Dataset

The dataset for training is stored in h5 datasets. A training dataset can be created from tiles via `python3 -m olmoearth_pretrain.internal.run_h5_conversion` script.


We have 2 versions of each dataset 1 with 256 x 256 tiles and 1 with 4x as many 128 by 128 tiles. The 128 by 128 tiles may be faster for data loading due to GB/s bottlenecks on weka.

OUT OF DATE!
- **Presto Dataset**: ~120k samples with Landsat, OpenStreetMap raster, Sentinel-1, Sentinel-2 L2A, SRTM, and WorldCover modalities sampled via locations used in Galileo paper
  - 256 Path: `/weka/dfive-default/helios/dataset/presto/rerun_1_h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/117473/`
  - 128 Path: `/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/469892`

- **OSM Sampling Dataset**: ~285k samples with Landsat, OpenStreetMap raster, Sentinel-1, Sentinel-2 L2A, SRTM, and WorldCover modalities sampled across OpenStreetmap classes
  - 256 Path: `/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/285288/`
  - 128 Path: `/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1141152`
- **OSM Big Dataset**: ~324k samples with Landsat, OpenStreetMap raster, Sentinel-1, Sentinel-2 L2A, SRTM, and WorldCover modalities  sampled across a wider set of opens treetmap classes
  - 256 Path: `/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/324482/`
  - 128 Path: `/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1297928`
- **Presto Neighbor Dataset**: ~877k samples with Landsat, OpenStreetMap raster, Sentinel-1, Sentinel-2 L2A, SRTM, and WorldCover modalities presto + the neighboring tiles
  - 256 Path: `/weka/dfive-default/helios/dataset/presto_neighbor/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/876937/`
  - 128 Path: `/weka/dfive-default/helios/dataset/presto_neighbor/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/3507748`
- **WorldCover Sampling Dataset**: ~1.6M samples with Landsat, OpenStreetMap raster, Sentinel-1, Sentinel-2 L2A, SRTM, and WorldCover modalities. WorldCover class based sampling and some additional random sampling over the rest of the world.
  - 256 Path: `/weka/dfive-default/helios/dataset/worldcover_sampling/h5py_data_w_missing_timesteps_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1592645/`
  - 128 Path: `/weka/dfive-default/helios/dataset/worldcover_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/6370580/`


## Running Eval Suite

[`olmoearth_pretrain/internal/full_eval_sweep.py`](olmoearth_pretrain/internal/full_eval_sweep.py) runs comprehensive evaluation sweeps across multiple downstream tasks for any OlmoEarth Pretrain checkpoint. It automatically sweeps over learning rates, pooling types, and normalization strategies.

### 1. How to run eval for a given checkpoint

Basic command to run evaluation sweep for a checkpoint:

```
python3 olmoearth_pretrain/internal/full_eval_sweep.py \
  --cluster=ai2/saturn-cirrascale \
  --checkpoint_path=/path/to/your/checkpoint/step450000 \
  --module_path=scripts/your_training_script.py \
```

For just default hyperparameters (faster, single run):
```bash
python3 olmoearth_pretrain/internal/full_eval_sweep.py \
  --cluster=ai2/saturn-cirrascale \
  --checkpoint_path=/path/to/your/checkpoint/step450000 \
  --module_path=scripts/your_training_script.py \
  --defaults_only
```

### 2. Example of how to add additional overrides

Pass additional training arguments after the main arguments:
```bash
python3 olmoearth_pretrain/internal/full_eval_sweep.py \
  --cluster=ai2/saturn-cirrascale \
  --checkpoint_path=/path/to/checkpoint \
  --module_path=scripts/your_script.py \
  --model.decoder_config.depth=1 \
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[mados,pastis_sentinel2,breizhcrops,sen1floods11,pastis_sentinel1_sentinel2\]  \
```

### 3. How to run panopticon

Use the `--panopticon` flag for Panopticon model evaluation:
```bash
python3 olmoearth_pretrain/internal/full_eval_sweep.py \
  --cluster=ai2/saturn-cirrascale \
  --panopticon \
  --model_name=panopticon
```

### 4. How to run different dino models

For DINO v3 evaluation:
```bash
python3 olmoearth_pretrain/internal/full_eval_sweep.py \
  --cluster=ai2/saturn-cirrascale \
  --dino_v3 \
  --model_name=dino_v3_large_sat \
  --model.model_name=DinoV3Models.LARGE_SATELLITE  \
```

### 5. How to run galileo

Use the `--galileo` flag for Galileo model evaluation:
```bash
python3 olmoearth_pretrain/internal/full_eval_sweep.py \
  --cluster=ai2/saturn-cirrascale \
  --galileo \
  --model_name=galileo_vit_base
  --model.patch_size=4
```

**Key Notes:**
- The script automatically determines appropriate normalization strategies for each model type (see [`olmoearth_pretrain/evals/datasets/normalize.py`](olmoearth_pretrain/evals/datasets/normalize.py))
  - OlmoEarth Pretrain: Use pretrained normalizer or NORM_METHOD.NORM_NO_CLIP with dataset stats
  - Galileo: Use galileo pretrained normalizer or  NORM_METHOD.NORM_NO_CLIP with dataset stats
  - Panopticon: Uses NORM_METHOD.STANDARDIZE with the dataset statistics
  - DinoV3: Uses NORM_METHOD.NORM_YES_CLIP_MIN_MAX_INT to get to 0-1 and then applies either the web or sat normalization values
- Supports both full hyperparameter sweeps and default-only runs
- Use `--dry_run` to preview commands without execution
- For local testing, use `--cluster=local`

See `olmoearth_pretrain/internal/full_eval_sweep.py` for complete argument list and implementation details.
