# OlmoEarth Pretrain

Allen Institute for AI's OlmoEarth Pretrain project

Earth system foundation model: data, training, and evaluation

launching training runs on beaker
## General Setup

1. Create a virtual environment in prefered directory with python 3.12 `python3.12 -m venv .venv-olmoearth_pretrain`
2. Activate the virtual environment `source .venv-olmoearth_pretrain/bin/activate`
3. Navigate to root directory of this repo and run `pip install -e .`
4. Run `pip install pre-commit`
5. Run `pre-commit install`

## Training Setup
1. Create a Github Token that is able to clone this repo on beaker. You can generate a token [here](https://github.com/settings/tokens) Following permissions are sufficient
    - repo
    - read:packages
    - read:org
    - write:org
    - read:project

    Authorize this token for the allenai org. by clicking on the Configure SSO drop down in [here](https://github.com/settings/tokens) for the token you created.
2. Set your default Beaker workspace and budget:
    `beaker config set default_workspace ai2/earth-systems`
    `beaker workspace set-budget ai2/earth-systems ai2/d5`
3. Set the following Beaker Secrets:
    - `beaker secret write <your_beaker_username>_WANDB_API_KEY <your_key>`
    - `beaker secret write <your_beaker_username>_BEAKER_TOKEN <your_token>`
    - `beaker secret write <your_beaker_username>_GITHUB_TOKEN <your_key>`

4. Create a script based on scripts/latent_mim.py and configure your experiment (you can override specific changes)


## Launch

### Pre-emptible Jobs

To launch pre-emptible jobs, we will use the main entrypoint in [olmoearth_pretrain/internal/experiment.py](olmoearth_pretrain/internal/experiment.py) and write python configuration files that use it like [scripts/latent_mim.py](scripts/latent_mim.py). Depednign on your experiment it might make sense to write a new script with different builders or to just overide as needed for an existing one.
Before launching your script **MAKE SURE YOUR CODE IS COMMITED AND PUSHED** as we are cloning the code on top of a docker image when we launch the job.

We can launch a script as follows:

`python3 scripts/base_debug_scripts/latent_mim.py launch test_run ai2/saturn-cirrascale`

This will launch a beaker job and stream the logs to your console until you cancel.
Add additional overides as needed.

### Sessions

[VSCODE/Cursor workflow setup](https://docs.google.com/document/d/1ydiCqIn45xlbrIcfPi8bILn_y00adTAHhIY1MPh9szE/edit?tab=t.0#heading=h.wua78h35aq1n) \
Be sure your session creation has included the following args
 - `  --secret-env WANDB_API_KEY=<your_beaker_username>_WANDB_API_KEY
    --secret-env BEAKER_TOKEN=<your_beaker_username>__BEAKER_TOKEN `

Note: In order to use flash attention in a session, use `"beaker://petew/olmo-core-tch270cu128"` as your base beaker image.
Then, set up a conda environment so you can use the flash attention code saved in the base image.
1. `conda init`
2. `exec bash`
3. `conda shell.bash activate base`
4. `pip install -e '.[all]'`

When launching runs in Sessions for debugging, use the following command,

`torchrun scripts/base_debug_scripts/latent_mim.py train test_run local`

Add additional overides as needed.

## Beaker Information

budget: `ai2/es-platform` \
workspace: `ai2/earth-systems` \
weka: `weka://dfive-default`

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
