# HELIOS

Highly Efficient Learning for Integrated Observation Systems (HELIOS)

Earth system foundation model: data, training, and evaluation

launching training runs on beaker
## General Setup

1. Create a virtual environment in prefered directory with python 3.12 `python3.12 -m venv .venv-helios`
2. Activate the virtual environment `source .venv-helios/bin/activate`
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

To launch pre-emptible jobs, we will use the main entrypoint in [helios/internal/experiment.py](helios/internal/experiment.py) and write python configuration files that use it like [scripts/latent_mim.py](scripts/latent_mim.py). Depednign on your experiment it might make sense to write a new script with different builders or to just overide as needed for an existing one.
Before launching your script **MAKE SURE YOUR CODE IS COMMITED AND PUSHED** as we are cloning the code on top of a docker image when we launch the job.

We can launch a script as follows:

`python3 scripts/base_debug_scripts/latent_mim.py launch test_run ai2/saturn-cirrascale`

This will launch a beaker job and stream the logs to your console until you cancel.
Add additional overides as needed.

### Sessions

Be sure your session creation has included the following args
 - `  --secret-env WANDB_API_KEY=<your_beaker_username>_WANDB_API_KEY
    --secret-env BEAKER_TOKEN=<your_beaker_username>__BEAKER_TOKEN `

When launching runs in Sessions for debugging, use the following command,

`torchrun scripts/base_debug_scripts/latent_mim.py train test_run local`

Add additional overides as needed.

## Beaker Information

budget: `ai2/d5` \
workspace: `ai2/earth-systems` \
weka: `weka://dfive-default`

## Helios Dataset

The dataset for training is stored in h5 datasets. A trainining datset can be created from tiles via `python3 internal/run_h5_conversion.py` script.

- 80K dataset with S1, S2, Landsat, NAIP, WORLDCOVER, and OSM `/weka/dfive-default/helios/dataset/20250223/`
- ~100k dataset with S1, S2, Landsat, NAIP, WORLDCOVER, Landsat, SRTM, and  OSM sampled based on Galileo locations `/weka/dfive-default/helios/dataset/presto/h5py_data/landsat_naip_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/118861`

- ~300k dataset with S1, S2, Landsat, NAIP, WORLDCOVER, Landsat, SRTM, and  OSM sampled base on open street map classes `/weka/dfive-default/helios/dataset/osm_sampling/h5py_data/landsat_naip_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/324192`
