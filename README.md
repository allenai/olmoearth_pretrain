# HELIOS
Highly Efficient Learning for Integrated Observation Systems (HELIOS)

earth system foundation model data, training, and eval

## General Setup

1. Create a virtual environment in prefered directory with python 3.12 `python3 -m venv .venv-helios` \
2. Navigate to root directory of this repo and run `pip install -e .`
3. Run `pre-commit install`

## Setup Instructions for running olmo_core_proto.py
1. Clone the [Olmo-core](https://github.com/allenai/OLMo-core/tree/v2) repo and switch to the v2 branch
    ```bash
    git clone --branch v2 https://github.com/allenai/OLMo-core.git
    cd OLMo-core
    ```
2. Navigate to the root directory of olmo-core repository and run `pip install -e .`
3. (Skip if dataset is on weka) Make sure you have access to the relevant bucket 'gcloud auth default login' or using beaker secrets
4. Set `WANDB_API_KEY' api key environment variable (or povide it via --secret-env flag when you start your beaker session)
    ```bash
    export WANDB_API_KEY=<your-key>
    ```
5. Adjust the variables to be changed per user in olmo_core_proto.py
6. run `python3 helios/olmo_core_proto.py` for single gpu and `torchrun helios/olmo_core_proto.py`



## Beaker Information
budget: ai2/d5 \
workspace: ai2/earth-systems \
weka: weka://dfive-default \
