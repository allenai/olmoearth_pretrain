# HELIOS
Highly Efficient Learning for Integrated Observation Systems (HELIOS)

earth system foundation model data, training, and eval


## Setup Instructions for running olmo_core_proto.py

1. Create a virtual environment in prefered directory with python 3.12 `python3 -m venv .venv-helios` \
2. Navigate to root directory of this repo and run `pip install -e .`
3. Clone the [Olmo-core](https://github.com/allenai/OLMo-core/tree/v2) repo and switch to the v2 branch
4. Navigate to the root directory of olmo-core repository and run `pip install -e .`
5. (Skip if dataset is on weka) Make sure you have access to the relevant bucket 'gcloud auth default login' or using beaker secrets
6. Set `WANDB_API_KEY' api key environment variable (or povide it via --secret-env flag when you start your beaker session)
7. Adjust the variables to be changed per user in olmo_core_proto.py
8. run `python3 helios/olmo_core_proto.py` for single gpu and `torchrun helios/olmo_core_proto.py`



## Beaker Information
budget: ai2/d5 \
workspace: ai2/earth-systems \
weka: weka://dfive-default \
