# Projects in Data Science Course 2026 - Geospatial Foundation Models

In this course we will learn about state-of-the-art geospatial foundation models. Our case study will be the omloearth foundation model by the Allen institute

## Setup

We will use `uv` to create a python environment with all of the required dependencies.

To create such an environments and test that it works, please follow these steps:

1. If uv is not installed on your machine, please go to the following [link](https://docs.astral.sh/uv/getting-started/installation/) to install it.

2. Clone the repo, create a pyhton environment and install dependencies by running this command:

```
git clone https://github.com/yinonbaron/WIS_data_science_olmoearth_pretrain_2026.git
cd WIS_data_science_olmoearth_pretrain_2026
uv sync --locked --extra all-no-flash --python 3.12
```
3. Test that the installation worked by running the following command:

```uv run test_install.py```

4. If the script runs with not It should run with no errors, you're done!