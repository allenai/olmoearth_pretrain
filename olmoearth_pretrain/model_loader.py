"""Load the OlmoEarth models from Hugging Face.

The weights are converted to pth file from distributed checkpoint like this:

    import json
    from pathlib import Path

    import torch

    from olmo_core.config import Config
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    checkpoint_path = Path("/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000")
    with (checkpoint_path / "config.json").open() as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    model = model_config.build()

    train_module_dir = checkpoint_path / "model_and_optim"
    load_model_and_optim_state(str(train_module_dir), model)
    torch.save(model.state_dict(), "OlmoEarth-v1-Nano.pth")
"""

import json
from enum import StrEnum

import torch
from huggingface_hub import hf_hub_download
from olmo_core.config import Config


class ModelID(StrEnum):
    """OlmoEarth pre-trained model ID."""

    OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"
    OLMOEARTH_V1_TINY = "OlmoEarth-v1-Tiny"
    OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"


def load_model(model_id: ModelID, load_weights: bool = True) -> torch.nn.Module:
    """Initialize and load the weights for the specified model ID.

    The config and weights will be downloaded from Hugging Face.

    Args:
        model_id: the model ID to load.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
            the config.json will still be downloaded from Hugging Face.
    """
    # We ignore bandit warnings here since we are just downloading config and weights,
    # not any code.
    repo_id = f"allenai/{model_id.value}"
    config_fname = hf_hub_download(repo_id=repo_id, filename="config.json")  # nosec
    with open(config_fname) as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    model: torch.nn.Module = model_config.build()

    if not load_weights:
        return model

    pth_fname = hf_hub_download(repo_id=repo_id, filename="weights.pth")  # nosec
    state_dict = torch.load(pth_fname, map_location="cpu")
    model.load_state_dict(state_dict)
    return model
