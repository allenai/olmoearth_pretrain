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
from os import PathLike

import torch
from huggingface_hub import hf_hub_download
from olmo_core.config import Config


class ModelID(StrEnum):
    """OlmoEarth pre-trained model ID."""

    OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"
    OLMOEARTH_V1_TINY = "OlmoEarth-v1-Tiny"
    OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"

    def repo_id(self) -> str:
        """Return the Hugging Face repo ID for this model."""
        return f"allenai/{self.value}"


def load_model(
    model_id_or_path: ModelID | PathLike, load_weights: bool = True
) -> torch.nn.Module:
    """Initialize and load the weights for the specified model.

    If the model ID is a ModelID, the config and weights will be downloaded from Hugging Face.
    Otherwise, the config and weights will be loaded from the specified path.

    Args:
        model_id_or_path: the model ID or path to load.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
            the config.json will still be downloaded from Hugging Face.
    """
    model_config = _load_model_config(model_id_or_path)
    model: torch.nn.Module = model_config.build()

    if not load_weights:
        return model

    state_dict = _load_state_dict(model_id_or_path)
    model.load_state_dict(state_dict)
    return model


def _load_model_config(model_id_or_path: ModelID | PathLike) -> Config:
    """Load the model config from the specified model ID out of hfhub or from a path."""
    if isinstance(model_id_or_path, ModelID):
        # We ignore bandit warnings here since we are just downloading config and weights,
        # not any code.
        config_fname = hf_hub_download(
            repo_id=model_id_or_path.repo_id(), filename="config.json"
        )  # nosec
    else:
        config_fname = model_id_or_path

    with open(config_fname) as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    return model_config


def _load_state_dict(
    model_id_or_path: ModelID | PathLike,
) -> dict[str, torch.Tensor]:
    """Load the model state dict from the specified model ID out of hfhub or from a path."""
    if isinstance(model_id_or_path, ModelID):
        # We ignore bandit warnings here since we are just downloading config and weights,
        # not any code.
        state_dict_fname = hf_hub_download(
            repo_id=model_id_or_path.repo_id(), filename="weights.pth"
        )  # nosec
    else:
        state_dict_fname = model_id_or_path
    state_dict = torch.load(state_dict_fname, map_location="cpu")
    return state_dict
