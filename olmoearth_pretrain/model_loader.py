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
from upath import UPath

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pth"


class ModelID(StrEnum):
    """OlmoEarth pre-trained model ID."""

    OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"
    OLMOEARTH_V1_TINY = "OlmoEarth-v1-Tiny"
    OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"

    def repo_id(self) -> str:
        """Return the Hugging Face repo ID for this model."""
        return f"allenai/{self.value}"


def load_model_from_id(model_id: ModelID, load_weights: bool = True) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from Hugging Face.

    Args:
        model_id: the model ID to load.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
            the config.json will still be downloaded from Hugging Face.
    """
    config_fpath = _resolve_artifact_path(model_id, CONFIG_FILENAME)
    model = _load_model_from_config(config_fpath)

    if not load_weights:
        return model

    state_dict_fpath = _resolve_artifact_path(model_id, WEIGHTS_FILENAME)
    state_dict = _load_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def load_model_from_path(
    model_path: PathLike | str, load_weights: bool = True
) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from a path.

    Args:
        model_path: the path to the model.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
    """
    config_fpath = _resolve_artifact_path(model_path, CONFIG_FILENAME)
    model = _load_model_from_config(config_fpath)

    if not load_weights:
        return model

    state_dict_fpath = _resolve_artifact_path(model_path, WEIGHTS_FILENAME)
    state_dict = _load_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def _resolve_artifact_path(
    model_id_or_path: ModelID | PathLike | str, filename: str
) -> PathLike:
    """Resolve the artifact file path for the specified model ID or path, downloading it from Hugging Face if necessary."""
    if isinstance(model_id_or_path, ModelID):
        return UPath(
            hf_hub_download(repo_id=model_id_or_path.repo_id(), filename=filename)  # nosec
        )
    base = UPath(model_id_or_path)
    return base / filename


def _load_model_from_config(path: PathLike) -> torch.nn.Module:
    """Load the model config from the specified path."""
    with open(path) as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])
    return model_config.build()


def _load_state_dict(path: PathLike) -> dict[str, torch.Tensor]:
    """Load the model state dict from the specified path."""
    state_dict = torch.load(path, map_location="cpu")
    return state_dict
