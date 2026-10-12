"""Tests for the train module's checkpoint load plan."""

from types import SimpleNamespace

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed.checkpoint.metadata import Metadata

from olmoearth_pretrain.train.train_module.train_module import OlmoEarthTrainModule


def _stub() -> SimpleNamespace:
    stub = SimpleNamespace(
        state_dict_load_opts=dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
    )
    stub._get_state_dict = lambda opts: {
        "model": {"encoder.s2.weight": torch.zeros(64, 12)},
        "optim": {"param_groups": [{}]},
    }
    return stub


def test_state_dict_to_load_honours_optim_false() -> None:
    """``load_optim_state=False`` drops the optimizer entry from the load plan."""
    state_dict = OlmoEarthTrainModule.state_dict_to_load(
        _stub(),  # type: ignore[arg-type]
        Metadata(state_dict_metadata={}),
        optim=False,
    )
    assert "optim" not in state_dict
    assert "encoder.s2.weight" in state_dict["model"]


def test_state_dict_to_load_keeps_optim_by_default() -> None:
    """Without an explicit ``optim=False`` the optimizer state is loaded."""
    state_dict = OlmoEarthTrainModule.state_dict_to_load(
        _stub(),  # type: ignore[arg-type]
        Metadata(state_dict_metadata={}),
    )
    assert "optim" in state_dict
