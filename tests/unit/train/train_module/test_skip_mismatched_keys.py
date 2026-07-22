"""Tests for the OE_LOAD_SKIP_MISMATCHED_KEYS partial-load escape hatch."""

from types import SimpleNamespace

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)

from olmoearth_pretrain.train.train_module.train_module import OlmoEarthTrainModule


def _tensor_meta(shape: list[int]) -> TensorStorageMetadata:
    return TensorStorageMetadata(
        properties=TensorProperties(dtype=torch.float32),
        size=torch.Size(shape),
        chunks=[
            ChunkStorageMetadata(
                offsets=torch.Size([0] * len(shape)), sizes=torch.Size(shape)
            )
        ],
    )


def _stub() -> SimpleNamespace:
    return SimpleNamespace(
        state_dict_load_opts=dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
    )


def test_drop_mismatched_keys() -> None:
    """Shape-mismatched and checkpoint-absent keys are dropped; matches kept."""
    metadata = Metadata(
        state_dict_metadata={
            # Saved before srtm grew terrain bands: [64, 1] vs the model's [64, 4].
            "model.encoder.srtm.weight": _tensor_meta([64, 1]),
            "model.encoder.s2.weight": _tensor_meta([64, 12]),
        }
    )
    state_dict = {
        "model": {
            "encoder.srtm.weight": torch.zeros(64, 4),
            "encoder.s2.weight": torch.zeros(64, 12),
            "encoder.new_param": torch.zeros(3),
        },
        "optim": {
            "state.encoder.srtm.weight.exp_avg": torch.zeros(64, 4),
            "state.encoder.s2.weight.exp_avg": torch.zeros(64, 12),
            "param_groups": [{}],
        },
    }
    stub = _stub()
    OlmoEarthTrainModule._drop_mismatched_keys(stub, state_dict, metadata)  # type: ignore[arg-type]

    assert set(state_dict["model"]) == {"encoder.s2.weight"}
    # The flattened optimizer format cannot be partially loaded, so optimizer
    # state is skipped entirely when any model key is dropped.
    assert "optim" not in state_dict
    # Dropped keys are missing from the load plan, so strictness must relax.
    assert stub.state_dict_load_opts.strict is False


def test_no_mismatch_is_a_no_op() -> None:
    """A fully matching checkpoint leaves the state dict and opts untouched."""
    metadata = Metadata(
        state_dict_metadata={"model.encoder.s2.weight": _tensor_meta([64, 12])}
    )
    state_dict = {
        "model": {"encoder.s2.weight": torch.zeros(64, 12)},
        "optim": {"param_groups": [{}]},
    }
    stub = _stub()
    OlmoEarthTrainModule._drop_mismatched_keys(stub, state_dict, metadata)  # type: ignore[arg-type]

    assert set(state_dict["model"]) == {"encoder.s2.weight"}
    assert set(state_dict["optim"]) == {"param_groups"}
    assert stub.state_dict_load_opts.strict is True
