"""utility Functions for hyper parameter sweeps."""

from collections.abc import Iterable
from typing import Any

import torch
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.train.train_module import EvalBatchSpec, TrainModule

EXIT_CONFIG_TYPES = ["zero", "half", "full", "varied"]


def build_token_exit_config(
    config_type: str, modality_names: list[str], encoder_depth: int
) -> str:
    """Build the token exit config for an experiment."""
    if config_type not in EXIT_CONFIG_TYPES:
        raise ValueError(f"Invalid config type: {config_type}")
    if config_type == "zero":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}=0"
            for modality_name in modality_names
        )
    elif config_type == "half":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth // 2}"
            for modality_name in modality_names
        )
    elif config_type == "full":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
            for modality_name in modality_names
        )
    elif config_type == "varied":
        varied_args = []
        for modality_name in modality_names:
            if modality_name not in ["latlon", "worldcover"]:
                varied_args.append(
                    f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
                )
            else:
                varied_args.append(f"--train_module.token_exit_cfg.{modality_name}=0")
        return " ".join(varied_args)
    else:
        raise ValueError(f"Invalid config type: {config_type}")


MODEL_SIZE_ARGS = {
    "nano": {
        "decoder_depth": 4,
        "encoder_embedding_size": 128,
        "decoder_embedding_size": 128,
        "encoder_depth": 4,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "tiny": {
        "decoder_depth": 12,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "tiny_more_heads": {
        "decoder_depth": 12,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "base": {
        "decoder_depth": 12,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large": {
        "decoder_depth": 24,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga": {
        "decoder_depth": 40,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "tiny_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "tiny_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "base_many_heads_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 8,
        "encoder_num_heads": 16,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "base_many_heads_shallower_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 8,
        "encoder_num_heads": 16,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


class MockOlmoEarthDataLoader(DataLoaderBase):
    """Minimal OlmoEarth dataloader that only satisfies the abstract interface."""

    def __init__(self) -> None:
        """Initialize the mock loader with trivial single-rank defaults."""
        super().__init__(
            work_dir="./",
            global_batch_size=128,
            dp_world_size=1,
            dp_rank=0,
            fs_local_rank=0,
        )
        self._seed = 42
        self._epoch = 0

    def _iter_batches(self) -> Iterable[Any]:
        return iter(())

    def state_dict(self) -> dict[str, Any]:
        """Return the minimal persisted state for the mock loader."""
        return {"seed": self._seed, "epoch": self._epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D401
        """No-op for the mock dataloader."""
        self._seed = state_dict.get("seed", self._seed)
        self._epoch = state_dict.get("epoch", self._epoch)

    def reshuffle(
        self, epoch: int | None = None, in_memory: bool = False, **_: Any
    ) -> None:
        """Record the provided epoch; other parameters are ignored."""
        if epoch is not None:
            self._epoch = epoch

    @property
    def total_batches(self) -> int:
        """Report zero batches, as the mock loader never yields data."""
        return 0

    def get_mock_batch(self) -> None:
        """Return no batch payload; this stub does not fabricate data."""
        return None


class MockLatentMIMTrainModule(TrainModule):
    """Minimal TrainModule stub for LatentMIM-style configs."""

    def __init__(self) -> None:
        """Initialize the mock train module."""
        super().__init__()
        self.model = torch.nn.Identity()

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        """Return a trivial eval batch specification."""
        return EvalBatchSpec(rank_batch_size=1)

    def state_dict(self, *, optim: bool | None = None) -> dict[str, Any]:
        """Return an empty state dict."""
        del optim
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Ignore any state dict content."""
        del state_dict

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        """No-op training step."""
        del batch, dry_run

    def eval_batch(self, batch: dict[str, Any], labels: Any | None = None) -> Any:
        """Return a constant tensor to satisfy interface expectations."""
        del batch, labels
        return torch.tensor(0.0)

    def optim_step(self) -> None:
        """No-op optimizer step."""

    def zero_grads(self) -> None:
        """No-op gradient reset."""
