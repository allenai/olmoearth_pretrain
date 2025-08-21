"""Updating the checkpointer to allow for partial loading of the model"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
)

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from cached_path import cached_path
from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_state_dict,
)
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
from olmo_core.distributed.utils import (
    get_rank,
    scatter_object,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import (
    is_url,
    normalize_path,
)
from olmo_core.train import checkpoint
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata


@torch.no_grad()
def load_state_dict(
    dir: str,
    state_dict: dict[str, Any],
    *,
    process_group: dist.ProcessGroup | None = None,
    pre_download: bool = False,
    work_dir: str | None = None,
    thread_count: int | None = None,
):
    """Load an arbitrary state dict in-place from a checkpoint saved with :func:`save_state_dict()`.

    :param dir: Path/URL to the checkpoint saved via :func:`save_state_dict()`.
    :param state_dict: The state dict to load the state into.
    :param process_group: The process group to use for distributed collectives.
    :param thread_count: Set the number of threads used for certain operations.
    """
    dir = normalize_path(dir)
    reader = RemoteFileSystemReader(
        dir, thread_count=thread_count, pre_download=pre_download, work_dir=work_dir
    )
    dist_cp.load(
        state_dict,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )


@dataclass
class HeliosCheckpointerConfig(checkpoint.CheckpointerConfig):
    """Config for the Helios checkpointer"""

    def build(
        self, process_group: dist.ProcessGroup | None = None, **kwargs
    ) -> HeliosCheckpointer:
        kwargs = {**self.as_dict(exclude_none=True, recurse=False), **kwargs}
        work_dir = kwargs.pop("work_dir", None)
        if work_dir is None:
            raise OLMoConfigurationError(
                "'work_dir' must be provided to build a Checkpointer"
            )
        return HeliosCheckpointer(
            work_dir=Path(work_dir), process_group=process_group, **kwargs
        )


@dataclass
class HeliosCheckpointer(checkpoint.Checkpointer):
    """Checkpointer that allows for partial loading of the model"""

    def load(
        self,
        dir: str,
        train_module: TrainModule,
        *,
        load_trainer_state: bool | None = None,
    ) -> dict[str, Any] | None:
        """Load model, optim, and other training state from a local or remote checkpoint directory
        created via :meth:`save()` or :meth:`save_async()`.
        """
        dir = normalize_path(dir)

        # Maybe load trainer state.
        trainer_state: dict[str, Any] | None = None
        if load_trainer_state is not False:
            # Try loading the given rank's state first, then fall back to rank 0 train state if it
            # doesn't exist, which can happen when we're restoring a checkpoint with a different world size.
            for path in (f"{dir}/train/rank{get_rank()}.pt", f"{dir}/train/rank0.pt"):
                try:
                    trainer_state = torch.load(
                        cached_path(path, quiet=True), weights_only=False
                    )
                    break
                except FileNotFoundError:
                    pass
                print("looking to use rank 0 trainer state")

            if load_trainer_state is True and trainer_state is None:
                raise FileNotFoundError(
                    f"Missing trainer state in checkpoint dir '{dir}'"
                )

        # Load train module state.
        train_module_dir = f"{dir}/model_and_optim"
        metadata: Metadata | None = None
        if get_rank(self.process_group) == 0:
            try:
                metadata = get_checkpoint_metadata(train_module_dir)
            except FileNotFoundError:
                # Try base directory, which could be the case if user is trying to load model weights
                # (possibly with optimizer state), and not an actual train checkpoint.
                if trainer_state is None:
                    metadata = get_checkpoint_metadata(dir)
                    train_module_dir = dir
                else:
                    raise

        train_module_dir = scatter_object(train_module_dir)
        if metadata is None:
            metadata = get_checkpoint_metadata(train_module_dir)

        state_dict = train_module.state_dict_to_load(metadata)
        load_state_dict(
            train_module_dir,
            state_dict,
            process_group=self.process_group,
            pre_download=is_url(dir) and self.pre_download,
            work_dir=self.work_dir,
            thread_count=self.load_thread_count,
        )
        train_module.load_state_dict(state_dict)

        return trainer_state
