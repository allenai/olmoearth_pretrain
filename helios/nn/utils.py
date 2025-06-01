"""Utilities for the nn module."""

import torch
from torch.distributed import DeviceMesh


# TODO: maybe this should just be functional or something
class DistributedMixins:
    """Mixin for distributed training."""

    def apply_ddp(
        self,
        dp_mesh: DeviceMesh | None = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
        find_unused_parameters: bool = False,
    ) -> None:
        """Apply DDP to the model.

        .. warning::
            Usually this does not need to be called directly, as :meth:`TransformerConfig.build()`
            will call it for you.
        """
        from torch.distributed._composable.replicate import replicate

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = (
                    "python_reducer_without_compiled_forward"  # type: ignore
                )
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore
        # Forwards kwargs to torch DDP class, find_unused_parameters=True is required for MAE
        # Small performance hit could be possible for other models
        replicate(
            self,
            device_mesh=dp_mesh,
            bucket_cap_mb=100,
            find_unused_parameters=find_unused_parameters,  # find_unused_parameters,
        )
