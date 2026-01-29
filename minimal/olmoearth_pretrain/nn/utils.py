"""Utilities for the nn module."""

from typing import Any

import torch
from torch.distributed import DeviceMesh


def unpack_encoder_output(
    output_dict: dict[str, Any],
) -> tuple:
    """Unpack the output of an encoder.

    Args:
        output_dict (dict[str, Any]): The output of an encoder.

    Returns:
        tuple[TokensAndMasks, TokensAndMasks, dict[str, Any]]: The unpacked output.
    """
    latent = output_dict.pop("tokens_and_masks", None)
    latent_projected_and_pooled = output_dict.pop("project_aggregated", None)
    # Pass through all other outputs that might be specific to an encoder decoder pair
    # remove token_norm_stats
    output_dict.pop("token_norm_stats", None)
    decoder_kwargs = output_dict
    return latent, latent_projected_and_pooled, decoder_kwargs


def get_cumulative_sequence_lengths(seq_lengths: torch.Tensor) -> torch.Tensor:
    """Get the cumulative sequence lengths of a tensor.

    Args:
        seq_lengths (torch.Tensor): The sequence lengths of a tensor.

    Returns:
        torch.Tensor: The cumulative sequence lengths of a tensor.
    """
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=seq_lengths.device),
            torch.cumsum(
                seq_lengths.masked_select(seq_lengths != 0), 0, dtype=torch.int32
            ),
        ]
    )


# TODO: maybe this should just be functional or something
class DistributedMixins:
    """Mixin for distributed training."""

    def apply_ddp(
        self,
        dp_mesh: DeviceMesh | None = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
        find_unused_parameters: bool = True,
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
            find_unused_parameters=find_unused_parameters,
        )
