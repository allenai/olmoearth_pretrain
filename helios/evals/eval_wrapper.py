"""Eval wrapper contract to be able to run evals on a model."""

from logging import getLogger
from typing import Any

import torch
from einops import reduce
from torch import nn

from helios.data.constants import Modality
from helios.evals.datasets.configs import TaskType
from helios.evals.models import DINOv2, GalileoWrapper, Panopticon
from helios.nn.flexihelios import FlexiHeliosBase, PoolingType, TokensAndMasks
from helios.nn.pooled_modality_predictor import EncodeEarlyAttnPool
from helios.nn.st_model import STBase
from helios.train.masking import MaskedHeliosSample
from einops import rearrange
import numpy as np

logger = getLogger(__name__)


class EvalWrapper:
    """Base class for eval wrappers.

    This is the common interface to run our evals on any model
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: TaskType,
        patch_size: int,
        pooling_type: PoolingType,
        concat_features: bool = False,
        use_pooled_tokens: bool = False,
    ):
        """Initialize the eval wrapper.

        Args:
            model: The model to evaluate.
            task_type: The type of task to evaluate.
            patch_size: The patch size to use for the model.
            pooling_type: The pooling type to use for the model.
            concat_features: Whether to concatenate features across modalities.
            use_pooled_tokens: Whether to use pooled tokens.
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.patch_size = patch_size
        self.pooling_type = pooling_type
        self.concat_features = concat_features
        self.spatial_pool = task_type == TaskType.SEGMENTATION
        self.use_pooled_tokens = use_pooled_tokens
        if self.use_pooled_tokens:
            assert isinstance(self.model, EncodeEarlyAttnPool), (
                "Pooled tokens are only supported for EncodeEarlyAttnPool"
            )

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        dev = getattr(self.model, "device", None)

        if isinstance(dev, torch.device):
            return dev

        # DinoV2 returns a string
        if isinstance(dev, str):
            return torch.device(dev)

        # For FSDP wrapped models, fall back to device of model parameters
        return next(self.model.parameters()).device

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model if the attribute is not found on the wrapper."""
        return getattr(self.model, name)

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass through the model produces the embedding specified by initialization."""
        raise NotImplementedError("Subclasses must implement this method")


class HeliosEvalWrapper(EvalWrapper):
    """Wrapper for Helios models."""

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass through the model produces the embedding specified by initialization."""
        if not self.use_pooled_tokens:
            batch_embeddings: TokensAndMasks = self.model(
                masked_helios_sample, patch_size=self.patch_size, always_pass_none_mask_to_transformer=True
            )["tokens_and_masks"]  # (bsz, dim)
            sentinel2_data = getattr(batch_embeddings, Modality.SENTINEL2_L2A.name, None)
            sentinel1_data = getattr(batch_embeddings, Modality.SENTINEL1.name, None)
            # get shapes of both data
            print(f"sentinel2_data shape: {sentinel2_data.shape}")
            print(f"sentinel1_data shape: {sentinel1_data.shape}")

            flattened_s2 = rearrange(sentinel2_data, "b ... c d -> b (...) c d")
            flattened_s1 = rearrange(sentinel1_data, "b ... c d -> b (...) c d")
            # stack in the c dimension
            # Combine flattened_s2 and flattened_s1 in the second to last dim (dim=-2)
            # flattened_s2: [32, 3072, 3, 768], flattened_s1: [32, 3072, 1, 768]
            # We want to concatenate at dim=-2 (the 3 and 1 dims)
            combined_data = torch.cat([flattened_s2, flattened_s1], dim=-2)
            print(f"combined_data shape: {combined_data.shape}")
            # write the stack data to an npy file
            np.save("./stacked_data.npy", combined_data.cpu().numpy())
            # compute the cosine similarity of every vector from the stacked data num vectors equaled to the stacked dim
            # adnd then log the distributions of the cosine similarities across al vectors
            # Concat features across modalities in space averaged across time
            batch_embeddings = batch_embeddings.pool_unmasked_tokens(
                self.pooling_type,
                spatial_pooling=self.spatial_pool,
                concat_features=self.concat_features,
            )
            # batch embedding shapes
            print(f"batch_embeddings shape: {batch_embeddings.shape}")
            raise ValueError("Stop here")
        else:
            pooled_tokens_dict = self.model(
                masked_helios_sample, patch_size=self.patch_size
            )["pooled_tokens_and_masks"]
            pooled_tokens = pooled_tokens_dict["modality_pooled_tokens"]
            # spatial pool is true means we want to keep the spatial dimensions
            # so here we just need to pool across time
            logger.info(f"pooled tokens shape in eval wrapper: {pooled_tokens.shape}")

            if self.spatial_pool:
                # B H W T C
                if pooled_tokens.shape[1] == 1 and pooled_tokens.ndim == 3:
                    # unsqueeze to get a W H C T
                    pooled_tokens = pooled_tokens.unsqueeze(1)
                pooled_tokens = reduce(
                    pooled_tokens, "b h w ... d -> b h w d", self.pooling_type
                )
            else:
                # Take the mean of all dims excetp the first and last
                pooled_tokens = reduce(
                    pooled_tokens, "b ... d -> b d", self.pooling_type
                )
            batch_embeddings = pooled_tokens
        return batch_embeddings


class PanopticonEvalWrapper(EvalWrapper):
    """Wrapper for Panopticon models."""

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass through the model produces the embedding specified by initialization."""
        if self.spatial_pool:
            # Intermediate features are not yet working because of some bug internal to the model
            batch_embeddings = self.model.forward_features(
                masked_helios_sample, pooling=self.pooling_type
            )
        else:
            batch_embeddings = self.model(
                masked_helios_sample, pooling=self.pooling_type
            )
        return batch_embeddings


class GalileoEvalWrapper(EvalWrapper):
    """Wrapper for Galileo models."""

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass through the model produces the embedding specified by initialization."""
        return self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )


class DINOv2EvalWrapper(EvalWrapper):
    """Wrapper for DINOv2 models."""

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass through the model produces the embedding specified by initialization."""
        # i need to do the apply imagenet normalizer thing in here
        if self.spatial_pool:
            # Intermediate features are not yet working because of some bug internal to the model
            batch_embeddings = self.model.forward_features(
                masked_helios_sample,
                pooling=self.pooling_type,
            )
        else:
            # should this call model ditectly
            batch_embeddings = self.model(
                masked_helios_sample,
                pooling=self.pooling_type,
            )
        return batch_embeddings


def get_eval_wrapper(model: nn.Module, **kwargs: Any) -> EvalWrapper:
    """Factory function to get the appropriate eval wrapper for a given model.

    Args:
        model: The model to evaluate.
        **kwargs: Additional keyword arguments.

    Returns:
        The appropriate eval wrapper for the given model.
    """
    if isinstance(model, FlexiHeliosBase) or isinstance(model, STBase):
        logger.info("Using HeliosEvalWrapper")
        return HeliosEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Panopticon):
        logger.info("Using PanopticonEvalWrapper")
        return PanopticonEvalWrapper(model=model, **kwargs)
    elif isinstance(model, DINOv2):
        logger.info("Using DINOv2EvalWrapper")
        return DINOv2EvalWrapper(model=model, **kwargs)
    elif isinstance(model, GalileoWrapper):
        logger.info("Using GalileoEvalWrapper")
        return GalileoEvalWrapper(model=model, **kwargs)
    else:
        raise NotImplementedError(f"No EvalWrapper for model type {type(model)}")
