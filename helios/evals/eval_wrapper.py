"""Eval wrapper contract to be able to run evals on a model."""

from logging import getLogger
from typing import Any

import torch
from torch import nn

from helios.evals.datasets.configs import TaskType
from helios.evals.models import DINOv2, Panopticon
from helios.nn.flexihelios import FlexiHeliosBase, PoolingType, TokensAndMasks
from helios.nn.st_model import STBase
from helios.train.masking import MaskedHeliosSample

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
    ):
        """Initialize the eval wrapper.

        Args:
            model: The model to evaluate.
            task_type: The type of task to evaluate.
            patch_size: The patch size to use for the model.
            pooling_type: The pooling type to use for the model.
            concat_features: Whether to concatenate features across modalities.
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.patch_size = patch_size
        self.pooling_type = pooling_type
        self.concat_features = concat_features
        self.spatial_pool = task_type == TaskType.SEGMENTATION

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
        batch_embeddings: TokensAndMasks = self.model(
            masked_helios_sample, patch_size=self.patch_size
        )[0]  # (bsz, dim)
        # Concat features across modalities in space averaged across time
        tokens_and_masks = batch_embeddings.pool_unmasked_tokens(
            self.pooling_type,
            spatial_pooling=self.spatial_pool,
            concat_features=self.concat_features,
        )
        return tokens_and_masks


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
    else:
        raise NotImplementedError(f"No EvalWrapper for model type {type(model)}")
