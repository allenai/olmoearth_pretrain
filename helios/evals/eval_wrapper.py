"""
Eval wrapper contract to be able to run evals on a model"""

from torch import nn
import torch
from typing import Any
from helios.evals.datasets.configs import TaskType
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample
from helios.nn.flexihelios import TokensAndMasks
from helios.nn.flexihelios import FlexiHeliosBase
from helios.evals.panopticon.panopticon import Panopticon
from helios.evals.dinov2.dinov2 import DINOv2
from logging import getLogger

logger = getLogger(__name__)



class EvalWrapper:
    def __init__(self, model: nn.Module, task_type: TaskType, patch_size: int, pooling_type: PoolingType, concat_features: bool = False, apply_imagenet_normalization: bool = False):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.patch_size = patch_size
        self.pooling_type = pooling_type
        self.concat_features = concat_features
        self.apply_imagenet_normalization = apply_imagenet_normalization
        self.spatial_pool = task_type == TaskType.SEGMENTATION

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        model = self.model
        if hasattr(model, "device"):
            return model.device
        else:
            return next(model.parameters()).device

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying model if the attribute is not found on the wrapper.
        """
        return getattr(self.model, name)

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        """
        Forward pass through the model produces the embedding specified by initialization
        """
        raise NotImplementedError("Subclasses must implement this method")




class HeliosEvalWrapper(EvalWrapper):

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        batch_embeddings: TokensAndMasks = self.model(
            masked_helios_sample, patch_size=self.patch_size
        )[0]  # (bsz, dim)
        # Concat features across modalities in space averaged across time
        batch_embeddings = batch_embeddings.pool_unmasked_tokens(
            self.pooling_type,
            spatial_pooling=self.spatial_pool,
            concat_features=self.concat_features,
        )
        return batch_embeddings


class PanopticonEvalWrapper(EvalWrapper):

    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        if self.spatial_pool:
            # Intermediate features are not yet working because of some bug internal to the model
            batch_embeddings = self.model.forward_features(masked_helios_sample, pooling=self.pooling_type)
        else:
            batch_embeddings = self.model(masked_helios_sample, pooling=self.pooling_type)
        return batch_embeddings


class DINOv2EvalWrapper(EvalWrapper):
    def __call__(self, masked_helios_sample: MaskedHeliosSample) -> torch.Tensor:
        # i need to do the apply imagenet normalizer thing in here
        if self.spatial_pool:
            # Intermediate features are not yet working because of some bug internal to the model
            batch_embeddings = self.model.forward_features(masked_helios_sample, pooling=self.pooling_type, apply_imagenet_normalization=self.apply_imagenet_normalization)
        else:
            # should this call model ditectly
            batch_embeddings = self.model.forward(masked_helios_sample, pooling=self.pooling_type, apply_imagenet_normalization=self.apply_imagenet_normalization)
        return batch_embeddings


def get_eval_wrapper(model: nn.Module, **kwargs) -> EvalWrapper:
    """
    Factory function to get the appropriate eval wrapper for a given model.
    """
    if isinstance(model, FlexiHeliosBase):
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