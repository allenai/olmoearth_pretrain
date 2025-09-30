"""Eval wrapper contract to be able to run evals on a model."""

from logging import getLogger
from typing import Any

import torch
from einops import rearrange, reduce
from torch import nn

from helios.evals.datasets.configs import TaskType
from helios.evals.models import (
    AnySat,
    CopernicusFM,
    Croma,
    DINOv2,
    DINOv3,
    GalileoWrapper,
    Panopticon,
    PrestoWrapper,
    PrithviV2,
    Satlas,
    Tessera,
)
from helios.nn.flexihelios import FlexiHeliosBase, PoolingType, TokensAndMasks
from helios.nn.pooled_modality_predictor import EncodeEarlyAttnPool
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
        use_pooled_tokens: bool = False,
        is_train: bool = True,
    ):
        """Initialize the eval wrapper.

        Args:
            model: The model to evaluate.
            task_type: The type of task to evaluate.
            patch_size: The patch size to use for the model.
            pooling_type: The pooling type to use for the model.
            concat_features: Whether to concatenate features across modalities.
            use_pooled_tokens: Whether to use pooled tokens.
            is_train: whether this is being used on the training data.
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
        self.is_train = is_train

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

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        raise NotImplementedError("Subclasses must implement this method")


class HeliosEvalWrapper(EvalWrapper):
    """Wrapper for Helios models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        if not self.use_pooled_tokens:
            batch_embeddings: TokensAndMasks = self.model(
                masked_helios_sample, patch_size=self.patch_size
            )["tokens_and_masks"]  # (bsz, dim)
            # Concat features across modalities in space averaged across time
            batch_embeddings = batch_embeddings.pool_unmasked_tokens(
                self.pooling_type,
                spatial_pooling=self.spatial_pool,
                concat_features=self.concat_features,
            )
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
        return batch_embeddings, labels


class PanopticonEvalWrapper(EvalWrapper):
    """Wrapper for Panopticon models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return batch_embeddings, labels


class GalileoEvalWrapper(EvalWrapper):
    """Wrapper for Galileo models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return embeddings, labels


class AnySatEvalWrapper(EvalWrapper):
    """Wrapper for AnySat model."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        if self.is_train and (self.task_type == TaskType.SEGMENTATION):
            # this is a special case for AnySat. Since it outputs per-pixel embeddings,
            # we subsample training pixels to keep the memory requirements reasonable.
            # From https://arxiv.org/abs/2502.09356:
            # """
            # for semantic segmentation, the AnySat features are per-pixel
            # instead of per-patch. For comparable training cost, we sam-
            # ple 6.25% of its pixel features per image when training, but
            # evaluate with all pixel features when testing. We confirmed
            # the fairness of this evaluation with the the AnySat authors
            # by personal communication.
            # """
            subsample_by = 1 / 16
            embeddings = rearrange(embeddings, "b h w d -> b (h w) d")
            labels = rearrange(labels, "b h w -> b (h w)")

            assert embeddings.shape[1] == labels.shape[1]
            num_tokens = embeddings.shape[1]
            num_tokens_to_keep = int(num_tokens * subsample_by)
            sampled_indices = torch.randperm(num_tokens)[:num_tokens_to_keep]
            embeddings = embeddings[:, sampled_indices]
            labels = labels[:, sampled_indices]

            new_hw = int(num_tokens_to_keep**0.5)
            # reshape to h w
            embeddings = rearrange(
                embeddings, "b (h w) d -> b h w d", h=new_hw, w=new_hw
            )
            labels = rearrange(labels, "b (h w) -> b h w", h=new_hw, w=new_hw)
        return embeddings, labels


class PrithviV2EvalWrapper(EvalWrapper):
    """Wrapper for PrithviV2 model."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return embeddings, labels


class DINOv2EvalWrapper(EvalWrapper):
    """Wrapper for DINOv2 models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return batch_embeddings, labels


class CromaEvalWrapper(EvalWrapper):
    """Wrapper for Croma models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class CopernicusFMWrapper(EvalWrapper):
    """Wrapper for CopernicusFM model."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class PrestoEvalWrapper(EvalWrapper):
    """Wrapper for Presto model."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class DINOv3EvalWrapper(EvalWrapper):
    """Wrapper for DINOv3 models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return batch_embeddings, labels


class SatlasEvalWrapper(EvalWrapper):
    """Wrapper for Satlas models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class TesseraEvalWrapper(EvalWrapper):
    """Wrapper for Tessera models."""

    def __call__(
        self, masked_helios_sample: MaskedHeliosSample, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_helios_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


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
    elif isinstance(model, DINOv3):
        logger.info("Using DINOv3EvalWrapper")
        return DINOv3EvalWrapper(model=model, **kwargs)
    elif isinstance(model, Croma):
        logger.info("Using CromaEvalWrapper")
        return CromaEvalWrapper(model=model, **kwargs)
    elif isinstance(model, GalileoWrapper):
        logger.info("Using GalileoEvalWrapper")
        return GalileoEvalWrapper(model=model, **kwargs)
    elif isinstance(model, CopernicusFM):
        logger.info("Using CopernicusFMWrapper")
        return CopernicusFMWrapper(model=model, **kwargs)
    elif isinstance(model, PrestoWrapper):
        logger.info("Using PrestoEvalWrapper")
        return PrestoEvalWrapper(model=model, **kwargs)
    elif isinstance(model, AnySat):
        logger.info("Using AnySatEvalWrapper")
        return AnySatEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Satlas):
        logger.info("Using SatlasEvalWrapper")
        return SatlasEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Tessera):
        logger.info("Using TesseraEvalWrapper")
        return TesseraEvalWrapper(model=model, **kwargs)
    elif isinstance(model, PrithviV2):
        logger.info("Using PrithviEvalWrapper")
        return PrithviV2EvalWrapper(model=model, **kwargs)
    else:
        raise NotImplementedError(f"No EvalWrapper for model type {type(model)}")
