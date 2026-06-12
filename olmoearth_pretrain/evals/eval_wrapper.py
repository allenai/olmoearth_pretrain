"""Eval wrapper contract to be able to run evals on a model."""

from logging import getLogger
from typing import Any, NamedTuple

import torch
from einops import rearrange, reduce
from torch import nn

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.models.registry import BASELINE_MODEL_SPECS
from olmoearth_pretrain.nn.flexi_vit import FlexiVitBase
from olmoearth_pretrain.nn.pooled_modality_predictor import EncodeEarlyAttnPool
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
from olmoearth_pretrain.nn.st_model import STBase

logger = getLogger(__name__)


def _model_inherits_from(model: nn.Module, class_name: str, module_prefix: str) -> bool:
    """Return True when a model inherits from a lazily identified adapter class."""
    return any(
        cls.__name__ == class_name
        and (
            cls.__module__ == module_prefix
            or cls.__module__.startswith(f"{module_prefix}.")
        )
        for cls in model.__class__.mro()
    )


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
            is_train: whether this is being used on the training data.
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.patch_size = patch_size
        self.pooling_type = pooling_type
        self.concat_features = concat_features
        self.spatial_pool = task_type in (TaskType.SEGMENTATION, TaskType.REGRESSION)
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

        if isinstance(dev, str):
            return torch.device(dev)

        # For FSDP wrapped models, fall back to device of model parameters
        return next(self.model.parameters()).device

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model if the attribute is not found on the wrapper."""
        return getattr(self.model, name)

    def _call_model_with_pooling(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
    ) -> torch.Tensor:
        """Call adapters that accept pooling and spatial-pooling controls."""
        return self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        raise NotImplementedError("Subclasses must implement this method")


class OlmoEarthEvalWrapper(EvalWrapper):
    """Wrapper for OlmoEarth Pretrain models."""

    @staticmethod
    def _has_missing_tokens(masked_olmoearth_sample: MaskedOlmoEarthSample) -> bool:
        for name, value in masked_olmoearth_sample.as_dict().items():
            if name.endswith("_mask") and value is not None:
                if (value == MaskValue.MISSING.value).any():
                    return True
        return False

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        if not self.use_pooled_tokens:
            fast_pass = not self._has_missing_tokens(masked_olmoearth_sample)
            batch_embeddings: TokensAndMasks = self.model(
                masked_olmoearth_sample, patch_size=self.patch_size, fast_pass=fast_pass
            )["tokens_and_masks"]  # (bsz, dim)
            # Concat features across modalities in space averaged across time
            batch_embeddings = pool_unmasked_tokens(
                batch_embeddings,
                self.pooling_type,
                spatial_pooling=self.spatial_pool,
                concat_features=self.concat_features,
            )
        else:
            pooled_tokens_dict = self.model(
                masked_olmoearth_sample, patch_size=self.patch_size, fast_pass=True
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
                # Take the mean of all dims except the first and last.
                pooled_tokens = reduce(
                    pooled_tokens, "b ... d -> b d", self.pooling_type
                )
            batch_embeddings = pooled_tokens
        return batch_embeddings, labels


HeliosEvalWrapper = _deprecated_class_alias(
    OlmoEarthEvalWrapper, "helios.evals.eval_wrapper.HeliosEvalWrapper"
)


class _PooledAdapterEvalWrapper(EvalWrapper):
    """Base wrapper for adapters that expose the shared eval embedding call."""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        return self._call_model_with_pooling(masked_olmoearth_sample), labels


class TerramindEvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for Terramind models."""


class _ForwardFeaturesWhenSpatialEvalWrapper(EvalWrapper):
    """Base wrapper for adapters that expose dense features separately."""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        if self.spatial_pool:
            batch_embeddings = self.model.forward_features(
                masked_olmoearth_sample, pooling=self.pooling_type
            )
        else:
            batch_embeddings = self.model(
                masked_olmoearth_sample, pooling=self.pooling_type
            )
        return batch_embeddings, labels


class PanopticonEvalWrapper(_ForwardFeaturesWhenSpatialEvalWrapper):
    """Wrapper for Panopticon models."""


class GalileoEvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for Galileo models."""


class AnySatEvalWrapper(EvalWrapper):
    """Wrapper for AnySat model."""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        embeddings = self._call_model_with_pooling(masked_olmoearth_sample)
        if is_train and (self.task_type == TaskType.SEGMENTATION):
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


class PrithviV2EvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for PrithviV2 model."""


class ClayEvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for Clay models."""


class CromaEvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for Croma models."""


class PrestoEvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for Presto model."""


class DINOv3EvalWrapper(_ForwardFeaturesWhenSpatialEvalWrapper):
    """Wrapper for DINOv3 models."""


class SatlasEvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for Satlas models."""


class TesseraEvalWrapper(_PooledAdapterEvalWrapper):
    """Wrapper for Tessera models."""


class _AdapterWrapperRegistration(NamedTuple):
    """Lazy adapter identity and the wrapper used for matching models."""

    class_name: str
    module_prefix: str
    wrapper_class: type[EvalWrapper]


_WRAPPER_CLASSES_BY_NAME: dict[str, type[EvalWrapper]] = {
    cls.__name__: cls
    for cls in (
        AnySatEvalWrapper,
        ClayEvalWrapper,
        CromaEvalWrapper,
        DINOv3EvalWrapper,
        GalileoEvalWrapper,
        PanopticonEvalWrapper,
        PrestoEvalWrapper,
        PrithviV2EvalWrapper,
        SatlasEvalWrapper,
        TerramindEvalWrapper,
        TesseraEvalWrapper,
    )
}

_ADAPTER_WRAPPERS: tuple[_AdapterWrapperRegistration, ...] = tuple(
    _AdapterWrapperRegistration(
        spec.adapter_class_name,
        spec.module_prefix,
        _WRAPPER_CLASSES_BY_NAME[spec.wrapper_class_name],
    )
    for spec in BASELINE_MODEL_SPECS
)


def get_eval_wrapper(model: nn.Module, **kwargs: Any) -> EvalWrapper:
    """Factory function to get the appropriate eval wrapper for a given model.

    Args:
        model: The model to evaluate.
        **kwargs: Additional keyword arguments.

    Returns:
        The appropriate eval wrapper for the given model.
    """
    if isinstance(model, FlexiVitBase) or isinstance(model, STBase):
        logger.info("Using OlmoEarthEvalWrapper")
        return OlmoEarthEvalWrapper(model=model, **kwargs)

    for registration in _ADAPTER_WRAPPERS:
        if _model_inherits_from(
            model,
            registration.class_name,
            registration.module_prefix,
        ):
            logger.info(f"Using {registration.wrapper_class.__name__}")
            return registration.wrapper_class(model=model, **kwargs)

    raise NotImplementedError(f"No EvalWrapper for model type {type(model)}")
