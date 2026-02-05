"""Utility functions for scaling hyperparameters with model size.

This module provides functions to automatically scale learning rates, warmup steps,
drop_path rates, and other hyperparameters based on model architecture size.
"""

import math
from typing import Literal

from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS


def get_model_scaling_factors(model_size_key: str) -> dict[str, float]:
    """Get scaling factors for a model size relative to nano baseline.
    
    Args:
        model_size_key: Key from MODEL_SIZE_ARGS (e.g., "nano", "base_shallow_decoder")
    
    Returns:
        Dictionary with scaling factors:
        - embedding_ratio: Ratio of embedding sizes (nano=1.0)
        - depth_ratio: Ratio of encoder depths (nano=1.0)
        - param_ratio: Approximate parameter count ratio
    """
    nano_config = MODEL_SIZE_ARGS["nano"]
    model_config = MODEL_SIZE_ARGS[model_size_key]
    
    embedding_ratio = model_config["encoder_embedding_size"] / nano_config["encoder_embedding_size"]
    depth_ratio = model_config["encoder_depth"] / nano_config["encoder_depth"]
    
    # Approximate parameter count (embedding_size^2 * depth is rough estimate)
    nano_params = nano_config["encoder_embedding_size"] ** 2 * nano_config["encoder_depth"]
    model_params = model_config["encoder_embedding_size"] ** 2 * model_config["encoder_depth"]
    param_ratio = model_params / nano_params
    
    return {
        "embedding_ratio": embedding_ratio,
        "depth_ratio": depth_ratio,
        "param_ratio": param_ratio,
    }


def scale_learning_rate(
    base_lr: float,
    model_size_key: str,
    scaling_method: Literal["sqrt", "linear", "inverse", "none"] = "sqrt",
) -> float:
    """Scale learning rate based on model size.
    
    Args:
        base_lr: Base learning rate (typically for nano model)
        model_size_key: Model size key from MODEL_SIZE_ARGS
        scaling_method: How to scale LR
            - "sqrt": Scale by sqrt(embedding_ratio) - recommended
            - "linear": Scale by embedding_ratio
            - "inverse": Scale by 1/embedding_ratio (more aggressive)
            - "none": No scaling
    
    Returns:
        Scaled learning rate
    """
    if scaling_method == "none":
        return base_lr
    
    factors = get_model_scaling_factors(model_size_key)
    embedding_ratio = factors["embedding_ratio"]
    
    if scaling_method == "sqrt":
        scale = math.sqrt(1.0 / embedding_ratio)
    elif scaling_method == "linear":
        scale = 1.0 / embedding_ratio
    elif scaling_method == "inverse":
        scale = 1.0 / embedding_ratio
    else:
        raise ValueError(f"Unknown scaling_method: {scaling_method}")
    
    return base_lr * scale


def scale_warmup_steps(
    base_warmup: int,
    model_size_key: str,
    scaling_method: Literal["depth", "embedding", "none"] = "depth",
) -> int:
    """Scale warmup steps based on model size.
    
    Args:
        base_warmup: Base warmup steps (typically for nano model)
        model_size_key: Model size key from MODEL_SIZE_ARGS
        scaling_method: How to scale warmup
            - "depth": Scale by depth_ratio (recommended)
            - "embedding": Scale by embedding_ratio
            - "none": No scaling
    
    Returns:
        Scaled warmup steps
    """
    if scaling_method == "none":
        return base_warmup
    
    factors = get_model_scaling_factors(model_size_key)
    
    if scaling_method == "depth":
        scale = factors["depth_ratio"]
    elif scaling_method == "embedding":
        scale = factors["embedding_ratio"]
    else:
        raise ValueError(f"Unknown scaling_method: {scaling_method}")
    
    return int(base_warmup * scale)


def scale_drop_path(
    base_drop_path: float,
    model_size_key: str,
    scaling_method: Literal["depth", "linear_depth", "none"] = "depth",
) -> float:
    """Scale drop_path rate based on model depth.
    
    Args:
        base_drop_path: Base drop_path rate (typically for nano model)
        model_size_key: Model size key from MODEL_SIZE_ARGS
        scaling_method: How to scale drop_path
            - "depth": Scale linearly with depth_ratio
            - "linear_depth": Use linear schedule 0.0 → scaled_max
            - "none": No scaling
    
    Returns:
        Scaled drop_path rate (or max for linear schedule)
    """
    if scaling_method == "none":
        return base_drop_path
    
    factors = get_model_scaling_factors(model_size_key)
    depth_ratio = factors["depth_ratio"]
    
    if scaling_method == "depth":
        return base_drop_path * depth_ratio
    elif scaling_method == "linear_depth":
        # Return max value for linear schedule
        return base_drop_path * depth_ratio
    else:
        raise ValueError(f"Unknown scaling_method: {scaling_method}")


def scale_weight_decay(
    base_weight_decay: float,
    model_size_key: str,
    scaling_method: Literal["slight_increase", "none"] = "slight_increase",
) -> float:
    """Scale weight decay based on model size.
    
    Args:
        base_weight_decay: Base weight decay (typically for nano model)
        model_size_key: Model size key from MODEL_SIZE_ARGS
        scaling_method: How to scale weight decay
            - "slight_increase": Slightly increase for larger models
            - "none": No scaling
    
    Returns:
        Scaled weight decay
    """
    if scaling_method == "none":
        return base_weight_decay
    
    factors = get_model_scaling_factors(model_size_key)
    embedding_ratio = factors["embedding_ratio"]
    
    if scaling_method == "slight_increase":
        # Increase by 10-50% for larger models
        increase_factor = 1.0 + 0.1 * min(embedding_ratio / 6.0, 0.5)
        return base_weight_decay * increase_factor
    else:
        raise ValueError(f"Unknown scaling_method: {scaling_method}")


def get_scaled_optim_config(
    model_size_key: str,
    base_lr: float = 0.0001,
    base_weight_decay: float = 0.02,
    lr_scaling: Literal["sqrt", "linear", "inverse", "none"] = "sqrt",
    wd_scaling: Literal["slight_increase", "none"] = "slight_increase",
) -> AdamWConfig:
    """Get scaled optimizer config for a model size.
    
    Args:
        model_size_key: Model size key from MODEL_SIZE_ARGS
        base_lr: Base learning rate (nano baseline)
        base_weight_decay: Base weight decay (nano baseline)
        lr_scaling: Learning rate scaling method
        wd_scaling: Weight decay scaling method
    
    Returns:
        Scaled AdamWConfig
    """
    lr = scale_learning_rate(base_lr, model_size_key, lr_scaling)
    weight_decay = scale_weight_decay(base_weight_decay, model_size_key, wd_scaling)
    
    return AdamWConfig(lr=lr, weight_decay=weight_decay, fused=False)


def get_scaled_scheduler(
    model_size_key: str,
    base_warmup: int = 8000,
    warmup_scaling: Literal["depth", "embedding", "none"] = "depth",
) -> CosWithWarmup:
    """Get scaled scheduler for a model size.
    
    Args:
        model_size_key: Model size key from MODEL_SIZE_ARGS
        base_warmup: Base warmup steps (nano baseline)
        warmup_scaling: Warmup scaling method
    
    Returns:
        Scaled CosWithWarmup scheduler
    """
    warmup_steps = scale_warmup_steps(base_warmup, model_size_key, warmup_scaling)
    return CosWithWarmup(warmup_steps=warmup_steps)


def get_scaled_drop_path(
    model_size_key: str,
    base_drop_path: float = 0.1,
    drop_path_scaling: Literal["depth", "linear_depth", "none"] = "depth",
) -> float:
    """Get scaled drop_path for a model size.
    
    Args:
        model_size_key: Model size key from MODEL_SIZE_ARGS
        base_drop_path: Base drop_path (nano baseline)
        drop_path_scaling: Drop path scaling method
    
    Returns:
        Scaled drop_path rate
    """
    return scale_drop_path(base_drop_path, model_size_key, drop_path_scaling)


# Example usage in a script:
"""
from scaling_utils import (
    get_scaled_optim_config,
    get_scaled_scheduler,
    get_scaled_drop_path,
)

# In build_train_module_config:
model_size_key = "base_shallow_decoder"  # or get from model config

optim_config = get_scaled_optim_config(
    model_size_key,
    base_lr=0.0001,
    base_weight_decay=0.02,
    lr_scaling="sqrt",  # Recommended
    wd_scaling="slight_increase",
)

scheduler = get_scaled_scheduler(
    model_size_key,
    base_warmup=8000,
    warmup_scaling="depth",  # Recommended
)

# In build_model_config:
drop_path = get_scaled_drop_path(
    model_size_key,
    base_drop_path=0.1,
    drop_path_scaling="depth",  # Recommended
)
"""

