""" Module for analyzing the token norms using forward hooks. """

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import re
import torch.nn as nn
from einops import rearrange
from olmo_core.utils import get_default_device

from helios.data.constants import Modality
from helios.data.dataset import GetItemArgs, HeliosDataset, HeliosSample, collate_helios
from helios.train.masking import MaskingConfig, MaskValue
from helios.nn.flexihelios import Encoder, save_token_norm_histograms
from helios.data.visualize import create_visualization
# Add matplotlib imports for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, histogram visualization will be disabled")

logger = logging.getLogger(__name__)


class TokenNormAnalysisHook:
    """Forward hook for analyzing token norms during model execution."""

    def __init__(
        self,
        save_dir: str = "./new_token_norm_histograms",
        bins: int = 75,
        num_samples_to_record: int = 10,
        enabled: bool = True,
    ):
        """Initialize the token norm analysis hook.

        Args:
            save_dir: Directory to save histogram plots
            bins: Number of bins for histograms
            num_samples_to_record: Maximum number of samples to record per step
            enabled: Whether the hook is enabled
        """
        self.save_dir = save_dir
        self.bins = bins
        self.num_samples_to_record = num_samples_to_record
        self.enabled = enabled
        self.global_step = 0
        self.sample_count = 0

        # Create save directory
        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)


    def __call__(self, module: nn.Module, input_args: Tuple[torch.Tensor, ...], input_kwargs: Dict[str, Any], output: torch.Tensor) -> None:
        """Forward hook function called after each block.

        Args:
            module: The module (Block) that was called
            input: Input tensors to the module
            output: Output tensor from the module
        """
        if not self.enabled or not MATPLOTLIB_AVAILABLE:
            return

        if self.sample_count >= self.num_samples_to_record:
            return
        context = input_kwargs['unwrap_context']

        # Get block index from the module name or use a counter
        block_idx = getattr(module, '_hook_block_idx', 0)

        with torch.no_grad():
            # Clone and detach the output tokens
            visualization_tokens = output.clone().detach()

            # Add removed tokens back using the Encoder's static method
            visualization_tokens, _ = Encoder.add_removed_tokens(
                visualization_tokens,
                context['indices'],
                context['new_mask']
            )

            # Split and expand per modality using the Encoder's static method
            visualization_tokens_dict = Encoder.split_and_expand_per_modality(
                visualization_tokens,
                context['modalities_to_dims_dict']
            )
            visualization_tokens_dict.update(context['original_masks_dict'])

            # Save histograms using the existing function
            save_token_norm_histograms(
                visualization_tokens_dict=visualization_tokens_dict,
                block_idx=block_idx,
                save_dir=self.save_dir,
                bins=self.bins,
                global_step=self.global_step,
                num_samples_to_record=1,  # We're processing one sample at a time
            )

    def increment_step(self):
        """Increment the global step counter."""
        self.global_step += 1
        self.sample_count = 0

    def increment_sample(self):
        """Increment the sample counter."""
        self.sample_count += 1



def register_token_norm_hooks(
    model: nn.Module,
    hook: TokenNormAnalysisHook,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks on all Block modules in the model.

    Args:
        model: The model to register hooks on
        hook: The hook instance to register

    Returns:
        List of removable handles for the registered hooks
    """
    handles = []

    # Find all Block modules and register hooks
    for name, module in model.named_modules():
        # only do this if name starts with encoder.blocks
        if re.match(r'encoder\.blocks\.\d+', name) and module.__class__.__name__ == 'Block':
            logger.info(f" name: {name} Module: {module.__class__.__name__}")
            # Set block index for identification
            block_idx = len(handles)
            module._hook_block_idx = block_idx

            # Register the hook
            handle = module.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
            logger.info(f"Registered hook on {name} (block {block_idx})")

    return handles


def analyze_token_norms_with_hooks(
    dataset: HeliosDataset,
    model: torch.nn.Module,
    patch_size: int,
    hw_p: int,
    num_samples: int | None = None,
    save_dir: str = "./token_norm_histograms",
    bins: int = 75,
    num_samples_to_record: int = 10,
) -> None:
    """Analyze the token norms using forward hooks.

    Args:
        dataset: The dataset to analyze
        model: The model to analyze
        patch_size: Patch size for processing
        hw_p: Height/width parameter
        num_samples: Number of samples to analyze (None for all)
        save_dir: Directory to save histograms
        bins: Number of bins for histograms
        num_samples_to_record: Maximum samples to record per step
    """
    # Create the hook
    hook = TokenNormAnalysisHook(
        save_dir=save_dir,
        bins=bins,
        num_samples_to_record=num_samples_to_record,
        enabled=True,
    )

    # Register hooks on all Block modules
    handles = register_token_norm_hooks(model, hook)

    try:
        # Set up masking
        masking_config = MaskingConfig(strategy_config={"type": "random"})
        masking_strategy = masking_config.build()

        if num_samples is None:
            num_samples = len(dataset)

        logger.info(f"Analyzing {num_samples} samples with hooks")
        model.eval()
        model.encoder.pass_unwrap_context = True
        device = get_default_device()
        logger.info(f"Default device: {device}")

        sample_count = 0
        for sample_index in range(num_samples):
            # Get sample
            args = GetItemArgs(idx=sample_index, patch_size=patch_size, sampled_hw_p=hw_p)
            patch_sample = dataset[args]
            # save the visualization before cropping
            visual_sample = patch_sample[1]
            sampled_hw = hw_p * patch_size

            # Log modality information
            for modality in patch_sample[1].modalities:
                logger.info(f"Modality: {modality}")
                logger.info(f"Shape: {getattr(patch_sample[1], modality).shape}")
                logger.info(f"Type: {getattr(patch_sample[1], modality).dtype}")

            # Create batch
            batch = HeliosSample(**patch_sample[1]._create_cropped_data_dict(
                start_h=0, start_w=0, sampled_hw=sampled_hw, start_t=0, max_t=12
            ))
            patch_sample = (patch_sample[0], batch)

            # save a visualization of the sample
            for timestep in range(12):
                fig = create_visualization(
                    sample=visual_sample,
                    timestep=timestep,
                )
                step_dir = os.path.join(save_dir, f"step_{sample_count}")
                os.makedirs(step_dir, exist_ok=True)
                out_path = os.path.join(step_dir, f"visualization_{timestep}.png")
                fig.savefig(out_path)
                logger.info(f"Saved visualization to {out_path}")
                plt.close(fig)
                logger.warning("only saving one timestep for now")
                break

            # Prepare MaskedHeliosSample
            _, batch = collate_helios([patch_sample])

            with torch.no_grad():
                batch = batch.to_device(device)
                masked_batch = masking_strategy.apply_mask(batch, patch_size=patch_size)
                unmasked_batch = masked_batch.unmask()



                logger.info("Running model")
                # Run the model - hooks will be triggered
                latent, decoded, _, reconstructed = model(unmasked_batch, patch_size)

                # Increment sample counter
                hook.increment_sample()

            # Increment step after processing sample
            hook.increment_step()
            sample_count += 1
    finally:
        # Clean up hooks
        for handle in handles:
            handle.remove()
        logger.info("Removed all token norm analysis hooks")


# Keep the original function for backward compatibility
def analyze_token_norms(
    dataset: HeliosDataset,
    model: torch.nn.Module,
    patch_size: int,
    hw_p: int,
    num_samples: int | None = None,
) -> None:
    """Analyze the token norms (original implementation for backward compatibility)."""
    analyze_token_norms_with_hooks(
        dataset=dataset,
        model=model,
        patch_size=patch_size,
        hw_p=hw_p,
        num_samples=num_samples,
    )