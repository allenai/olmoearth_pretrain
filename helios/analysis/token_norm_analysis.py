""" Module for analyzing the token norms. """

import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
from helios.train.masking import MaskingConfig
import torch
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from einops import rearrange
from matplotlib.figure import Figure
from upath import UPath
from olmo_core.utils import get_default_device

from helios.data.constants import Modality
from helios.data.dataset import GetItemArgs, HeliosDataset, HeliosSample, collate_helios
from helios.data.utils import convert_to_db


logger = logging.getLogger(__name__)


def analyze_token_norms(
    dataset: HeliosDataset,
    model: torch.nn.Module,
    patch_size: int,
    hw_p: int,
    num_samples: int | None = None,
) -> None:
    """Analyze the token norms."""

    masking_config = MaskingConfig(strategy_config={"type": "random"})
    masking_strategy = masking_config.build()
    if num_samples is None:
        num_samples = len(dataset)
    logger.info(f"Analyzing {num_samples} samples")
    model.eval()
    # log default device
    logger.info(f"Default device: {get_default_device()}")
    device = get_default_device()
    for sample_index in range(num_samples):
        # without a token budget we simply are not using the patch size or the sampled hw p
        args = GetItemArgs(idx=sample_index, patch_size=patch_size,sampled_hw_p=hw_p)
        # we may also want to grab the specific sample from here
        patch_sample = dataset[args]
        sampled_hw = hw_p * patch_size
        for modality in patch_sample[1].modalities:
            logger.info(f"Modality: {modality}")
            logger.info(f"Shape: {getattr(patch_sample[1], modality).shape}")
            logger.info(f"Type: {getattr(patch_sample[1], modality).dtype}")
        batch = HeliosSample(**patch_sample[1]._create_cropped_data_dict(start_h=0, start_w=0,
                                        sampled_hw=sampled_hw, start_t=0, max_t=12))
        patch_sample = (patch_sample[0], batch)
        # Prepare MaskedHeliosSample so that missing tokens are masked
        _, batch = collate_helios([patch_sample])
        with torch.no_grad():
            batch = batch.to_device(device)
            masked_batch = masking_strategy.apply_mask(batch, patch_size=patch_size)
            # now unmask the batch except missing tokens
            unmasked_batch = masked_batch.unmask()
            # Attach hooks to the model so that all the code runs within the hooks

            latent, decoded, _, reconstructed = model(unmasked_batch, patch_size)

        # call some sample visualization for this to happen