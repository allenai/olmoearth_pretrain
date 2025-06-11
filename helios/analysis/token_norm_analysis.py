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

from helios.data.constants import Modality
from helios.data.dataset import GetItemArgs, HeliosDataset, collate_helios
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
    for sample_index in range(num_samples):
        # without a token budget we simply are not using the patch size or the sampled hw p
        args = GetItemArgs(idx=sample_index, patch_size=patch_size,sampled_hw_p=hw_p)
        # we may also want to grab the specific sample from here
        patch_sample = dataset[args]
        # Prepare MaskedHeliosSample so that missing tokens are masked
        _, batch = collate_helios([patch_sample])
        sampled_hw = hw_p * patch_size
        max_t = 12
        batch = batch._create_cropped_data_dict(start_h=0, start_w=0,
                                        sampled_hw=sampled_hw, start_t=0, max_t=max_t)
        logger.info(f"Batch: {batch.device}")

        for modality in batch.modalities:
            logger.info(f"Modality: {modality}")
            logger.info(f"Shape: {getattr(batch, modality).shape}")
        with torch.no_grad():
            masked_batch = masking_strategy.apply_mask(batch, patch_size=patch_size)
            # now unmask the batch except missing tokens
            unmasked_batch = masked_batch.unmask()
            # Attach hooks to the model so that all the code runs within the hooks

            latent, decoded, _, reconstructed = model(unmasked_batch, patch_size)

        # call some sample visualization for this to happen