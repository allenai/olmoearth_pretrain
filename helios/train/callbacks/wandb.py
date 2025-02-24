"""Helios specific wandb callback."""

import logging

import matplotlib.pyplot as plt
from olmo_core.train.callbacks.wandb import WandBCallback

from helios.data.dataloader import HeliosDataLoader
from helios.data.utils import plot_latlon_distribution

logger = logging.getLogger(__name__)


class HeliosWandBCallback(WandBCallback):
    """Helios specific wandb callback."""

    upload_dataset_distribution_pre_train: bool = True

    def pre_train(self) -> None:
        """Pre-train callback for the wandb callback."""
        super().pre_train()
        if self.upload_dataset_distribution_pre_train:
            assert isinstance(self.trainer.data_loader, HeliosDataLoader)
            dataset = self.trainer.data_loader.dataset
            logger.info("Gathering locations of entire dataset")
            latlons = dataset.get_geographic_distribution()
            # this should just be a general utility function
            logger.info(f"Uploading dataset distribution to wandb: {latlons.shape}")
            fig = plot_latlon_distribution(
                latlons, "Geographic Distribution of Dataset"
            )
            # Log to wandb
            self.wandb.log(
                {"dataset/pretraining_geographic_distribution": self.wandb.Image(fig)}
            )
            plt.close(fig)
