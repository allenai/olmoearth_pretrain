"""Helios specific wandb callback."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError
from olmo_core.train.callbacks.wandb import WANDB_API_KEY_ENV_VAR, WandBCallback

from helios.data.dataloader import HeliosDataLoader
from helios.data.utils import plot_latlon_distribution, plot_modality_data_distribution

logger = logging.getLogger(__name__)


class HeliosWandBCallback(WandBCallback):
    """Helios specific wandb callback."""

    upload_dataset_distribution_pre_train: bool = True
    restart_on_same_run: bool = True

    def pre_train(self) -> None:
        """Pre-train callback for the wandb callback."""
        if self.enabled and get_rank() == 0:
            self.wandb
            if WANDB_API_KEY_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{WANDB_API_KEY_ENV_VAR}'")

            wandb_dir = Path(self.trainer.save_folder) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            resume_id = None
            if self.restart_on_same_run:
                runid_file = wandb_dir / "wandb_runid.txt"
                if runid_file.exists():
                    resume_id = runid_file.read_text().strip()

            self.wandb.init(
                dir=wandb_dir,
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=self.name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                id=resume_id,
                resume="allow",
            )

            if not resume_id and self.restart_on_same_run:
                runid_file.write_text(self.run.id)

            self._run_path = self.run.path  # type: ignore
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
                    {
                        "dataset/pretraining_geographic_distribution": self.wandb.Image(
                            fig
                        )
                    }
                )
                plt.close(fig)
                logger.info("Gathering normalized data distribution")
                sample_data = dataset.get_sample_data_for_histogram()
                for modality, modality_data in sample_data.items():
                    fig = plot_modality_data_distribution(modality, modality_data)
                    self.wandb.log(
                        {
                            f"dataset/pretraining_{modality}_distribution": self.wandb.Image(
                                fig
                            )
                        }
                    )
                    plt.close(fig)
