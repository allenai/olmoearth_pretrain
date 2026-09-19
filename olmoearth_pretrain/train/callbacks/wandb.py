"""OlmoEarth Pretrain specific wandb callback."""

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError
from olmo_core.train.callbacks.beaker import BeakerCallback
from olmo_core.train.callbacks.wandb import WANDB_API_KEY_ENV_VAR, WandBCallback
from tqdm import tqdm

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.data.constants import IMAGE_TILE_SIZE, Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoader
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.data.utils import (
    plot_latlon_distribution,
    plot_modality_data_distribution,
)

logger = logging.getLogger(__name__)


def get_sample_data_for_histogram(
    dataset: OlmoEarthDataset, num_samples: int = 100, num_values: int = 100
) -> dict[str, Any]:
    """Get the sample data per modality per band for showing the histogram.

    Args:
        dataset: The dataset to sample from.
        num_samples: The number of samples to sample from the dataset.
        num_values: The number of values to sample from each modality per band.

    Returns:
        dict: A dictionary containing the sample data per modality per band.
    """
    if num_samples > len(dataset):
        raise ValueError(
            f"num_samples {num_samples} is greater than the number of samples in the dataset {len(dataset)}"
        )
    indices_to_sample = random.sample(list(range(len(dataset))), k=num_samples)
    sample_data: dict[str, Any] = {}

    # Assume samples could include different modalities and bands
    # TODO: compute the histogram for each modality and band directly
    for i in tqdm(indices_to_sample):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]
        for modality in sample.modalities:
            if modality == "latlon":
                continue
            modality_data = sample.as_dict()[modality]
            if modality_data is None:
                continue
            modality_spec = Modality.get(modality)
            modality_bands = modality_spec.band_order
            if modality not in sample_data:
                sample_data[modality] = {band: [] for band in modality_bands}
            # for each band, flatten the data and extend the list
            for idx, band in enumerate(modality_bands):
                sample_data[modality][band].extend(
                    random.sample(
                        modality_data[:, :, :, idx].flatten().tolist(), num_values
                    )
                )
    return sample_data


@dataclass
class OlmoEarthWandBCallback(WandBCallback):
    """OlmoEarth Pretrain specific wandb callback."""

    upload_dataset_distribution_pre_train: bool = True
    upload_modality_data_band_distribution_pre_train: bool = False
    restart_on_same_run: bool = True
    # Optional explicit path to the file storing this run's wandb id. When set,
    # the run always resumes from (and writes) this id. This lets multiple
    # separate jobs (e.g. the in-loop beaker eval jobs) share one wandb run id so
    # their metrics consolidate into a single run instead of one run per job.
    runid_path: str | None = None

    def pre_train(self) -> None:
        """Pre-train callback for the wandb callback."""
        if self.enabled and get_rank() == 0:
            self.wandb
            if WANDB_API_KEY_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{WANDB_API_KEY_ENV_VAR}'")

            wandb_dir = Path(self.trainer.save_folder) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            resume_id = None
            # A shared runid_path (set for in-loop beaker eval jobs) makes every
            # job resume one wandb run; otherwise fall back to the per-run file
            # when restart_on_same_run is set.
            runid_file = (
                Path(self.runid_path)
                if self.runid_path
                else wandb_dir / "wandb_runid.txt"
            )
            use_runid_file = self.runid_path is not None or self.restart_on_same_run
            if use_runid_file and runid_file.exists():
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
                settings=self.wandb.Settings(init_timeout=240),
            )

            if not resume_id and use_runid_file:
                runid_file.parent.mkdir(parents=True, exist_ok=True)
                runid_file.write_text(self.run.id)

            self._run_path = self.run.path  # type: ignore
            self._seed_beaker_config()
            if self.upload_dataset_distribution_pre_train:
                assert isinstance(self.trainer.data_loader, OlmoEarthDataLoader)
                dataset = self.trainer.data_loader.dataset
                logger.info("Gathering locations of entire dataset")
                latlons = dataset.latlon_distribution
                assert latlons is not None
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
                # Delete the latlon distribution from the dataset so it doesn't get pickled into data worker processes
                del dataset.latlon_distribution
                if self.upload_modality_data_band_distribution_pre_train:
                    logger.info("Gathering normalized data distribution")
                    sample_data = get_sample_data_for_histogram(dataset)
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

    def _seed_beaker_config(self) -> None:
        """Pre-seed the Beaker experiment id/URL in the wandb run config.

        olmo-core's BeakerCallback (which runs after this callback, since its
        priority is lower) writes ``beaker_experiment_url`` and
        ``beaker_experiment_id`` into the wandb run config without
        ``allow_val_change``. When a run is relaunched as a new Beaker experiment
        while resuming the same wandb run id (via the runid file), those keys
        already hold the previous experiment's values and wandb raises a
        ConfigError, killing rank 0. Writing the current values here with
        ``allow_val_change=True`` means the BeakerCallback's update later finds
        identical values and is a no-op, and the wandb config keeps pointing at
        the live experiment.
        """
        beaker_callback: BeakerCallback | None = None
        for callback in self.trainer.callbacks.values():
            if isinstance(callback, BeakerCallback):
                beaker_callback = callback
                break
        if beaker_callback is None or not beaker_callback.enabled:
            return

        try:
            from olmo_core.launch.beaker import (
                get_beaker_client,
                get_beaker_experiment_id,
            )

            # Mirror BeakerCallback.pre_train exactly so the values match.
            experiment_id = beaker_callback.experiment_id
            if experiment_id is None:
                experiment_id = get_beaker_experiment_id()
            if experiment_id is None:
                return
            with get_beaker_client() as beaker:
                workload = beaker.workload.get(experiment_id)
                beaker_url = beaker.workload.url(workload)
        except Exception as e:
            logger.warning(f"Failed to resolve Beaker experiment for wandb config: {e}")
            return

        self.run.config.update(  # type: ignore
            {
                "beaker_experiment_url": beaker_url,
                "beaker_experiment_id": experiment_id,
            },
            allow_val_change=True,
        )
        logger.info(
            f"Seeded wandb config with beaker_experiment_url={beaker_url} "
            f"beaker_experiment_id={experiment_id}"
        )


HeliosWandBCallback = _deprecated_class_alias(
    OlmoEarthWandBCallback, "helios.train.callbacks.wandb.HeliosWandBCallback"
)
