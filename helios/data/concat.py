"""Concat dataset for Helios."""

import bisect
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from olmo_core.config import Config
from torch.utils.data import ConcatDataset, Dataset

from helios.data.constants import ModalitySpec

from .dataset import GetItemArgs

logger = logging.getLogger(__name__)


class HeliosConcatDataset(ConcatDataset):
    """Dataset based on ConcatDataset for concatenating multiple HeliosDatasets.

    The resulting HeliosConcatDataset acts as a concatenated version of the individual
    HeliosDatasets.

    We need to use custom HeliosConcatDataset because we have a custom way to access
    __getitem__ (instead of just integer index), and we need to support various
    functions and attributes expected by the HeliosDataLoader and various callbacks.
    """

    def __getitem__(self, args: GetItemArgs) -> Any:
        """Get the sample at the given index."""
        # Adapted from ConcatDataset.
        # The only change we make is to extract the index from args, and then get a
        # tuple with updated index at the end to pass to the sub dataset.
        idx = args.idx
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_args = args._replace(idx=sample_idx)
        return self.datasets[dataset_idx][sample_args]

    @property
    def fingerprint_version(self) -> str:
        """The version of the fingerprint."""
        # We make sure fingerprint version is the same for all sub datasets.
        version = self.datasets[0].fingerprint_version
        for dataset in self.datasets:
            if dataset.fingerprint_version != version:
                raise ValueError(
                    "expected all sub datasets to have the same fingerprint_version"
                )
        return version

    @property
    def fingerprint(self) -> str:
        """Can be used to identify/compare a dataset."""
        # Compute fingerprint that combines the fingerprints of sub datasets.
        sha256_hash = hashlib.sha256()
        for dataset in self.datasets:
            if not hasattr(dataset, "fingerprint"):
                raise ValueError(
                    "expected all sub datasets to have fingerprint property"
                )
            sha256_hash.update(dataset.fingerprint.encode())
        return sha256_hash.hexdigest()

    def prepare(self) -> None:
        """Prepare the dataset."""
        # The datasets should already be prepared before initializing
        # HeliosConcatDataset since otherwise they would not have a length, but we
        # prepare here just in case.
        for dataset in self.datasets:
            dataset.prepare()

        # We need to compute latlon_distribution attribute since it is expected by some
        # callback.
        dataset_latlons = []
        for dataset in self.datasets:
            dataset_latlons.append(dataset.latlon_distribution)
        self.latlon_distribution = np.concatenate(dataset_latlons, axis=0)

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Return the supported modalities."""
        # We make sure supported modalities is same for all sub datasets.
        supported_modalities = self.datasets[0].supported_modalities
        for dataset in self.datasets:
            if dataset.supported_modalities != supported_modalities:
                raise ValueError(
                    "expected all sub datasets to have the same supported modalities"
                )
        return supported_modalities


@dataclass
class HeliosConcatDatasetConfig(Config):
    """Configuration for the HeliosConcatDataset."""

    dataset_configs: list[Config]

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.dataset_configs) == 0:
            raise ValueError("at least one dataset config must be provided")

    def build(self) -> HeliosConcatDataset:
        """Build the dataset."""
        self.validate()
        logging.info(f"concatenating {len(self.dataset_configs)} sub datasets")
        datasets: list[Dataset] = []
        for dataset_config in self.dataset_configs:
            dataset = dataset_config.build()
            # Dataset must be prepared before passing to HeliosConcatDataset so it has
            # a defined length.
            dataset.prepare()
            datasets.append(dataset)
        return HeliosConcatDataset(datasets)
