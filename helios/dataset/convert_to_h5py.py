"""Module for converting a dataset of GeoTiffs into a training dataset set up of h5py files."""

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import pandas as pd
from einops import rearrange
from olmo_core.config import Config
from tqdm import tqdm
from upath import UPath

from helios.data.constants import Modality, ModalitySpec, TimeSpan
from helios.data.utils import convert_to_db
from helios.dataset.parse import parse_helios_dataset
from helios.dataset.sample import (
    ModalityTile,
    SampleInformation,
    image_tiles_to_samples,
    load_image_for_sample,
)
from helios.dataset.utils import get_modality_specs_from_names

logger = logging.getLogger(__name__)


@dataclass
class ConvertToH5pyConfig(Config):
    """Configuration for converting GeoTiffs to H5py files.

    See https://docs.h5py.org/en/stable/high/dataset.html for more information on compression settings.
    """

    tile_path: str
    supported_modality_names: list[str]  # List of modality names
    multiprocessed_h5_creation: bool = True
    compression: str | None = None  # Compression algorithm
    compression_opts: int | None = None  # Compression level (0-9 for gzip)
    shuffle: bool | None = None  # Enable shuffle filter (only used with compression)

    def build(self) -> "ConvertToH5py":
        """Build the ConvertToH5py object."""
        return ConvertToH5py(
            tile_path=UPath(self.tile_path),
            supported_modalities=get_modality_specs_from_names(
                self.supported_modality_names
            ),
            multiprocessed_h5_creation=self.multiprocessed_h5_creation,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
        )


class ConvertToH5py:
    """Class for converting a dataset of GeoTiffs into a training dataset set up of h5py files."""

    h5py_folder: str = "h5py_data"
    latlon_distribution_fname: str = "latlon_distribution.npy"
    sample_metadata_fname: str = "sample_metadata.csv"
    sample_file_pattern: str = "sample_{index}.h5"
    compression_settings_fname: str = "compression_settings.json"

    def __init__(
        self,
        tile_path: UPath,
        supported_modalities: list[ModalitySpec],
        multiprocessed_h5_creation: bool = True,
        compression: str | None = None,
        compression_opts: int | None = None,
        shuffle: bool | None = None,
    ) -> None:
        """Initialize the ConvertToH5py object.

        Args:
            tile_path: The path to the tile directory, containing csvs for each modality and tiles
            supported_modalities: The list of modalities to convert to h5py that will be available in this dataset
            multiprocessed_h5_creation: Whether to create the h5py files in parallel
            compression: Compression algorithm to use (None, "gzip", "lzf", "szip")
            compression_opts: Compression level (0-9 for gzip), only used with gzip compression
            shuffle: Enable shuffle filter, only used with compression
        """
        self.tile_path = tile_path
        self.supported_modalities = supported_modalities
        logger.info(f"Supported modalities: {self.supported_modalities}")
        self.multiprocessed_h5_creation = multiprocessed_h5_creation
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.h5py_dir: UPath | None = None

    @property
    def compression_settings_suffix(self) -> str:
        """String representation of the compression settings.

        Use for folder naming.
        """
        compression_str = ""
        if self.compression is not None:
            compression_str = f"_{self.compression}"
        if self.compression_opts is not None:
            compression_str += f"_{self.compression_opts}"
        if self.shuffle is not None:
            compression_str += "_shuffle"
        return compression_str

    def _get_samples(self) -> list[SampleInformation]:
        """Get the samples from the raw dataset (image tile directory)."""
        tiles = parse_helios_dataset(self.tile_path, self.supported_modalities)
        samples = image_tiles_to_samples(tiles, self.supported_modalities)
        logger.info(f"Total samples: {len(samples)}")
        logger.info("Distribution of samples before filtering:\n")
        self._log_modality_distribution(samples)
        return samples

    def process_sample_into_h5(
        self, index_sample_tuple: tuple[int, SampleInformation]
    ) -> None:
        """Process a sample into an h5 file."""
        i, sample = index_sample_tuple
        h5_file_path = self._get_h5_file_path(i)
        if h5_file_path.exists():
            return
        self._create_h5_file(sample, h5_file_path)

    def create_h5_dataset(self, samples: list[SampleInformation]) -> None:
        """Create a dataset of the samples in h5 format in a shared weka directory under the given fingerprint."""
        total_sample_indices = len(samples)

        if self.multiprocessed_h5_creation:
            num_processes = max(1, mp.cpu_count() - 4)
            logger.info(f"Creating H5 dataset using {num_processes} processes")
            with mp.Pool(processes=num_processes) as pool:
                # Process samples in parallel and track progress with tqdm
                _ = list(
                    tqdm(
                        pool.imap(self.process_sample_into_h5, enumerate(samples)),
                        total=total_sample_indices,
                        desc="Creating H5 files",
                    )
                )
        else:
            for i, sample in enumerate(samples):
                logger.info(f"Processing sample {i}")
                self.process_sample_into_h5((i, sample))

    def save_sample_metadata(self, samples: list[SampleInformation]) -> None:
        """Save metadata about which samples contain which modalities."""
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")
        csv_path = self.h5py_dir / self.sample_metadata_fname
        logger.info(f"Writing metadata CSV to {csv_path}")

        # Create a DataFrame to store the metadata
        metadata_dict: dict = {
            "sample_index": [],
        }

        # Add columns for each supported modality
        for modality in self.supported_modalities:
            metadata_dict[modality.name] = []

        # Populate the DataFrame with metadata from each sample
        for i, sample in enumerate(samples):
            metadata_dict["sample_index"].append(i)

            # Set modality presence (1 if present, 0 if not)
            for modality in self.supported_modalities:
                metadata_dict[modality.name].append(
                    1 if modality in sample.modalities else 0
                )

        # Write the DataFrame to a CSV file
        df = pd.DataFrame(metadata_dict)
        df.to_csv(csv_path, index=False)

    def _get_h5_file_path(self, index: int) -> UPath:
        """Get the h5 file path."""
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")
        return self.h5py_dir / self.sample_file_pattern.format(index=index)

    @property
    def latlon_distribution_path(self) -> UPath:
        """Get the path to the latlon distribution file."""
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")
        return self.h5py_dir / self.latlon_distribution_fname

    def save_latlon_distribution(self, samples: list[SampleInformation]) -> None:
        """Save the latlon distribution to a file."""
        logger.info(f"Saving latlon distribution to {self.latlon_distribution_path}")
        latlons = np.array([sample.get_latlon() for sample in samples])
        with self.latlon_distribution_path.open("wb") as f:
            np.save(f, latlons)

    def _create_h5_file(
        self, sample: SampleInformation, h5_file_path: UPath
    ) -> dict[str, Any]:
        """Create the h5 file."""
        sample_dict = {}
        sample_dict["latlon"] = sample.get_latlon().astype(np.float32)
        sample_dict["timestamps"] = sample.get_timestamps()
        for modality in sample.modalities:
            sample_modality = sample.modalities[modality]
            image = self.load_sample(sample_modality, sample)
            # Convert Sentinel1 data to dB
            if modality == Modality.SENTINEL1:
                image = convert_to_db(image)
            sample_dict[modality.name] = image
        # w+b as sometimes metadata needs to be read as well for different chunking/compression settings
        with h5_file_path.open("w+b") as f:
            with h5py.File(f, "w") as h5file:
                for modality_name, image in sample_dict.items():
                    logger.info(
                        f"Writing modality {modality_name} to h5 file path {h5_file_path}"
                    )
                    # Create dataset with optional compression
                    create_kwargs: dict[str, Any] = {}

                    if self.compression is not None:
                        create_kwargs["compression"] = self.compression
                        # Only use compression_opts with gzip
                        if (
                            self.compression == "gzip"
                            and self.compression_opts is not None
                        ):
                            create_kwargs["compression_opts"] = self.compression_opts
                        # Only use shuffle with compression
                        if self.shuffle is not None:
                            create_kwargs["shuffle"] = self.shuffle

                    h5file.create_dataset(modality_name, data=image, **create_kwargs)
        return sample_dict

    def _log_modality_distribution(self, samples: list[SampleInformation]) -> None:
        """Log the modality distribution."""
        # Log modality distribution
        modality_counts: dict[str, int] = {}
        modality_combinations: dict[frozenset[str], int] = {}

        for sample in samples:
            # Count individual modalities
            for modality in sample.modalities:
                modality_counts[modality.name] = (
                    modality_counts.get(modality.name, 0) + 1
                )

            # Count modality combinations
            combination = frozenset(m.name for m in sample.modalities)
            modality_combinations[combination] = (
                modality_combinations.get(combination, 0) + 1
            )

        # Log individual modality counts
        for modality_name, count in modality_counts.items():
            percentage = (count / len(samples)) * 100
            logger.info(
                f"Modality {modality_name}: {count} samples ({percentage:.1f}%)"
            )

        # Log modality combinations
        logger.info("\nModality combinations:")
        for combination, count in modality_combinations.items():
            percentage = (count / len(samples)) * 100
            logger.info(
                f"{'+'.join(sorted(combination))}: {count} samples ({percentage:.1f}%)"
            )

    def set_h5py_dir(self, num_samples: int) -> None:
        """Set the h5py directory.

        This can only be set once to ensure consistency.

        Args:
            num_samples: Number of samples in the dataset
        """
        if self.h5py_dir is not None:
            logger.warning("h5py_dir is already set, ignoring new value")
            return

        h5py_dir = (
            self.tile_path
            / f"{self.h5py_folder}{self.compression_settings_suffix}"
            / "_".join(
                sorted([modality.name for modality in self.supported_modalities])
            )
            / str(num_samples)
        )
        self.h5py_dir = h5py_dir
        logger.info(f"Setting h5py_dir to {self.h5py_dir}")
        os.makedirs(self.h5py_dir, exist_ok=True)

    @classmethod
    def load_sample(
        cls, sample_modality: ModalityTile, sample: SampleInformation
    ) -> np.ndarray:
        """Load the sample."""
        image = load_image_for_sample(sample_modality, sample)

        if image.ndim == 4:
            modality_data = rearrange(image, "t c h w -> h w t c")
        else:
            modality_data = rearrange(image, "c h w -> h w c")
        return modality_data

    def _filter_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """Filter samples to adjust to the HeliosSample format."""
        logger.info(f"Number of samples before filtering: {len(samples)}")
        filtered_samples = []
        for sample in samples:
            if not all(
                modality in self.supported_modalities
                for modality in sample.modalities
                # TODO: clarify usage of ignore when parsing
                if not modality.ignore_when_parsing
            ):
                logger.info("Skipping sample because it has unsupported modalities")
                continue

            if sample.time_span != TimeSpan.YEAR:
                logger.info(
                    "Skipping sample because it is not the yearly frequency data"
                )
                continue

            multitemporal_modalities = [
                modality for modality in sample.modalities if modality.is_multitemporal
            ]
            total_multitemporal_modalities = len(multitemporal_modalities)
            # Pop off any modalities that don't have 12 months of data
            for modality in multitemporal_modalities:
                if len(sample.modalities[modality].images) != 12:
                    logger.info(
                        f"Skipping {modality} because it has less than 12 months of data"
                    )
                    sample.modalities.pop(modality)
                    total_multitemporal_modalities -= 1
            # If there's no multitemporal modalities, skip the sample
            if total_multitemporal_modalities == 0:
                logger.info(
                    "Skipping sample because it has no multitemporal modalities"
                )
                continue

            filtered_samples.append(sample)
        logger.info(f"Number of samples after filtering: {len(filtered_samples)}")
        logger.info("Distribution of samples after filtering:")
        self._log_modality_distribution(filtered_samples)
        return filtered_samples

    def get_and_filter_samples(self) -> list[SampleInformation]:
        """Get and filter samples.

        This parses csvs, loads images, and filters samples to adjust to the HeliosSample format.
        """
        samples = self._get_samples()
        return self._filter_samples(samples)

    def save_compression_settings(self) -> None:
        """Save compression settings to a JSON file."""
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")

        settings = {
            "compression": (
                str(self.compression) if self.compression is not None else None
            ),
            "compression_opts": (
                int(self.compression_opts)
                if self.compression_opts is not None
                else None
            ),
            "shuffle": bool(self.shuffle) if self.shuffle is not None else None,
        }

        settings_path = self.h5py_dir / self.compression_settings_fname
        logger.info(f"Saving compression settings to {settings_path}")
        with settings_path.open("w") as f:
            json.dump(settings, f, indent=2)

    def prepare_h5_dataset(self, samples: list[SampleInformation]) -> None:
        """Prepare the h5 dataset."""
        self.set_h5py_dir(len(samples))
        self.save_compression_settings()  # Save settings before creating data
        self.save_sample_metadata(samples)
        self.save_latlon_distribution(samples)
        logger.info("Attempting to create H5 files may take some time...")
        self.create_h5_dataset(samples)

    def run(self) -> None:
        """Run the conversion."""
        samples = self.get_and_filter_samples()
        self.prepare_h5_dataset(samples)
