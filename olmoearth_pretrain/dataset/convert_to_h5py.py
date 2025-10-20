"""Module for converting a dataset of GeoTiffs into a training dataset set up of h5py files."""

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from einops import rearrange
from olmo_core.config import Config
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import (
    IMAGE_TILE_SIZE,
    SENTINEL1_NODATA,
    YEAR_NUM_TIMESTEPS,
    Modality,
    ModalitySpec,
    TimeSpan,
)
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.dataset.parse import parse_dataset
from olmoearth_pretrain.dataset.sample import (
    ModalityTile,
    SampleInformation,
    image_tiles_to_samples,
    load_image_for_sample,
)
from olmoearth_pretrain.dataset.utils import get_modality_specs_from_names

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)


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
    chunk_options: tuple | None = (
        None  # Chunking configuration. None: disabled. True: auto (data_item.shape). tuple: specific shape.
    )
    tile_size: int = IMAGE_TILE_SIZE
    # Processes may go to sleep state if we use too many processes
    reserved_cores: int = (
        10  # Number of cores to reserve and not used for multiprocessing
    )
    required_modality_names: list[str] = field(
        default_factory=lambda: list()
    )  # Samples without all of these are skipped

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
            chunk_options=self.chunk_options,
            tile_size=self.tile_size,
            reserved_cores=self.reserved_cores,
            required_modalities=get_modality_specs_from_names(
                self.required_modality_names
            ),
        )


class ConvertToH5py:
    """Class for converting a dataset of GeoTiffs into a training dataset set up of h5py files."""

    h5py_folder: str = "h5py_data_w_missing_timesteps"
    latlon_distribution_fname: str = "latlon_distribution.npy"
    sample_metadata_fname: str = "sample_metadata.csv"
    sample_file_pattern: str = "sample_{index}.h5"
    compression_settings_fname: str = "compression_settings.json"
    missing_timesteps_mask_group_name: str = "missing_timesteps_masks"

    def __init__(
        self,
        tile_path: UPath,
        supported_modalities: list[ModalitySpec],
        multiprocessed_sample_processing: bool = True,
        multiprocessed_h5_creation: bool = True,
        compression: str | None = None,
        compression_opts: int | None = None,
        shuffle: bool | None = None,
        chunk_options: tuple | bool | None = None,
        tile_size: int = IMAGE_TILE_SIZE,
        reserved_cores: int = 10,
        required_modalities: list[ModalitySpec] = [],
    ) -> None:
        """Initialize the ConvertToH5py object.

        Args:
            tile_path: The path to the tile directory, containing csvs for each modality and tiles
            supported_modalities: The list of modalities to convert to h5py that will be available in this dataset
            multiprocessed_sample_processing: Whether to process samples in parallel
            multiprocessed_h5_creation: Whether to create the h5py files in parallel
            compression: Compression algorithm to use (None, "gzip", "lzf", "szip")
            compression_opts: Compression level (0-9 for gzip), only used with gzip compression
            shuffle: Enable shuffle filter, only used with compression
            chunk_options: Chunking configuration.
                         None: chunking disabled.
                         True: auto-chunk (chunks will match dataset shape).
                         tuple: specify a chunk shape. If tuple rank differs from data rank,
                                it's adjusted (padded with full dimension sizes or truncated).
            tile_size: The size of the tile to split the image into. It is based on the IMAGE_TILE_SIZE, so
                higher-resolution modalities like NAIP would be split up correspondingly.
            reserved_cores: The number of cores to reserve and not use for multiprocessing.
            required_modalities: Samples without all of these modalities will be skipped.
        """
        self.tile_path = tile_path
        self.supported_modalities = supported_modalities
        logger.info(f"Supported modalities: {self.supported_modalities}")
        self.multiprocessed_sample_processing = multiprocessed_sample_processing
        self.multiprocessed_h5_creation = multiprocessed_h5_creation
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.chunk_options = chunk_options
        self.h5py_dir: UPath | None = None
        self.required_modalities = required_modalities
        if IMAGE_TILE_SIZE % tile_size != 0:
            raise ValueError(
                f"Tile size {tile_size} must be a factor of {IMAGE_TILE_SIZE}"
            )
        self.tile_size = tile_size
        # Tile_size_split_factor is the factor by which the tile size is split into subtiles
        self.num_subtiles_per_dim = IMAGE_TILE_SIZE // tile_size
        self.num_subtiles = self.num_subtiles_per_dim**2
        self.reserved_cores = reserved_cores

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

    @property
    def image_tile_size_suffix(self) -> str:
        """String representation of the image tile size."""
        return f"_{self.tile_size}_x_{self.num_subtiles}"

    def _get_samples(self) -> list[SampleInformation]:
        """Get the samples from the raw dataset (image tile directory)."""
        tiles = parse_dataset(self.tile_path, self.supported_modalities)
        samples = image_tiles_to_samples(tiles, self.supported_modalities)
        logger.info(f"Total samples: {len(samples)}")
        logger.info("Distribution of samples before filtering:\n")
        self._log_modality_distribution(samples)
        return samples

    def process_sample_into_h5(
        self, index_sample_tuple: tuple[int, tuple[int, SampleInformation]]
    ) -> None:
        """Process a sample into an h5 file."""
        i, (sublock_index, sample) = index_sample_tuple
        h5_file_path = self._get_h5_file_path(i)
        # Comment out the exist check because if somehow the h5 generation is interrupted
        # When rerun the process, there're maybe some broken h5 files (partially written, invalid)
        # if h5_file_path.exists():
        #     return
        self._create_h5_file(sample, h5_file_path, sublock_index)

    def create_h5_dataset(self, samples: list[tuple[int, SampleInformation]]) -> None:
        """Create a dataset of the samples in h5 format in a shared weka directory under the given fingerprint."""
        total_sample_indices = len(samples)

        if self.multiprocessed_h5_creation:
            num_processes = max(1, mp.cpu_count() - self.reserved_cores)
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
            for i, (sublock_index, sample) in enumerate(samples):
                logger.info(f"Processing sample {i}")
                self.process_sample_into_h5((i, (sublock_index, sample)))

    def save_sample_metadata(
        self, samples: list[tuple[int, SampleInformation]]
    ) -> None:
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
        for i, (_, sample) in enumerate(samples):
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

    def save_latlon_distribution(
        self, samples: list[tuple[int, SampleInformation]]
    ) -> None:
        """Save the latlon distribution to a file."""
        logger.info(f"Saving latlon distribution to {self.latlon_distribution_path}")
        latlons = np.array([sample.get_latlon() for _, sample in samples])
        with self.latlon_distribution_path.open("wb") as f:
            np.save(f, latlons)

    def _find_longest_timestamps_array(
        self, spacetime_varying_modalities: dict[ModalitySpec, np.ndarray]
    ) -> np.ndarray:
        """Find the timestamps for the sample with the most timestamps."""
        return spacetime_varying_modalities[
            max(
                spacetime_varying_modalities,
                key=lambda k: len(spacetime_varying_modalities[k]),
            )
        ]

    def _create_missing_timesteps_masks(
        self,
        spacetime_varying_modalities: dict[ModalitySpec, np.ndarray],
        longest_timestamps_array: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Create missing timesteps masks for each modality."""
        missing_timesteps_masks_data: dict[str, np.ndarray] = {}
        for mod_spec, mod_timestamps in spacetime_varying_modalities.items():
            # Create a boolean mask indicating presence of each timestamp from longest_timestamps_array
            # in the current modality's timestamps.
            # np.all(..., axis=1) checks for full row match (day, month, year)
            # np.any(...) checks if any of mod_timestamps' rows match the current longest_ts
            mask = np.array(
                [
                    np.any(np.all(longest_ts == mod_timestamps, axis=1))
                    for longest_ts in longest_timestamps_array
                ],
                dtype=bool,
            )
            missing_timesteps_masks_data[mod_spec.name] = mask
        return missing_timesteps_masks_data

    def _remove_bad_modalities_from_sample(
        self, sample: SampleInformation
    ) -> SampleInformation:
        """Remove bad modalities from the sample."""
        modalities_to_remove = set()
        for modality in sample.modalities:
            sample_modality = sample.modalities[modality]
            image = self.load_sample(sample_modality, sample)
            # Remove modalities that contains any nan
            if np.any(np.isnan(image)):
                logger.warning(
                    f"Image for modality {modality.name} contains NaN values, removing this modality"
                )
                modalities_to_remove.add(modality)
            # Remove ERA5 and OSM Raster if they are all zeros
            if (
                modality == Modality.ERA5_10
                or modality == Modality.OPENSTREETMAP_RASTER
            ) and np.all(image == 0):
                logger.warning(
                    f"Image for modality {modality.name} is all zeros, removing this modality"
                )
                modalities_to_remove.add(modality)
            # Remove S1 if it contains any nodata values
            if modality == Modality.SENTINEL1 and np.any(image == SENTINEL1_NODATA):
                logger.warning(
                    f"Image for modality {modality.name} contains nodata values, removing this modality"
                )
                modalities_to_remove.add(modality)
        for modality in modalities_to_remove:
            del sample.modalities[modality]
        return sample

    def _process_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """Process samples before the filtering process."""
        total_sample_indices = len(samples)
        if self.multiprocessed_sample_processing:
            num_processes = max(1, mp.cpu_count() - self.reserved_cores)
            logger.info(f"Processing samples using {num_processes} processes")
            with mp.Pool(processes=num_processes) as pool:
                # Process samples in parallel and track progress with tqdm
                processed_samples = list(
                    tqdm(
                        pool.imap(self._remove_bad_modalities_from_sample, samples),
                        total=total_sample_indices,
                        desc="Processing samples",
                    )
                )
        else:
            processed_samples = []
            for i, sample in enumerate(samples):
                logger.info(f"Processing sample {i}")
                processed_sample = self._remove_bad_modalities_from_sample(sample)
                processed_samples.append(processed_sample)
        return processed_samples

    def _create_h5_file(
        self, sample: SampleInformation, h5_file_path: UPath, sublock_index: int
    ) -> dict[str, Any]:
        """Create the h5 file."""
        sample_dict = {}
        sample_dict["latlon"] = sample.get_latlon().astype(np.float32)
        multi_temporal_timestamps_dict = sample.get_timestamps()

        # Compute longest timestamps from only spacetime varying modalities
        # This is used to deal with the case that when ERA5 is available, it always has full 12 months
        # But other spacetime varying modalities may have way less months
        spacetime_varying_modalities = {
            modality: timestamps
            for modality, timestamps in multi_temporal_timestamps_dict.items()
            if modality.is_spacetime_varying
        }
        # Note that with the longest timestamps array, we cut all modalities to this range
        # Anything outside of this range is considered missing
        longest_timestamps_array = self._find_longest_timestamps_array(
            spacetime_varying_modalities
        )
        missing_timesteps_masks_data = self._create_missing_timesteps_masks(
            spacetime_varying_modalities, longest_timestamps_array
        )

        sample_dict["timestamps"] = longest_timestamps_array

        # Load image data for all modalities in the sample
        for modality in sample.modalities:
            sample_modality = sample.modalities[modality]
            image = self.load_sample(sample_modality, sample)

            if modality == Modality.SENTINEL1:
                # Convert Sentinel1 data to dB
                image = convert_to_db(image)

            if modality.is_spatial:
                # Calculate row and column indices for grid
                if image.shape[0] != image.shape[1]:
                    raise ValueError("Expected image width to match image height")
                if image.shape[0] % self.num_subtiles_per_dim != 0:
                    raise ValueError(
                        f"Got image size {image.shape[0]} which is not multiple of subtile count {self.num_subtiles_per_dim}"
                    )
                tile_size = image.shape[0] // self.num_subtiles_per_dim
                row = (sublock_index // self.num_subtiles_per_dim) * tile_size
                col = (sublock_index % self.num_subtiles_per_dim) * tile_size
                logger.info(f"Sublock index: {sublock_index}, row: {row}, col: {col}")
                logger.info(f"Image shape: {image.shape}")
                image = image[row : row + tile_size, col : col + tile_size, ...]
                logger.info(f"Image shape after slicing: {image.shape}")

            sample_dict[modality.name] = image

        # w+b as sometimes metadata needs to be read as well for different chunking/compression settings
        with h5_file_path.open("w+b") as f:
            with h5py.File(f, "w") as h5file:
                # Write datasets for latlon, timestamps, and modality images
                for item_name, data_item in sample_dict.items():
                    logger.info(
                        f"Writing item {item_name} to h5 file path {h5_file_path}"
                    )
                    # Create dataset with optional compression
                    create_kwargs: dict[str, Any] = {}
                    # Maybe want to move this into a seperate class for ease of use
                    if self.compression is not None:
                        # Gzip is natively supported by h5py
                        if self.compression == "gzip":
                            create_kwargs["compression"] = self.compression
                            if self.compression_opts is not None:
                                create_kwargs["compression_opts"] = (
                                    self.compression_opts
                                )
                            if self.shuffle is not None:
                                create_kwargs["shuffle"] = self.shuffle
                        # For other compression algorithms, we switch to hdf5plugin
                        elif self.compression == "zstd":
                            create_kwargs["compression"] = hdf5plugin.Zstd(
                                clevel=self.compression_opts
                            )
                        elif self.compression == "lz4":
                            create_kwargs["compression"] = hdf5plugin.LZ4(nbytes=0)
                        else:
                            raise ValueError(
                                f"Unsupported compression: {self.compression}"
                            )

                        # Apply chunking based on self.chunk_options
                        if self.chunk_options is True:  # auto-chunk
                            create_kwargs["chunks"] = True  # need to configure
                        elif (
                            isinstance(self.chunk_options, tuple)
                            and self.chunk_options is not None
                        ):  # Specific chunk shape
                            num_data_dims = len(data_item.shape)
                            final_chunks_list = []
                            for i in range(num_data_dims):
                                if i < len(self.chunk_options):
                                    final_chunks_list.append(self.chunk_options[i])
                                else:
                                    # If chunk_options is shorter, pad with full data dimension size
                                    final_chunks_list.append(data_item.shape[i])
                            logger.info(f"Final chunks list: {final_chunks_list}")
                            create_kwargs["chunks"] = tuple(final_chunks_list)
                        else:
                            logger.info(
                                f"Chunk options: using chunk size {data_item.shape}"
                            )
                            create_kwargs["chunks"] = (
                                data_item.shape
                            )  # use the dataset item shape as the chunk so it effectively does no chunking

                    # Create the dataset per item
                    logger.info(
                        f"Creating dataset for {item_name} with kwargs: {create_kwargs}"
                    )
                    h5file.create_dataset(item_name, data=data_item, **create_kwargs)

                # Store missing timesteps masks in a dedicated group
                if missing_timesteps_masks_data:
                    masks_group = h5file.create_group(
                        self.missing_timesteps_mask_group_name
                    )
                    for mod_name, mask_array in missing_timesteps_masks_data.items():
                        logger.info(
                            f"Writing missing timesteps mask for {mod_name} to h5 file path {h5_file_path}"
                        )
                        # Boolean masks typically don't benefit from compression/shuffle
                        masks_group.create_dataset(mod_name, data=mask_array)
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

        required_modalities_suffix = ""
        if self.required_modalities:
            required_modalities_suffix = "_required_" + "_".join(
                sorted([modality.name for modality in self.required_modalities])
            )
        h5py_dir = (
            self.tile_path
            / f"{self.h5py_folder}{self.compression_settings_suffix}{self.image_tile_size_suffix}"
            / (
                "_".join(
                    sorted([modality.name for modality in self.supported_modalities])
                )
                + required_modalities_suffix
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
        elif image.ndim == 3:
            modality_data = rearrange(image, "c h w -> h w c")
        elif image.ndim == 2:
            # It is already in the correct shape (t, c)
            modality_data = image
        else:
            raise ValueError(
                f"Unexpected image shape {image.shape} for modality {sample_modality.modality.name}"
            )
        return modality_data

    def _filter_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """Filter samples to adjust to the OlmoEarthSample format."""
        logger.info(f"Number of samples before filtering: {len(samples)}")

        # Remove bad modalities from samples
        # This ensures that the metadata is consistent with the actual data saved in h5
        processed_samples = self._process_samples(samples)
        filtered_samples = []
        for sample in processed_samples:
            if not all(
                modality in self.supported_modalities
                for modality in sample.modalities
                # TODO: clarify usage of ignore when parsing
                if not modality.ignore_when_parsing
            ):
                logger.info("Skipping sample because it has unsupported modalities")
                continue
            if any(
                modality not in sample.modalities
                for modality in self.required_modalities
            ):
                logger.info(
                    "Skipping sample because it does not have a required modality"
                )
                continue

            if sample.time_span != TimeSpan.YEAR:
                logger.debug(
                    "Skipping sample because it is not the yearly frequency data"
                )
                continue

            multi_temporal_timestamps_dict = sample.get_timestamps()
            spacetime_varying_modalities = {
                modality: timestamps
                for modality, timestamps in multi_temporal_timestamps_dict.items()
                if modality.is_spacetime_varying
            }

            if len(spacetime_varying_modalities) == 0:
                logger.info(
                    "Skipping sample because it has no spacetime varying modalities"
                )
                continue

            # To align with ERA5, which either missing (for ocean) or has 12 timesteps
            # We require at least one spacetime varying modality to have 12 timesteps
            # e.g., for Presto dataset, only 43 samples not meeting this requirement
            longest_timestamps_array = self._find_longest_timestamps_array(
                spacetime_varying_modalities
            )
            if len(longest_timestamps_array) < YEAR_NUM_TIMESTEPS:
                logger.info(
                    "Skipping sample because it does not have at least 12 timesteps"
                )
                continue

            filtered_samples.append(sample)

        logger.info("Distribution of samples after filtering:")
        self._log_modality_distribution(filtered_samples)
        return filtered_samples

    def get_and_filter_samples(self) -> list[SampleInformation]:
        """Get and filter samples.

        This parses csvs, loads images, and filters samples to adjust to the OlmoEarthSample format.
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
        tuples = []
        for sample in samples:
            for j in range(self.num_subtiles):
                tuples.append((j, sample))
        self.set_h5py_dir(len(tuples))
        self.save_compression_settings()  # Save settings before creating data
        self.save_sample_metadata(tuples)
        self.save_latlon_distribution(tuples)
        logger.info("Attempting to create H5 files may take some time...")
        self.create_h5_dataset(tuples)

    def run(self) -> None:
        """Run the conversion."""
        samples = self.get_and_filter_samples()
        self.prepare_h5_dataset(samples)
