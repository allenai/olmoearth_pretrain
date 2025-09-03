"""This script is used to compare the performance of different compression strategies for H5 files.

It is used to find the best compression strategy for a given dataset.
"""

import os
import tempfile
import time

import h5py
import numpy as np

# Attempt to import hdf5plugin to make its filters available to h5py
try:
    import hdf5plugin

    print("hdf5plugin imported successfully.")
except ImportError:
    print(
        "Warning: hdf5plugin not found. Additional compression filters may not be available."
    )
    hdf5plugin = None


def load_all_datasets_from_h5(source_h5_path):
    """Loads all datasets from an existing H5 file into a dictionary."""
    print(f"Loading all datasets from: {source_h5_path}")
    data_dict = {}
    try:
        with h5py.File(source_h5_path, "r") as hf_source:

            def visit_object(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data_dict[name] = obj[()]

            hf_source.visititems(visit_object)
        if not data_dict:
            print(f"Warning: No datasets found in {source_h5_path}")
        else:
            print(f"Successfully loaded {len(data_dict)} dataset(s).")
        return data_dict
    except Exception as e:
        print(f"Error loading data from {source_h5_path}: {e}")
        raise


def create_h5_file_with_multiple_datasets(
    file_path,
    datasets_dict,
    strategy_chunks_config,
    strategy_compression,
    strategy_compression_opts,
):
    """Creates an H5 file, writing multiple datasets, each with the same specified strategy.

    Args:
        file_path (str): Path to the H5 file to create.
        datasets_dict (dict): Dictionary of {'dataset_name': np.ndarray}.
        strategy_chunks_config (str, tuple, or bool): General guidance for chunks for the strategy.
                                            Can be None (contiguous), 'dataset_shape' (single chunk per dataset),
                                            True (auto-chunking by h5py).
        strategy_compression (str, optional): Compression filter.
        strategy_compression_opts (int, optional): Compression level.
    """
    with h5py.File(file_path, "w") as hf:
        for ds_name, ds_data in datasets_dict.items():
            current_chunks_config = None
            if (
                strategy_chunks_config == "dataset_shape"
            ):  # Indicates single chunk per dataset
                current_chunks_config = ds_data.shape
            elif strategy_chunks_config is True:  # Auto-chunking
                current_chunks_config = True
            elif isinstance(strategy_chunks_config, tuple):  # A fixed chunk tuple
                current_chunks_config = strategy_chunks_config
            # If strategy_chunks_config is None, current_chunks_config remains None (contiguous if no compression)

            # HDF5 requires chunking if compression is enabled.
            # If strategy_chunks_config is None AND compression is on, h5py will auto-chunk.
            # To be explicit for our "single chunk per dataset" when compressed, we set it.
            # If strategy_chunks_config is True, h5py handles chunking.
            if (
                strategy_compression is not None
                and current_chunks_config is None
                and not strategy_chunks_config
            ):
                # This condition ensures that if we have compression and didn't explicitly ask for auto-chunking (True)
                # or specific chunking ('dataset_shape' or a tuple), we default to dataset_shape to be clear.
                # However, the strategies list should ideally provide explicit chunking for compressed datasets.
                # For 'No Compression, Contiguous', chunks_config is None and strategy_compression is None.
                # For strategies like 'Gzip L4, Auto-Chunking', current_chunks_config will be True.
                current_chunks_config = ds_data.shape

            hf.create_dataset(
                ds_name,
                data=ds_data,
                chunks=current_chunks_config,
                compression=strategy_compression,
                compression_opts=strategy_compression_opts,
            )


def time_h5_load_all_datasets(file_path):
    """Loads all datasets from an H5 file into a dictionary and times the operation.

    Returns:
        float: Time taken to load all datasets.
        dict: Dictionary of loaded datasets.
    """
    loaded_data_dict = {}
    with h5py.File(file_path, "r") as hf:
        start_time = time.perf_counter()

        def visit_object(name, obj):
            if isinstance(obj, h5py.Dataset):
                loaded_data_dict[name] = obj[()]

        hf.visititems(visit_object)
        end_time = time.perf_counter()
        duration = end_time - start_time
    return duration, loaded_data_dict


def verify_data_dictionaries(original_dict, loaded_dict):
    """Verifies if two dictionaries of datasets are equivalent."""
    if original_dict.keys() != loaded_dict.keys():
        print("  Data verification failed: Dataset names mismatch.")
        print(f"    Original keys: {set(original_dict.keys())}")
        print(f"    Loaded keys: {set(loaded_dict.keys())}")
        return False
    for key in original_dict:
        if not np.array_equal(original_dict[key], loaded_dict[key]):
            print(f"  Data verification failed: Content mismatch for dataset '{key}'.")
            print(
                f"    Original shape: {original_dict[key].shape}, Loaded shape: {loaded_dict[key].shape}"
            )
            return False
    print(
        "  Data verification successful (all dataset names, shapes, and content match)."
    )
    return True


def main():
    """Main function to run the H5 compression strategy comparison."""
    print(
        "HDF5 Compression Strategy I/O Performance Comparison (Multiple Source Files - Averaged Results)\n"
    )

    # --- User Configuration: Specify your input H5 files ---
    # !!! USER: Update this list with paths to your actual H5 files !!!
    # Example with two files, extend this list as needed (e.g., for 10 files)
    base_path = "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_gzip_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/285288/"
    input_h5_file_paths = [
        base_path + "sample_174448.h5",
        base_path + "sample_70033.h5",
        base_path + "sample_171662.h5",
        base_path + "sample_66172.h5",
        base_path + "sample_84226.h5",
        base_path + "sample_192056.h5",
        base_path + "sample_1673.h5",
        base_path + "sample_7009.h5",
        base_path + "sample_63905.h5",
        base_path + "sample_247681.h5",
        base_path + "sample_83807.h5",
        base_path + "sample_84198.h5",
        base_path + "sample_191705.h5",
        base_path + "sample_257579.h5",
        base_path + "sample_13183.h5",
        base_path + "sample_225436.h5",
        base_path + "sample_136767.h5",
        base_path + "sample_247339.h5",
        base_path + "sample_282122.h5",
        base_path + "sample_89125.h5",
        base_path + "sample_37069.h5",
        base_path + "sample_1858.h5",
        base_path + "sample_114950.h5",
        base_path + "sample_55340.h5",
        base_path + "sample_116604.h5",
        base_path + "sample_147073.h5",
        base_path + "sample_119293.h5",
        base_path + "sample_51114.h5",
        base_path + "sample_118622.h5",
        base_path + "sample_89117.h5",
    ]
    if not input_h5_file_paths or not input_h5_file_paths[0]:  # Basic check
        print(
            "Error: No input H5 file paths provided. Please update 'input_h5_file_paths' in the script."
        )
        return
    # --- End User Configuration ---

    aggregated_results_data = {}  # To store data across all files for averaging

    # --- Compression Strategies to Test ---
    # Format: (description_str, chunks_guidance, compression_kwargs_dict)
    # chunks_guidance: None (contiguous), 'dataset_shape' (single chunk per DS), True (auto-chunking)
    compression_strategies = [
        ("No Compression, Contiguous", None, {}),
        ("No Compression, Single Chunk pDS", "dataset_shape", {}),
        (
            "Gzip L1, Single Chunk pDS",
            "dataset_shape",
            {"compression": "gzip", "compression_opts": 1},
        ),
        (
            "Gzip L2, Single Chunk pDS",
            "dataset_shape",
            {"compression": "gzip", "compression_opts": 2},
        ),
        (
            "Gzip L3, Single Chunk pDS",
            "dataset_shape",
            {"compression": "gzip", "compression_opts": 3},
        ),
        (
            "Gzip L4, Single Chunk pDS",
            "dataset_shape",
            {"compression": "gzip", "compression_opts": 4},
        ),
        (
            "Gzip L9, Single Chunk pDS",
            "dataset_shape",
            {"compression": "gzip", "compression_opts": 9},
        ),
        (
            "Gzip L3, Auto-Chunking",
            True,
            {"compression": "gzip", "compression_opts": 3},
        ),
        (
            "Gzip L4, Auto-Chunking",
            True,
            {"compression": "gzip", "compression_opts": 4},
        ),
        ("LZF, Single Chunk pDS", "dataset_shape", {"compression": "lzf"}),
        ("LZF, Auto-Chunking", True, {"compression": "lzf"}),
    ]

    if hdf5plugin:
        plugin_strategies = [
            (
                "Zstd L1, Single Chunk pDS",
                "dataset_shape",
                {"compression": hdf5plugin.Zstd(clevel=1)},
            ),
            (
                "Zstd L1, Auto-Chunking",
                True,
                {"compression": hdf5plugin.Zstd(clevel=1)},
            ),
            (
                "Zstd L2, Single Chunk pDS",
                "dataset_shape",
                {"compression": hdf5plugin.Zstd(clevel=2)},
            ),
            (
                "Zstd L2, Auto-Chunking",
                True,
                {"compression": hdf5plugin.Zstd(clevel=2)},
            ),
            (
                "Zstd L3, Single Chunk pDS",
                "dataset_shape",
                {"compression": hdf5plugin.Zstd(clevel=3)},
            ),
            (
                "Zstd L9, Single Chunk pDS",
                "dataset_shape",
                {"compression": hdf5plugin.Zstd(clevel=9)},
            ),
            (
                "Zstd L3, Auto-Chunking",
                True,
                {"compression": hdf5plugin.Zstd(clevel=3)},
            ),
            (
                "Blosc L5 Sh(B), Single Chunk pDS",
                "dataset_shape",
                {
                    "compression": hdf5plugin.Blosc(
                        clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                    )
                },
            ),
            (
                "Blosc L5 Sh(B), Auto-Chunking",
                True,
                {
                    "compression": hdf5plugin.Blosc(
                        clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                    )
                },
            ),
            (
                "Blosc2 L5 Sh(B), Single Chunk pDS",
                "dataset_shape",
                {
                    "compression": hdf5plugin.Blosc2(
                        clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE
                    )
                },
            ),
            (
                "Blosc2 L5 Sh(B), Auto-Chunking",
                True,
                {
                    "compression": hdf5plugin.Blosc2(
                        clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE
                    )
                },
            ),
            (
                "BZip2 L5, Single Chunk pDS",
                "dataset_shape",
                {"compression": hdf5plugin.BZip2(blocksize=5)},
            ),
            (
                "BZip2 L5, Auto-Chunking",
                True,
                {"compression": hdf5plugin.BZip2(blocksize=5)},
            ),
            (
                "LZ4 L0, Single Chunk pDS",
                "dataset_shape",
                {"compression": hdf5plugin.LZ4()},
            ),
            ("LZ4 L0, Auto-Chunking", True, {"compression": hdf5plugin.LZ4()}),
            (
                "Bitshuffle(St), Single Chunk pDS",
                "dataset_shape",
                {"compression": hdf5plugin.Bitshuffle()},
            ),
            (
                "Bitshuffle(St), Auto-Chunking",
                True,
                {"compression": hdf5plugin.Bitshuffle()},
            ),
            (
                "Bitshuffle+LZ4, Auto-Chunk",
                True,
                {"compression": hdf5plugin.Bitshuffle(lz4=True)},
            ),
            (
                "Bitshuffle+Zstd, Auto-Chunk",
                True,
                {"compression": hdf5plugin.Bitshuffle(cname="zstd", clevel=3)},
            ),
        ]
        compression_strategies.extend(plugin_strategies)
    else:
        print("\nSkipping hdf5plugin specific filters as the plugin is not available.")

    for file_idx, input_h5_file_path in enumerate(input_h5_file_paths):
        print(
            f"\n--- Processing Source File {file_idx + 1}/{len(input_h5_file_paths)}: {input_h5_file_path} ---"
        )
        try:
            source_datasets_dict = load_all_datasets_from_h5(input_h5_file_path)
            if not source_datasets_dict:
                print(
                    f"Warning: No data loaded from source H5 file {input_h5_file_path}. Skipping this file."
                )
                continue
        except Exception as e:
            print(
                f"Error loading source data from {input_h5_file_path}: {e}. Skipping this file."
            )
            continue

        total_source_data_size_mb = sum(
            arr.nbytes for arr in source_datasets_dict.values()
        ) / (1024 * 1024)
        print(
            f"Using data from {len(source_datasets_dict)} dataset(s) in '{input_h5_file_path}'."
        )
        print(
            f"Total size of loaded source data for this file: {total_source_data_size_mb:.2f} MB\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            print(
                f"Using temporary directory for test H5 files for {os.path.basename(input_h5_file_path)}: {tmpdir}\n"
            )

            for (
                description,
                chunks_guidance,
                compression_kwargs,
            ) in compression_strategies:
                print(
                    f"--- Testing Strategy: {description} (for {os.path.basename(input_h5_file_path)}) ---"
                )
                safe_desc = (
                    description.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "")
                    .replace("-", "")
                    .replace(".", "_")
                )
                file_name = (
                    f"test_{os.path.basename(input_h5_file_path)}_{safe_desc}.h5"
                )
                temp_h5_file_path = os.path.join(tmpdir, file_name)

                try:
                    create_h5_file_with_multiple_datasets(
                        temp_h5_file_path,
                        source_datasets_dict,
                        strategy_chunks_config=chunks_guidance,
                        strategy_compression=compression_kwargs.get("compression"),
                        strategy_compression_opts=compression_kwargs.get(
                            "compression_opts"
                        ),
                    )
                    file_size_mb = os.path.getsize(temp_h5_file_path) / (1024 * 1024)
                    print(
                        f"  Created temp H5 file: {temp_h5_file_path}, Size: {file_size_mb:.2f} MB"
                    )
                except Exception as e:
                    print(
                        f"  Error creating temp H5 file {temp_h5_file_path} with strategy '{description}': {e}"
                    )
                    print(
                        f"    Filter: {compression_kwargs.get('compression')}, Opts: {compression_kwargs.get('compression_opts')}, Chunks: {chunks_guidance}"
                    )
                    continue  # Skip to next strategy for this file

                try:
                    load_time, loaded_dict_from_temp = time_h5_load_all_datasets(
                        temp_h5_file_path
                    )
                    print(
                        f"  Time to load all datasets from temp file: {load_time:.4f}s"
                    )

                    is_lossy = compression_kwargs.get("compression") in [
                        "sperr",
                        "sz",
                        "sz3",
                        "zfp",
                    ]  # Assuming these are identifier for lossy
                    if hasattr(
                        compression_kwargs.get("compression"), "filter_name"
                    ):  # For hdf5plugin objects
                        is_lossy = compression_kwargs.get(
                            "compression"
                        ).filter_name in ["sperr", "sz", "sz3", "zfp"]

                    if is_lossy:
                        print(
                            "  Skipping exact content verification for known lossy filter."
                        )
                        if source_datasets_dict.keys() != loaded_dict_from_temp.keys():
                            print(
                                "  Data verification failed: Dataset names mismatch for lossy."
                            )
                        else:
                            match = True
                            for key in source_datasets_dict:
                                if (
                                    source_datasets_dict[key].shape
                                    != loaded_dict_from_temp[key].shape
                                ):
                                    print(
                                        f"  Data verification failed: Shape mismatch for lossy dataset '{key}'."
                                    )
                                    match = False
                                    break
                            if match:
                                print(
                                    "  Data verification basic (names, shapes) successful for lossy filter."
                                )
                    else:
                        verify_data_dictionaries(
                            source_datasets_dict, loaded_dict_from_temp
                        )

                    # Aggregate results
                    if description not in aggregated_results_data:
                        aggregated_results_data[description] = {
                            "load_times": [],
                            "file_sizes_mb": [],
                            "successful_runs": 0,
                            "compression": compression_kwargs.get("compression"),
                            "comp_opts": str(
                                compression_kwargs.get("compression_opts")
                            ),
                            "chunks_guidance": str(chunks_guidance),
                        }

                    aggregated_results_data[description]["load_times"].append(load_time)
                    aggregated_results_data[description]["file_sizes_mb"].append(
                        file_size_mb
                    )
                    aggregated_results_data[description]["successful_runs"] += 1

                except Exception as e:
                    print(
                        f"  Error loading/verifying data from temp H5 file {temp_h5_file_path}: {e}"
                    )
                print("-" * 30)

    print("\n--- Summary of Averaged Results (Across All Processed Files) ---")

    final_summary_results = []
    if aggregated_results_data:
        for desc, data in aggregated_results_data.items():
            if data["successful_runs"] > 0:
                avg_load_time = np.mean(data["load_times"])
                avg_file_size_mb = np.mean(data["file_sizes_mb"])
                final_summary_results.append(
                    {
                        "description": desc,
                        "load_time": avg_load_time,
                        "file_size_mb": avg_file_size_mb,
                        "compression": data["compression"],
                        "comp_opts": data["comp_opts"],
                        "chunks_guidance": data["chunks_guidance"],
                        "num_files": data["successful_runs"],
                    }
                )
            else:
                print(f"Strategy '{desc}' had no successful runs across files.")

    if final_summary_results:
        # Sort by Average Load Time
        results_sorted_by_load_time = sorted(
            final_summary_results, key=lambda x: x["load_time"]
        )
        print("\n--- Sorted by Average Load Time ---")
        header = f"{'Strategy Description':<50} | {'Avg Load (s)':<15} | {'Avg Size (MB)':<15} | {'Files Agg':<10} | {'Compression':<30} | {'Opts':<25} | {'Chunks Guidance':<20}"
        print(header)
        print("-" * len(header))
        for res in results_sorted_by_load_time:
            compression_str = str(res.get("compression", "None"))
            if hasattr(
                res.get("compression"), "filter_name"
            ):  # Check for hdf5plugin objects
                compression_str = res.get("compression").filter_name

            comp_opts_str = str(res.get("comp_opts", "None"))
            chunks_guidance_str = str(res.get("chunks_guidance", "None"))
            num_files_str = str(res.get("num_files", 0))

            display_opts_str = comp_opts_str
            if len(display_opts_str) > 23:
                display_opts_str = display_opts_str[:20] + "..."
            display_compression_str = compression_str
            if len(display_compression_str) > 28:
                display_compression_str = display_compression_str[:25] + "..."

            print(
                f"{res['description']:<50} | {res['load_time']:<15.4f} | {res['file_size_mb']:<15.2f} | {num_files_str:<10} | {display_compression_str:<30} | {display_opts_str:<25} | {chunks_guidance_str:<20}"
            )

        # Sort by Average File Size
        results_sorted_by_file_size = sorted(
            final_summary_results, key=lambda x: x["file_size_mb"]
        )
        print("\n--- Sorted by Average File Size (MB) ---")
        print(header)  # Re-print header
        print("-" * len(header))
        for res in results_sorted_by_file_size:
            compression_str = str(res.get("compression", "None"))
            if hasattr(
                res.get("compression"), "filter_name"
            ):  # Check for hdf5plugin objects
                compression_str = res.get("compression").filter_name

            comp_opts_str = str(res.get("comp_opts", "None"))
            chunks_guidance_str = str(res.get("chunks_guidance", "None"))
            num_files_str = str(res.get("num_files", 0))

            display_opts_str = comp_opts_str
            if len(display_opts_str) > 23:
                display_opts_str = display_opts_str[:20] + "..."
            display_compression_str = compression_str
            if len(display_compression_str) > 28:
                display_compression_str = display_compression_str[:25] + "..."

            print(
                f"{res['description']:<50} | {res['file_size_mb']:<15.2f} | {res['load_time']:<15.4f} | {num_files_str:<10} | {display_compression_str:<30} | {display_opts_str:<25} | {chunks_guidance_str:<20}"
            )
    else:
        print(
            "No results to display. Check for errors during file processing or if no strategies had successful runs."
        )

    print("\nScript finished.")


if __name__ == "__main__":
    main()
