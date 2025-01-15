"""Collator functions for the dataset."""

from typing import NamedTuple

import torch
from torch.nn import functional as F

from helios.data.dataset import (
    DATA_SOURCE_TO_VARIATION_TYPE,
    ArrayWithMetadata,
    DatasetOutput,
)
from helios.data.masking import apply_random_masking


class CollateFnOutput(NamedTuple):
    """Output of the collate function.

    Based on presto-v3
    """

    space_time_x: torch.Tensor  # Spatiotime features
    space_x: torch.Tensor  # space-only features
    time_x: torch.Tensor  # time-only features
    static_x: torch.Tensor  # Static features
    space_time_mask: torch.Tensor  # Mask for spatiotime features
    space_mask: torch.Tensor  # Mask for space features
    time_mask: torch.Tensor  # Mask for time features
    static_mask: torch.Tensor  # Mask for static features
    time_info: torch.Tensor  # Time information associated with each index
    patch_size: float  # Size of patches
    collate_info: dict | None  # Additional collation metadata


def variable_time_collate_fn(
    items: list[DatasetOutput],
    pad_token_value: float = 0.0,
    patch_size: int = 4,
    encode_ratio: float = 0.5,
    decode_ratio: float = 0.5,
) -> CollateFnOutput:
    """Collate function for inputs with variable time data."""
    assert items
    max_len = max(item.sample_metadata["num_timesteps"] for item in items)
    print(f"Max len: {max_len}")
    all_space_time_x: list[torch.Tensor] = []
    all_space_x: list[torch.Tensor] = []
    all_time_x: list[torch.Tensor] = []
    all_static_x: list[torch.Tensor] = []
    for item in items:
        data_sources = item.get_data_sources()
        for data_source in data_sources:
            variation_type = DATA_SOURCE_TO_VARIATION_TYPE[data_source]
            # Note: layers are not necessarily sorted in time order
            if variation_type == "space_time_varying":
                array_with_metadata: ArrayWithMetadata = getattr(item, data_source)
                num_timesteps = item.sample_metadata["num_timesteps"]
                print(f"Num timesteps: {num_timesteps}")
                space_time_x = torch.tensor(array_with_metadata.array)
                # h, w,t,c
                pad_shape = (0, 0, 0, int(max_len - num_timesteps), 0, 0)
                all_space_time_x.append(
                    F.pad(
                        space_time_x,
                        pad_shape,
                        value=pad_token_value,
                    )
                )
            elif variation_type == "time_varying":
                raise NotImplementedError("Time varying data not implemented")
            elif variation_type == "space_varying":
                raise NotImplementedError("Space varying data not implemented")
            elif variation_type == "static":
                raise NotImplementedError("Static data not implemented")
            else:
                raise ValueError(
                    f"Unknown data source variation type: {variation_type}"
                )
    space_time_x = (
        torch.stack(all_space_time_x) if all_space_time_x else torch.tensor([])
    )
    space_x = torch.stack(all_space_x) if all_space_x else torch.tensor([])
    time_x = torch.stack(all_time_x) if all_time_x else torch.tensor([])
    static_x = torch.stack(all_static_x) if all_static_x else torch.tensor([])
    # patching and masking can occur here for now
    # We need to figure out how mmany patches per dimension for the space stuff
    # Then figure out what the new image sized will be which is patch size times the space patches per dim
    b, h, w, t, c = space_time_x.shape
    assert h == w, "space dimensions must be equal"
    # Unneeded until we add subset code and the like
    # space_dim_size = h
    # space_patches_per_dim = int(space_dim_size / patch_size)
    # patched_image_size = patch_size * space_patches_per_dim

    # TODO: COllecting the time info
    time_info = torch.tensor([])
    # Random Masking only

    masked_output = apply_random_masking(
        space_time_x=space_time_x,
        space_x=space_x,
        time_x=time_x,
        static_x=static_x,
        time_info=time_info,
        patch_size=patch_size,
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    )

    collated_output = CollateFnOutput(
        space_time_x=masked_output.space_time_x,
        space_x=masked_output.space_x,
        time_x=masked_output.time_x,
        static_x=masked_output.static_x,
        space_time_mask=masked_output.space_time_mask,
        space_mask=masked_output.space_mask,
        time_mask=masked_output.time_mask,
        static_mask=masked_output.static_mask,
        time_info=masked_output.time_info,
        patch_size=patch_size,
        collate_info=None,
    )

    return collated_output


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from helios.data.dataset import HeliosDataset

    data_index_path = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    dataset = HeliosDataset(data_index_path)
    print(f"Dataset length: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=variable_time_collate_fn
    )
    for batch in dataloader:
        batch.space_time_x.shape
    # import time

    # time_to_load_sample = []
    # items = []
    # np.random.seed(42)
    # batch_size = 4
    # for i in np.random.randint(0, len(dataset), size=batch_size):
    #     start_time = time.time()
    #     items.append(dataset[i])
    #     end_time = time.time()
    #     time_taken = end_time - start_time
    #     print(f"Time taken: {time_taken} seconds")
    #     time_to_load_sample.append(time_taken)
    # collated_output = variable_time_collate_fn(items)

    # # print the shapes and sums of all tnesors in collated output in a clear and readable format
    # for key, value in collated_output._asdict().items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: {value.shape} {value.sum()}")
    #     else:
    #         print(f"{key}: {value}")

    # print(
    #     f"Time taken: {np.mean(time_to_load_sample)} seconds "
    #     f"and {np.std(time_to_load_sample)} seconds"
    # )
    # # Do we want to sample monthly data and freq data in the same batch?
