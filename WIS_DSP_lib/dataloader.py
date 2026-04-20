import json
import random
from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from WIS_DSP_lib.constants import *

class Dataset:
    def __init__(self, h5py_dir: Path):
        self.h5py_dir = h5py_dir
        self.size = int(self.h5py_dir.name)
        self.norm_min_vals, self.norm_max_vals = self.load_norm_bounds(MODALITY)

    def load_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        with (self.h5py_dir / f"sample_{index}.h5").open("rb") as handle:
            with h5py.File(handle, "r") as h5file:
                timestamps = h5file["timestamps"][()]
                data = h5file[MODALITY][()]
                missing_timesteps_mask = h5file["missing_timesteps_masks"][MODALITY][()].astype(bool) if "missing_timesteps_masks" in h5file else None
        
        return timestamps, data, missing_timesteps_mask

    def normalize(self, sample: dict) -> dict:
        missing_mask = sample[MODALITY] == MISSING_VALUE
        normalized = (sample[MODALITY].astype(np.float64) - self.norm_min_vals) / (self.norm_max_vals - self.norm_min_vals)
        out_sample = sample.copy()
        out_sample[MODALITY] = np.where(missing_mask, sample[MODALITY], normalized).astype(np.float32)
        return out_sample
    
    def crop_timestamps_and_masks(self,
        timestamps: np.ndarray,
        missing_timesteps_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if missing_timesteps_mask is None:
            return timestamps[:MAX_SEQUENCE_LENGTH], None
        valid_timesteps = np.where(missing_timesteps_mask)[0]
        if len(valid_timesteps) == 0:
            raise ValueError("sentinel2_l2a is missing for every timestep")
        first_valid_timestep = int(valid_timesteps[0])
        last_valid_timestep = int(valid_timesteps[-1])
        return (
            timestamps[first_valid_timestep : last_valid_timestep + 1],
            missing_timesteps_mask[first_valid_timestep : last_valid_timestep + 1],
        )

    def fill_missing_timesteps(self, 
        data: np.ndarray,
        missing_timestep_mask: np.ndarray,
    ) -> np.ndarray:
        height, width, _, channels = data.shape
        full_timesteps_data = np.full(
            (height, width, MAX_SEQUENCE_LENGTH, channels),
            MISSING_VALUE,
            dtype=np.float32,
        )
        present_indices = np.where(missing_timestep_mask)[0]
        num_to_copy = min(len(present_indices), data.shape[2])
        if num_to_copy > 0:
            full_timesteps_data[:, :, present_indices[:num_to_copy], :] = data[:, :, :num_to_copy, :]
        return full_timesteps_data
    
    def pad_timestamps(self, timestamps: np.ndarray) -> tuple[np.ndarray, int]:
        if timestamps.shape[0] < MAX_SEQUENCE_LENGTH:
            timestamps = np.pad(
                timestamps,
                pad_width=((0, MAX_SEQUENCE_LENGTH - timestamps.shape[0]), (0, 0)),
                mode="edge",
            )
        return timestamps, timestamps.shape[0]

    def subset_sample(self,
        timestamps: np.ndarray,
        data: np.ndarray,
        current_length: int,
        missing_timesteps_mask: np.ndarray | None,
    ) -> dict:
        max_t = self.compute_max_t(current_length)

        valid_start_ts = self.get_valid_start_ts(missing_timesteps_mask, max_t, current_length)
        start_t = int(np.random.choice(valid_start_ts))

        sampled_hw = SAMPLED_HW_P * PATCH_SIZE
        start_h = int(np.random.choice(data.shape[0] - sampled_hw + 1))
        start_w = int(np.random.choice(data.shape[1] - sampled_hw + 1))

        return {
            "timestamps": timestamps[start_t : start_t + max_t],
            MODALITY: data[
                start_h : start_h + sampled_hw,
                start_w : start_w + sampled_hw,
                start_t : start_t + max_t,
                :,
            ],
        }

    def load_norm_bounds(self, modality: str) -> tuple[np.ndarray, np.ndarray]:
        with COMPUTED_NORM_CONFIG_PATH.open() as handle:
            norm_config = json.load(handle)[modality]
        means = np.array([norm_config[band]["mean"] for band in BAND_ORDER[modality]])
        stds = np.array([norm_config[band]["std"] for band in BAND_ORDER[modality]])
        return means - (2 * stds), means + (2 * stds)

    def get_valid_start_ts(self,
        missing_timesteps: np.ndarray | None, max_t: int, current_length: int
    ) -> list[int]:
        if current_length > max_t:
            if missing_timesteps is None:
                valid_start_ts = list(range(current_length - max_t + 1))
            else:
                valid_timesteps = np.flatnonzero(missing_timesteps)
                valid_timesteps = valid_timesteps[valid_timesteps + max_t <= current_length]
                valid_start_ts = sorted(set(valid_timesteps.tolist()))
        else:
            valid_start_ts = [0]
        if not valid_start_ts:
            raise ValueError(
                "No valid start timesteps found for sentinel2_l2a with "
                f"max_t={max_t} and current_length={current_length}"
            )
        return valid_start_ts

    def compute_max_t(self, current_length: int) -> int:
        tokens_per_timestep = (SAMPLED_HW_P**2) * len(BANDSETS[MODALITY])
        max_t = TOKEN_BUDGET // tokens_per_timestep
        if max_t < 1:
            raise ValueError("Patch size is too small for the configured token budget")
        return min(max_t, current_length)

    def __getitem__(self, index: int) -> dict:
        
        # load the sample data, timestamps, and missing timesteps mask
        timestamps, data, missing_timesteps_mask = self.load_sample(index)

        # crop timestamps and missing timestep mask to remove leading and trailing missing timesteps
        timestamps, missing_timesteps_mask = self.crop_timestamps_and_masks(timestamps, missing_timesteps_mask)
        
        # pad the timestamps to MAX_SEQUENCE_LENGTH and get the current length before padding
        timestamps, current_length = self.pad_timestamps(timestamps)

        # fill in the data for missing timesteps with the MISSING_VALUE, if there are any missing timesteps
        if missing_timesteps_mask is not None and (
            (not np.all(missing_timesteps_mask))
            or len(missing_timesteps_mask) < MAX_SEQUENCE_LENGTH
        ):
            data = self.fill_missing_timesteps(
                data,
                missing_timesteps_mask,
            )
        
        # subset the sample to a random valid spatiotemporal crop
        sample = self.subset_sample(
            timestamps=timestamps,
            data=data,
            current_length=current_length,
            missing_timesteps_mask=missing_timesteps_mask,
        )
        
        # normalize the sample using the pre-computed normalization bounds
        sample = self.normalize(sample)
        return sample

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int):
        self.global_indices = np.arange(dataset.size, dtype=np.uint32)
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        # Mirror the torch RNG draw performed when PyTorch creates a DataLoader iterator.
        torch.empty((), dtype=torch.int64).random_()
        batch: list[dict] = []
        instances_processed = 0
        for index in self.global_indices:
            batch.append(self.dataset[index])
            instances_processed += 1
            if len(batch) == self.batch_size:
                yield self.collate_double_masked_batched(batch)
                batch = []
    
    def to_stacked_tensor(self, batch, key):
        return torch.stack([torch.from_numpy(sample[key]) for  sample in batch], dim=0)
    
    def collate_double_masked_batched(self,
        batch: list[dict],
    ) -> tuple[dict, dict]:
        stacked_batch = {
            "timestamps": self.to_stacked_tensor(batch, "timestamps"),
            "sentinel2_l2a": self.to_stacked_tensor(batch, "sentinel2_l2a"),
        }
        return self.apply_mask(stacked_batch), self.apply_mask(stacked_batch)
    
    def apply_mask(self, batch: dict) -> dict:
        mask = self.create_random_mask(batch[MODALITY].shape, encode_frac=0.5, decode_frac=0.5)
        mask = self.fill_mask_with_missing_values(batch[MODALITY], mask)
        mask = self.apply_bandset_mask_rules(mask)
        return {
            "timestamps": batch["timestamps"],
            "sentinel2_l2a": batch["sentinel2_l2a"],
            "sentinel2_l2a_mask": mask,
        }
    
    def create_random_mask(self, shape: torch.Size, encode_frac: float, decode_frac: float) -> torch.Tensor:
        mask_shape = list(shape)
        mask_shape[-1] = len(BANDSETS[MODALITY])
        mask_shape[1] //= PATCH_SIZE
        mask_shape[2] //= PATCH_SIZE
        num_tokens = int(np.prod(mask_shape[1:]))

        flat_mask_tokens = self.get_encode_decode_target_token_mask(encode_frac, decode_frac, num_tokens)
        masks = [flat_mask_tokens[torch.randperm(num_tokens)] for _ in range(shape[0])]
        mask = torch.stack(masks).view(*mask_shape)
        mask = torch.repeat_interleave(mask, repeats=PATCH_SIZE, dim=1)
        mask = torch.repeat_interleave(mask, repeats=PATCH_SIZE, dim=2)
        return mask
    
    def get_encode_decode_target_token_mask(self, encode_frac: float, decode_frac: float, num_tokens: int) -> torch.Tensor:
        encode_tokens = int(num_tokens * encode_frac)
        decode_tokens = int(num_tokens * decode_frac)
        target_tokens = num_tokens - (encode_tokens + decode_tokens)
        mask_tokens = torch.cat(
            [
                torch.full((encode_tokens,), MaskValue.ONLINE_ENCODER),
                torch.full((decode_tokens,), MaskValue.DECODER),
                torch.full((target_tokens,), MaskValue.TARGET_ENCODER_ONLY),
            ]
        )
        return mask_tokens
    
    def fill_mask_with_missing_values(self,
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        missing_mask = torch.zeros_like(mask, dtype=torch.bool)
        for bandset_idx, band_indices in enumerate(BANDSETS[MODALITY]):
            bandset = data[..., band_indices]
            missing_mask[..., bandset_idx] = (bandset == MISSING_VALUE).any(dim=-1)
        if missing_mask.any():
            mask = mask.clone()
            mask[missing_mask] = MaskValue.MISSING
        return mask
    
    def apply_bandset_mask_rules(self, mask: torch.Tensor) -> torch.Tensor:
        out_mask = mask.clone()
        present_bandsets = self.get_present_bandsets(mask)
        encoded_decoded_bandsets = self.select_encoded_decoded_bandsets(present_bandsets)
        for sample_idx in range(mask.shape[0]):
            encoded_bandsets, decoded_bandsets = encoded_decoded_bandsets[sample_idx]
            if not present_bandsets[sample_idx]:
                continue
            for bandset_idx in range(mask.shape[-1]):
                is_encoded = (MODALITY, bandset_idx) in encoded_bandsets
                is_decoded = (MODALITY, bandset_idx) in decoded_bandsets
                if not is_encoded:
                    online_encoder_mask = (
                        mask[sample_idx, ..., bandset_idx] == MaskValue.ONLINE_ENCODER
                    )
                    out_mask[sample_idx, ..., bandset_idx] = torch.where(
                        online_encoder_mask,
                        torch.tensor(MaskValue.TARGET_ENCODER_ONLY, dtype=out_mask.dtype),
                        out_mask[sample_idx, ..., bandset_idx],
                    )
                    continue
                if not is_decoded:
                    decoder_mask = mask[sample_idx, ..., bandset_idx] == MaskValue.DECODER
                    out_mask[sample_idx, ..., bandset_idx] = torch.where(
                        decoder_mask,
                        torch.tensor(MaskValue.TARGET_ENCODER_ONLY, dtype=out_mask.dtype),
                        out_mask[sample_idx, ..., bandset_idx],
                    )

        flat_mask = torch.flatten(out_mask, start_dim=1)
        num_encoded = (flat_mask == MaskValue.ONLINE_ENCODER).sum(dim=-1)
        num_decoded = (flat_mask == MaskValue.DECODER).sum(dim=-1)
        for index in torch.argwhere(num_encoded == 0).flatten().tolist():
            out_mask[index : index + 1] = self.random_fill_unmasked(
                out_mask[index : index + 1],
                encode_frac=0.5,
                decode_frac=0.0,
            )
        for index in torch.argwhere(num_decoded == 0).flatten().tolist():
            out_mask[index : index + 1] = self.random_fill_unmasked(
                out_mask[index : index + 1],
                encode_frac=0.5,
                decode_frac=0.0,    
            )
        return out_mask
    
    def get_present_bandsets(self,mask: torch.Tensor) -> list[list[tuple[str, int]]]:
        present_bandsets = [[] for _ in range(mask.shape[0])]
        for sample_idx in range(mask.shape[0]):
            for bandset_idx in range(mask.shape[-1]):
                has_any_encoded_tokens = torch.sum(mask[sample_idx, ..., bandset_idx] == MaskValue.ONLINE_ENCODER) > 0
                if has_any_encoded_tokens:
                    present_bandsets[sample_idx].append((MODALITY, bandset_idx))
        return present_bandsets

    def select_encoded_decoded_bandsets(self, 
        present_bandsets: list[list[tuple[str, int]]],
        ) -> list[tuple[set[tuple[str, int]], set[tuple[str, int]]]]:
        selected = []
        for bandsets_for_sample in present_bandsets:
            if len(bandsets_for_sample) == 1:
                selected.append((set(bandsets_for_sample), set()))
                continue
            if len(bandsets_for_sample) == 2:
                selected.append(
                    (
                        {bandsets_for_sample[0]},
                        {bandsets_for_sample[1]},
                    )
                )
                continue
            num_bandsets_to_encode = int(np.random.randint(2, len(bandsets_for_sample) + 1))
            encoded_indices = np.random.choice(
                len(bandsets_for_sample),
                size=num_bandsets_to_encode,
                replace=False,
            )
            encoded_bandsets = {bandsets_for_sample[i] for i in encoded_indices}
            num_bandsets_to_decode = int(np.random.randint(1, len(bandsets_for_sample) + 1))
            decoded_indices = np.random.choice(
                len(bandsets_for_sample),
                size=num_bandsets_to_decode,
                replace=False,
            )
            decoded_bandsets = {bandsets_for_sample[i] for i in decoded_indices}
            selected.append((encoded_bandsets, decoded_bandsets))
        return selected
    
    def random_fill_unmasked(self, mask: torch.Tensor, encode_frac: float = 0.5, decode_frac: float = 0.5) -> torch.Tensor:
        if mask.shape[0] != 1:
            raise ValueError("random_fill_unmasked expects a single-sample mask")
        compact_mask = mask[:, 0::PATCH_SIZE, 0::PATCH_SIZE]
        original_shape = compact_mask.shape
        flat_mask = compact_mask.flatten()
        not_missing = flat_mask != MaskValue.MISSING
        num_not_missing_tokens = int(not_missing.sum().item())
        if num_not_missing_tokens == 0:
            return mask
        if num_not_missing_tokens == 1:
            replacement = torch.cat([torch.full((1,), MaskValue.ONLINE_ENCODER)])
        else:
            replacement = self.get_encode_decode_target_token_mask(encode_frac, decode_frac, num_not_missing_tokens)
            replacement = replacement[torch.randperm(num_not_missing_tokens)]
        filled = flat_mask.clone()
        filled[not_missing] = replacement
        filled = filled.view(original_shape)
        filled = torch.repeat_interleave(filled, repeats=PATCH_SIZE, dim=1)
        filled = torch.repeat_interleave(filled, repeats=PATCH_SIZE, dim=2)
        return filled