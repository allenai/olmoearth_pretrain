"""Extract encoder embeddings from a pretrained checkpoint and save to disk.

Uses seasonal sampling: instead of all 12 monthly timesteps, selects one
representative timestep per season (4 total). For each 3-month window the
first timestep with complete multi-temporal data is chosen, falling back to
the window start when no month has full coverage.
"""

import logging

import numpy as np
from script import (
    build_common_components as _build_common_components,
)
from script import (
    build_dataset_config,
)

import olmoearth_pretrain.internal.experiment as _experiment_module
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

ENCODE_ONLY_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components restricted to encode-only modalities."""
    common = _build_common_components(script, cmd, run_name, cluster, overrides)
    common.training_modalities = ENCODE_ONLY_MODALITIES
    return common


SEASON_WINDOWS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]


def _select_seasonal_indices(
    sample: OlmoEarthSample,
) -> tuple[list[int], list[bool]]:
    """Pick one timestep per season, preferring the earliest with full coverage.

    For each 3-month window, iterates through month indices that fall within
    the sample's actual number of timesteps and checks that every multitemporal
    modality present has real (non-MISSING) data at that timestep.

    Always returns exactly 4 entries (one per season). Seasons that fall
    entirely outside the available timesteps use a placeholder index (0) and
    are flagged as invalid so the caller can fill them with MISSING_VALUE.

    Returns:
        (indices, valid): both length-4 lists. ``valid[i]`` is False when
        season *i* had no in-range timesteps.
    """
    T = len(sample.timestamps) if sample.timestamps is not None else 12

    selected: list[int] = []
    valid: list[bool] = []
    for window in SEASON_WINDOWS:
        valid_months = [m for m in window if m < T]
        if not valid_months:
            selected.append(0)
            valid.append(False)
            continue
        chosen = valid_months[0]
        for month_idx in valid_months:
            all_available = True
            for field_name in sample._fields:
                if field_name in ("timestamps", "latlon"):
                    continue
                data = getattr(sample, field_name)
                if data is None:
                    continue
                mod_spec = Modality.get(field_name)
                if not mod_spec.is_multitemporal:
                    continue
                if mod_spec.is_spatial:
                    timestep_missing = np.all(data[:, :, month_idx, :] == MISSING_VALUE)
                else:
                    timestep_missing = np.all(data[month_idx] == MISSING_VALUE)
                if timestep_missing:
                    all_available = False
                    break
            if all_available:
                chosen = month_idx
                break
        selected.append(chosen)
        valid.append(True)
    return selected, valid


def _slice_sample_seasonal(
    sample: OlmoEarthSample, indices: list[int], valid: list[bool]
) -> OlmoEarthSample:
    """Reduce a sample to exactly 4 seasonal timesteps.

    Timesteps corresponding to ``valid[i] == False`` are filled with
    MISSING_VALUE so that every sample has a uniform time dimension of 4.
    """
    invalid_positions = [i for i, v in enumerate(valid) if not v]
    new_dict: dict[str, np.ndarray | None] = {}
    for field_name in sample._fields:
        data = getattr(sample, field_name)
        if data is None:
            new_dict[field_name] = None
        elif field_name == "timestamps":
            sliced = data[indices].copy()
            for i in invalid_positions:
                sliced[i] = 0
            new_dict[field_name] = sliced
        elif field_name == "latlon":
            new_dict[field_name] = data
        else:
            mod_spec = Modality.get(field_name)
            if mod_spec.is_multitemporal:
                if mod_spec.is_spatial:
                    sliced = data[:, :, indices, :].copy()
                    for i in invalid_positions:
                        sliced[:, :, i, :] = MISSING_VALUE
                else:
                    sliced = data[indices].copy()
                    for i in invalid_positions:
                        sliced[i] = MISSING_VALUE
                new_dict[field_name] = sliced
            else:
                new_dict[field_name] = data
    return OlmoEarthSample(**new_dict)


# ---------------------------------------------------------------------------
# Monkey-patch _IndexedDataset so the existing extract() function in
# experiment.py transparently gets seasonal-filtered samples.
# ---------------------------------------------------------------------------
_OriginalIndexedDataset = _experiment_module._IndexedDataset


class _SeasonalIndexedDataset(_OriginalIndexedDataset):
    """Extends _IndexedDataset with per-season timestep selection."""

    def __getitem__(self, idx: int) -> tuple:
        idx, patch_size, sample = super().__getitem__(idx)
        indices, valid = _select_seasonal_indices(sample)
        sample = _slice_sample_seasonal(sample, indices, valid)
        return idx, patch_size, sample


_experiment_module._IndexedDataset = _SeasonalIndexedDataset


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config -- must match the checkpoint's architecture."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_linear_patch_embed=False,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_extract_config(common: CommonComponents) -> dict:
    """Build the extraction-specific config fields."""
    return {
        "checkpoint_path": "/weka/dfive-default/olmoearth_pretrain/checkpoints/PLACEHOLDER/stepXXXXXX",
        "output_dir": f"{common.save_folder}/embeddings",
        "patch_size": MAX_PATCH_SIZE,
        "batch_size": 64,
        "num_workers": 8,
        "sampled_hw_p": 4,
    }


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        dataset_config_builder=build_dataset_config,
        extract_config_builder=build_extract_config,
    )
