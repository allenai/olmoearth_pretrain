"""Eval dataset adapter for the ERA5 climate-zone probe.

Loads imagery inputs from an ERA5-inclusive pretraining HDF5 dataset and pairs
each scene with a precomputed climate-zone id -- a KMeans cluster over the
72-dim ERA5 12-month climate-normal signature, produced offline by
``scripts/tools/era5_climate_zone_eval.py build-zones`` (an npz mapping
``indices`` -> ``zones``). The probe measures whether the pooled scene embedding
linearly separates climate zones WITHOUT ERA5 ever being fed as an input, so
``era5_10`` is intentionally absent from ``training_modalities`` here: the label
is the climate zone, the inputs are the ordinary imagery modalities.

This is the in-loop counterpart of the linear-probe accuracy / macro-F1 reported
by the offline ``score`` command; the offline tool additionally computes
unsupervised NMI/ARI and a same/diff-zone cosine gap, which this per-checkpoint
eval does not.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.evals.datasets.pretrain_subset import (
    DEFAULT_HW_P,
    DEFAULT_PATCH_SIZE,
    PretrainSubsetDataset,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_SAMPLES = 4096
DEFAULT_VALID_SAMPLES = 1024
DEFAULT_TEST_SAMPLES = 1024


class ClimateZoneEvalDataset(Dataset):
    """Pair pretraining imagery with an offline ERA5 climate-zone label.

    Selects the samples whose h5 index carries a precomputed zone id, assigns a
    deterministic disjoint train/valid/test split, and returns normalized masked
    inputs plus the scalar zone id for a pooled linear-probe classification.
    """

    def __init__(
        self,
        h5py_dir: str,
        training_modalities: list[str],
        zones_npz_path: str,
        split: str = "train",
        patch_size: int = DEFAULT_PATCH_SIZE,
        hw_p: int = DEFAULT_HW_P,
        label_seed: int = 42,
        train_samples: int = DEFAULT_TRAIN_SAMPLES,
        valid_samples: int = DEFAULT_VALID_SAMPLES,
        test_samples: int = DEFAULT_TEST_SAMPLES,
        require_input_modalities_present: bool = True,
    ) -> None:
        """Initialize a deterministic climate-zone probe subset.

        Args:
            h5py_dir: Path to the ERA5-inclusive pretraining HDF5 directory.
            training_modalities: Imagery modalities used as model inputs. ERA5 is
                deliberately NOT among these -- it is the supervision target, not
                an input, so the probe tests climate-awareness of the embedding.
            zones_npz_path: Offline npz with ``indices`` (h5 sample ids) and
                ``zones`` (int zone id per index) from ``build-zones``.
            split: ``train``, ``valid``/``val``, or ``test``.
            patch_size: Patch size passed through to ``OlmoEarthDataset``.
            hw_p: Spatial patch count passed through to ``OlmoEarthDataset``.
            label_seed: Random seed for split assignment.
            train_samples: Max train samples after split assignment.
            valid_samples: Max validation samples after split assignment.
            test_samples: Max test samples after split assignment.
            require_input_modalities_present: Drop samples missing any input
                modality (applied after split assignment so split membership
                stays aligned across input-modality choices).
        """
        self.patch_size = patch_size
        self.hw_p = hw_p

        self._dataset = OlmoEarthDataset(
            h5py_dir=UPath(h5py_dir),
            training_modalities=training_modalities,
            dtype=np.float32,
            normalize=True,
        )
        self._dataset.prepare()
        assert self._dataset.sample_indices is not None, (
            "OlmoEarthDataset.prepare() must populate sample_indices."
        )
        sample_indices = np.asarray(self._dataset.sample_indices)

        zone_by_h5idx = self._load_zone_labels(zones_npz_path)

        # Positions into sample_indices whose h5 sample has an offline zone label.
        labeled_mask = np.fromiter(
            (int(h5idx) in zone_by_h5idx for h5idx in sample_indices.tolist()),
            dtype=bool,
            count=len(sample_indices),
        )
        eligible_positions = np.where(labeled_mask)[0]
        if eligible_positions.size == 0:
            raise ValueError(
                f"No samples in {h5py_dir} overlap the zone labels in "
                f"{zones_npz_path}; check the two point at the same corpus."
            )

        if require_input_modalities_present:
            eligible_positions = np.asarray(
                PretrainSubsetDataset._filter_positions_with_inputs_present(
                    self._dataset,
                    eligible_positions.tolist(),
                    training_modalities,
                ),
                dtype=np.int64,
            )

        selected = PretrainSubsetDataset._select_split_indices(
            total=len(eligible_positions),
            split=split,
            seed=label_seed,
            train_samples=train_samples,
            valid_samples=valid_samples,
            test_samples=test_samples,
        )
        self._indices = eligible_positions[
            np.asarray(selected, dtype=np.int64)
        ].tolist()
        self._labels = [
            zone_by_h5idx[int(sample_indices[pos])] for pos in self._indices
        ]

    @staticmethod
    def _load_zone_labels(zones_npz_path: str) -> dict[int, int]:
        """Load the offline ``indices -> zones`` mapping into a dict."""
        with UPath(zones_npz_path).open("rb") as f:
            with np.load(f) as npz:
                if "indices" not in npz or "zones" not in npz:
                    raise ValueError(
                        f"{zones_npz_path} must contain 'indices' and 'zones' "
                        f"arrays; got {list(npz.keys())}."
                    )
                h5_idx = np.asarray(npz["indices"], dtype=np.int64)
                zones = np.asarray(npz["zones"], dtype=np.int64)
        if h5_idx.shape != zones.shape:
            raise ValueError(
                f"zone npz 'indices' {h5_idx.shape} and 'zones' {zones.shape} "
                f"must have the same shape."
            )
        return {int(i): int(z) for i, z in zip(h5_idx.tolist(), zones.tolist())}

    def __len__(self) -> int:
        """Return number of samples in the split."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return a masked input sample and its scalar climate-zone label."""
        real_idx = self._indices[idx]
        args = GetItemArgs(
            idx=real_idx,
            patch_size=self.patch_size,
            sampled_hw_p=self.hw_p,
        )
        _, sample = self._dataset[args]
        masked = PretrainSubsetDataset._missing_aware_masked_sample(sample)
        return masked, torch.tensor(self._labels[idx], dtype=torch.long)
