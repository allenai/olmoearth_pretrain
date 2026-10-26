"""Per-modality capture dates packed into the shared ``timestamps`` array.

v1.3 gives every modality token of a time slot the same ``[day, month-1, year]`` stamp.
For the "v1.3 + per-modality DOY" experiment (2026-10-26) each multitemporal modality
token instead carries its own capture date. Rather than adding a field to
``OlmoEarthSample`` (most code special-cases only ``timestamps``/``latlon`` and treats
every other field as a modality), the dates ride in extra columns of ``timestamps``:

    timestamps[..., 0:3]                     shared slot stamp (unchanged meaning)
    timestamps[..., 3 + 3k : 6 + 3k]         stamp of PER_MODALITY_TIMESTAMP_MODALITIES[k]

so the array is ``(..., T, 3 + 3K)``. Every consumer that reads columns 0..2, slices the
time axis, pads it or passes it through is unaffected. Where a modality has no capture in
a slot, its block repeats the shared stamp (those tokens are missing anyway). A
3-column array means "no per-modality dates": every modality uses the shared stamp.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from olmoearth_pretrain.data.constants import Modality

# Fixed column layout; append only (a checkpoint trained with a layout reads by position).
PER_MODALITY_TIMESTAMP_MODALITIES: tuple[str, ...] = (
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
)
STAMP_WIDTH = 3
PACKED_TIMESTAMP_WIDTH = STAMP_WIDTH * (1 + len(PER_MODALITY_TIMESTAMP_MODALITIES))

ArrayT = TypeVar("ArrayT")


def has_per_modality_timestamps(timestamps: object) -> bool:
    """True if ``timestamps`` carries per-modality blocks (last dim > 3)."""
    return timestamps.shape[-1] > STAMP_WIDTH  # type: ignore[attr-defined]


def shared_timestamps(timestamps: ArrayT) -> ArrayT:
    """The shared ``(..., T, 3)`` slot stamp (numpy or torch)."""
    return timestamps[..., :STAMP_WIDTH]  # type: ignore[index]


def modality_timestamps(timestamps: ArrayT, modality_name: str) -> ArrayT:
    """``(..., T, 3)`` stamps for one modality: its block if packed, else the shared one."""
    width = timestamps.shape[-1]  # type: ignore[attr-defined]
    if width > STAMP_WIDTH and modality_name in PER_MODALITY_TIMESTAMP_MODALITIES:
        k = PER_MODALITY_TIMESTAMP_MODALITIES.index(modality_name)
        start = STAMP_WIDTH * (1 + k)
        if start + STAMP_WIDTH <= width:
            return timestamps[..., start : start + STAMP_WIDTH]  # type: ignore[index]
    return timestamps[..., :STAMP_WIDTH]  # type: ignore[index]


def pack_per_modality_timestamps(
    shared: np.ndarray, per_modality: dict[str, np.ndarray]
) -> np.ndarray:
    """``(T, 3)`` shared stamps + ``{modality: (T, 3)}`` -> ``(T, PACKED_TIMESTAMP_WIDTH)``.

    Modalities of the layout absent from ``per_modality`` get the shared stamp. Keys
    outside the layout are an error (they would be silently dropped).
    """
    shared = np.asarray(shared).reshape(-1, STAMP_WIDTH)
    unknown = set(per_modality) - set(PER_MODALITY_TIMESTAMP_MODALITIES)
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} not in PER_MODALITY_TIMESTAMP_MODALITIES "
            f"{PER_MODALITY_TIMESTAMP_MODALITIES}"
        )
    blocks = [shared]
    for name in PER_MODALITY_TIMESTAMP_MODALITIES:
        block = per_modality.get(name)
        if block is None:
            blocks.append(shared)
            continue
        block = np.asarray(block).reshape(-1, STAMP_WIDTH)
        assert block.shape == shared.shape, (name, block.shape, shared.shape)
        blocks.append(block.astype(shared.dtype))
    return np.concatenate(blocks, axis=-1)
