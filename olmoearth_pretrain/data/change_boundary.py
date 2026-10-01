"""Helpers for the ``open_set_change_boundary`` modality.

Paired pre/post change samples (see ``open_set_segmentation_data``) carry a single
boundary date per sample, stored as ``[day, month, year]`` in the same convention as
``timestamps`` (0-based month). A timestep is *post*-change iff its timestamp is on
or after the boundary. Non-change samples have the modality missing-filled
(``MISSING_VALUE``), in which case there is no pre/post split.

These helpers are shared by the dataset subsetting (keep at least one timestep on
each side), the masking strategy (encode at least one timestep on each side) and
the open-set probe (only supervise change labels when both sides were visible).
"""

from __future__ import annotations

import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.types import ArrayTensor


def _date_key(day_month_year: ArrayTensor) -> torch.Tensor:
    """Encode ``[..., 3]`` (day, month, year) as a sortable ``[...]`` integer."""
    dmy = torch.as_tensor(day_month_year).to(torch.int64)
    return dmy[..., 2] * 10_000 + dmy[..., 1] * 100 + dmy[..., 0]


def has_change_boundary(boundary: ArrayTensor | None) -> torch.Tensor:
    """Presence of a ``[..., 3]`` boundary (not missing-filled) as a ``[...]`` bool."""
    if boundary is None:
        return torch.zeros((), dtype=torch.bool)
    return (torch.as_tensor(boundary) != MISSING_VALUE).all(dim=-1)


def timestep_is_post(timestamps: ArrayTensor, boundary: ArrayTensor) -> torch.Tensor:
    """Per-timestep post-change flags.

    Args:
        timestamps: ``[..., T, 3]`` (day, month, year) timestamps.
        boundary: ``[..., 3]`` boundary in the same convention (missing-filled
            entries yield arbitrary flags; mask them with :func:`has_change_boundary`).

    Returns:
        ``[..., T]`` bool tensor, True where ``timestamp >= boundary``.
    """
    return _date_key(timestamps) >= _date_key(boundary).unsqueeze(-1)


def restrict_start_ts_to_change_boundary(
    valid_start_ts: list[int],
    max_t: int,
    timestamps: ArrayTensor,
    boundary: ArrayTensor | None,
) -> list[int]:
    """Keep only the start timesteps whose window covers both a pre and a post step.

    Used by the dataset subsetting so a change sample's ``max_t``-long temporal crop
    always contains at least one timestep before the boundary and one on or after
    it. Falls back to ``valid_start_ts`` unchanged when the sample has no boundary,
    when ``max_t < 2``, or when no start satisfies the constraint.

    Args:
        valid_start_ts: Candidate start timesteps (from ``get_valid_start_ts``).
        max_t: Length of the temporal crop.
        timestamps: ``[T, 3]`` timestamps of the sample.
        boundary: ``[3]`` boundary, or None / missing-filled for non-change samples.
    """
    if max_t < 2 or boundary is None or not bool(has_change_boundary(boundary)):
        return valid_start_ts
    is_post = timestep_is_post(timestamps, boundary)
    keep = [
        start
        for start in valid_start_ts
        if not bool(is_post[start : start + max_t].all())
        and bool(is_post[start : start + max_t].any())
    ]
    return keep or valid_start_ts
