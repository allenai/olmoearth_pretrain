"""Turn an every-capture ("allcap") H5 sample into the standard 12-slot v1.3 sample.

The allcap corpus (``/weka/dfive-default/helios/dataset/osm_allcaptures``) stores every
Sentinel-2 / Sentinel-1 / Landsat-L2 capture over a 360-day span per window, each
modality with its own ``timestamps_<modality>`` array of ``[day, month-1, year]`` rows
and no shared timeline. The v1.3 loader, collate, masking and model all expect ONE
shared ``timestamps`` array of at most ``MAX_SEQUENCE_LENGTH`` (12) slots plus a
per-modality ``missing_timesteps_masks`` entry, with the modality arrays stored
compactly (present timesteps only). This module selects captures and emits exactly
that, so everything downstream of ``OlmoEarthDataset.read_h5_file`` is unchanged.

Two selection rules (experiment E5, "which 12 captures and which dates"):

``least_cloudy_slot`` (arm A, the v1.3 mosaic rule with single captures)
    The span is cut into 12 slots of 30 days starting at the earliest capture of any
    sensor (the window's own -180 d start is not stored in the H5; the first capture is
    within a few days of it). Per slot: the S2 capture with the lowest
    ``sentinel2_scl_cloud_fraction`` and the S1 capture nearest the slot midpoint, both
    stamped at the slot start. Deterministic.

``random_clear`` (arm B, real captures with real dates)
    Up to 12 S2 captures drawn uniformly without replacement among those with cloud
    fraction <= ``tau``; each becomes a slot stamped with its real date. Because the
    model reads one shared timeline (RoPE days and month embeddings both index the same
    ``timestamps``), S1 cannot carry its own date in the same slot; each S2 slot takes
    the nearest unused S1 capture within +/- ``S1_MATCH_TOLERANCE_DAYS`` (greedy by
    distance) or marks S1 missing there. If fewer than 12 clear S2 captures exist, the
    free slots get randomly drawn unused S1 captures with their own real dates (S2
    missing there), and the rest of the 12 slots are missing, as in v1.3. Slots are
    sorted by date. Random on every read, i.e. per sample per epoch.

Two more rules complete a 2x2 with A and B (E5 follow-up, 2026-10-19), since A and B
differ both in whether captures are redrawn each read and in real vs slot-start dates:

``least_cloudy_slot_real_dates`` (arm A_real)
    Arm A's captures exactly, each slot stamped with its capture's real date.

``random_clear_slot`` (arm B_slot)
    Arm A's slot grid and slot-start stamps, but per slot a random clear S2 capture
    (``tau``; least-cloudy fallback) and a random S1 capture, redrawn on every read.

Arm C (``read_allcap_sample_two_draws``) reads the H5 once and applies the arm B rule
twice with two independent generators, giving two "views" of the same pixel-year that
differ only in which captures (and dates) were drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from olmoearth_pretrain.data.constants import MAX_SEQUENCE_LENGTH, Modality

ALLCAP_LEAST_CLOUDY_SLOT = "least_cloudy_slot"
ALLCAP_RANDOM_CLEAR = "random_clear"
# E5 follow-up 2x2 (2026-10-19): A and B differ in fresh-vs-fixed captures AND real-vs-slot dates.
ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES = "least_cloudy_slot_real_dates"  # A's captures, real dates
ALLCAP_RANDOM_CLEAR_SLOT = "random_clear_slot"  # A's slot grid and stamps, captures redrawn per read
ALLCAP_SELECTIONS = (
    ALLCAP_LEAST_CLOUDY_SLOT,
    ALLCAP_RANDOM_CLEAR,
    ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES,
    ALLCAP_RANDOM_CLEAR_SLOT,
)

NUM_SLOTS = MAX_SEQUENCE_LENGTH
SLOT_DAYS = 30
# Arm B: an S1 capture is paired with an S2 slot only if within half a slot of it.
S1_MATCH_TOLERANCE_DAYS = 15

S2 = Modality.SENTINEL2_L2A.name
S1 = Modality.SENTINEL1.name
CLOUD_FRACTION_KEY = "sentinel2_scl_cloud_fraction"
SCL_TIMESTAMPS_KEY = "timestamps_sentinel2_scl"
# Every-capture modalities in the allcap H5 that this module never emits.
_TIME_SERIES_KEYS_FOR_SPAN = ("sentinel2_l2a", "sentinel1", "landsat_l2")


def timestamps_to_epoch_days(ts: np.ndarray) -> np.ndarray:
    """``[day, month-1, year]`` rows -> int64 days since 1970-01-01 (exact calendar)."""
    ts = np.asarray(ts, dtype=np.int64).reshape(-1, 3)
    months = ((ts[:, 2] - 1970) * 12 + ts[:, 1]).astype("datetime64[M]")
    days = months.astype("datetime64[D]") + (ts[:, 0] - 1).astype("timedelta64[D]")
    return days.astype(np.int64)


def epoch_days_to_timestamps(days: np.ndarray) -> np.ndarray:
    """Inverse of :func:`timestamps_to_epoch_days`, as int32 ``[day, month-1, year]``."""
    dt = np.asarray(days, dtype=np.int64).astype("datetime64[D]")
    year = dt.astype("datetime64[Y]")
    month = dt.astype("datetime64[M]")
    out = np.stack(
        [
            (dt - month.astype("datetime64[D]")).astype(np.int64) + 1,
            (month - year.astype("datetime64[M]")).astype(np.int64),
            year.astype(np.int64) + 1970,
        ],
        axis=-1,
    )
    return out.astype(np.int32).reshape(-1, 3)


@dataclass
class SlotSelection:
    """Per-slot capture choice. ``-1`` marks a modality missing in that slot."""

    days: np.ndarray  # (n,) int64 epoch days of each slot's timestamp
    s2_idx: np.ndarray  # (n,) int64 index into the S2 capture axis, or -1
    s1_idx: np.ndarray  # (n,) int64 index into the S1 capture axis, or -1

    def __post_init__(self) -> None:
        """Validate shapes."""
        n = len(self.days)
        assert len(self.s2_idx) == n and len(self.s1_idx) == n
        assert n <= NUM_SLOTS, f"{n} slots > {NUM_SLOTS}"


def _clean_cloud(cloud: np.ndarray) -> np.ndarray:
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1)
    return np.where(np.isfinite(cloud), cloud, 1.0)


def select_least_cloudy_per_slot(
    s2_days: np.ndarray,
    s2_cloud: np.ndarray,
    s1_days: np.ndarray,
    span_start: int,
) -> SlotSelection:
    """Arm A: per 30-day slot, least-cloudy S2 and midpoint-nearest S1, stamped at slot start.

    Captures after the 12th slot (only possible if ``span_start`` is late) are clamped
    into the last slot. Ties go to the earliest capture.
    """
    s2_days = np.asarray(s2_days, dtype=np.int64)
    s1_days = np.asarray(s1_days, dtype=np.int64)
    cloud = _clean_cloud(s2_cloud)
    assert len(cloud) == len(s2_days)
    slot_starts = span_start + SLOT_DAYS * np.arange(NUM_SLOTS, dtype=np.int64)
    s2_slot = np.clip((s2_days - span_start) // SLOT_DAYS, 0, NUM_SLOTS - 1)
    s1_slot = np.clip((s1_days - span_start) // SLOT_DAYS, 0, NUM_SLOTS - 1)
    s2_idx = np.full(NUM_SLOTS, -1, dtype=np.int64)
    s1_idx = np.full(NUM_SLOTS, -1, dtype=np.int64)
    for k in range(NUM_SLOTS):
        cand = np.flatnonzero(s2_slot == k)
        if len(cand):
            s2_idx[k] = cand[np.argmin(cloud[cand])]
        cand = np.flatnonzero(s1_slot == k)
        if len(cand):
            mid = slot_starts[k] + SLOT_DAYS // 2
            s1_idx[k] = cand[np.argmin(np.abs(s1_days[cand] - mid))]
    return SlotSelection(days=slot_starts, s2_idx=s2_idx, s1_idx=s1_idx)


def select_least_cloudy_per_slot_real_dates(
    s2_days: np.ndarray,
    s2_cloud: np.ndarray,
    s1_days: np.ndarray,
    span_start: int,
) -> SlotSelection:
    """2x2 arm A_real: arm A's captures exactly, each slot stamped with a real capture date.

    The slot's date is its S2 capture's date, else its S1 capture's date (S1-only slot),
    else the slot start (an empty slot, which carries no tokens). The S1 capture of a
    slot that also has S2 shares the S2 date, as both share the slot start in arm A
    (within one 30-day slot either way). Slots stay in grid order, which is date order.
    """
    sel = select_least_cloudy_per_slot(s2_days, s2_cloud, s1_days, span_start)
    s2_days = np.asarray(s2_days, dtype=np.int64)
    s1_days = np.asarray(s1_days, dtype=np.int64)
    days = sel.days.copy()
    has_s2 = sel.s2_idx >= 0
    s1_only = ~has_s2 & (sel.s1_idx >= 0)
    days[has_s2] = s2_days[sel.s2_idx[has_s2]]
    days[s1_only] = s1_days[sel.s1_idx[s1_only]]
    return SlotSelection(days=days, s2_idx=sel.s2_idx, s1_idx=sel.s1_idx)


def select_random_clear_per_slot(
    s2_days: np.ndarray,
    s2_cloud: np.ndarray,
    s1_days: np.ndarray,
    span_start: int,
    tau: float,
    rng: np.random.Generator | Any,
) -> SlotSelection:
    """2x2 arm B_slot: arm A's slot grid and slot-start stamps, captures redrawn every read.

    Per 30-day slot: a uniformly random S2 capture among that slot's clear ones (cloud
    fraction <= ``tau``), or arm A's least-cloudy one if the slot has none clear; and a
    uniformly random S1 capture among that slot's S1 captures (arm A takes the one
    nearest the midpoint). Both are stamped at the slot start, as in arm A.
    """
    s2_days = np.asarray(s2_days, dtype=np.int64)
    s1_days = np.asarray(s1_days, dtype=np.int64)
    cloud = _clean_cloud(s2_cloud)
    assert len(cloud) == len(s2_days)
    base = select_least_cloudy_per_slot(s2_days, cloud, s1_days, span_start)
    s2_slot = np.clip((s2_days - span_start) // SLOT_DAYS, 0, NUM_SLOTS - 1)
    s1_slot = np.clip((s1_days - span_start) // SLOT_DAYS, 0, NUM_SLOTS - 1)
    s2_idx = base.s2_idx.copy()
    s1_idx = base.s1_idx.copy()
    for k in range(NUM_SLOTS):
        clear = np.flatnonzero((s2_slot == k) & (cloud <= tau))
        if len(clear):
            s2_idx[k] = int(rng.choice(clear))
        cand = np.flatnonzero(s1_slot == k)
        if len(cand):
            s1_idx[k] = int(rng.choice(cand))
    return SlotSelection(days=base.days, s2_idx=s2_idx, s1_idx=s1_idx)


def select_random_clear(
    s2_days: np.ndarray,
    s2_cloud: np.ndarray,
    s1_days: np.ndarray,
    tau: float,
    rng: np.random.Generator | Any,
    s1_tolerance_days: int = S1_MATCH_TOLERANCE_DAYS,
) -> SlotSelection:
    """Arm B: up to 12 random clear S2 captures with real dates; S1 paired or S1-only.

    ``rng`` is a ``np.random.Generator`` or the ``np.random`` module (anything with
    ``choice``). See the module docstring for the full rule.
    """
    s2_days = np.asarray(s2_days, dtype=np.int64)
    s1_days = np.asarray(s1_days, dtype=np.int64)
    cloud = _clean_cloud(s2_cloud)
    assert len(cloud) == len(s2_days)

    clear = np.flatnonzero(cloud <= tau)
    n2 = min(NUM_SLOTS, len(clear))
    if n2 > 0:
        s2_sel = np.asarray(rng.choice(clear, size=n2, replace=False), dtype=np.int64)
    elif len(s1_days) == 0 and len(s2_days) > 0:
        # No clear S2 and no S1 at all: keep the single least-cloudy S2 capture so the
        # sample still has encoder tokens (rare: persistently cloudy and no S1).
        s2_sel = np.array([int(np.argmin(cloud))], dtype=np.int64)
    else:
        s2_sel = np.zeros(0, dtype=np.int64)
    s2_sel = s2_sel[np.argsort(s2_days[s2_sel], kind="stable")]
    slot_days = list(s2_days[s2_sel])
    slot_s2 = list(s2_sel)
    slot_s1 = [-1] * len(s2_sel)

    # Greedy nearest-first pairing of S1 captures to S2 slots, each used at most once.
    used_s1 = np.zeros(len(s1_days), dtype=bool)
    if len(s2_sel) and len(s1_days):
        dist = np.abs(s2_days[s2_sel][:, None] - s1_days[None, :])
        slot_i, s1_j = np.nonzero(dist <= s1_tolerance_days)
        order = np.argsort(dist[slot_i, s1_j], kind="stable")
        for i, j in zip(slot_i[order], s1_j[order]):
            if slot_s1[i] == -1 and not used_s1[j]:
                slot_s1[i] = int(j)
                used_s1[j] = True

    # Fewer than 12 S2 slots: fill free slots with random unused S1 captures (real dates).
    free = NUM_SLOTS - len(slot_days)
    unused = np.flatnonzero(~used_s1)
    if free > 0 and len(unused):
        extra = np.asarray(
            rng.choice(unused, size=min(free, len(unused)), replace=False),
            dtype=np.int64,
        )
        for j in extra:
            slot_days.append(s1_days[j])
            slot_s2.append(-1)
            slot_s1.append(int(j))

    days = np.asarray(slot_days, dtype=np.int64)
    order = np.argsort(days, kind="stable")
    return SlotSelection(
        days=days[order],
        s2_idx=np.asarray(slot_s2, dtype=np.int64)[order],
        s1_idx=np.asarray(slot_s1, dtype=np.int64)[order],
    )


def align_cloud_fraction(
    s2_ts: np.ndarray, scl_ts: np.ndarray | None, cloud: np.ndarray | None
) -> np.ndarray:
    """Cloud fraction per S2 capture, matched through the SCL timestamps.

    ``sentinel2_scl_cloud_fraction`` follows ``timestamps_sentinel2_scl``, which in
    ~0.3% of samples has one capture more than ``timestamps_sentinel2_l2a`` (and in
    rare samples is absent). Captures are matched by date in order; an S2 capture
    without an SCL match gets NaN (treated as fully cloudy by the selectors).
    """
    s2_ts = np.asarray(s2_ts).reshape(-1, 3)
    out = np.full(len(s2_ts), np.nan, dtype=np.float64)
    if scl_ts is None or cloud is None:
        return out
    scl_ts = np.asarray(scl_ts).reshape(-1, 3)
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1)
    if len(scl_ts) != len(cloud):
        return out
    if len(scl_ts) == len(s2_ts) and np.array_equal(scl_ts, s2_ts):
        return cloud
    s2_days = timestamps_to_epoch_days(s2_ts)
    scl_days = timestamps_to_epoch_days(scl_ts)
    j = 0
    for i, d in enumerate(s2_days):
        while j < len(scl_days) and scl_days[j] < d:
            j += 1
        if j < len(scl_days) and scl_days[j] == d:
            out[i] = cloud[j]
            j += 1
    return out


def span_start_day(timestamps_by_modality: dict[str, np.ndarray]) -> int:
    """Earliest capture date over the given per-modality timestamp arrays."""
    firsts = [
        int(timestamps_to_epoch_days(ts).min())
        for ts in timestamps_by_modality.values()
        if ts is not None and len(ts)
    ]
    if not firsts:
        raise ValueError("allcap sample has no time-series captures")
    return min(firsts)


def select_slots(
    selection: str,
    s2_ts: np.ndarray,
    s2_cloud: np.ndarray,
    s1_ts: np.ndarray,
    span_start: int,
    tau: float,
    rng: np.random.Generator | Any,
) -> SlotSelection:
    """Dispatch to the arm's selection rule on raw ``[day, month-1, year]`` arrays."""
    s2_days = timestamps_to_epoch_days(s2_ts)
    s1_days = timestamps_to_epoch_days(s1_ts)
    if selection == ALLCAP_LEAST_CLOUDY_SLOT:
        return select_least_cloudy_per_slot(s2_days, s2_cloud, s1_days, span_start)
    if selection == ALLCAP_RANDOM_CLEAR:
        return select_random_clear(s2_days, s2_cloud, s1_days, tau, rng)
    if selection == ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES:
        return select_least_cloudy_per_slot_real_dates(s2_days, s2_cloud, s1_days, span_start)
    if selection == ALLCAP_RANDOM_CLEAR_SLOT:
        return select_random_clear_per_slot(s2_days, s2_cloud, s1_days, span_start, tau, rng)
    raise ValueError(f"unknown allcap selection {selection!r}; use {ALLCAP_SELECTIONS}")


def selection_to_sample(
    sel: SlotSelection,
    series: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the v1.3 ``(sample_dict, missing_timesteps_masks)`` from a selection.

    ``series`` maps ``sentinel2_l2a`` / ``sentinel1`` to their full ``(H, W, T, C)``
    capture stacks (absent keys are treated as having no captures). Modality arrays are
    stored compactly (present slots only, in slot order) exactly like the v1.3 H5s; a
    modality with no present slot is left out so the dataset fills it as missing.
    """
    sample: dict[str, Any] = {"timestamps": epoch_days_to_timestamps(sel.days)}
    masks: dict[str, np.ndarray] = {}
    for name, idx in ((S2, sel.s2_idx), (S1, sel.s1_idx)):
        if name not in series:
            continue
        present = idx >= 0
        if not present.any():
            continue
        sample[name] = series[name][:, :, idx[present], :]
        masks[name] = present
    return sample, masks


def _read_selection_inputs(
    h5file: Any, training_modalities: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-sensor timestamps, aligned S2 cloud fractions and span start of one H5.

    S1 timestamps always enter the span start (so arm A's slot grid does not depend on
    whether S1 is trained), but if ``sentinel1`` is not a training modality the
    selectors get no S1 captures (Ilse's S2-only rule, c83250ae7).
    """
    ts = {
        key: h5file[f"timestamps_{key}"][()]
        for key in _TIME_SERIES_KEYS_FOR_SPAN
        if f"timestamps_{key}" in h5file
    }
    empty_ts = np.zeros((0, 3), dtype=np.int32)
    s2_ts = ts.get(S2, empty_ts)
    s1_ts = ts.get(S1, empty_ts)
    s2_cloud = align_cloud_fraction(
        s2_ts,
        h5file[SCL_TIMESTAMPS_KEY][()] if SCL_TIMESTAMPS_KEY in h5file else None,
        h5file[CLOUD_FRACTION_KEY][()] if CLOUD_FRACTION_KEY in h5file else None,
    )
    span_start = span_start_day(ts)
    if S1 not in training_modalities:
        s1_ts = empty_ts
    return s2_ts, s2_cloud, s1_ts, span_start


def _read_series(
    h5file: Any, training_modalities: list[str], sels: list[SlotSelection]
) -> dict[str, np.ndarray]:
    """The S2/S1 stacks that at least one of ``sels`` uses."""
    series = {}
    for name, attr in ((S2, "s2_idx"), (S1, "s1_idx")):
        if (
            name in training_modalities
            and name in h5file
            and any((getattr(sel, attr) >= 0).any() for sel in sels)
        ):
            # One chunk spans all T, so reading the whole stack costs the same
            # decompression as a subset and avoids h5py's per-index hyperslab path.
            series[name] = h5file[name][()]
    return series


def _read_static(h5file: Any, training_modalities: list[str]) -> dict[str, Any]:
    """The non-time-series training modalities (maps), as stored."""
    out = {}
    for name in training_modalities:
        if name in (S2, S1) or name not in h5file:
            continue
        spec = Modality.get(name)
        if spec.is_spacetime_varying or spec.is_time_only_varying:
            raise ValueError(
                f"allcap loader only builds S2/S1 time series; got multitemporal {name}"
            )
        out[name] = h5file[name][()]
    return out


def read_allcap_sample(
    h5file: Any,
    training_modalities: list[str],
    selection: str,
    tau: float,
    rng: np.random.Generator | Any = np.random,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one allcap H5 (an open ``h5py.File``) as a 12-slot v1.3 sample.

    Only the datasets that are needed are read: the per-modality timestamps, the S2
    cloud fractions, the S2/S1 stacks if they are training modalities, and the
    non-time-series training modalities (maps) as stored. Landsat and SCL imagery are
    never decompressed.

    S2-only (``sentinel1`` not in ``training_modalities``): the S1 stack is never
    opened, and the selectors see no S1 captures, so no slot exists only for S1 (arm
    B fills at most 12 clear S2 captures; arm A's per-slot S2 choice is unchanged).
    The S1 *timestamps* (a few hundred bytes) still enter the span start, so arm A's
    slot grid is identical to the S1+S2 variant.
    """
    s2_ts, s2_cloud, s1_ts, span_start = _read_selection_inputs(
        h5file, training_modalities
    )
    sel = select_slots(selection, s2_ts, s2_cloud, s1_ts, span_start, tau, rng)
    series = _read_series(h5file, training_modalities, [sel])
    sample, masks = selection_to_sample(sel, series)
    sample.update(_read_static(h5file, training_modalities))
    return sample, masks


def read_allcap_sample_two_draws(
    h5file: Any,
    training_modalities: list[str],
    selection: str,
    tau: float,
    rngs: tuple[np.random.Generator, np.random.Generator],
) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Arm C: ONE read of an allcap H5, two independent slot selections ("views").

    Each view is what :func:`read_allcap_sample` returns, with its own selection drawn
    from its own generator. The S2/S1 stacks and the static maps are read (and
    decompressed) once and shared; only the chosen captures, their dates and the
    missing-timestep masks differ between the views.
    """
    s2_ts, s2_cloud, s1_ts, span_start = _read_selection_inputs(
        h5file, training_modalities
    )
    sels = [
        select_slots(selection, s2_ts, s2_cloud, s1_ts, span_start, tau, rng)
        for rng in rngs
    ]
    series = _read_series(h5file, training_modalities, sels)
    static = _read_static(h5file, training_modalities)
    views = []
    for sel in sels:
        sample, masks = selection_to_sample(sel, series)
        sample.update(static)
        views.append((sample, masks))
    return views
