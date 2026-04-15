"""Optional modality subsampling.

At training time we may want each batch to contain only a subset of the
available input modalities — e.g. some batches with Sentinel-2 only, some
with Sentinel-1 only, some with Landsat only — so the model learns to
operate on whichever sensors a downstream user has access to.

The subsampler operates at the boundary between the dataset and the encoder.
It does *not* drop the label-source modalities (OSM/CDL/...), only the
input modalities the encoder is supposed to consume.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample

# Default set of *input* modalities the encoder can consume. Label-source
# modalities (openstreetmap_raster, cdl, eurocrops, worldcover) are
# explicitly excluded; the sampler/dataset still need them for supervision.
DEFAULT_INPUT_MODALITIES: tuple[str, ...] = (
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    "naip",
    "naip_10",
)


@dataclass(frozen=True)
class ModalitySubsampleConfig:
    """Configuration for stochastic modality subsampling.

    Attributes:
        input_modalities: The modalities considered "inputs" to the encoder.
            Only these are eligible to be dropped. Modalities not in this set
            (label sources, latlon, timestamps, era5, srtm, ...) are passed
            through untouched.
        min_kept: Minimum number of input modalities retained per call.
            Must be at least 1 — dropping all sensors would yield an empty
            input.
        max_kept: Maximum number of input modalities retained, or None for
            no cap (keep up to all available).
        p_subsample: Probability of subsampling on a given call. With
            probability ``1 - p_subsample`` the sample is returned unchanged.
    """

    input_modalities: tuple[str, ...] = DEFAULT_INPUT_MODALITIES
    min_kept: int = 1
    max_kept: int | None = None
    p_subsample: float = 1.0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.min_kept < 1:
            raise ValueError("min_kept must be >= 1")
        if self.max_kept is not None and self.max_kept < self.min_kept:
            raise ValueError("max_kept must be >= min_kept")
        if not 0.0 <= self.p_subsample <= 1.0:
            raise ValueError("p_subsample must be in [0, 1]")


def _drop_to_none(sample: OlmoEarthSample, drop: Iterable[str]) -> OlmoEarthSample:
    """Return a copy of ``sample`` with the named modalities replaced by None."""
    drop_set = set(drop)
    fields = sample.as_dict(include_nones=True)
    new_fields = {k: (None if k in drop_set else v) for k, v in fields.items()}
    return OlmoEarthSample(**new_fields)


def _drop_to_none_masked(
    sample: MaskedOlmoEarthSample, drop: Iterable[str]
) -> MaskedOlmoEarthSample:
    """Drop both the modality field and its ``<modality>_mask`` from a MaskedOlmoEarthSample."""
    drop_set = set(drop)
    drop_set |= {f"{name}_mask" for name in drop}
    fields = sample.as_dict(include_nones=True)
    new_fields = {k: (None if k in drop_set else v) for k, v in fields.items()}
    return MaskedOlmoEarthSample(**new_fields)


def subsample_modalities_masked(
    sample: MaskedOlmoEarthSample,
    config: ModalitySubsampleConfig,
    rng: random.Random | None = None,
) -> MaskedOlmoEarthSample:
    """Stochastically drop input modalities from a ``MaskedOlmoEarthSample``."""
    if rng is None:
        rng = random.Random()

    if rng.random() >= config.p_subsample:
        return sample

    present_inputs = [
        m for m in config.input_modalities if getattr(sample, m, None) is not None
    ]
    if len(present_inputs) <= config.min_kept:
        return sample

    upper = (
        len(present_inputs)
        if config.max_kept is None
        else min(config.max_kept, len(present_inputs))
    )
    n_keep = rng.randint(config.min_kept, upper)
    keep = set(rng.sample(present_inputs, n_keep))
    drop = [m for m in present_inputs if m not in keep]
    if not drop:
        return sample
    return _drop_to_none_masked(sample, drop)


def subsample_modalities(
    sample: OlmoEarthSample,
    config: ModalitySubsampleConfig,
    rng: random.Random | None = None,
) -> OlmoEarthSample:
    """Stochastically drop input modalities from a sample.

    Returns the original sample (no copy) when no subsampling is performed.
    """
    if rng is None:
        rng = random.Random()

    if rng.random() >= config.p_subsample:
        return sample

    present_inputs = [
        m for m in config.input_modalities if getattr(sample, m, None) is not None
    ]
    if len(present_inputs) <= config.min_kept:
        return sample

    upper = (
        len(present_inputs)
        if config.max_kept is None
        else min(config.max_kept, len(present_inputs))
    )
    n_keep = rng.randint(config.min_kept, upper)
    keep = set(rng.sample(present_inputs, n_keep))
    drop = [m for m in present_inputs if m not in keep]
    if not drop:
        return sample
    return _drop_to_none(sample, drop)


@dataclass
class ModalitySubsampler:
    """Callable wrapper around ``subsample_modalities`` carrying its config + rng."""

    config: ModalitySubsampleConfig = field(default_factory=ModalitySubsampleConfig)
    seed: int | None = None
    _rng: random.Random | None = field(default=None, init=False, repr=False)

    def __call__(self, sample: OlmoEarthSample) -> OlmoEarthSample:
        """Subsample modalities on a single sample (or batch)."""
        if self._rng is None:
            self._rng = random.Random(self.seed)
        return subsample_modalities(sample, self.config, self._rng)
