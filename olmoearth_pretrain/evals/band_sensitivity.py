"""Per-band occlusion sensitivity for a frozen encoder's spectral pathway.

Accuracy tells you *that* a task regressed; this tells you *which input bands
the encoder still reads*. For each band of one modality the eval split is
re-embedded with that band zeroed, and we measure how far the embedding moves
and how much a KNN probe loses. Zeroing the already-normalized tensor is exactly
what training-time band dropout does, so the perturbation is one the model has
seen rather than an arbitrary out-of-distribution poke.

The motivating case is m-eurosat. Its classes turn on fine spectral contrast
(red-edge B05-B07, SWIR B11/B12), but those bands are strongly correlated with
the visible/NIR ones, so a pretext objective can be driven down while quietly
collapsing them away -- and with all twelve S2 bands entering a single bandset,
that collapse has to happen inside one small projection. ``effective_num_bands``
is the summary to watch: the exponentiated entropy of the per-band reliance
profile, i.e. how many bands the encoder behaves as though it reads. A model
spreading reliance across all twelve scores 12; one that has fallen back on
RGB+NIR scores around 4.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from torch import Tensor

from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def occlude_band(
    masked_olmoearth_sample: MaskedOlmoEarthSample,
    modality: str,
    band_index: int,
) -> MaskedOlmoEarthSample:
    """Return a copy of the sample with one band of one modality zeroed.

    Zeroing happens on the normalized tensor the model consumes, matching
    training-time band dropout. The modality mask is left untouched: the band
    still occupies its channel and the token is still encoded, so what we
    measure is the loss of that band's information rather than a change in
    sequence length.
    """
    data = getattr(masked_olmoearth_sample, modality, None)
    if data is None:
        raise ValueError(
            f"Cannot occlude '{modality}': the sample carries no such modality "
            f"(present: {masked_olmoearth_sample.modalities})"
        )
    if not -data.shape[-1] <= band_index < data.shape[-1]:
        raise IndexError(
            f"Band index {band_index} out of range for '{modality}' with "
            f"{data.shape[-1]} bands"
        )
    occluded = data.clone()
    occluded[..., band_index] = 0.0
    return masked_olmoearth_sample._replace(**{modality: occluded})


def embedding_drift(reference: Tensor, occluded: Tensor) -> dict[str, float]:
    """Measure how far occluding a band moved each sample's embedding.

    ``emb_cos`` near 1 and ``emb_rel_l2`` near 0 mean the band is not read at
    all. Both are reported because they answer slightly different questions:
    cosine ignores any change in embedding magnitude, which the cosine KNN probe
    also ignores, while the relative L2 captures it.
    """
    ref = reference.float()
    occ = occluded.float()
    cos = torch.nn.functional.cosine_similarity(ref, occ, dim=-1)
    rel_l2 = (occ - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-12)
    return {
        "emb_cos": cos.mean().item(),
        "emb_rel_l2": rel_l2.mean().item(),
    }


def reliance_profile(per_band: Mapping[str, float]) -> dict[str, float]:
    """Summarize a per-band reliance profile as an effective band count.

    ``per_band`` maps band name to a non-negative measure of how much removing
    that band perturbs the model (embedding drift, or accuracy lost). Negative
    values -- which occur for accuracy drops purely as probe noise -- are floored
    at zero. The values are normalized to a distribution and reported as
    ``exp(entropy)``, so the number is directly readable as "the encoder behaves
    as though it reads this many bands", comparable across checkpoints and
    across runs with different absolute sensitivity.
    """
    if not per_band:
        return {}
    values = torch.tensor(
        [max(float(v), 0.0) for v in per_band.values()], dtype=torch.float64
    )
    total = values.sum()
    if total <= 0.0:
        # Nothing moves when any single band is removed. That is degenerate
        # rather than uniform, so report the floor instead of a flat profile
        # that would read as maximally healthy.
        return {"effective_num_bands": 0.0, "max_band_share": 0.0}
    p = values / total
    nonzero = p[p > 0]
    entropy = -(nonzero * nonzero.log()).sum()
    return {
        "effective_num_bands": entropy.exp().item(),
        "max_band_share": p.max().item(),
    }
