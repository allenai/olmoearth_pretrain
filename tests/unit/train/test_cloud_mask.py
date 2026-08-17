"""Test the cloud-aware decoder-token skip and the cloud side-payload plumbing.

Covers the pieces that make ``scripts/vnext/2026_07_22_cloud_mask`` work:
``align_cloud_to_nominal`` (time de-compaction), the ``*_cloud`` fields staying out
of ``.modalities``, collate + transform carrying cloud in lockstep with its
modality, and ``RandomTimeWithDecodeMaskingStrategy`` dropping exactly the
mostly-cloud DECODER tokens.
"""

import numpy as np
import torch

from olmoearth_pretrain.data.cloud_mask_cache import (
    CLEAR,
    NODATA,
    SHADOW,
    THICK_CLOUD,
    align_cloud_to_nominal,
)
from olmoearth_pretrain.data.collate import (
    collate_olmoearth_pretrain,
    collate_single_masked_batched,
    extract_cloud_payload,
)
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.transform import FlipAndRotateSpace
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskValue,
    RandomTimeWithDecodeMaskingStrategy,
)

MAX_T = 6


def _timestamps(b: int, t: int) -> torch.Tensor:
    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    return torch.cat([days, months, years], dim=1)


def _sample(
    h: int = 8,
    w: int = 8,
    t: int = 2,
    cloud_s2: np.ndarray | None = None,
) -> OlmoEarthSample:
    """A single (un-batched) sample with S2 + worldcover and optional S2 cloud."""
    kwargs = {
        "sentinel2_l2a": np.ones(
            (h, w, t, Modality.SENTINEL2_L2A.num_bands), dtype=np.float32
        ),
        "worldcover": np.ones(
            (h, w, 1, Modality.WORLDCOVER.num_bands), dtype=np.float32
        ),
        "timestamps": _timestamps(1, t)[0].T.numpy(),
    }
    if cloud_s2 is not None:
        kwargs["sentinel2_l2a_cloud"] = cloud_s2
    return OlmoEarthSample(**kwargs)


def test_align_cloud_to_nominal_matches_missing_timestep_fill() -> None:
    """De-compaction places stored timestep k at the k-th present nominal slot."""
    h = w = 4
    present_mask = np.array([True, False, True, True, False, False])
    stored = np.stack(
        [np.full((h, w), fill_value=i + 1, dtype=np.uint8) for i in range(3)], axis=-1
    )

    aligned = align_cloud_to_nominal(stored, present_mask, MAX_T)

    assert aligned.shape == (h, w, MAX_T)
    assert aligned.dtype == np.uint8
    for k, t in enumerate(np.where(present_mask)[0]):
        assert np.array_equal(aligned[:, :, t], stored[:, :, k])
    for t in np.where(~present_mask)[0]:
        assert np.all(aligned[:, :, t] == NODATA)


def test_align_cloud_to_nominal_handles_fewer_stored_than_present() -> None:
    """Extra present slots stay NODATA rather than raising or wrapping."""
    stored = np.zeros((2, 2, 1), dtype=np.uint8)
    aligned = align_cloud_to_nominal(stored, np.array([True, True, False]), 3)
    assert np.array_equal(aligned[:, :, 0], stored[:, :, 0])
    assert np.all(aligned[:, :, 1:] == NODATA)


def test_cloud_fields_excluded_from_modalities() -> None:
    """Cloud is a side-payload: never tokenized, never normalized."""
    sample = _sample(cloud_s2=np.zeros((8, 8, 2, 1), dtype=np.uint8))
    assert "sentinel2_l2a_cloud" not in sample.modalities
    assert "sentinel2_l2a_cloud" not in sample.modalities_with_timestamps
    assert "sentinel2_l2a" in sample.modalities


def test_from_olmoearthsample_ignores_cloud_fields() -> None:
    """Every eval builds masked samples this way; cloud fields must not leak in.

    ``OlmoEarthSample`` declares ``*_cloud`` for all samples (None when unused), so
    a loop over ``as_dict(include_nones=True)`` sees them even for eval datasets
    that know nothing about clouds -- and ``MaskedOlmoEarthSample`` has no such
    field.
    """
    for cloud in (None, np.zeros((8, 8, 2, 1), dtype=np.uint8)):
        sample = _sample(cloud_s2=cloud)
        _, stacked = collate_olmoearth_pretrain([(4, sample)])
        masked = MaskedOlmoEarthSample.from_olmoearthsample(stacked)
        assert masked.sentinel2_l2a_mask is not None
        assert not [f for f in masked._fields if f.endswith("_cloud")]


def test_collate_stacks_cloud_when_every_sample_has_it() -> None:
    """The all-present case: cloud is stacked and handed to the masking strategy."""
    cloud = np.zeros((8, 8, 2, 1), dtype=np.uint8)
    with_cloud = [(4, _sample(cloud_s2=cloud)) for _ in range(3)]
    _, stacked = collate_olmoearth_pretrain(with_cloud)
    assert stacked.sentinel2_l2a_cloud is not None
    assert stacked.sentinel2_l2a_cloud.shape == (3, 8, 8, 2, 1)
    payload = extract_cloud_payload(stacked)
    assert payload is not None
    assert sorted(payload) == ["sentinel2_l2a_cloud"]


def test_collate_fills_missing_cloud_with_nodata_instead_of_dropping_the_batch() -> (
    None
):
    """One sample without a cloud map must not disable masking for the whole batch.

    ~2% of the OSM-sampling set has no Landsat, so under the old all-or-nothing rule
    (drop the field unless EVERY sample has it) Landsat masking was active in only
    ~0.98**64 = 27% of batches at 64 samples per rank -- silently.
    """
    cloud = np.full((8, 8, 2, 1), THICK_CLOUD, dtype=np.uint8)
    mixed = [(4, _sample(cloud_s2=cloud)) for _ in range(2)] + [(4, _sample())]
    _, stacked = collate_olmoearth_pretrain(mixed)

    assert stacked.sentinel2_l2a_cloud is not None, "batch lost cloud to one bad sample"
    assert stacked.sentinel2_l2a_cloud.shape == (3, 8, 8, 2, 1)
    # The two real maps survive verbatim; the absent one becomes NODATA.
    assert (stacked.sentinel2_l2a_cloud[:2] == THICK_CLOUD).all()
    assert (stacked.sentinel2_l2a_cloud[2] == NODATA).all()

    payload = extract_cloud_payload(stacked)
    assert payload is not None and sorted(payload) == ["sentinel2_l2a_cloud"]


def test_nodata_filled_sample_loses_no_decoder_tokens() -> None:
    """The NODATA fill must be inert: that sample is masked exactly as if no cloud."""
    patch_size = 4
    h = w = 16
    cloud = np.full((h, w, 2, 1), THICK_CLOUD, dtype=np.uint8)
    mixed = [(patch_size, _sample(h=h, w=w, t=2, cloud_s2=cloud))] + [
        (patch_size, _sample(h=h, w=w, t=2))
    ]
    _, stacked = collate_olmoearth_pretrain(mixed)
    payload = extract_cloud_payload(stacked)
    assert payload is not None

    strategy = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=[Modality.WORLDCOVER.name]
    )
    torch.manual_seed(0)
    np.random.seed(0)
    baseline = strategy.apply_mask(stacked, patch_size)
    torch.manual_seed(0)
    np.random.seed(0)
    skipped = strategy.apply_mask(stacked, patch_size, cloud=payload)

    base, skip = baseline.sentinel2_l2a_mask, skipped.sentinel2_l2a_mask
    assert base is not None and skip is not None
    # Sample 0 (all thick cloud) loses its decode targets ...
    assert (base[0] != skip[0]).any()
    # ... and sample 1 (NODATA-filled) is untouched.
    assert torch.equal(base[1], skip[1])


def test_transform_moves_cloud_in_lockstep_with_its_modality() -> None:
    """Flip/rotate must apply the same op to cloud, or the skip misaligns."""
    h = w = 8
    # Encode position in both arrays so any geometric mismatch shows up.
    pos = np.arange(h * w, dtype=np.float32).reshape(h, w)
    s2 = np.zeros((h, w, 1, Modality.SENTINEL2_L2A.num_bands), dtype=np.float32)
    s2[:, :, 0, 0] = pos
    cloud = (pos.astype(np.int64) % 4).astype(np.uint8)[:, :, None, None]
    sample = OlmoEarthSample(
        sentinel2_l2a=s2,
        timestamps=_timestamps(1, 1)[0].T.numpy(),
        sentinel2_l2a_cloud=cloud,
    )

    transform = FlipAndRotateSpace()
    for op in transform.transformations:
        transform.transformations = [op]  # force this op
        _, stacked = collate_olmoearth_pretrain([(4, sample)])
        out = transform.apply(stacked)
        out_s2, out_cloud = out.sentinel2_l2a, out.sentinel2_l2a_cloud
        assert out_s2 is not None and out_cloud is not None
        expected = (out_s2[:, :, :, 0, 0].long() % 4).to(torch.uint8)
        assert torch.equal(out_cloud[:, :, :, 0, 0], expected), (
            f"cloud not transformed identically to its modality under {op.__name__}"
        )


def _cloudy_batch(
    b: int = 2, h: int = 16, w: int = 16, t: int = 2, patch_size: int = 4
) -> tuple[OlmoEarthSample, dict[str, torch.Tensor]]:
    """Batch whose S2 cloud map is fully cloudy on the left half, clear on the right."""
    cloud = np.full((h, w, t, 1), CLEAR, dtype=np.uint8)
    cloud[:, : w // 2] = THICK_CLOUD
    cloud[: h // 2, : w // 4] = SHADOW  # shadow also counts as cloud
    batch = [(patch_size, _sample(h=h, w=w, t=t, cloud_s2=cloud)) for _ in range(b)]
    _, stacked = collate_olmoearth_pretrain(batch)
    payload = extract_cloud_payload(stacked)
    assert payload is not None
    return stacked, payload


def _cloudy_ndvi_batch(
    b: int = 2, h: int = 16, w: int = 16, t: int = 2, patch_size: int = 4
) -> tuple[OlmoEarthSample, dict[str, torch.Tensor]]:
    """Like _cloudy_batch, but the sample also carries the derived ndvi decode target.

    Mirrors cand_ndvi: ndvi is a decode-only modality on S2's exact grid, and there is
    no ``ndvi_cloud`` payload -- it must be covered by the S2 cloud map.
    """
    cloud = np.full((h, w, t, 1), CLEAR, dtype=np.uint8)
    cloud[:, : w // 2] = THICK_CLOUD
    batch = []
    for _ in range(b):
        sample = _sample(h=h, w=w, t=t, cloud_s2=cloud)
        sample = sample._replace(
            ndvi=np.ones((h, w, t, Modality.NDVI.num_bands), dtype=np.float32)
        )
        batch.append((patch_size, sample))
    _, stacked = collate_olmoearth_pretrain(batch)
    payload = extract_cloud_payload(stacked)
    assert payload is not None
    assert "ndvi_cloud" not in payload, "ndvi must borrow the S2 cloud map, not its own"
    return stacked, payload


def test_cloud_skip_covers_the_s2_derived_ndvi_decode_target() -> None:
    """Ndvi DECODER tokens over cloud become MISSING, driven by the S2 cloud map.

    Without this the NDVI supervision head is asked to predict a vegetation index
    through cloud, which is exactly what the cloud skip exists to prevent.
    """
    patch_size = 4
    stacked, cloud = _cloudy_ndvi_batch(patch_size=patch_size)
    strategy = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=[Modality.WORLDCOVER.name, Modality.NDVI.name]
    )

    torch.manual_seed(0)
    np.random.seed(0)
    baseline = strategy.apply_mask(stacked, patch_size)
    torch.manual_seed(0)
    np.random.seed(0)
    skipped = strategy.apply_mask(stacked, patch_size, cloud=cloud)

    base_mask, skip_mask = baseline.ndvi_mask, skipped.ndvi_mask
    assert base_mask is not None and skip_mask is not None

    changed = base_mask != skip_mask
    assert changed.any(), "no ndvi token was dropped, so the test proves nothing"

    # Only DECODER -> MISSING, and only over the cloudy half.
    assert torch.equal(
        changed,
        (base_mask == MaskValue.DECODER.value) & (skip_mask == MaskValue.MISSING.value),
    )
    w = base_mask.shape[2]
    assert not changed[:, :, w // 2 :].any(), "clear half must be untouched"

    # Every cloudy-half ndvi DECODER token is gone.
    assert not (skip_mask[:, :, : w // 2] == MaskValue.DECODER.value).any(), (
        "ndvi still has decode targets over cloud"
    )

    # Patch-uniform, so the stride reduction in tokenization sees the drop.
    blocks = skip_mask.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )
    assert bool((blocks.amin(dim=(-1, -2)) == blocks.amax(dim=(-1, -2))).all())


def test_cloud_skip_drops_exactly_the_cloudy_decoder_tokens() -> None:
    """DECODER -> MISSING on mostly-cloud tokens only; nothing else changes."""
    patch_size = 4
    stacked, cloud = _cloudy_batch(patch_size=patch_size)
    strategy = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=[Modality.WORLDCOVER.name]
    )

    torch.manual_seed(0)
    np.random.seed(0)
    baseline = strategy.apply_mask(stacked, patch_size)
    torch.manual_seed(0)
    np.random.seed(0)
    skipped = strategy.apply_mask(stacked, patch_size, cloud=cloud)

    base_mask = baseline.sentinel2_l2a_mask
    skip_mask = skipped.sentinel2_l2a_mask
    assert base_mask is not None and skip_mask is not None

    # Cloud must not leak into the masked sample.
    assert not [
        f
        for f in skipped._fields
        if f.endswith("_cloud") and getattr(skipped, f, None) is not None
    ]

    # Only DECODER -> MISSING transitions, and only on the cloudy half.
    changed = base_mask != skip_mask
    assert torch.equal(
        changed,
        (base_mask == MaskValue.DECODER.value) & (skip_mask == MaskValue.MISSING.value),
    )
    w = base_mask.shape[2]
    assert not changed[:, :, w // 2 :].any(), "clear half must be untouched"
    assert (base_mask[:, :, : w // 2] == MaskValue.DECODER.value).sum() == (
        skip_mask[:, :, : w // 2] == MaskValue.MISSING.value
    ).sum() - (base_mask[:, :, : w // 2] == MaskValue.MISSING.value).sum()

    # Encoder tokens are never touched (the skip only affects loss targets).
    for value in (MaskValue.ONLINE_ENCODER, MaskValue.TARGET_ENCODER_ONLY):
        assert (base_mask == value.value).sum() == (skip_mask == value.value).sum()

    # Masks must stay patch-uniform so the token-level reduction sees the drop.
    blocks = skip_mask.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )
    assert bool((blocks.amin(dim=(-1, -2)) == blocks.amax(dim=(-1, -2))).all())

    # And something was actually dropped, otherwise the test proves nothing.
    assert changed.any()


def test_cloud_skip_respects_threshold() -> None:
    """A threshold above the cloud fraction of every token drops nothing."""
    patch_size = 4
    stacked, cloud = _cloudy_batch(patch_size=patch_size)
    only_decode = [Modality.WORLDCOVER.name]

    torch.manual_seed(0)
    np.random.seed(0)
    lenient = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=only_decode, cloud_skip_threshold=1.0
    ).apply_mask(stacked, patch_size, cloud=cloud)
    torch.manual_seed(0)
    np.random.seed(0)
    strict = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=only_decode, cloud_skip_threshold=0.0
    ).apply_mask(stacked, patch_size, cloud=cloud)

    lenient_mask, strict_mask = lenient.sentinel2_l2a_mask, strict.sentinel2_l2a_mask
    assert lenient_mask is not None and strict_mask is not None
    # frac > 1.0 is never true -> no drops; frac > 0.0 -> every cloudy token drops.
    assert int((lenient_mask == MaskValue.MISSING.value).sum()) == 0
    assert int((strict_mask == MaskValue.MISSING.value).sum()) > 0


def test_cloud_skip_ignores_nodata_and_clear() -> None:
    """NODATA (255) is not cloud, so those decoder tokens survive."""
    patch_size, h, w, t = 4, 8, 8, 1
    cloud = np.full((h, w, t, 1), NODATA, dtype=np.uint8)
    _, stacked = collate_olmoearth_pretrain(
        [(patch_size, _sample(h=h, w=w, t=t, cloud_s2=cloud)) for _ in range(2)]
    )
    payload = extract_cloud_payload(stacked)
    strategy = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=[Modality.WORLDCOVER.name]
    )

    torch.manual_seed(0)
    np.random.seed(0)
    baseline = strategy.apply_mask(stacked, patch_size)
    torch.manual_seed(0)
    np.random.seed(0)
    skipped = strategy.apply_mask(stacked, patch_size, cloud=payload)
    assert torch.equal(baseline.sentinel2_l2a_mask, skipped.sentinel2_l2a_mask)


def test_single_masked_collate_passes_cloud_through_transform() -> None:
    """The full collate path runs end to end and keeps cloud off the model input."""
    patch_size, h, w, t = 4, 16, 16, 2
    cloud = np.full((h, w, t, 1), THICK_CLOUD, dtype=np.uint8)
    batch = [(patch_size, _sample(h=h, w=w, t=t, cloud_s2=cloud)) for _ in range(2)]
    strategy = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=[Modality.WORLDCOVER.name]
    )
    _, masked = collate_single_masked_batched(batch, FlipAndRotateSpace(), strategy)
    assert masked.sentinel2_l2a_mask is not None
    # Every S2 pixel is thick cloud -> no S2 decoder token may survive.
    assert not (masked.sentinel2_l2a_mask == MaskValue.DECODER.value).any()
    assert not [
        f
        for f in masked._fields
        if f.endswith("_cloud") and getattr(masked, f, None) is not None
    ]


def test_missing_pixels_stay_missing_after_cloud_skip() -> None:
    """A modality-missing timestep is unaffected by the cloud skip."""
    patch_size, h, w, t = 4, 8, 8, 2
    sample = _sample(h=h, w=w, t=t, cloud_s2=np.full((h, w, t, 1), CLEAR, np.uint8))
    assert sample.sentinel2_l2a is not None
    s2 = np.asarray(sample.sentinel2_l2a).copy()
    s2[:, :, 1, :] = MISSING_VALUE
    sample = sample._replace(sentinel2_l2a=s2)
    _, stacked = collate_olmoearth_pretrain([(patch_size, sample) for _ in range(2)])
    payload = extract_cloud_payload(stacked)
    strategy = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=[Modality.WORLDCOVER.name]
    )
    torch.manual_seed(0)
    np.random.seed(0)
    masked = strategy.apply_mask(stacked, patch_size, cloud=payload)
    mask = masked.sentinel2_l2a_mask
    assert mask is not None
    assert (mask[:, :, :, 1] == MaskValue.MISSING.value).all()
