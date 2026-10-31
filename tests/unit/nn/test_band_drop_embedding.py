"""Tests for the learned dropped-band embedding (``band_drop_embedding``)."""

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import MultiModalPatchEmbeddings
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig

TOKENIZATION = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[Modality.SENTINEL2_L2A.band_order]
        ),
        "landsat": ModalityTokenization(band_groups=[Modality.LANDSAT.band_order]),
    }
)
S2_BANDS = len(Modality.SENTINEL2_L2A.band_order)


def _embeddings(band_drop_embedding: bool, **kwargs) -> MultiModalPatchEmbeddings:
    torch.manual_seed(0)
    return MultiModalPatchEmbeddings(
        supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat"],
        max_patch_size=8,
        embedding_size=16,
        tokenization_config=TOKENIZATION,
        band_dropout_modalities=["sentinel2_l2a", "landsat"],
        band_drop_embedding=band_drop_embedding,
        **kwargs,
    )


def _sample(b: int = 4, h: int = 8, w: int = 8, t: int = 3) -> MaskedOlmoEarthSample:
    s2 = torch.randn(b, h, w, t, S2_BANDS)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=torch.zeros(b, h, w, t, 1, dtype=torch.long),
        timestamps=torch.tensor([[15, 6, 2023]] * t).expand(b, t, 3),
    )


def test_tables_only_for_band_dropout_modalities() -> None:
    """One zero-initialized table per multi-band band set of the dropout modalities."""
    emb = _embeddings(True)
    assert set(emb.band_drop_embeddings) == {"sentinel2_l2a__0", "landsat__0"}
    assert emb.band_drop_embeddings["sentinel2_l2a__0"].shape == (S2_BANDS, 16)
    assert all((p == 0).all() for p in emb.band_drop_embeddings.values())
    assert len(_embeddings(False).band_drop_embeddings) == 0


def test_zero_init_matches_no_embedding() -> None:
    """At init the model equals the one without the embedding, also with dropped bands."""
    sample = _sample()
    sample.sentinel2_l2a[:, ..., 3] = 0
    a = _embeddings(True).eval()
    b = _embeddings(False).eval()
    torch.testing.assert_close(
        a.apply_embedding_to_modality("sentinel2_l2a", sample, 4)[0],
        b.apply_embedding_to_modality("sentinel2_l2a", sample, 4)[0],
    )


def test_adds_rows_of_all_zero_bands_per_sample() -> None:
    """Sample i gets the sum of the rows of exactly its all-zero bands, at every token."""
    emb = _embeddings(True).eval()
    table = emb.band_drop_embeddings["sentinel2_l2a__0"]
    with torch.no_grad():
        table.copy_(torch.randn_like(table))
    sample = _sample()
    sample.sentinel2_l2a[0, ..., [2, 5]] = 0  # sample 0: bands 2 and 5 zero everywhere
    sample.sentinel2_l2a[1, :, :, 0, 7] = 0  # sample 1: band 7 zero at one timestep only
    base = _embeddings(False).eval()
    diff = (
        emb.apply_embedding_to_modality("sentinel2_l2a", sample, 4)[0]
        - base.apply_embedding_to_modality("sentinel2_l2a", sample, 4)[0]
    )
    expected0 = (table[2] + table[5]).detach()
    torch.testing.assert_close(diff[0], expected0.expand_as(diff[0]))
    assert diff[1:].abs().max() < 1e-6


def test_training_band_dropout_flags_dropped_bands_and_gets_grad() -> None:
    """In training the bands zeroed by band dropout are flagged; only their rows get grads."""
    emb = _embeddings(True, band_dropout_rate=0.5).train()
    sample = _sample(b=16)
    tokens, _ = emb.apply_embedding_to_modality("sentinel2_l2a", sample, 4)
    tokens.sum().backward()
    grad = emb.band_drop_embeddings["sentinel2_l2a__0"].grad
    assert grad is not None
    # Rate 0.5 over 16 samples x 12 bands: some bands were dropped somewhere.
    assert (grad.abs().sum(-1) > 0).any()


def test_no_dropout_no_zero_bands_is_noop_with_grad_path() -> None:
    """Nothing dropped -> zero gradient, but the parameter is still in the graph."""
    emb = _embeddings(True).train()
    tokens, _ = emb.apply_embedding_to_modality("sentinel2_l2a", _sample(), 4)
    tokens.sum().backward()
    grad = emb.band_drop_embeddings["sentinel2_l2a__0"].grad
    assert grad is not None and (grad == 0).all()
