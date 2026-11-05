"""Training-time neighborhood attention (``nn/neighborhood_attention.py``)."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import (
    AttentionPoolerConfig,
    Encoder,
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.neighborhood_attention import dense_neighborhood_mask
from olmoearth_pretrain.nn.searchlight import SearchlightSettings

T = 3
MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
W = 5


def _sample(
    size: int,
    batch: int = 1,
    missing_t: int | None = None,
    missing_in: tuple[str, ...] = (),
    decode_frac: float = 0.0,
    seed: int = 1234,
) -> MaskedOlmoEarthSample:
    """``size`` px domains; ``decode_frac`` of the pixels' tokens are masked (DECODER)."""
    g = torch.Generator().manual_seed(seed)
    fields: dict[str, torch.Tensor] = {}
    for name in MODALITIES:
        bands = Modality.get(name).num_bands
        mask = torch.full(
            (batch, size, size, T, bands),
            MaskValue.ONLINE_ENCODER.value,
            dtype=torch.long,
        )
        if decode_frac > 0:
            # Mask whole (pixel, timestep) entries so every band of a token agrees.
            hide = torch.rand(batch, size, size, T, 1, generator=g) < decode_frac
            mask = torch.where(hide, MaskValue.DECODER.value, mask).expand_as(mask)
            mask = mask.clone()
        if missing_t is not None and name in missing_in:
            mask[:, :, :, missing_t] = MaskValue.MISSING.value
        fields[name] = torch.randn(batch, size, size, T, bands, generator=g)
        fields[f"{name}_mask"] = mask
    fields["timestamps"] = torch.tensor([[[1, t, 2020] for t in range(T)]] * batch)
    return MaskedOlmoEarthSample(**fields)


def _encoder(na: int | None, depth: int = 2, pooler_depth: int = 1) -> Encoder:
    """A small v1.3-shaped encoder: 3D mixed RoPE ViT + per-depth-read AttentionPooler."""
    torch.manual_seed(0)
    return (
        EncoderConfig(
            supported_modality_names=MODALITIES,
            embedding_size=32,
            num_heads=2,
            depth=depth,
            mlp_ratio=2.0,
            max_patch_size=8,
            min_patch_size=1,
            max_sequence_length=12,
            drop_path=0.0,
            position_encoding="rope_3d_mixed",
            attention_pooler_config=AttentionPoolerConfig(
                pooler_dim=16,
                pooler_depth=pooler_depth,
                attn_dim=32,
                per_depth_read_proj=True,
                student_dims=[8],
                student_output_norm=True,
            ),
            neighborhood_attention_cells=na,
        )
        .build()
        .eval()
    )


def _assert_same(a: dict, b: dict, atol: float = 1e-5) -> None:
    for key in ("location_tokens", "student_location_tokens"):
        torch.testing.assert_close(a[key], b[key], atol=atol, rtol=atol)
    for name in MODALITIES:
        torch.testing.assert_close(
            getattr(a["tokens_and_masks"], name),
            getattr(b["tokens_and_masks"], name),
            atol=atol,
            rtol=atol,
        )


def test_dense_mask_is_the_shifted_box() -> None:
    """Odd box, shifted inward at the edges; padding keys never attended."""
    h, w = 7, 9
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cells = torch.stack([rows, cols], -1).reshape(1, -1, 2).float()
    valid = torch.ones(1, h * w, dtype=torch.bool)
    valid[0, 5] = False
    allowed = dense_neighborhood_mask(cells, cells, valid, (h, w), W)[0, 0]
    allowed = allowed.view(h, w, h, w)
    start = lambda i, n: min(max(i - 2, 0), n - W)  # noqa: E731
    for r in range(h):
        for c in range(w):
            box = torch.zeros(h, w, dtype=torch.bool)
            box[start(r, h) : start(r, h) + W, start(c, w) : start(c, w) + W] = True
            box.view(-1)[5] = False
            assert torch.equal(allowed[r, c], box), (r, c)
    # Rows 2..4 of a 7-row axis see exactly 2 cells on each side.
    assert start(3, h) == 1


@pytest.mark.parametrize(
    ("patch_size", "location_query_patch_size", "size"),
    [(2, 1, 8), (2, None, 10), (4, 2, 16), (1, None, 5)],
)
@pytest.mark.parametrize("decode_frac", [0.0, 0.5])
def test_grid_within_the_box_is_full_attention(
    patch_size: int,
    location_query_patch_size: int | None,
    size: int,
    decode_frac: float,
) -> None:
    """A grid of at most W cells: every box is the whole grid, so NA == stock."""
    sample = _sample(size, batch=2, decode_frac=decode_frac)
    kwargs = dict(
        patch_size=patch_size,
        input_res=10,
        location_query_patch_size=location_query_patch_size,
    )
    with torch.no_grad():
        _assert_same(_encoder(W)(sample, **kwargs), _encoder(None)(sample, **kwargs))


@pytest.mark.parametrize(
    "missing_in", [(), (Modality.SENTINEL1.name,)], ids=["none", "s1_missing_t"]
)
@pytest.mark.parametrize(
    ("patch_size", "location_query_patch_size"), [(2, 1), (2, None), (1, None)]
)
def test_matches_searchlight(
    patch_size: int, location_query_patch_size: int | None, missing_in: tuple
) -> None:
    """Fully visible domain wider than the box: training NA == Searchlight inference."""
    encoder = _encoder(W)
    sample = _sample(12 * patch_size, missing_t=1, missing_in=missing_in)
    kwargs = dict(
        patch_size=patch_size,
        input_res=10,
        location_query_patch_size=location_query_patch_size,
    )
    with torch.no_grad():
        na = encoder(sample, **kwargs)
        sl = encoder(
            sample,
            **kwargs,
            searchlight=SearchlightSettings(
                neighborhood_attention_size_px=W * patch_size
            ),
        )
        full = _encoder(None)(sample, **kwargs)
    _assert_same(na, sl)
    # ... and NA really is active on this domain.
    assert not torch.allclose(na["location_tokens"], full["location_tokens"], atol=1e-3)


def test_outputs_are_local() -> None:
    """Inputs beyond the receptive field cannot change an output.

    Depth 1 + a 1-layer pooler: a location query at cell 0 depends on input cells
    <= 4 (query self-attn) + 2 (read) + 2 (ViT) = 8 at most; perturb cell 13.
    """
    encoder = _encoder(W, depth=1, pooler_depth=1)
    a = _sample(14)
    b = _sample(14)
    b.sentinel2_l2a[:, 13] += 5.0
    b.sentinel1[:, :, 13] += 5.0
    kwargs = dict(patch_size=1, input_res=10)
    with torch.no_grad():
        out_a, out_b = encoder(a, **kwargs), encoder(b, **kwargs)
    la, lb = out_a["location_tokens"], out_b["location_tokens"]
    torch.testing.assert_close(la[:, :4, :4], lb[:, :4, :4], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(la[:, 13], lb[:, 13], atol=1e-3)


def test_padding_does_not_leak_across_the_batch() -> None:
    """Ragged visible counts (MIM masking): a sample's output ignores its neighbours."""
    encoder = _encoder(W)
    batch = _sample(12, batch=2, decode_frac=0.5, seed=7)
    solo = _sample(12, batch=2, decode_frac=0.5, seed=7)
    # Sample 1 has a different visible count, so sample 0 is padded in the batch.
    solo_fields = {k: v[:1] for k, v in solo.as_dict().items() if v is not None}
    kwargs = dict(patch_size=1, input_res=10)
    with torch.no_grad():
        out_batch = encoder(batch, **kwargs)
        out_solo = encoder(MaskedOlmoEarthSample(**solo_fields), **kwargs)
    torch.testing.assert_close(
        out_batch["location_tokens"][:1],
        out_solo["location_tokens"],
        atol=1e-5,
        rtol=1e-5,
    )


def _latent_mim(na: int | None):
    torch.manual_seed(0)
    encoder_config = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=8,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        attention_pooler_config=AttentionPoolerConfig(
            pooler_dim=16, pooler_depth=1, attn_dim=32, per_depth_read_proj=True
        ),
        neighborhood_attention_cells=na,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=MODALITIES,
        encoder_embedding_size=32,
        decoder_embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope_mixed",
        use_attention_pooler=True,
        pooler_dim=16,
        neighborhood_attention_cells=na,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config, decoder_config=decoder_config
    ).build()


def test_decoder_within_the_box_is_full_attention() -> None:
    """End-to-end LatentMIM forward: NA == full attention on a grid <= W."""
    sample = _sample(8, batch=2, decode_frac=0.5)
    with torch.no_grad():
        out_na = _latent_mim(W).eval().forward(sample, 2)
        out_full = _latent_mim(None).eval().forward(sample, 2)
    for name in MODALITIES:
        torch.testing.assert_close(
            getattr(out_na[1], name), getattr(out_full[1], name), atol=1e-5, rtol=1e-5
        )


def test_latent_mim_backward_on_a_large_grid() -> None:
    """Forward + backward with NA active everywhere (grid 12 > W)."""
    model = _latent_mim(W).train()
    sample = _sample(24, batch=2, decode_frac=0.5)
    _, decoded, *_ = model.forward(sample, 2)
    loss = sum(
        getattr(decoded, name).float().pow(2).mean()
        for name in MODALITIES
        if getattr(decoded, name) is not None
    )
    loss.backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
