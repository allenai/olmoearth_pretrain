"""Tests for RC Lighthouse inference (``nn/lighthouse_rc.py``), on the dense CPU path."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder, PerceiverConfig
from olmoearth_pretrain.nn.lighthouse_rc import (
    RCLighthouseSettings,
    _block_tables,
    _codes,
    _fov_start,
    _rule,
    _slot_layout,
    lighthouse_rc_reach_px,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

VIT_DEPTH = 2
LATENT_DEPTH = 2


def _encoder(pixel_latents: bool) -> Encoder:
    torch.manual_seed(0)
    return (
        Encoder(
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.SENTINEL1],
            embedding_size=32,
            max_patch_size=4,
            min_patch_size=1,
            num_heads=4,
            mlp_ratio=2.0,
            max_sequence_length=12,
            depth=VIT_DEPTH,
            drop_path=0.0,
            position_encoding="rope_3d_mixed",
            perceiver_config=PerceiverConfig(
                register_dim=32,
                latent_depth=LATENT_DEPTH,
                per_depth_read_proj=True,
                attn_dim=32,
                pixel_latents=pixel_latents or None,
                eval_latent_stride=1 if pixel_latents else None,
                student_dims=[8],
                student_output_norm=True,
            ),
        )
        .double()
        .eval()
    )


def _sample(
    H: int, W: int, T: int = 3, seed: int = 1, missing: bool = False
) -> MaskedOlmoEarthSample:
    g = torch.Generator().manual_seed(seed)
    nb2 = Modality.SENTINEL2_L2A.num_bands
    nb1 = Modality.SENTINEL1.num_bands
    s1_mask = torch.full((1, H, W, T, nb1), MaskValue.ONLINE_ENCODER.value)
    if missing:
        # One S1 timestep missing over part of the domain, one pixel entirely.
        s1_mask[:, : H // 2, :, 1] = MaskValue.MISSING.value
        s1_mask[:, H - 1, W - 1] = MaskValue.MISSING.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(1, H, W, T, nb2, generator=g, dtype=torch.float64),
        sentinel2_l2a_mask=torch.full(
            (1, H, W, T, nb2), MaskValue.ONLINE_ENCODER.value
        ),
        sentinel1=torch.randn(1, H, W, T, nb1, generator=g, dtype=torch.float64),
        sentinel1_mask=s1_mask,
        timestamps=torch.tensor([[[1, 0, 2020], [1, 3, 2020], [1, 6, 2020]]]).long(),
    )


def _crop(
    sample: MaskedOlmoEarthSample, r0: int, r1: int, c0: int, c1: int
) -> MaskedOlmoEarthSample:
    out = {}
    for k, v in sample.as_dict().items():
        if v is None:
            continue
        out[k] = v if k == "timestamps" else v[:, r0:r1, c0:c1]
    return MaskedOlmoEarthSample(**out)


def _run(
    encoder: Encoder,
    sample: MaskedOlmoEarthSample,
    patch_size: int,
    fov_px: int | None,
    origin: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    encoder.lighthouse = (
        RCLighthouseSettings(
            fov_px=fov_px, dense=True, mlp_chunk=37, q_chunk=97, origin_px=origin
        )
        if fov_px is not None
        else None
    )
    try:
        with torch.no_grad():
            # fast_pass=False with batch 1: the stock path drops MISSING tokens.
            out = encoder(sample, patch_size=patch_size, input_res=10, fast_pass=False)
    finally:
        encoder.lighthouse = None
    return out["student_registers"][0]


@pytest.mark.parametrize("pixel_latents", [False, True])
@pytest.mark.parametrize("patch_size", [1, 2, 4])
@pytest.mark.parametrize("missing", [False, True])
def test_one_window_domain_reproduces_the_stock_forward(
    pixel_latents: bool, patch_size: int, missing: bool
) -> None:
    """With the domain exactly one FOV wide, every FOV is the whole domain."""
    encoder = _encoder(pixel_latents)
    sample = _sample(8, 8, missing=missing)
    ref = _run(encoder, sample, patch_size, None)
    out = _run(encoder, sample, patch_size, fov_px=8)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("pixel_latents", [False, True])
@pytest.mark.parametrize("patch_size", [1, 2])
def test_chunk_with_the_reach_as_halo_is_exact(
    pixel_latents: bool, patch_size: int
) -> None:
    """A chunk carrying the receptive-field reach as halo reproduces its core."""
    encoder = _encoder(pixel_latents)
    fov_px = 4 * patch_size
    reach = lighthouse_rc_reach_px(fov_px, patch_size, VIT_DEPTH, LATENT_DEPTH, 4)
    side = 2 * reach + 8
    sample = _sample(side + 4, side + 6, missing=True)
    full = _run(encoder, sample, patch_size, fov_px)
    unit = 1 if pixel_latents else patch_size  # output pixels per embedding
    core = (reach + 2 * patch_size, reach + 6 * patch_size)

    def window(lo: int, hi: int) -> torch.Tensor:
        out = _run(
            encoder, _crop(sample, lo, hi, lo, hi), patch_size, fov_px, origin=(lo, lo)
        )
        c = slice((core[0] - lo) // unit, (core[1] - lo) // unit)
        return out[c, c]

    ref = full[core[0] // unit : core[1] // unit, core[0] // unit : core[1] // unit]
    torch.testing.assert_close(
        window(core[0] - reach, core[1] + reach), ref, atol=1e-6, rtol=1e-6
    )
    # A halo short of the reach is NOT exact.
    short = window(core[0] - 2 * patch_size, core[1] + 2 * patch_size)
    assert not torch.allclose(short, ref, atol=1e-4)


def _brute_rule(
    q_rows: np.ndarray,
    q_cols: np.ndarray,
    k_rows: np.ndarray,
    k_cols: np.ndarray,
    fov: int,
    n_h: int,
    n_w: int,
    quantum: int = 1,
) -> np.ndarray:
    r0 = _fov_start(q_rows, fov, n_h, quantum)[:, None]
    c0 = _fov_start(q_cols, fov, n_w, quantum)[:, None]
    return (
        (k_rows[None] >= r0)
        & (k_rows[None] < r0 + fov)
        & (k_cols[None] >= c0)
        & (k_cols[None] < c0 + fov)
    )


@pytest.mark.parametrize(
    ("n_h", "n_w", "fov", "per_cell", "q_tile", "block", "quantum", "drop"),
    [
        (13, 10, 4, 5, (1, 10), 16, 1, 0.2),  # tokens x tokens
        (9, 12, 4, 7, (2, 3), 16, 1, 0.2),  # tiled latents x tokens
        (20, 20, 8, 3, (1, 20), 32, 1, 0.2),
        (17, 23, 8, 1, (4, 8), 32, 1, 0.2),  # one latent per cell
        (18, 22, 8, 2, (4, 4), 32, 4, 0.2),  # quantized FOV, missing tokens
        (16, 24, 8, 2, (4, 4), 32, 4, 0.0),  # quantized, complete: all blocks full
        (17, 21, 8, 1, (4, 8), 16, 2, 0.0),  # quantized latents x tokens
    ],
)
def test_block_tables_are_exact(
    n_h: int,
    n_w: int,
    fov: int,
    per_cell: int,
    q_tile: tuple[int, int],
    block: int,
    quantum: int,
    drop: float,
) -> None:
    """Listed blocks cover every allowed pair; full blocks hold only allowed pairs."""
    rng = np.random.default_rng(0)
    cells = np.repeat(np.arange(n_h * n_w), per_cell)
    cells = cells[rng.random(cells.size) >= drop]  # missing tokens
    cells = cells[rng.permutation(cells.size)]
    k_tile = (1, n_w) if quantum == 1 else (quantum, quantum)
    k = _slot_layout(cells // n_w, cells % n_w, n_w, k_tile, block)
    lat = np.repeat(np.arange(n_h * n_w), per_cell if q_tile == k_tile else 1)
    q = _slot_layout(lat // n_w, lat % n_w, n_w, q_tile, block)
    q.quantum = quantum
    tab = _block_tables(q, k, fov, n_h, n_w, block)

    allowed = np.zeros((q.length, k.length), dtype=bool)
    qv, kv = q.valid, k.valid
    allowed[np.ix_(qv, kv)] = _brute_rule(
        q.row[qv], q.col[qv], k.row[kv], k.col[kv], fov, n_h, n_w, quantum
    )
    nq, nk = q.length // block, k.length // block
    listed = np.zeros((nq, nk), dtype=bool)
    full = np.zeros((nq, nk), dtype=bool)
    for qb in range(nq):
        listed[qb, tab["part_idx"][qb, : tab["part_num"][qb]]] = True
        full[qb, tab["full_idx"][qb, : tab["full_num"][qb]]] = True
    assert not (listed & full).any()
    covered = np.kron(listed | full, np.ones((block, block), dtype=bool))
    assert not (allowed & ~covered).any()
    full_px = np.kron(full, np.ones((block, block), dtype=bool))
    assert allowed[full_px & qv[:, None]].all()
    if quantum > 1 and drop == 0 and (n_h % quantum, n_w % quantum) == (0, 0):
        # Whole quantum tiles, no padding: nothing needs the mask.
        assert not listed.any()

    # The packed rule the kernel evaluates equals the brute-force rule.
    q_code, k_code = _codes(q, k, fov, n_h, n_w, torch.device("cpu"))
    rule = _rule(fov, q_code[:, None], k_code[None, :]).numpy()
    assert np.array_equal(rule[qv], allowed[qv])
    # Every valid query sees exactly the keys a training window would show it.
    counts = allowed.sum(1)[qv]
    window = np.zeros((n_h, n_w), dtype=np.int64)
    np.add.at(window, (k.row[kv], k.col[kv]), 1)
    r0 = _fov_start(q.row[qv], fov, n_h, quantum)
    c0 = _fov_start(q.col[qv], fov, n_w, quantum)
    expect = [window[a : a + fov, b : b + fov].sum() for a, b in zip(r0, c0)]
    assert np.array_equal(counts, expect)
    # The query's own cell is always inside its FOV, within the quantum of centre.
    assert ((q.row[qv] >= r0) & (q.row[qv] < r0 + fov)).all()


@pytest.mark.parametrize("pixel_latents", [False, True])
def test_quantum_one_window_is_still_exact(pixel_latents: bool) -> None:
    """A one-window domain has one FOV, so any quantum reproduces the stock forward."""
    encoder = _encoder(pixel_latents)
    sample = _sample(8, 8, missing=True)
    ref = _run(encoder, sample, 1, None)
    encoder.lighthouse = RCLighthouseSettings(fov_px=8, dense=True, fov_quantum=4)
    with torch.no_grad():
        out = encoder(sample, patch_size=1, input_res=10, fast_pass=False)
    encoder.lighthouse = None
    torch.testing.assert_close(out["student_registers"][0], ref, atol=1e-6, rtol=1e-6)


def test_eval_wrapper_lighthouse_matches_per_sample_forward() -> None:
    """The eval wrapper's Lighthouse path = one Lighthouse forward per sample."""
    from olmoearth_pretrain.evals.datasets.configs import TaskType
    from olmoearth_pretrain.evals.eval_wrapper import OlmoEarthEvalWrapper
    from olmoearth_pretrain.nn.pooling import PoolingType

    encoder = _encoder(pixel_latents=False)
    encoder.use_perceiver = True  # what the wrapper checks on the full model
    samples = [_sample(12, 12, seed=s) for s in (1, 2)]
    batch = MaskedOlmoEarthSample(
        **{
            k: torch.cat([s.as_dict()[k] for s in samples])
            for k, v in samples[0].as_dict().items()
            if v is not None
        }
    )
    wrapper = OlmoEarthEvalWrapper(
        model=encoder,
        task_type=TaskType.SEGMENTATION,
        patch_size=1,
        pooling_type=PoolingType.MEAN,
        eval_on_student_registers=True,
        lighthouse_fov_px=8,
    )
    labels = torch.zeros(2, 12, 12)
    with torch.no_grad():
        emb, _ = wrapper(batch, labels, is_train=False)
    ref = torch.stack([_run(encoder, s, 1, fov_px=8) for s in samples])
    torch.testing.assert_close(emb, ref)
    assert encoder.lighthouse is None


def test_retile_dense_matches_tile_order() -> None:
    """Row-major windows per sample, embeddings and labels cut identically."""
    from olmoearth_pretrain.train.callbacks.evaluator_callback import retile_dense

    emb = torch.arange(2 * 4 * 6 * 3).reshape(2, 4, 6, 3)
    lab = emb[..., 0]
    e, lb = retile_dense(emb, lab, 2)
    assert e.shape == (12, 2, 2, 3) and lb.shape == (12, 2, 2)
    torch.testing.assert_close(e[1], emb[0, 0:2, 2:4])
    torch.testing.assert_close(e[3], emb[0, 2:4, 0:2])
    torch.testing.assert_close(lb, e[..., 0])


@pytest.mark.parametrize(
    ("n_h", "n_w", "fov", "per_cell", "block"), [(13, 10, 4, 5, 16), (20, 23, 8, 3, 32)]
)
def test_column_mask_equals_full_rule_on_listed_blocks(
    n_h: int, n_w: int, fov: int, per_cell: int, block: int
) -> None:
    """On the blocks the tables list, the column-only mask = the full FOV rule."""
    from olmoearth_pretrain.nn.lighthouse_rc import _column_codes

    rng = np.random.default_rng(1)
    cells = np.repeat(np.arange(n_h * n_w), per_cell)
    cells = cells[rng.random(cells.size) > 0.2]
    lay = _slot_layout(cells // n_w, cells % n_w, n_w, (1, n_w), block)
    tab = _block_tables(lay, lay, fov, n_h, n_w, block)
    cols = _column_codes(lay, lay, fov, n_w, block, torch.device("cpu"))
    assert cols is not None
    c0, kcol = cols[0].numpy(), cols[1].numpy()
    q_code, k_code = _codes(lay, lay, fov, n_h, n_w, torch.device("cpu"))
    full_rule = _rule(fov, q_code[:, None], k_code[None, :]).numpy()
    col_rule = (kcol[None, :] >= c0[:, None]) & (kcol[None, :] < c0[:, None] + fov)
    qv = lay.valid
    for qb in range(lay.length // block):
        qs = slice(qb * block, (qb + 1) * block)
        rows_q = qv[qs]
        for kb in tab["part_idx"][qb, : tab["part_num"][qb]]:
            ks = slice(kb * block, (kb + 1) * block)
            assert np.array_equal(col_rule[qs, ks][rows_q], full_rule[qs, ks][rows_q])
        # Every query (padding too) keeps at least one key.
        listed = np.concatenate(
            [
                tab["part_idx"][qb, : tab["part_num"][qb]],
                tab["full_idx"][qb, : tab["full_num"][qb]],
            ]
        )
        keys = np.zeros(lay.length, bool)
        for kb in listed:
            keys[kb * block : (kb + 1) * block] = True
        part_keys = np.zeros(lay.length, bool)
        for kb in tab["part_idx"][qb, : tab["part_num"][qb]]:
            part_keys[kb * block : (kb + 1) * block] = True
        full_keys = keys & ~part_keys
        seen = col_rule[qs][:, part_keys].sum(1) + full_keys.sum()
        assert (seen > 0).all()


def _reference_na3d(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size: tuple
) -> torch.Tensor:
    """NATTEN na3d semantics for kernel (W, W, Z): dense SDPA with the window mask."""
    _, n_h, n_w, z, heads, d = q.shape
    fov = kernel_size[0]
    assert kernel_size == (fov, fov, z)
    idx = torch.arange(n_h * n_w * z)
    cell = idx // z
    r, c = cell // n_w, cell % n_w
    r0 = (r - fov // 2).clamp(0, n_h - fov)
    c0 = (c - fov // 2).clamp(0, n_w - fov)
    mask = (
        (r[None] >= r0[:, None])
        & (r[None] < r0[:, None] + fov)
        & (c[None] >= c0[:, None])
        & (c[None] < c0[:, None] + fov)
    )

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(1, -1, heads, d).transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(
        flat(q), flat(k), flat(v), attn_mask=mask
    )
    return out.transpose(1, 2).reshape(q.shape)


@pytest.mark.parametrize("pixel_latents", [False, True])
@pytest.mark.parametrize("patch_size", [1, 2])
@pytest.mark.parametrize("missing", [False, True])
def test_natten_path_equals_flex_path(
    monkeypatch: pytest.MonkeyPatch, pixel_latents: bool, patch_size: int, missing: bool
) -> None:
    """The NATTEN grid path (with a reference na3d) = the masked path, multi-window.

    ``missing`` drops one S1 timestep over part of the domain (per-pixel counts
    differ), so the ViT must fall back to the masked path there while the latent
    self-attention (always complete) still takes the grid path.
    """
    import olmoearth_pretrain.nn.lighthouse_rc as lrc

    encoder = _encoder(pixel_latents)
    sample = _sample(12 * patch_size, 10 * patch_size, missing=missing)
    fov_px = 4 * patch_size
    ref = _run(encoder, sample, patch_size, fov_px)
    monkeypatch.setattr(lrc, "_natten_na3d", lambda backend, device: _reference_na3d)
    encoder.lighthouse = RCLighthouseSettings(
        fov_px=fov_px, dense=True, attention_backend="natten"
    )
    with torch.no_grad():
        out = encoder(sample, patch_size=patch_size, input_res=10, fast_pass=False)
    stats = encoder.last_lighthouse_stats
    encoder.lighthouse = None
    torch.testing.assert_close(out["student_registers"][0], ref, atol=1e-6, rtol=1e-6)
    assert stats["vit_natten"] == float(not missing)
    assert stats["latent_natten"] == 1.0
