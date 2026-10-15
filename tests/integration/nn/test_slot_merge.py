"""Per-cell slot merge (``nn/slot_merge.py``) and its use inside the encoder."""

import math

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, PerceiverConfig
from olmoearth_pretrain.nn.slot_merge import SlotMerge, SlotMergeConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
S2_ONE_BANDSET = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
        )
    }
)
T, PX, PS, D = 6, 8, 2, 32


def _merge(**kwargs: int) -> SlotMerge:
    torch.manual_seed(0)
    cfg = SlotMergeConfig(**{"ratio": 3, "min_slots": 4, "iters": 3, **kwargs})
    return cfg.build(D)


def _cells(counts: list[list[int]]) -> tuple[torch.Tensor, ...]:
    """Padded inputs with ``counts[b][c]`` valid tokens in cell c of sample b."""
    B = len(counts)
    L = max(sum(c) for c in counts) + 2  # some padding
    x = torch.randn(B, L, D)
    pos = torch.zeros(B, L, 3)
    valid = torch.zeros(B, L, dtype=torch.bool)
    cell = torch.full((B, L), -1, dtype=torch.long)
    mod = torch.zeros(B, L, dtype=torch.long)
    for b, per_cell in enumerate(counts):
        i = 0
        for c, n in enumerate(per_cell):
            for j in range(n):
                valid[b, i], cell[b, i], mod[b, i] = True, c, j % 2
                pos[b, i] = torch.tensor([float(j // 2) * 30.0, float(c), -float(c)])
                i += 1
    return x, pos, valid, cell, mod


def test_slot_counts_and_bypass() -> None:
    """Cells merge to max(ceil(n/r), min(n, min_slots)); small cells pass through."""
    merge = _merge()
    counts = [[12, 3, 4, 7], [1, 12, 0, 5]]
    x, pos, valid, cell, mod = _cells(counts)
    out, out_pos, out_valid, _ = merge(x, pos, valid, cell, mod)
    for b, per_cell in enumerate(counts):
        expected = sum(max(math.ceil(n / 3), min(n, 4)) for n in per_cell)
        assert int(out_valid[b].sum()) == expected
        # bypassed tokens (cells with n <= 4) come through unchanged
        for c, n in enumerate(per_cell):
            if 0 < n <= 4:
                src = x[b][valid[b] & (cell[b] == c)]
                got = out[b][out_valid[b] & (out_pos[b, :, 1] == c)]
                torch.testing.assert_close(got, src)


def test_slot_positions_sit_in_their_cell_and_time_range() -> None:
    """A slot keeps its cell's row/col and a time inside the cell's time span."""
    merge = _merge()
    x, pos, valid, cell, mod = _cells([[12, 9]])
    out, out_pos, out_valid, _ = merge(x, pos, valid, cell, mod)
    for c in (0, 1):
        slot_rows = out_pos[0][out_valid[0] & (out_pos[0, :, 1] == c)]
        times = pos[0][valid[0] & (cell[0] == c), 0]
        assert (
            slot_rows.shape[0] == math.ceil([12, 9][c] / 3) or slot_rows.shape[0] == 4
        )
        assert torch.all(slot_rows[:, 2] == -float(c))
        assert torch.all(slot_rows[:, 0] >= times.min() - 1e-4)
        assert torch.all(slot_rows[:, 0] <= times.max() + 1e-4)


def test_unmerge_returns_to_the_input_layout() -> None:
    """Bypassed tokens get their own output; merged ones a convex mix of their slots."""
    merge = _merge()
    counts = [[12, 3, 7], [2, 12, 0]]
    x, pos, valid, cell, mod = _cells(counts)
    out, out_pos, out_valid, unmerge = merge(x, pos, valid, cell, mod)
    back = unmerge(out)
    assert back.shape == x.shape
    assert torch.all(back[~valid] == 0)
    for b, per_cell in enumerate(counts):
        for c, n in enumerate(per_cell):
            sel = valid[b] & (cell[b] == c)
            if 0 < n <= 4:  # bypass: identity
                torch.testing.assert_close(back[b][sel], x[b][sel])
    # every slot output = the same vector -> every merged token gets exactly it
    const = torch.randn(D)
    y = torch.where(out_valid[..., None], const.expand_as(out), torch.zeros_like(out))
    back = unmerge(y)
    for b, per_cell in enumerate(counts):
        for c, n in enumerate(per_cell):
            if n > 4:
                sel = valid[b] & (cell[b] == c)
                torch.testing.assert_close(
                    back[b][sel], const.expand(int(sel.sum()), D), atol=1e-5, rtol=1e-5
                )


def test_every_parameter_gets_a_gradient_even_without_merging() -> None:
    """DDP needs grads for all parameters; a batch with only small cells must not drop them."""
    for counts in ([[12, 2]], [[2, 3]]):
        merge = _merge()
        x, pos, valid, cell, mod = _cells(counts)
        x.requires_grad_(True)
        out, _, out_valid, _ = merge(x, pos, valid, cell, mod)
        out[out_valid].sum().backward()
        for name, p in merge.named_parameters():
            assert p.grad is not None, (counts, name)


def _encoder(slot_merge: bool) -> Encoder:
    torch.manual_seed(0)
    return EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=D,
        num_heads=2,
        depth=3,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        tokenization_config=S2_ONE_BANDSET,
        perceiver_config=PerceiverConfig(
            register_dim=16,
            latent_depth=1,
            pixel_latents=True,
            random_latent_stride=True,
            max_latents=64,
            eval_latent_stride=1,
            student_dims=[8],
            student_output_norm=True,
        ),
        slot_merge_config=SlotMergeConfig(merge_after=1) if slot_merge else None,
    ).build()


def _sample(B: int = 2, encode_frac: float = 0.6) -> MaskedOlmoEarthSample:
    """S2 + S1 with a per-token (cell, timestep) encoder mask, varying per cell."""
    g = torch.Generator().manual_seed(1)
    fields: dict = {}
    n = PX // PS
    for name in MODALITIES:
        bands = Modality.get(name).num_bands
        tok = torch.rand(B, n, n, T, generator=g) < encode_frac
        tok_mask = torch.where(
            tok, MaskValue.ONLINE_ENCODER.value, MaskValue.DECODER.value
        )
        pix = tok_mask.repeat_interleave(PS, 1).repeat_interleave(PS, 2)
        fields[name] = torch.randn(B, PX, PX, T, bands, generator=g)
        fields[f"{name}_mask"] = pix[..., None].expand(-1, -1, -1, -1, bands).clone()
    fields["timestamps"] = torch.tensor([[[1, t, 2020] for t in range(T)]]).expand(
        B, -1, -1
    )
    return MaskedOlmoEarthSample(**fields)


def test_encoder_with_slot_merge_trains_and_infers() -> None:
    """Train-mode forward/backward with ragged visible cells; eval and fast_pass run."""
    enc = _encoder(slot_merge=True).train()
    base = _encoder(slot_merge=False).train()
    sample = _sample()
    out = enc(sample, patch_size=PS)
    ref = base(sample, patch_size=PS)
    assert out["registers"].shape == ref["registers"].shape
    loss = out["registers"].float().pow(2).mean()
    loss.backward()
    assert enc.slot_merge is not None
    for name, p in enc.slot_merge.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # per-token outputs keep the input layout (the decoder reads their masks)
    assert (
        out["tokens_and_masks"].sentinel2_l2a.shape
        == ref["tokens_and_masks"].sentinel2_l2a.shape
    )
    enc.eval()
    full = _sample(encode_frac=1.1)
    with torch.no_grad():
        a = enc(full, patch_size=PS)
        b = enc(full, patch_size=PS, fast_pass=True)
    torch.testing.assert_close(a["registers"], b["registers"], atol=1e-5, rtol=1e-5)


def test_config_dict_is_coerced() -> None:
    """A checkpoint config stores the merge as a dict; EncoderConfig rebuilds it."""
    cfg = SlotMergeConfig(merge_after=2, ratio=6, min_slots=3, iters=1)
    enc_cfg = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=D,
        num_heads=2,
        depth=3,
        position_encoding="rope_3d_mixed",
        perceiver_config=PerceiverConfig(register_dim=16, latent_depth=1),
        slot_merge_config=cfg.as_config_dict(),  # type: ignore[arg-type]
    )
    assert enc_cfg.slot_merge_config == cfg
    assert enc_cfg.build().slot_merge is not None
