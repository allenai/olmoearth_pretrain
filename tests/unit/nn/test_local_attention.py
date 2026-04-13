"""Unit tests for local attention mask building and encoder integration."""

from typing import Any

import pytest
import torch
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import BlockMask

from olmoearth_pretrain.data.constants import ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import (
    _NON_SPATIAL_SENTINEL,
    Encoder,
    _ModalityBlockMeta,
    build_analytical_block_mask,
    collapse_block_aligned,
    compute_block_aligned_positions,
    expand_block_aligned,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def _cuda_functional() -> bool:
    """Check that CUDA is truly available and the driver works."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(
    not _cuda_functional(), reason="flex_attention requires a functional CUDA device"
)


@pytest.fixture(autouse=True)
def _allow_nondeterministic_cublas() -> Any:
    """flex_attention uses CuBLAS ops that aren't deterministic.

    Temporarily relax the global deterministic mode set by conftest so CUDA
    tests in this module can run.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(prev)


def _to_cuda(sample: MaskedOlmoEarthSample) -> MaskedOlmoEarthSample:
    """Move all tensor fields of a MaskedOlmoEarthSample to CUDA."""
    return MaskedOlmoEarthSample(
        **{
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in sample._asdict().items()
        }
    )


def _eval_mask_mod(block_mask: BlockMask, N: int, B: int = 1) -> torch.Tensor:
    """Evaluate a BlockMask's mask_mod on all element pairs to get full dense mask.

    Returns (N, N) bool tensor if B=1, or (B, N, N) if B>1.
    Tensors are created on the same device as the BlockMask so the mask_mod
    closure (which may capture CUDA tensors) doesn't hit a device mismatch.
    """
    device = block_mask.kv_num_blocks.device
    q_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, N)
    kv_idx = torch.arange(N, device=device).unsqueeze(0).expand(N, N)
    if B == 1:
        b = torch.zeros(N, N, dtype=torch.long, device=device)
        h = torch.zeros(N, N, dtype=torch.long, device=device)
        return block_mask.mask_mod(b, h, q_idx, kv_idx)
    results = []
    for bi in range(B):
        b = torch.full((N, N), bi, dtype=torch.long, device=device)
        h = torch.zeros(N, N, dtype=torch.long, device=device)
        results.append(block_mask.mask_mod(b, h, q_idx, kv_idx))
    return torch.stack(results)


# ---------------------------------------------------------------------------
# Tests for collapse_block_aligned / expand_block_aligned
# ---------------------------------------------------------------------------


class TestCollapseBlockAligned:
    """Tests for spatial-block-aligned collapsing and expanding."""

    def _make_6d_dict(
        self,
        B: int = 1,
        H: int = 4,
        W: int = 4,
        T: int = 1,
        bs: int = 1,
        D: int = 8,
    ) -> dict[str, torch.Tensor]:
        """Build a minimal token dict with unique per-token values."""
        n = B * H * W * T * bs * D
        tok = torch.arange(n, dtype=torch.float32).reshape(B, H, W, T, bs, D)
        msk = torch.full(
            (B, H, W, T, bs), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
        return {"spatial": tok, "spatial_mask": msk}

    def _make_3d_dict(
        self, B: int = 1, N: int = 3, D: int = 8
    ) -> dict[str, torch.Tensor]:
        """Build a non-spatial token dict."""
        tok = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)
        msk = torch.full((B, N), MaskValue.ONLINE_ENCODER.value, dtype=torch.long)
        return {"nonspatial": tok, "nonspatial_mask": msk}

    def test_roundtrip_6d(self) -> None:
        """Collapse then expand should recover the original row-major flat tensor."""
        x = self._make_6d_dict(B=2, H=4, W=4, T=2, bs=3, D=8)
        law = 2
        tokens, validity, meta = collapse_block_aligned(x, law, ["spatial"])
        recovered = expand_block_aligned(tokens, meta)

        original_flat = rearrange(x["spatial"], "b h w t bs d -> b (h w t bs) d")
        assert torch.allclose(recovered, original_flat)

    def test_roundtrip_mixed(self) -> None:
        """Mixed spatial + non-spatial roundtrip."""
        B, D = 1, 4
        spatial = self._make_6d_dict(B=B, H=4, W=4, T=1, bs=1, D=D)
        nonspatial = self._make_3d_dict(B=B, N=3, D=D)
        x = {**spatial, **nonspatial}
        tokens, validity, meta = collapse_block_aligned(
            x, law=2, modalities_to_process=["spatial", "nonspatial"]
        )
        recovered = expand_block_aligned(tokens, meta)

        spatial_flat = rearrange(spatial["spatial"], "b h w t bs d -> b (h w t bs) d")
        nonspatial_flat = nonspatial["nonspatial"]
        expected = torch.cat([spatial_flat, nonspatial_flat], dim=1)
        assert torch.allclose(recovered, expected)

    def test_non_spatial_padding(self) -> None:
        """Non-spatial modalities should be padded to block_size = law^2."""
        x = self._make_3d_dict(B=1, N=3, D=4)
        law = 2
        tokens, validity, meta = collapse_block_aligned(x, law, ["nonspatial"])
        assert tokens.shape[1] == 4  # ceil(3/4)*4 = 4
        assert meta[0].padded_n_tokens == 4
        assert meta[0].original_n_tokens == 3
        # Padding positions should be invalid.
        assert validity[0, 3].item() is False

    def test_spatial_no_padding_needed(self) -> None:
        """Spatial modalities always produce a multiple of law^2 tokens."""
        x = self._make_6d_dict(B=1, H=4, W=4, T=2, bs=1, D=4)
        tokens, _, meta = collapse_block_aligned(
            x, law=2, modalities_to_process=["spatial"]
        )
        assert meta[0].padded_n_tokens == meta[0].original_n_tokens

    def test_validity_mask(self) -> None:
        """Validity should reflect ONLINE_ENCODER mask values."""
        x = self._make_6d_dict(B=1, H=4, W=4, T=2, bs=1, D=4)
        # Mark some tokens as MISSING.
        x["spatial_mask"][0, :, :, 1, :] = MaskValue.MISSING.value
        tokens, validity, meta = collapse_block_aligned(
            x, law=2, modalities_to_process=["spatial"]
        )
        # Half the tokens (T=1 of 2) should be invalid.
        n_valid = validity.sum().item()
        n_total = meta[0].original_n_tokens
        assert n_valid == n_total // 2

    def test_raises_on_bad_dims(self) -> None:
        """Should raise if H or W not divisible by law."""
        tok = torch.randn(1, 5, 4, 1, 1, 8)
        msk = torch.zeros(1, 5, 4, 1, 1, dtype=torch.long)
        x = {"bad": tok, "bad_mask": msk}
        with pytest.raises(ValueError, match="multiples of local_attention_window"):
            collapse_block_aligned(x, law=2, modalities_to_process=["bad"])

    def test_block_alignment_6d(self) -> None:
        """Verify tokens are grouped into law x law spatial blocks."""
        law = 2
        x = self._make_6d_dict(B=1, H=4, W=4, T=1, bs=1, D=8)
        tokens, _, meta = collapse_block_aligned(x, law, ["spatial"])

        rows, cols = compute_block_aligned_positions(meta)
        block_size = law * law
        n_blocks = tokens.shape[1] // block_size

        for blk in range(n_blocks):
            start = blk * block_size
            blk_rows = rows[start : start + block_size]
            blk_cols = cols[start : start + block_size]
            # All tokens in a block should be within a law x law spatial square.
            assert (blk_rows.max() - blk_rows.min()) < law
            assert (blk_cols.max() - blk_cols.min()) < law


# ---------------------------------------------------------------------------
# Tests for compute_block_aligned_positions
# ---------------------------------------------------------------------------


class TestComputeBlockAlignedPositions:
    """Tests for position computation in block-aligned order."""

    def test_simple_4x4(self) -> None:
        """Check (row, col) for a simple 4x4 grid with law=2."""
        meta = [
            _ModalityBlockMeta(
                name="m",
                is_spatial=True,
                original_n_tokens=16,
                padded_n_tokens=16,
                num_bh=2,
                num_bw=2,
                t_x_bs=1,
                t=1,
                bs=1,
                law=2,
                original_shape=(1, 4, 4, 1, 1, 8),
            )
        ]
        rows, cols = compute_block_aligned_positions(meta)
        assert rows.shape == (16,)

        # Block 0: spatial square (bh=0, bw=0) -> rows [0,1], cols [0,1]
        assert rows[:4].tolist() == [0, 0, 1, 1]
        assert cols[:4].tolist() == [0, 1, 0, 1]

        # Block 1: spatial square (bh=0, bw=1) -> rows [0,1], cols [2,3]
        assert rows[4:8].tolist() == [0, 0, 1, 1]
        assert cols[4:8].tolist() == [2, 3, 2, 3]

    def test_non_spatial_gets_sentinel(self) -> None:
        """Non-spatial modality tokens should get _NON_SPATIAL_SENTINEL."""
        meta = [
            _ModalityBlockMeta(
                name="ns",
                is_spatial=False,
                original_n_tokens=3,
                padded_n_tokens=4,
            )
        ]
        rows, cols = compute_block_aligned_positions(meta)
        assert (rows == _NON_SPATIAL_SENTINEL).all()
        assert rows.shape == (4,)

    def test_multiple_timesteps(self) -> None:
        """With T=2, same spatial positions appear in consecutive blocks."""
        meta = [
            _ModalityBlockMeta(
                name="m",
                is_spatial=True,
                original_n_tokens=8,
                padded_n_tokens=8,
                num_bh=1,
                num_bw=1,
                t_x_bs=2,
                t=2,
                bs=1,
                law=2,
                original_shape=(1, 2, 2, 2, 1, 4),
            )
        ]
        rows, cols = compute_block_aligned_positions(meta)
        # Block 0 (t=0): positions (0,0),(0,1),(1,0),(1,1)
        # Block 1 (t=1): same positions
        assert rows[:4].tolist() == rows[4:8].tolist()
        assert cols[:4].tolist() == cols[4:8].tolist()


# ---------------------------------------------------------------------------
# Tests for build_analytical_block_mask
# ---------------------------------------------------------------------------


class TestBuildAnalyticalBlockMask:
    """Tests for analytically constructed BlockMask."""

    def _build_simple_mask(
        self,
        H: int = 4,
        W: int = 4,
        T: int = 1,
        bs: int = 1,
        law: int = 2,
        B: int = 1,
        D: int = 4,
    ) -> tuple[BlockMask, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a BlockMask for a single spatial modality."""
        tok = torch.randn(B, H, W, T, bs, D)
        msk = torch.full(
            (B, H, W, T, bs), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
        x = {"spatial": tok, "spatial_mask": msk}
        tokens, validity, meta = collapse_block_aligned(x, law, ["spatial"])
        rows, cols = compute_block_aligned_positions(meta)
        block_mask = build_analytical_block_mask(meta, law, validity, rows, cols)
        return block_mask, rows, cols, validity

    def test_returns_block_mask(self) -> None:
        """Should return a BlockMask instance."""
        bm, _, _, _ = self._build_simple_mask()
        assert isinstance(bm, BlockMask)

    def test_spatial_proximity(self) -> None:
        """Valid token pairs within window should attend; outside should not."""
        law = 2
        bm, rows, cols, validity = self._build_simple_mask(H=4, W=4, law=law)
        N = rows.shape[0]
        dense = _eval_mask_mod(bm, N)
        radius = law // 2

        for i in range(N):
            for j in range(N):
                dr = abs(rows[i].item() - rows[j].item())
                dc = abs(cols[i].item() - cols[j].item())
                expected = (dr <= radius and dc <= radius) or (i == j)
                assert dense[i, j].item() == expected, (
                    f"mask[{i},{j}]={dense[i, j].item()}, expected={expected}, "
                    f"dr={dr}, dc={dc}"
                )

    def test_symmetry(self) -> None:
        """The mask should be symmetric for valid tokens."""
        bm, rows, cols, _ = self._build_simple_mask(H=6, W=6, T=2, bs=1, law=3)
        N = rows.shape[0]
        dense = _eval_mask_mod(bm, N)
        assert (dense == dense.T).all()

    def test_non_spatial_isolated(self) -> None:
        """Non-spatial tokens attend to each other but not spatial tokens."""
        B, D, law = 1, 4, 2
        spatial_tok = torch.randn(B, 4, 4, 1, 1, D)
        spatial_msk = torch.full(
            (B, 4, 4, 1, 1), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
        ns_tok = torch.randn(B, 3, D)
        ns_msk = torch.full((B, 3), MaskValue.ONLINE_ENCODER.value, dtype=torch.long)
        x = {
            "spatial": spatial_tok,
            "spatial_mask": spatial_msk,
            "nonspatial": ns_tok,
            "nonspatial_mask": ns_msk,
        }
        tokens, validity, meta = collapse_block_aligned(
            x, law, ["spatial", "nonspatial"]
        )
        rows, cols = compute_block_aligned_positions(meta)
        bm = build_analytical_block_mask(meta, law, validity, rows, cols)

        N = tokens.shape[1]
        dense = _eval_mask_mod(bm, N)

        # Non-spatial starts at offset = spatial padded tokens
        ns_start = meta[0].padded_n_tokens
        ns_end = ns_start + meta[1].original_n_tokens

        # Non-spatial tokens should attend to each other.
        for i in range(ns_start, ns_end):
            for j in range(ns_start, ns_end):
                assert dense[i, j].item() is True

        # Non-spatial should NOT attend to spatial.
        for i in range(ns_start, ns_end):
            for j in range(meta[0].original_n_tokens):
                assert dense[i, j].item() is False
                assert dense[j, i].item() is False

    def test_missing_tokens_masked(self) -> None:
        """Missing tokens should only self-attend."""
        B, law = 2, 2
        tok = torch.randn(B, 4, 4, 1, 1, 4)
        msk = torch.full(
            (B, 4, 4, 1, 1), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
        # Mark all tokens in batch 0 as MISSING.
        msk[0] = MaskValue.MISSING.value
        x = {"spatial": tok, "spatial_mask": msk}
        tokens, validity, meta = collapse_block_aligned(x, law, ["spatial"])
        rows, cols = compute_block_aligned_positions(meta)
        bm = build_analytical_block_mask(meta, law, validity, rows, cols)

        N = tokens.shape[1]
        dense = _eval_mask_mod(bm, N, B=B)

        # Batch 0: only diagonal should be True.
        b0 = dense[0]
        assert b0.diagonal().all()
        off_diag = b0.clone()
        off_diag.fill_diagonal_(False)
        assert not off_diag.any()

        # Batch 1: should have normal spatial attention.
        b1 = dense[1]
        assert b1.sum() > N

    def test_block_sparsity_structure(self) -> None:
        """Verify kv_indices encode the correct 3x3 neighborhood."""
        law = 2
        bm, _, _, _ = self._build_simple_mask(H=6, W=6, law=law)
        block_size = law * law
        # num_bh=3, num_bw=3, T=1, bs=1 -> 9 spatial blocks -> 9 flex-blocks
        num_blocks = 36 // block_size  # 6*6*1*1 / 4 = 9
        assert num_blocks == 9

        # Corner block (bh=0, bw=0): neighbors are (0,0),(0,1),(1,0),(1,1) = 4 blocks
        assert bm.kv_num_blocks[0, 0, 0].item() == 4
        # Edge block (bh=0, bw=1): neighbors include (0,0),(0,1),(0,2),(1,0),(1,1),(1,2) = 6
        assert bm.kv_num_blocks[0, 0, 1].item() == 6
        # Center block (bh=1, bw=1): all 9 neighbors
        # Block index for (bh=1, bw=1) is 1*3+1 = 4
        assert bm.kv_num_blocks[0, 0, 4].item() == 9


# ---------------------------------------------------------------------------
# Encoder integration tests (requires CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
class TestEncoderLocalAttention:
    """Integration tests for Encoder with local_attention_window."""

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
        """Create a small encoder on CUDA for testing."""
        return Encoder(
            embedding_size=16,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
        ).cuda()

    @pytest.fixture
    def modality_band_set_len_and_total_bands(
        self,
        supported_modalities: list[ModalitySpec],
    ) -> dict[str, tuple[int, int]]:
        """Get band set counts and total band counts per modality."""
        return {
            modality.name: (len(modality.band_sets), modality.num_bands)
            for modality in supported_modalities
        }

    @torch.inference_mode()
    def test_forward_with_local_attention(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Encoder forward with local_attention_window should produce valid output."""
        sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
            modality_band_set_len_and_total_bands["sentinel2_l2a"]
        )
        latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
            "latlon"
        ]
        B, H, W, T = 2, 16, 16, 3
        patch_size = 4
        sample = _to_cuda(
            _make_sample(B, H, W, T, sentinel2_l2a_num_bands, latlon_num_bands)
        )
        output = encoder(
            sample, patch_size=patch_size, fast_pass=True, local_attention_window=2
        )
        tokens_and_masks = output["tokens_and_masks"]
        assert tokens_and_masks.sentinel2_l2a.shape == (
            B,
            H // patch_size,
            W // patch_size,
            T,
            sentinel2_l2a_num_band_sets,
            16,
        )

    @torch.inference_mode()
    def test_local_attention_changes_output(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Local attention should produce different output than global attention."""
        _, sentinel2_l2a_num_bands = modality_band_set_len_and_total_bands[
            "sentinel2_l2a"
        ]
        _, latlon_num_bands = modality_band_set_len_and_total_bands["latlon"]
        B, H, W, T = 1, 16, 16, 3
        patch_size = 4
        sample = _to_cuda(
            _make_sample(B, H, W, T, sentinel2_l2a_num_bands, latlon_num_bands)
        )
        out_global = encoder(sample, patch_size=patch_size, fast_pass=True)
        out_local = encoder(
            sample, patch_size=patch_size, fast_pass=True, local_attention_window=2
        )
        global_tokens = out_global["tokens_and_masks"].sentinel2_l2a
        local_tokens = out_local["tokens_and_masks"].sentinel2_l2a
        assert not torch.allclose(global_tokens, local_tokens, atol=1e-5)

    def test_flash_attn_guard(self, supported_modalities: list[ModalitySpec]) -> None:
        """Should raise ValueError when local_attention_window + flash_attn."""
        encoder = Encoder(
            embedding_size=16,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            use_flash_attn=True,
        ).cuda()
        sample = _to_cuda(
            MaskedOlmoEarthSample(
                sentinel2_l2a=torch.randn(1, 8, 8, 1, 13),
                sentinel2_l2a_mask=torch.zeros(1, 8, 8, 1, 13, dtype=torch.long),
                latlon=torch.randn(1, 2),
                latlon_mask=torch.zeros(1, 2),
                timestamps=torch.tensor([[[15, 7, 2023]]]),
            )
        )
        with pytest.raises(ValueError, match="flash attention"):
            encoder(sample, patch_size=4, fast_pass=True, local_attention_window=4)

    def test_register_tokens_guard(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Should raise ValueError when local_attention_window + register tokens."""
        encoder = Encoder(
            embedding_size=16,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            num_register_tokens=4,
        ).cuda()
        sample = _to_cuda(
            MaskedOlmoEarthSample(
                sentinel2_l2a=torch.randn(1, 8, 8, 1, 13),
                sentinel2_l2a_mask=torch.zeros(1, 8, 8, 1, 13, dtype=torch.long),
                latlon=torch.randn(1, 2),
                latlon_mask=torch.zeros(1, 2),
                timestamps=torch.tensor([[[15, 7, 2023]]]),
            )
        )
        with pytest.raises(ValueError, match="register tokens"):
            encoder(sample, patch_size=4, fast_pass=True, local_attention_window=4)

    @torch.inference_mode()
    def test_local_attention_with_missing_tokens(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    ) -> None:
        """Local attention + fast_pass=False (missing tokens) should succeed."""
        sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
            modality_band_set_len_and_total_bands["sentinel2_l2a"]
        )
        _, latlon_num_bands = modality_band_set_len_and_total_bands["latlon"]
        B, H, W, T = 2, 16, 16, 3
        patch_size = 4

        sample = _make_sample(B, H, W, T, sentinel2_l2a_num_bands, latlon_num_bands)
        # Mark one full timestep as MISSING for the first batch element.
        assert sample.sentinel2_l2a_mask is not None
        sample.sentinel2_l2a_mask[0, :, :, 2, :] = MaskValue.MISSING.value
        sample = _to_cuda(sample)

        output = encoder(
            sample,
            patch_size=patch_size,
            fast_pass=False,
            local_attention_window=2,
        )
        tokens_and_masks = output["tokens_and_masks"]
        assert tokens_and_masks.sentinel2_l2a.shape == (
            B,
            H // patch_size,
            W // patch_size,
            T,
            sentinel2_l2a_num_band_sets,
            16,
        )


def _make_sample(
    B: int,
    H: int,
    W: int,
    T: int,
    sentinel2_l2a_num_bands: int,
    latlon_num_bands: int,
) -> MaskedOlmoEarthSample:
    """Build a minimal sample for testing."""
    sentinel2_l2a = torch.randn(B, H, W, T, sentinel2_l2a_num_bands)
    sentinel2_l2a_mask = torch.full(
        (B, H, W, T, sentinel2_l2a_num_bands),
        MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )
    latlon = torch.randn(B, latlon_num_bands)
    latlon_mask = torch.full(
        (B, latlon_num_bands),
        MaskValue.ONLINE_ENCODER.value,
        dtype=torch.float32,
    )
    timestamps = torch.tensor(
        [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
    )
    timestamps = repeat(timestamps, "t d -> b t d", b=B)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
        timestamps=timestamps,
    )


def _mock_patch_and_pos_encoding(
    monkeypatch: pytest.MonkeyPatch,
    encoder: Encoder,
    n_band_sets: int,
    n_ll_band_sets: int,
    patch_size: int,
) -> None:
    """Mock patch embedding and positional encoding to preserve pixel-encoded positions.

    Expects sentinel2_l2a pixel band 0 = row index, band 1 = col index per patch.
    Produces tokens where embedding dim 0 = row, dim 1 = col.
    Latlon tokens get (-1_000_000, -1_000_000) as non-spatial sentinel.
    """
    embed_size = encoder.embedding_size

    def mock_patch_forward(
        input_data: MaskedOlmoEarthSample, ps: int
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        assert input_data.sentinel2_l2a is not None
        assert input_data.sentinel2_l2a_mask is not None
        assert input_data.latlon_mask is not None
        data = input_data.sentinel2_l2a
        mask_data = input_data.sentinel2_l2a_mask
        sampled = data[:, ::ps, ::ps, :, :]  # (B, h, w, T, n_bands)
        B_size = data.shape[0]
        tokens_list, masks_list = [], []
        for bs_idx in range(n_band_sets):
            bs_mask = mask_data[:, ::ps, ::ps, :, bs_idx]
            tok = torch.zeros(*sampled.shape[:4], embed_size, device=data.device)
            tok[..., 0] = sampled[..., 0]
            tok[..., 1] = sampled[..., 1]
            # Zero out positions for MISSING tokens so tok > 0 ≡ valid.
            valid = (bs_mask == MaskValue.ONLINE_ENCODER.value).unsqueeze(-1)
            tok = tok * valid
            tokens_list.append(tok)
            masks_list.append(bs_mask)
        output["sentinel2_l2a"] = torch.stack(tokens_list, dim=-2)
        output["sentinel2_l2a_mask"] = torch.stack(masks_list, dim=-1)

        ll_tok = torch.zeros(B_size, n_ll_band_sets, embed_size, device=data.device)
        ll_tok[..., 0] = -1_000_000
        ll_tok[..., 1] = -1_000_000
        output["latlon"] = ll_tok
        output["latlon_mask"] = input_data.latlon_mask[:, :n_ll_band_sets]
        return output

    monkeypatch.setattr(encoder.patch_embeddings, "forward", mock_patch_forward)
    monkeypatch.setattr(
        encoder.composite_encodings,
        "forward",
        lambda tokens_dict, *args, **kwargs: tokens_dict,
    )


def _make_position_encoded_sample(
    B: int,
    H: int,
    W: int,
    T: int,
    n_bands: int,
    latlon_num_bands: int,
    patch_size: int,
) -> MaskedOlmoEarthSample:
    """Build a sample with (row, col) encoded in pixel bands 0 and 1 per patch."""
    h_grid, w_grid = H // patch_size, W // patch_size
    sentinel2_l2a = torch.zeros(B, H, W, T, n_bands)
    for r in range(h_grid):
        for c in range(w_grid):
            sentinel2_l2a[
                :,
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
                :,
                0,
            ] = r + 1
            sentinel2_l2a[
                :,
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
                :,
                1,
            ] = c + 1
    sentinel2_l2a_mask = torch.full(
        (B, H, W, T, n_bands),
        MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )
    latlon = torch.randn(B, latlon_num_bands)
    latlon_mask = torch.full(
        (B, latlon_num_bands),
        MaskValue.ONLINE_ENCODER.value,
        dtype=torch.float32,
    )
    timestamps = repeat(
        torch.tensor([[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long),
        "t d -> b t d",
        b=B,
    )
    return MaskedOlmoEarthSample(
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
        timestamps=timestamps,
    )


@requires_cuda
class TestMaskAlignmentWithTokens:
    """Verify the local attention mask aligns with actual token spatial positions.

    Encodes (row, col) into pixel values, mocks patch embedding to preserve them,
    then reads positions back from captured tokens and checks the mask.
    """

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
        """Create a small encoder on CUDA for testing."""
        return Encoder(
            embedding_size=16,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
        ).cuda()

    @pytest.fixture
    def modality_band_set_len_and_total_bands(
        self,
        supported_modalities: list[ModalitySpec],
    ) -> dict[str, tuple[int, int]]:
        """Get band set counts and total band counts per modality."""
        return {
            modality.name: (len(modality.band_sets), modality.num_bands)
            for modality in supported_modalities
        }

    @torch.inference_mode()
    def test_mask_alignment_fast_pass_true(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With fast_pass=True, captured BlockMask must align with pixel-encoded positions."""
        n_band_sets, n_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"]
        n_ll_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
            "latlon"
        ]
        B, H, W, T = 1, 16, 16, 3
        patch_size = 4
        window_size = 2

        sample = _to_cuda(
            _make_position_encoded_sample(
                B, H, W, T, n_bands, latlon_num_bands, patch_size
            )
        )

        _mock_patch_and_pos_encoding(
            monkeypatch, encoder, n_band_sets, n_ll_band_sets, patch_size
        )

        captured: dict[str, Any] = {}
        orig_forward = encoder.blocks[0].forward

        def capturing_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            captured["attn_mask"] = kwargs.get("attn_mask")
            captured["x"] = args[0] if args else kwargs.get("x")
            return orig_forward(*args, **kwargs)

        monkeypatch.setattr(encoder.blocks[0], "forward", capturing_forward)

        encoder(
            sample,
            patch_size=patch_size,
            fast_pass=True,
            local_attention_window=window_size,
        )

        block_mask = captured["attn_mask"]
        assert isinstance(block_mask, BlockMask)
        tokens: torch.Tensor = captured["x"]
        N_padded = tokens.shape[1]

        tok_rows = tokens[0, :, 0].cpu().round().long()
        tok_cols = tokens[0, :, 1].cpu().round().long()

        dense = _eval_mask_mod(block_mask, N_padded)

        # Positions are 1-indexed so tok > 0 ≡ valid spatial token.
        valid_idx = torch.where(tok_rows > 0)[0]
        radius = window_size // 2

        for ii in range(len(valid_idx)):
            for jj in range(len(valid_idx)):
                i, j = valid_idx[ii].item(), valid_idx[jj].item()
                expected = (
                    abs(tok_rows[i].item() - tok_rows[j].item()) <= radius
                    and abs(tok_cols[i].item() - tok_cols[j].item()) <= radius
                )
                assert dense[i, j].item() == expected, (
                    f"mask[{i},{j}] = {dense[i, j].item()}, expected {expected}"
                )

    @torch.inference_mode()
    def test_mask_alignment_fast_pass_false(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With missing tokens (fast_pass=False), captured BlockMask must be correct."""
        n_band_sets, n_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"]
        n_ll_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
            "latlon"
        ]
        B, H, W, T = 2, 16, 16, 3
        patch_size = 4
        window_size = 2

        sample = _make_position_encoded_sample(
            B, H, W, T, n_bands, latlon_num_bands, patch_size
        )
        # Mark timestep 2 as MISSING for batch element 0 only.
        assert sample.sentinel2_l2a_mask is not None
        sample.sentinel2_l2a_mask[0, :, :, 2, :] = MaskValue.MISSING.value
        sample = _to_cuda(sample)

        _mock_patch_and_pos_encoding(
            monkeypatch, encoder, n_band_sets, n_ll_band_sets, patch_size
        )

        captured: dict[str, Any] = {}
        orig_forward = encoder.blocks[0].forward

        def capturing_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            captured["attn_mask"] = kwargs.get("attn_mask")
            captured["x"] = args[0] if args else kwargs.get("x")
            return orig_forward(*args, **kwargs)

        monkeypatch.setattr(encoder.blocks[0], "forward", capturing_forward)

        encoder(
            sample,
            patch_size=patch_size,
            fast_pass=False,
            local_attention_window=window_size,
        )

        block_mask = captured["attn_mask"]
        assert isinstance(block_mask, BlockMask)
        tokens: torch.Tensor = captured["x"]
        N_padded = tokens.shape[1]

        dense = _eval_mask_mod(block_mask, N_padded, B=B)
        radius = window_size // 2

        for b in range(B):
            tok_rows = tokens[b, :, 0].cpu().round().long()
            tok_cols = tokens[b, :, 1].cpu().round().long()

            # Positions are 1-indexed; MISSING tokens zeroed by mock → tok > 0 ≡ valid.
            valid_idx = torch.where(tok_rows > 0)[0]

            for ii in range(len(valid_idx)):
                for jj in range(len(valid_idx)):
                    i, j = valid_idx[ii].item(), valid_idx[jj].item()
                    expected = (
                        abs(tok_rows[i].item() - tok_rows[j].item()) <= radius
                        and abs(tok_cols[i].item() - tok_cols[j].item()) <= radius
                    )
                    assert dense[b, i, j].item() == expected, (
                        f"Batch {b}: mask[{i},{j}] = {dense[b, i, j].item()}, "
                        f"expected {expected}"
                    )
