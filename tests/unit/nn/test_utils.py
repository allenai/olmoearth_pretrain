"""Tests for olmoearth_pretrain.nn.utils."""

import pytest
import torch

from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths

try:
    import flash_attn

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import FlexiVitBase, Predictor


class TestGetCumulativeSequenceLengths:
    """Tests for get_cumulative_sequence_lengths."""

    def test_basic(self) -> None:
        seq_lengths = torch.tensor([3, 5, 2])
        cu = get_cumulative_sequence_lengths(seq_lengths)
        assert cu.tolist() == [0, 3, 8, 10]
        assert cu.dtype == torch.int32

    def test_preserves_zero_length_sequences(self) -> None:
        """Zero-length sequences must be preserved for cross-attention alignment."""
        seq_lengths = torch.tensor([3, 0, 4])
        cu = get_cumulative_sequence_lengths(seq_lengths)
        assert cu.tolist() == [0, 3, 3, 7]
        assert len(cu) == len(seq_lengths) + 1

    def test_all_zeros(self) -> None:
        seq_lengths = torch.tensor([0, 0, 0])
        cu = get_cumulative_sequence_lengths(seq_lengths)
        assert cu.tolist() == [0, 0, 0, 0]
        assert len(cu) == 4

    def test_single_sequence(self) -> None:
        seq_lengths = torch.tensor([7])
        cu = get_cumulative_sequence_lengths(seq_lengths)
        assert cu.tolist() == [0, 7]

    def test_single_zero(self) -> None:
        seq_lengths = torch.tensor([0])
        cu = get_cumulative_sequence_lengths(seq_lengths)
        assert cu.tolist() == [0, 0]

    def test_mixed_zeros(self) -> None:
        """Multiple zeros interspersed with real lengths."""
        seq_lengths = torch.tensor([5, 0, 3, 0, 0, 2])
        cu = get_cumulative_sequence_lengths(seq_lengths)
        assert cu.tolist() == [0, 5, 5, 8, 8, 8, 10]
        assert len(cu) == 7

    def test_length_always_batch_plus_one(self) -> None:
        """cu_seqlens must always be batch_size + 1, regardless of zeros."""
        for batch_size in [1, 4, 16]:
            for zero_frac in [0.0, 0.25, 0.5, 1.0]:
                seq_lengths = torch.randint(0, 10, (batch_size,))
                n_zeros = int(batch_size * zero_frac)
                seq_lengths[:n_zeros] = 0
                cu = get_cumulative_sequence_lengths(seq_lengths)
                assert len(cu) == batch_size + 1, (
                    f"batch_size={batch_size}, zeros={n_zeros}: "
                    f"expected {batch_size + 1} entries, got {len(cu)}"
                )


class TestCrossAttentionCuSeqlensAlignment:
    """Tests that cu_seqlens_q and cu_seqlens_k stay aligned for cross-attention."""

    def test_alignment_when_decoder_has_zero_but_encoder_does_not(self) -> None:
        """Reproduces the bug: sample has encoder tokens but no decoder tokens."""
        seqlens_decode = torch.tensor([5, 0, 3])
        seqlens_encode = torch.tensor([3, 4, 2])

        cu_q = get_cumulative_sequence_lengths(seqlens_decode)
        cu_k = get_cumulative_sequence_lengths(seqlens_encode)

        assert len(cu_q) == len(cu_k), (
            f"cu_seqlens_q has {len(cu_q)} entries but cu_seqlens_k has {len(cu_k)}"
        )

    def test_alignment_when_encoder_has_zero_but_decoder_does_not(self) -> None:
        """Sample has decoder tokens but no encoder tokens (missing S2/S1/landsat)."""
        seqlens_decode = torch.tensor([5, 3, 2])
        seqlens_encode = torch.tensor([3, 0, 4])

        cu_q = get_cumulative_sequence_lengths(seqlens_decode)
        cu_k = get_cumulative_sequence_lengths(seqlens_encode)

        assert len(cu_q) == len(cu_k), (
            f"cu_seqlens_q has {len(cu_q)} entries but cu_seqlens_k has {len(cu_k)}"
        )

    def test_alignment_with_mixed_zeros(self) -> None:
        """Both sides have zeros but at different positions."""
        seqlens_decode = torch.tensor([0, 3, 5, 0, 2])
        seqlens_encode = torch.tensor([4, 0, 2, 3, 0])

        cu_q = get_cumulative_sequence_lengths(seqlens_decode)
        cu_k = get_cumulative_sequence_lengths(seqlens_encode)

        assert len(cu_q) == len(cu_k) == 6

    def test_cu_seqlens_last_equals_total_tokens(self) -> None:
        """Last entry of cu_seqlens must equal total packed tokens."""
        seqlens = torch.tensor([3, 0, 4, 0, 2])
        cu = get_cumulative_sequence_lengths(seqlens)
        assert cu[-1].item() == seqlens.sum().item()


class TestPackTokensWithZeroLengthSequences:
    """Test that pack_tokens + cu_seqlens work correctly with zero-length sequences."""

    def test_pack_with_empty_samples(self) -> None:
        """Packed tensor should skip empty samples; cu_seqlens should still align."""
        batch, max_len, dim = 3, 5, 8
        tokens = torch.randn(batch, max_len, dim)

        mask = torch.zeros(batch, max_len, dtype=torch.bool)
        mask[0, :3] = True
        mask[2, :2] = True

        seq_lengths = mask.sum(dim=-1)  # [3, 0, 2]
        cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)

        packed = FlexiVitBase.pack_tokens(tokens, mask)

        assert packed.shape == (5, dim)  # 3 + 0 + 2
        assert cu_seqlens.tolist() == [0, 3, 3, 5]
        assert len(cu_seqlens) == batch + 1

    def test_pack_unpack_roundtrip_with_empty_samples(self) -> None:
        """Tokens should survive pack -> unpack even with empty samples."""
        batch, max_len, dim = 4, 6, 8
        torch.manual_seed(99)
        tokens = torch.randn(batch, max_len, dim)

        mask = torch.zeros(batch, max_len, dtype=torch.bool)
        mask[0, :4] = True
        mask[1] = False  # empty
        mask[2, :2] = True
        mask[3, :5] = True

        og_shape = tokens.shape
        packed = FlexiVitBase.pack_tokens(tokens, mask)
        unpacked = FlexiVitBase.unpack_tokens(packed, mask, og_shape)

        for i in range(batch):
            n = mask[i].sum().item()
            torch.testing.assert_close(
                unpacked[i, :n],
                tokens[i, :n],
                msg=f"Sample {i} failed roundtrip",
            )
        assert (unpacked[1] == 0).all(), "Empty sample should be all zeros"


@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash-attn not installed")
class TestFlashAttnCrossAttentionAlignment:
    """Integration test: decoder cross-attention with zero-length sequences.

    Requires flash-attn to be installed.
    """

    def test_predictor_forward_with_missing_encoder_tokens(self) -> None:
        """Run a Predictor forward pass where one sample has 0 encoder tokens.

        This previously caused cu_seqlens_q/cu_seqlens_k length mismatch.
        """
        supported_modalities = [Modality.SENTINEL2_L2A, Modality.LATLON]
        predictor = Predictor(
            supported_modalities=supported_modalities,
            encoder_embedding_size=16,
            decoder_embedding_size=16,
            depth=1,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.0,
            output_embedding_size=16,
            use_flash_attn=True,
        )
        predictor.eval()

        B, H, W, T, D = 3, 2, 2, 2, 16
        C = Modality.SENTINEL2_L2A.band_sets[0].num_bands
        from olmoearth_pretrain.train.masking import MaskValue

        s2_tokens = torch.randn(B, H, W, T, C, D)
        s2_mask = torch.zeros(B, H, W, T, C, dtype=torch.long)

        s2_mask[0, 0, :, :, :] = MaskValue.DECODER.value
        s2_mask[0, 1, :, :, :] = MaskValue.ONLINE_ENCODER.value

        # Sample 1: ALL decode, NO encode — triggers the bug
        s2_mask[1, :, :, :, :] = MaskValue.DECODER.value

        s2_mask[2, 0, :, :, :] = MaskValue.ONLINE_ENCODER.value
        s2_mask[2, 1, :, :, :] = MaskValue.DECODER.value

        latlon = torch.randn(B, 2, D)
        latlon_mask = torch.zeros(B, 2, dtype=torch.long)

        timestamps = torch.ones(B)

        x = {
            "sentinel2_l2a": s2_tokens,
            "sentinel2_l2a_mask": s2_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }

        with torch.no_grad():
            result = predictor.apply_attn(x, timestamps, patch_size=4, input_res=10)

        assert "sentinel2_l2a" in result
