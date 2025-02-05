"""Integration tests for the model.

Any methods that piece together multiple steps or are the entire forward pass for a module should be here
"""

import pytest
import torch

from helios.nn.model import Encoder, TokensAndMasks
from helios.train.masking import MaskValue


# TODO: We should have a loaded Test batch with real data for this one
class TestEncoder:
    @pytest.fixture
    def encoder(self) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        modalities_dict = dict({"s2": dict({"rgb": [0, 1, 2], "nir": [3]})})
        return Encoder(
            embedding_size=16,
            max_patch_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            modalities_to_channel_groups_dict=modalities_dict,
            max_sequence_length=12,
            base_patch_size=4,
            use_channel_embs=True,
        )

    def test_apply_attn(self, encoder: Encoder) -> None:
        """Test applying attention layers with masking via the apply_attn method.

        Args:
            encoder: Test encoder instance
        """
        s2_tokens = torch.randn(1, 2, 2, 3, 2, 16)
        s2_mask = torch.zeros(1, 2, 2, 3, 2, dtype=torch.long)

        # Mask the first and second "positions" in this 2x2 grid.
        s2_mask[0, 0, 0, 0] = 1  # mask first token
        s2_mask[0, 0, 1, 0] = 1  # mask second token

        # Construct the TokensAndMasks namedtuple with mock modality data + mask.
        x = TokensAndMasks(s2=s2_tokens, s2_mask=s2_mask)

        timestamps = (
            torch.tensor(
                [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
            )
            .unsqueeze(0)
            .permute(0, 2, 1)
        )  # [B, 3, T]
        patch_size = 4
        input_res = 10

        output = encoder.apply_attn(
            x=x, timestamps=timestamps, patch_size=patch_size, input_res=input_res
        )

        assert isinstance(
            output, TokensAndMasks
        ), "apply_attn should return a TokensAndMasks object."

        # Ensure shape is preserved in the output tokens.
        assert (
            output.s2.shape == s2_tokens.shape
        ), f"Expected output 's2' shape {s2_tokens.shape}, got {output.s2.shape}."

        # Confirm the mask was preserved and that masked tokens are zeroed out in the output.
        assert (output.s2_mask == s2_mask).all(), "Mask should be preserved in output"
        assert (
            output.s2[s2_mask >= MaskValue.TARGET_ENCODER_ONLY] == 0
        ).all(), "Masked tokens should be 0 in output"

    def test_forward(self, encoder: Encoder) -> None:
        """Test full forward pass.

        Args:
            encoder: Test encoder instance
        """
        pass
