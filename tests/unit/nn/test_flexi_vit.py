"""Unit tests for the flexi_vit module."""

import logging
from typing import Any

import pytest
import torch
from einops import repeat

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.encodings import (
    get_1d_sincos_pos_encoding,
    timestamps_to_days,
)
from olmoearth_pretrain.nn.flexi_vit import (
    CompositeEncodings,
    Encoder,
    EncoderConfig,
    FlexiVitBase,
    MultiModalPatchEmbeddings,
    PoolingType,
    Predictor,
    PredictorConfig,
    ProjectAndAggregate,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.pooling import pool_unmasked_tokens
from olmoearth_pretrain.train.masking import MaskValue

logger = logging.getLogger(__name__)


class TestCompositeEncodings:
    """Unit tests for the CompositeEncodings class."""

    @pytest.fixture
    def composite_encodings(
        self,
    ) -> CompositeEncodings:
        """Create composite encoder fixture for testing."""
        composite_encodings = CompositeEncodings(
            embedding_size=16,
            supported_modalities=[
                Modality.SENTINEL2_L2A,
                Modality.LATLON,
                Modality.WORLDCOVER,
            ],
            max_sequence_length=12,
            random_channel_embeddings=True,
        )
        return composite_encodings

    def test_apply_encodings_per_modality_latlon(
        self,
        composite_encodings: CompositeEncodings,
    ) -> None:
        """Test applying encodings to different modalities."""
        B, D = 4, 16
        patch_size = 4
        input_res = 10
        latlon_tokens = torch.randn(B, 1, D)
        ll_enc = composite_encodings._apply_encodings_per_modality(
            "latlon", latlon_tokens, None, patch_size, input_res
        )
        assert not (ll_enc == 0).all()
        assert not (ll_enc == latlon_tokens).all()
        assert latlon_tokens.shape == ll_enc.shape

    def test_apply_encodings_per_modality_sentinel2_l2a(
        self, composite_encodings: CompositeEncodings
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, T, C, D = 4, 4, 4, 3, 3, 16
        patch_size = 4
        input_res = 10
        timestamps = torch.tensor(
            [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
        )
        timestamps = repeat(timestamps, "... -> b ...", b=B)
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, C, D)
        enc = composite_encodings._apply_encodings_per_modality(
            "sentinel2_l2a", sentinel2_l2a_tokens, timestamps, patch_size, input_res
        )
        assert not (enc == 0).all()

    def test_apply_encodings_per_modality_worldcover(
        self,
        composite_encodings: CompositeEncodings,
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, C, D = 4, 4, 4, 1, 16
        patch_size = 4
        input_res = 10
        worldcover_tokens = torch.randn(B, H, W, C, D)
        wc_enc = composite_encodings._apply_encodings_per_modality(
            "worldcover", worldcover_tokens, None, patch_size, input_res
        )
        assert not (wc_enc == 0).all()
        assert not (wc_enc == worldcover_tokens).all()
        assert worldcover_tokens.shape == wc_enc.shape

    def test_apply_encodings_per_modality_grad(
        self, composite_encodings: CompositeEncodings
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, T, C, D = 4, 4, 4, 3, 3, 16
        patch_size = 4
        input_res = 10
        timestamps = torch.tensor(
            [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
        )
        timestamps = repeat(timestamps, "... -> b ...", b=B)
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, C, D)
        assert (
            composite_encodings.per_modality_channel_embeddings["sentinel2_l2a"].grad
            is None
        )
        enc = composite_encodings._apply_encodings_per_modality(
            "sentinel2_l2a", sentinel2_l2a_tokens, timestamps, patch_size, input_res
        )
        loss = enc.sum()
        loss.backward()
        assert (
            composite_encodings.per_modality_channel_embeddings["sentinel2_l2a"].grad
            is not None
        )

    def test_dynamic_pos_embed_matches_static(self) -> None:
        """On-the-fly sinusoidal encoding matches a pre-allocated table for overlapping positions."""
        dim = 48
        table = get_1d_sincos_pos_encoding(torch.arange(12), dim)
        for t in [1, 5, 12, 17, 24]:
            dynamic = get_1d_sincos_pos_encoding(torch.arange(t), dim)
            overlap = min(t, 12)
            assert torch.allclose(dynamic[:overlap], table[:overlap], atol=1e-6)

    def test_temporal_encoding_works_beyond_max_sequence_length(
        self,
    ) -> None:
        """Forward pass works when t exceeds the configured max_sequence_length."""
        ce = CompositeEncodings(
            embedding_size=16,
            supported_modalities=[Modality.SENTINEL2_L2A],
            max_sequence_length=12,
            random_channel_embeddings=True,
        )
        B, H, W, T, C, D = 2, 4, 4, 17, 3, 16
        tokens = torch.randn(B, H, W, T, C, D)
        timestamps = torch.zeros(B, T, 3, dtype=torch.long)
        timestamps[:, :, 1] = torch.arange(T) % 12
        result = ce._apply_encodings_per_modality(
            "sentinel2_l2a", tokens, timestamps, patch_size=4, input_res=10
        )
        assert result.shape == tokens.shape
        assert not (result == tokens).all()

    def test_temporal_encoding_values_match_expected(self) -> None:
        """Temporal position encoding values match get_1d_sincos_pos_encoding directly."""
        embedding_size = 16
        n = embedding_size // 4
        ce = CompositeEncodings(
            embedding_size=embedding_size,
            supported_modalities=[Modality.SENTINEL2_L2A],
            max_sequence_length=12,
            random_channel_embeddings=True,
        )
        B, H, W, T, C, D = 1, 2, 2, 5, 3, embedding_size
        tokens = torch.zeros(B, H, W, T, C, D)
        timestamps = torch.zeros(B, T, 3, dtype=torch.long)
        timestamps[:, :, 1] = torch.arange(T)
        result = ce._apply_encodings_per_modality(
            "sentinel2_l2a", tokens, timestamps, patch_size=4, input_res=10
        )
        expected_time = get_1d_sincos_pos_encoding(torch.arange(T), n)
        actual_time = result[0, 0, 0, :, 0, n : 2 * n]
        assert torch.allclose(actual_time, expected_time, atol=1e-5)


# TODO: Add tests for when the inputs are completely masked or different dims or something
class TestFlexiVitBase:
    """Unit tests for the FlexiVitBase class."""

    @pytest.fixture
    def flexi_helios_base(
        self, supported_modalities: list[ModalitySpec]
    ) -> FlexiVitBase:
        """Create encoder fixture for testing."""
        flexi_helios_base = FlexiVitBase(
            embedding_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
        )
        return flexi_helios_base

    def test_collapse_and_combine_hwtc(self, flexi_helios_base: FlexiVitBase) -> None:
        """Test collapsing tokens from different modalities into single tensor."""
        B, D = 2, 4
        sentinel2_l2a_tokens = torch.randn(B, 2, 1, 1, 2, D)
        sentinel2_l2a_mask = torch.randint(0, 2, (B, 2, 1, 1, 2)).float()
        latlon = torch.randn(B, 1, D)
        latlon_mask = torch.randint(0, 2, (B, 1)).float()
        x = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        tokens, masks = flexi_helios_base.collapse_and_combine_hwtc(x)
        assert tokens.shape == (B, 5, D)
        assert masks.shape == (B, 5)

    def test_split_and_expand_per_modality(self) -> None:
        """Test splitting combined tensor back into per-modality tensors."""
        B, D = 2, 4  # Batch size and embedding dimension
        modality_1_channel_groups = 3
        modality_2_channel_groups = 5
        modalities_to_dims_dict: dict[str, tuple[int, ...]] = {
            "modality1": (B, 2, 2, 1, modality_1_channel_groups, D),
            "modality2": (B, 1, 1, 2, modality_2_channel_groups, D),
        }

        modality1_data = torch.randn(B, 4 * modality_1_channel_groups, D)
        modality2_data = torch.randn(B, 4 * modality_2_channel_groups, D)

        x = torch.cat([modality1_data, modality2_data], dim=1)

        # Now call the function
        modality_tokens_dict = FlexiVitBase.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )

        modality1_tokens = modality_tokens_dict["modality1"]
        modality2_tokens = modality_tokens_dict["modality2"]
        assert list(modality1_tokens.shape) == [
            2,
            2,
            2,
            1,
            3,
            4,
        ], f"Incorrect shape for modality1 tokens: {modality1_tokens.shape}"
        assert list(modality2_tokens.shape) == [
            2,
            1,
            1,
            2,
            5,
            4,
        ], f"Incorrect shape for modality2 tokens: {modality2_tokens.shape}"

    def test_3d_rope_positions_share_timestamp_slots_across_modalities(self) -> None:
        """All multitemporal modalities should use the same timestamp slot values."""
        model = FlexiVitBase(
            embedding_size=32,
            num_heads=2,
            mlp_ratio=2.0,
            depth=1,
            drop_path=0.0,
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.LANDSAT],
            max_sequence_length=12,
            position_encoding="rope_3d",
        )
        timestamps = torch.tensor(
            [[[1, 0, 2023], [1, 1, 2023], [1, 6, 2023]]], dtype=torch.long
        )
        tokens_only_dict = {
            "sentinel2_l2a": torch.zeros(1, 1, 1, 3, 1, 32),
            "landsat": torch.zeros(1, 1, 1, 3, 1, 32),
        }
        masks_dict = {
            "sentinel2_l2a_mask": torch.zeros(1, 1, 1, 3, 1),
            "landsat_mask": torch.zeros(1, 1, 1, 3, 1),
        }

        positions = model.build_rope_positions(
            tokens_only_dict=tokens_only_dict,
            original_masks_dict=masks_dict,
            patch_size=4,
            input_res=10,
            timestamps=timestamps,
        )

        assert positions is not None
        expected_days = timestamps_to_days(timestamps)[0].repeat_interleave(2)
        actual_days = torch.sort(positions[0, :, 0]).values
        assert torch.allclose(actual_days, torch.sort(expected_days).values)
        assert torch.equal(positions[0, :, 1:], torch.zeros_like(positions[0, :, 1:]))

    def test_3d_rope_requires_timestamps(self) -> None:
        """3D RoPE cannot build the temporal coordinate without timestamps."""
        model = FlexiVitBase(
            embedding_size=32,
            num_heads=2,
            mlp_ratio=2.0,
            depth=1,
            drop_path=0.0,
            supported_modalities=[Modality.SENTINEL2_L2A],
            max_sequence_length=12,
            position_encoding="rope_3d",
        )
        tokens_only_dict = {"sentinel2_l2a": torch.zeros(1, 1, 1, 3, 1, 32)}
        masks_dict = {"sentinel2_l2a_mask": torch.zeros(1, 1, 1, 3, 1)}

        with pytest.raises(ValueError, match="3D RoPE requires timestamps"):
            model.build_rope_positions(
                tokens_only_dict=tokens_only_dict,
                original_masks_dict=masks_dict,
                patch_size=4,
                input_res=10,
                timestamps=None,
            )


class TestEncoder:
    """Unit tests for the Encoder class."""

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        return Encoder(
            embedding_size=8,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
        )

    def test_create_token_exit_ids_normal_usage(self, encoder: Encoder) -> None:
        """Test creating exit IDs for early token exiting - normal usage.

        Tests normal usage with full token_exit_cfg.
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, D)
        latlon_tokens = torch.randn(B, 1, D)
        x = {"sentinel2_l2a": sentinel2_l2a_tokens, "latlon": latlon_tokens}

        token_exit_cfg = {"sentinel2_l2a": 1, "latlon": 2}
        exit_ids_dict = encoder.create_token_exit_ids(x, token_exit_cfg)
        assert "sentinel2_l2a" in exit_ids_dict, (
            "Expected 'sentinel2_l2a' key in the result dict"
        )
        sentinel2_l2a_exit_ids = exit_ids_dict["sentinel2_l2a"]
        assert sentinel2_l2a_exit_ids.shape == sentinel2_l2a_tokens.shape, (
            "Shape of exit IDs should match the shape of the modality tokens."
        )

        assert (exit_ids_dict["sentinel2_l2a"] == 1).all()
        assert (exit_ids_dict["latlon"] == 2).all()

    def test_create_token_exit_ids_missing_exit_cfg_band_group(
        self, encoder: Encoder
    ) -> None:
        """Test creating exit IDs for early token exiting - error cases.

        Tests error handling for:
        - Missing band group in token_exit_cfg (KeyError)
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, D)
        x = {"sentinel2_l2a": sentinel2_l2a_tokens}

        with pytest.raises(KeyError):
            incomplete_exit_cfg = {"rgb": 1}  # Missing the "nir" key
            encoder.create_token_exit_ids(x, incomplete_exit_cfg)

    def test_remove_masked_tokens(self) -> None:
        """Test removing masked tokens and tracking indices."""
        d = 2
        x = torch.tensor([[0, 1, 0], [1, 0, 1]]).float()
        x = repeat(x, "b n -> b n d", d=d)
        print(f"x shape: {x.shape}")
        mask = torch.tensor([[0, 1, 0], [1, 0, 1]]).bool()

        expected_tokens = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
        num_tokens_to_keep = torch.sum(mask)
        expected_indices = torch.tensor([[1, 0, 2], [0, 2, 1]])
        expected_updated_mask = torch.tensor([[1, 0], [1, 1]]).bool()
        tokens, indices, updated_mask, seqlens, max_length = (
            Encoder.remove_masked_tokens(x, mask)
        )
        kept_unmasked_tokens = torch.sum(updated_mask)
        assert torch.equal(tokens, expected_tokens)
        assert torch.equal(indices, expected_indices)
        assert torch.equal(updated_mask, expected_updated_mask)
        assert kept_unmasked_tokens == num_tokens_to_keep
        assert seqlens.shape == (2,)
        assert max_length == 2

    def test_add_removed_tokens(self) -> None:
        """Test adding removed tokens back into tensor."""
        partial_tokens = torch.tensor(
            [
                [[1.0, 11.0], [2.0, 22.0]],
                [[5.0, 55.0], [6.0, 66.0]],
            ]
        )
        indices = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 2],
            ]
        )
        partial_mask = torch.tensor(
            [
                [1, 1],
                [1, 0],
            ]
        ).bool()

        expected_out = torch.tensor(
            [
                [[1.0, 11.0], [2.0, 22.0], [0.0, 0.0]],
                [[0.0, 0.0], [5.0, 55.0], [0.0, 0.0]],
            ]
        )
        expected_mask = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 0],
            ]
        ).bool()

        out, full_mask = Encoder.add_removed_tokens(
            partial_tokens, indices, partial_mask
        )
        assert torch.equal(out, expected_out)
        assert torch.equal(full_mask, expected_mask)

    def test_encoder_config(self, supported_modalities: list[ModalitySpec]) -> None:
        """Tests we can build with default args."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(supported_modality_names)
        _ = config.build()

    def test_encoder_config_rope(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Tests we can build an encoder with 2D RoPE."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names,
            embedding_size=16,
            num_heads=2,
            position_encoding="rope",
            rope_base=5000.0,
            rope_coordinate_scale=0.5,
        )
        encoder = config.build()
        assert encoder.position_encoding == "rope"
        assert encoder.rope_base == 5000.0
        assert encoder.rope_coordinate_scale == 0.5
        assert encoder.blocks[0].attn.rope_base == 5000.0

    def test_encoder_config_rope_requires_valid_head_dim(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """2D RoPE needs each attention head to split cleanly across x/y axes."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names,
            embedding_size=12,
            num_heads=2,
            position_encoding="rope",
        )
        with pytest.raises(ValueError, match="head_dim divisible by 4"):
            config.build()

    def test_encoder_config_rope_mixed(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Tests we can build an encoder with RoPE-Mixed (learnable freqs)."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names,
            embedding_size=16,
            num_heads=2,
            position_encoding="rope_mixed",
            rope_mixed_base=5.0,
        )
        encoder = config.build()
        assert encoder.position_encoding == "rope_mixed"
        assert encoder.rope_mixed_base == 5.0
        attn = encoder.blocks[0].attn
        assert attn.position_encoding == "rope_mixed"
        assert attn.rope_mixed_freqs is not None
        # (2, num_heads, head_dim // 2)
        assert attn.rope_mixed_freqs.shape == (2, 2, 4)
        assert attn.rope_mixed_freqs.requires_grad is True

    def test_encoder_config_rope_mixed_requires_valid_head_dim(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """RoPE-Mixed init also needs head_dim divisible by 4."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            supported_modality_names,
            embedding_size=12,
            num_heads=2,
            position_encoding="rope_mixed",
        )
        with pytest.raises(ValueError, match="head_dim divisible by 4"):
            config.build()

    def test_position_encoding_deprecated_alias(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """The legacy ``spatial_pos_encoding`` name still works but warns."""
        supported_modality_names = [m.name for m in supported_modalities]
        with pytest.warns(DeprecationWarning, match="spatial_pos_encoding"):
            config = EncoderConfig(
                supported_modality_names,
                embedding_size=16,
                num_heads=2,
                spatial_pos_encoding="rope",
                rope_base=5000.0,
            )
        # Reconciled onto the canonical field; legacy field cleared.
        assert config.position_encoding == "rope"
        assert config.spatial_pos_encoding is None
        encoder = config.build()
        assert encoder.position_encoding == "rope"

    def test_position_encoding_legacy_from_dict(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Old checkpoint configs carrying the legacy key still deserialize.

        ``Config.from_dict`` drops keys that are not dataclass fields, so the
        deprecated ``spatial_pos_encoding`` must remain a field for old
        checkpoints to keep loading rather than silently falling back to the
        ``absolute`` default.
        """
        supported_modality_names = [m.name for m in supported_modalities]
        with pytest.warns(DeprecationWarning, match="spatial_pos_encoding"):
            config = EncoderConfig.from_dict(
                {
                    "supported_modality_names": supported_modality_names,
                    "embedding_size": 16,
                    "num_heads": 2,
                    "spatial_pos_encoding": "rope",
                }
            )
        assert config.position_encoding == "rope"
        assert config.spatial_pos_encoding is None


class TestWindowedAttention:
    """Unit tests for neighborhood (windowed) attention in the Encoder."""

    EMBED = 16
    NUM_HEADS = 2

    def _make_encoder(
        self,
        supported_modalities: list[ModalitySpec],
        windowed_attention_size: int | None,
        seed: int = 0,
        **kwargs: Any,
    ) -> Encoder:
        torch.manual_seed(seed)
        return Encoder(
            embedding_size=self.EMBED,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=self.NUM_HEADS,
            mlp_ratio=2.0,
            depth=2,
            drop_path=0.0,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            windowed_attention_size=windowed_attention_size,
            **kwargs,
        )

    def _make_tokens(
        self, encoder: Encoder, batch: int, h: int, w: int, t: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build an ``apply_attn`` input dict: S2 (spatial) + latlon (static)."""
        s2_bs = Modality.SENTINEL2_L2A.num_band_sets
        ll_bs = Modality.LATLON.num_band_sets
        torch.manual_seed(1)
        x = {
            "sentinel2_l2a": torch.randn(batch, h, w, t, s2_bs, self.EMBED),
            "sentinel2_l2a_mask": torch.full(
                (batch, h, w, t, s2_bs), float(MaskValue.ONLINE_ENCODER.value)
            ),
            "latlon": torch.randn(batch, ll_bs, self.EMBED),
            "latlon_mask": torch.full(
                (batch, ll_bs), float(MaskValue.ONLINE_ENCODER.value)
            ),
        }
        timestamps = torch.tensor([[15, m, 2023] for m in range(1, t + 1)]).expand(
            batch, -1, -1
        )
        return x, timestamps

    def test_build_window_coordinates(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Spatial tokens get patch (row, col, 1); static tokens get (0, 0, 0)."""
        encoder = self._make_encoder(supported_modalities, windowed_attention_size=3)
        B, H, W, T = 2, 2, 3, 2
        x, _ = self._make_tokens(encoder, B, H, W, T)
        tokens_only, masks_only, dims = encoder.split_tokens_masks_and_dims(x)
        coords = encoder.build_window_coordinates(tokens_only, masks_only)

        s2_bs = Modality.SENTINEL2_L2A.num_band_sets
        n_s2 = H * W * T * s2_bs
        n_ll = Modality.LATLON.num_band_sets
        assert coords.shape == (B, n_s2 + n_ll, 5)
        # Coordinates follow the same modality order as the collapsed tokens, so
        # split them back per modality the same way apply_attn does for tokens.
        per_modality = FlexiVitBase.split_and_expand_per_modality(coords, dims)
        s2_coords = per_modality["sentinel2_l2a"]  # (B, H, W, T, b_s, 5)
        assert s2_coords.shape == (B, H, W, T, s2_bs, 5)
        expected = torch.stack(
            torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        )  # (H, W, 2)
        expected = repeat(expected, "h w p -> b h w t c p", b=B, t=T, c=s2_bs)
        assert torch.equal(s2_coords[..., :2], expected)
        assert (s2_coords[..., 2] == 1).all()
        # Timestep slot index broadcast over (b, h, w, b_s).
        expected_t = repeat(
            torch.arange(T, dtype=torch.float32),
            "t -> b h w t c",
            b=B,
            h=H,
            w=W,
            c=s2_bs,
        )
        assert torch.equal(s2_coords[..., 4], expected_t)
        # One modality index per modality, constant within it, distinct across them.
        ll_coords = per_modality["latlon"]
        assert (s2_coords[..., 3] == s2_coords[0, 0, 0, 0, 0, 3]).all()
        assert (ll_coords[..., 3] == ll_coords[0, 0, 3]).all()
        assert s2_coords[0, 0, 0, 0, 0, 3] != ll_coords[0, 0, 3]
        # Non-spatial tokens: zero (row, col), is_spatial=0, t=0.
        assert (ll_coords[..., :3] == 0).all()
        assert (ll_coords[..., 4] == 0).all()

    def test_build_window_attn_mask_neighborhood(self) -> None:
        """Radius-1 mask: 8-neighborhood on the grid, static tokens global."""
        # 3x3 grid, one token per cell, plus one static token at the end.
        rows, cols = torch.meshgrid(
            torch.arange(3, dtype=torch.float32),
            torch.arange(3, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([rows.flatten(), cols.flatten(), torch.ones(9)], dim=-1)
        coords = torch.cat([coords, torch.zeros(1, 3)], dim=0)[None]  # (1, 10, 3)

        mask = Encoder._build_window_attn_mask(coords, None, window_size=3)
        assert mask.shape == (1, 1, 10, 10)
        assert mask.dtype == torch.bool
        m = mask[0, 0]

        def idx(r: int, c: int) -> int:
            return r * 3 + c

        # Center cell sees every grid cell.
        assert m[idx(1, 1), :9].all()
        # Corner (0, 0) sees itself, (0,1), (1,0), (1,1) and nothing else on the grid.
        corner_visible = m[idx(0, 0), :9].nonzero().flatten().tolist()
        assert corner_visible == [idx(0, 0), idx(0, 1), idx(1, 0), idx(1, 1)]
        # Edge (0, 1) does not see (2, x).
        assert not m[idx(0, 1), idx(2, 0)]
        assert not m[idx(0, 1), idx(2, 1)]
        # Everyone sees the static token, and the static token sees everyone.
        assert m[:, 9].all()
        assert m[9, :].all()
        # Symmetric on the grid.
        assert torch.equal(m[:9, :9], m[:9, :9].T)

        # Radius 2 (5x5) makes the whole 3x3 grid mutually visible.
        assert Encoder._build_window_attn_mask(coords, None, window_size=5).all()

    def test_build_window_attn_mask_excludes_padding_keys(self) -> None:
        """Padding keys are never attended, even inside the window."""
        coords = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]
        )  # two adjacent spatial tokens + one static token
        key_valid = torch.tensor([[True, False, True]])
        mask = Encoder._build_window_attn_mask(coords, key_valid, window_size=3)[0, 0]
        assert not mask[:, 1].any(), "padding key must be hidden from all queries"
        assert mask[:, 0].all() and mask[:, 2].all()

    def test_prepend_register_mask_4d(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Register tokens get all-True rows and columns in the 4D window mask."""
        encoder = self._make_encoder(
            supported_modalities, windowed_attention_size=3, num_register_tokens=2
        )
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        mask[0, 0, 0, 0] = True
        out = encoder._prepend_register_mask(mask)
        assert out.shape == (1, 1, 5, 5)
        assert out[0, 0, :2, :].all(), "register queries attend everything"
        assert out[0, 0, :, :2].all(), "register keys visible to every query"
        assert torch.equal(out[0, 0, 2:, 2:], mask[0, 0])
        # 2D path is unchanged.
        out2d = encoder._prepend_register_mask(torch.tensor([[True, False]]))
        assert torch.equal(out2d, torch.tensor([[True, True, True, False]]))

    def test_large_window_matches_full_attention(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """A window wider than the grid reproduces the unwindowed encoder exactly.

        Run in training mode (no drop_path/dropout is configured) so that both
        paths apply the padding key mask, with a batch whose samples have
        different numbers of encoded tokens so that padding is present.
        """
        full = self._make_encoder(supported_modalities, None, seed=0)
        windowed = self._make_encoder(supported_modalities, 99, seed=0)
        windowed.load_state_dict(full.state_dict())
        full.train()
        windowed.train()

        B, H, W, T = 2, 3, 3, 2
        x, timestamps = self._make_tokens(full, B, H, W, T)
        # Hide a few tokens from the encoder in sample 1 only -> padding in sample 1.
        x["sentinel2_l2a_mask"][1, 0, :, 0, :] = MaskValue.DECODER.value

        out_full, _, _ = full.apply_attn(
            x, timestamps=timestamps, patch_size=4, input_res=10
        )
        out_win, _, _ = windowed.apply_attn(
            x, timestamps=timestamps, patch_size=4, input_res=10
        )
        for key in ("sentinel2_l2a", "latlon"):
            assert torch.allclose(out_full[key], out_win[key], atol=1e-5), key

    def test_small_window_changes_output(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """A 3x3 window on a larger grid must change the encoded tokens."""
        full = self._make_encoder(supported_modalities, None, seed=0)
        windowed = self._make_encoder(supported_modalities, 3, seed=0)
        windowed.load_state_dict(full.state_dict())
        full.eval()
        windowed.eval()

        B, H, W, T = 1, 4, 4, 2
        x, timestamps = self._make_tokens(full, B, H, W, T)
        with torch.no_grad():
            out_full, _, _ = full.apply_attn(
                x, timestamps=timestamps, patch_size=4, input_res=10
            )
            out_win, _, _ = windowed.apply_attn(
                x, timestamps=timestamps, patch_size=4, input_res=10
            )
        assert out_win["sentinel2_l2a"].shape == out_full["sentinel2_l2a"].shape
        assert not torch.allclose(
            out_full["sentinel2_l2a"], out_win["sentinel2_l2a"], atol=1e-5
        )

    def test_windowed_layers_subset(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Windowing only some layers: blocks outside the subset stay full."""
        encoder = self._make_encoder(
            supported_modalities, 3, windowed_attention_layers=[1]
        )
        assert encoder.windowed_attention_layers == frozenset({1})
        encoder.eval()
        B, H, W, T = 1, 4, 4, 1
        x, timestamps = self._make_tokens(encoder, B, H, W, T)
        with torch.no_grad():
            out, _, _ = encoder.apply_attn(
                x, timestamps=timestamps, patch_size=4, input_res=10
            )
        assert out["sentinel2_l2a"].shape == x["sentinel2_l2a"].shape

    def test_windowed_attention_with_register_tokens(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Register tokens and the 4D window mask compose in a forward pass."""
        encoder = self._make_encoder(supported_modalities, 3, num_register_tokens=2)
        encoder.train()
        B, H, W, T = 2, 4, 4, 1
        x, timestamps = self._make_tokens(encoder, B, H, W, T)
        x["sentinel2_l2a_mask"][1, 0, 0, 0, :] = MaskValue.DECODER.value
        out, _, _ = encoder.apply_attn(
            x, timestamps=timestamps, patch_size=4, input_res=10
        )
        assert out["sentinel2_l2a"].shape == x["sentinel2_l2a"].shape
        assert torch.isfinite(out["sentinel2_l2a"]).all()

    def test_encoder_config_windowed(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Config builds an encoder with the window settings applied."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            names,
            embedding_size=16,
            num_heads=2,
            depth=2,
            windowed_attention_size=5,
            windowed_attention_layers=[0],
        )
        encoder = config.build()
        assert encoder.windowed_attention_size == 5
        assert encoder.windowed_attention_layers == frozenset({0})

    @pytest.mark.parametrize("size", [1, 2, 4])
    def test_encoder_config_rejects_bad_window_size(
        self, supported_modalities: list[ModalitySpec], size: int
    ) -> None:
        """Window size must be odd and at least 3."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            names, embedding_size=16, num_heads=2, windowed_attention_size=size
        )
        with pytest.raises(ValueError, match="odd int >= 3"):
            config.validate()

    def test_encoder_config_rejects_flash_attn_with_window(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Flash attention ignores attn_mask, so it cannot be windowed."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            names,
            embedding_size=16,
            num_heads=2,
            windowed_attention_size=3,
            use_flash_attn=True,
        )
        with pytest.raises(ValueError, match="use_flash_attn"):
            config.validate()

    def test_encoder_config_rejects_bad_layers(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Layer indices must lie in [0, depth), and require a window size."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            names,
            embedding_size=16,
            num_heads=2,
            depth=2,
            windowed_attention_size=3,
            windowed_attention_layers=[0, 2],
        )
        with pytest.raises(ValueError, match="windowed_attention_layers"):
            config.validate()
        config = EncoderConfig(
            names, embedding_size=16, num_heads=2, windowed_attention_layers=[0]
        )
        with pytest.raises(ValueError, match="requires windowed_attention_size"):
            config.validate()

    # --- per-slice attention on the non-windowed blocks ---------------------------

    def test_build_slice_attn_mask(self) -> None:
        """Same (modality, timestep) attends; other slices do not; static is global."""
        # Two modalities x two timesteps, two tokens each (different positions), plus
        # one static token at the end. Columns: (row, col, is_spatial, mod, t).
        rows = []
        for mod in (0, 1):
            for t in (0, 1):
                rows.append([0.0, 0.0, 1.0, float(mod), float(t)])
                rows.append([3.0, 3.0, 1.0, float(mod), float(t)])
        rows.append([0.0, 0.0, 0.0, 2.0, 0.0])
        coords = torch.tensor(rows)[None]  # (1, 9, 5)
        mask = Encoder._build_slice_attn_mask(coords, None)
        assert mask.shape == (1, 1, 9, 9)
        assert mask.dtype == torch.bool
        m = mask[0, 0]

        def idx(mod: int, t: int, k: int) -> int:
            return (mod * 2 + t) * 2 + k

        for mod in (0, 1):
            for t in (0, 1):
                a, b = idx(mod, t, 0), idx(mod, t, 1)
                # Within a slice: mutually visible regardless of spatial distance.
                assert m[a, b] and m[b, a] and m[a, a]
                # Same modality, other timestep: hidden.
                assert not m[a, idx(mod, 1 - t, 0)]
                # Same timestep, other modality: hidden.
                assert not m[a, idx(1 - mod, t, 0)]
                # Other modality and timestep: hidden.
                assert not m[a, idx(1 - mod, 1 - t, 1)]
        # Everyone sees the static token, and the static token sees everyone.
        assert m[:, 8].all()
        assert m[8, :].all()
        assert torch.equal(m, m.T)

        # Padding keys are hidden from all queries, even inside their slice.
        key_valid = torch.ones(1, 9, dtype=torch.bool)
        key_valid[0, idx(0, 0, 1)] = False
        masked = Encoder._build_slice_attn_mask(coords, key_valid)[0, 0]
        assert not masked[:, idx(0, 0, 1)].any()
        assert masked[idx(0, 0, 0), idx(0, 0, 0)]

    def test_per_slice_single_slice_matches_full_attention(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """One spatial modality at T=1 is a single slice: per-slice == full attention.

        Block 0 is windowed with a window wider than the grid (so it is also full),
        block 1 is per-slice. With padding present in one sample, the result must
        match the unwindowed encoder exactly.
        """
        full = self._make_encoder(supported_modalities, None, seed=0)
        alt = self._make_encoder(
            supported_modalities,
            99,
            seed=0,
            windowed_attention_layers=[0],
            non_windowed_attention="per_slice",
        )
        alt.load_state_dict(full.state_dict())
        full.train()
        alt.train()

        B, H, W, T = 2, 3, 3, 1
        x, timestamps = self._make_tokens(full, B, H, W, T)
        x["sentinel2_l2a_mask"][1, 0, :, 0, :] = MaskValue.DECODER.value

        out_full, _, _ = full.apply_attn(
            x, timestamps=timestamps, patch_size=4, input_res=10
        )
        out_alt, _, _ = alt.apply_attn(
            x, timestamps=timestamps, patch_size=4, input_res=10
        )
        for key in ("sentinel2_l2a", "latlon"):
            assert torch.allclose(out_full[key], out_alt[key], atol=1e-5), key

    def test_per_slice_changes_output(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """With T=2 the per-slice block must differ from full attention."""
        full = self._make_encoder(supported_modalities, None, seed=0)
        alt = self._make_encoder(
            supported_modalities,
            99,
            seed=0,
            windowed_attention_layers=[0],
            non_windowed_attention="per_slice",
        )
        alt.load_state_dict(full.state_dict())
        full.eval()
        alt.eval()

        B, H, W, T = 1, 3, 3, 2
        x, timestamps = self._make_tokens(full, B, H, W, T)
        with torch.no_grad():
            out_full, _, _ = full.apply_attn(
                x, timestamps=timestamps, patch_size=4, input_res=10
            )
            out_alt, _, _ = alt.apply_attn(
                x, timestamps=timestamps, patch_size=4, input_res=10
            )
        assert out_alt["sentinel2_l2a"].shape == out_full["sentinel2_l2a"].shape
        assert torch.isfinite(out_alt["sentinel2_l2a"]).all()
        assert not torch.allclose(
            out_full["sentinel2_l2a"], out_alt["sentinel2_l2a"], atol=1e-5
        )

    def test_per_slice_with_register_tokens(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Register tokens compose with both 4D masks in a forward pass."""
        encoder = self._make_encoder(
            supported_modalities,
            3,
            num_register_tokens=2,
            windowed_attention_layers=[0],
            non_windowed_attention="per_slice",
        )
        encoder.train()
        B, H, W, T = 2, 4, 4, 2
        x, timestamps = self._make_tokens(encoder, B, H, W, T)
        x["sentinel2_l2a_mask"][1, 0, 0, 0, :] = MaskValue.DECODER.value
        out, _, _ = encoder.apply_attn(
            x, timestamps=timestamps, patch_size=4, input_res=10
        )
        assert out["sentinel2_l2a"].shape == x["sentinel2_l2a"].shape
        assert torch.isfinite(out["sentinel2_l2a"]).all()

    def test_encoder_config_per_slice(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Config builds an encoder with per-slice non-windowed blocks."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            names,
            embedding_size=16,
            num_heads=2,
            depth=4,
            windowed_attention_size=3,
            windowed_attention_layers=[0, 2],
            non_windowed_attention="per_slice",
        )
        encoder = config.build()
        assert encoder.non_windowed_attention == "per_slice"
        assert encoder.windowed_attention_layers == frozenset({0, 2})
        # Default is full attention on the non-windowed blocks.
        assert EncoderConfig(names, embedding_size=16).non_windowed_attention == "full"

    def test_encoder_config_rejects_bad_non_windowed_attention(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Unknown modes are rejected; per_slice needs a window size and layer subset."""
        names = [m.name for m in supported_modalities]
        config = EncoderConfig(
            names,
            embedding_size=16,
            num_heads=2,
            windowed_attention_size=3,
            windowed_attention_layers=[0],
            non_windowed_attention="bogus",
        )
        with pytest.raises(ValueError, match="non_windowed_attention must be one of"):
            config.validate()
        # per_slice with every block windowed (no layer subset) has nothing to act on.
        config = EncoderConfig(
            names,
            embedding_size=16,
            num_heads=2,
            windowed_attention_size=3,
            non_windowed_attention="per_slice",
        )
        with pytest.raises(ValueError, match="requires windowed_attention_size and"):
            config.validate()
        # per_slice without any windowing at all.
        config = EncoderConfig(
            names, embedding_size=16, num_heads=2, non_windowed_attention="per_slice"
        )
        with pytest.raises(ValueError, match="requires windowed_attention_size and"):
            config.validate()


class TestPredictor:
    """Unit tests for the Predictor class."""

    @pytest.fixture
    def predictor(self, supported_modalities: list[ModalitySpec]) -> Predictor:
        """Create predictor fixture for testing."""
        return Predictor(
            supported_modalities=supported_modalities,
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            depth=2,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.1,
            output_embedding_size=8,
        )

    def test_add_masks(self, predictor: Predictor) -> None:
        """Test adding masks to tokens."""
        B, H, W, T, C, D = (
            1,
            2,
            2,
            1,
            2,
            8,
        )  # Changed D from 16 to 8 to match predictor's embedding size
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, C, D)
        sentinel2_l2a_mask = torch.zeros(B, H, W, T, C, dtype=torch.float32)
        # Set one pixel to be decoded (mask value 2)
        sentinel2_l2a_mask[0, 0, 0, 0, 0] = MaskValue.DECODER.value

        latlon = torch.randn(B, 2, D)
        latlon_mask = torch.zeros(B, 2, dtype=torch.float32)

        tokens_and_masks = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        replaced_dict = predictor.add_masks(tokens_and_masks)

        # We expect replaced_dict to have the key "sentinel2_l2a"
        assert "sentinel2_l2a" in replaced_dict, (
            "Expected replaced_dict to have key 'sentinel2_l2a'"
        )

        replaced_sentinel2_l2a = replaced_dict["sentinel2_l2a"]
        assert replaced_sentinel2_l2a.shape == sentinel2_l2a_tokens.shape, (
            f"Expected shape {sentinel2_l2a_tokens.shape}, "
            f"got {replaced_sentinel2_l2a.shape}"
        )

        # Check the single pixel we set to be decoded
        replaced_location = replaced_sentinel2_l2a[0, 0, 0, 0, 0, :]

        # Check an unchanged location
        unchanged_location = replaced_sentinel2_l2a[0, 0, 0, 0, 1, :]

        assert torch.allclose(replaced_location, predictor.mask_token, atol=1e-6), (
            "Tokens at masked location should be replaced with mask token."
        )
        assert torch.allclose(
            unchanged_location, sentinel2_l2a_tokens[0, 0, 0, 0, 1, :], atol=1e-6
        ), "Tokens at non-masked location should remain the same."

    def test_split_x_y(self) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups."""
        tokens = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18]]
        ).unsqueeze(-1)
        # should we handle target encoder values here?
        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                ],
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                ],
            ]
        )

        # Add some missing tokens (value MISSING)
        mask[0, 0] = MaskValue.MISSING.value  # First token in first batch is missing
        mask[1, 1] = MaskValue.MISSING.value  # Second token in second batch is missing

        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_decoded_tokens,
            max_length_of_unmasked_tokens,
        ) = Predictor.split_x_y(tokens, mask)
        # Check shapes
        assert unmasked_tokens.shape == (2, 6, 1)
        assert tokens_to_decode.shape == (2, 3, 1)

        expected_unmasked_tokens = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [10, 12, 13, 14, 15, 16]]
        )
        assert torch.equal(unmasked_tokens.squeeze(-1), expected_unmasked_tokens)
        assert torch.equal(
            unmasked_tokens_mask, torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        )

        expected_tokens_to_decode = torch.tensor([[7, 8, 9], [17, 18, 11]])
        assert torch.equal(tokens_to_decode.squeeze(-1), expected_tokens_to_decode)
        assert torch.equal(tokens_to_decode_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
        assert torch.equal(seqlens_tokens_to_decode, torch.tensor([3, 2]))
        assert torch.equal(seqlens_unmasked_tokens, torch.tensor([5, 6]))
        assert max_length_of_decoded_tokens == 3
        assert max_length_of_unmasked_tokens == 6

    def test_split_and_recombine_with_missing_tokens(self) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups with missing tokens."""
        # Create a batch with two samples, with tokens in a non-sorted order
        # and different numbers of missing tokens per batch
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],  # First batch
                [10, 11, 12, 13, 14, 15, 16, 17, 18],  # Second batch
            ]
        ).unsqueeze(-1)

        # Create masks with different patterns of missing tokens
        # First batch: 1 missing token, 3 decoder tokens, 5 encoder tokens
        # Second batch: 3 missing tokens, 2 decoder tokens, 4 encoder tokens
        # The tokens are intentionally not sorted by mask value
        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
                [
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
            ]
        )

        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            _,
            _,
            _,
            _,
        ) = Predictor.split_x_y(tokens, mask)

        # Check shapes
        assert tokens_to_decode.shape == (2, 3, 1)
        assert unmasked_tokens.shape == (2, 5, 1)

        expected_unmasked_tokens = torch.tensor([[1, 3, 5, 7, 9], [17, 11, 14, 16, 18]])
        assert torch.equal(unmasked_tokens.squeeze(-1), expected_unmasked_tokens)

        expected_unmasked_tokens_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        assert torch.equal(unmasked_tokens_mask, expected_unmasked_tokens_mask)

        expected_tokens_to_decode = torch.tensor([[2, 6, 8], [12, 15, 10]])
        assert torch.equal(tokens_to_decode.squeeze(-1), expected_tokens_to_decode)

        expected_tokens_to_decode_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        assert torch.equal(tokens_to_decode_mask, expected_tokens_to_decode_mask)

        # Test that we can combine the tokens back correctly
        combined_tokens = Predictor.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        # Check the shape of the combined tokens
        assert combined_tokens.shape == tokens.shape
        missing_mask = mask == MaskValue.MISSING.value
        # Check that all values are the same but missing values in mask are set to 0
        assert (combined_tokens[missing_mask] == 0).all()
        target_encoder_only_mask = mask == MaskValue.TARGET_ENCODER_ONLY.value
        missing_or_target_encoder_only_mask = missing_mask | target_encoder_only_mask
        # Ensuring Encode decode tokens are put back together correctly
        assert torch.equal(
            combined_tokens[~missing_or_target_encoder_only_mask],
            tokens[~missing_or_target_encoder_only_mask],
        )

    def test_split_and_recombine_with_missing_tokens_and_target_encoder_only_tokens(
        self,
    ) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups with missing tokens and target encoder only tokens."""
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],  # First batch
                [10, 11, 12, 13, 14, 15, 16, 17, 18],  # Second batch
            ]
        ).unsqueeze(-1)

        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
                [
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.TARGET_ENCODER_ONLY.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
            ]
        )

        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            _,
            _,
            _,
            _,
        ) = Predictor.split_x_y(tokens, mask)

        combined_tokens = Predictor.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        # check that it is zero wherever there is a target encoder only token
        target_encoder_only_mask = mask == MaskValue.TARGET_ENCODER_ONLY.value
        missing_mask = mask == MaskValue.MISSING.value
        missing_or_target_encoder_only_mask = missing_mask | target_encoder_only_mask
        assert (combined_tokens[missing_or_target_encoder_only_mask] == 0).all()
        assert torch.equal(
            combined_tokens[~missing_or_target_encoder_only_mask],
            tokens[~missing_or_target_encoder_only_mask],
        )

    def test_combine_x_y(self) -> None:
        """Test combining the decoded, unmasked, and missing groups back into the original tokens."""
        # x is the query (i.e. the masked tokens)
        tokens_to_decode = torch.tensor([[14, 15, 16], [15, 16, 1]]).unsqueeze(-1)
        # y is the keys and values (i.e. the unmasked tokens)
        unmasked_tokens = torch.tensor([[5, 6, 7, 8], [4, 5, 6, 7]]).unsqueeze(-1)
        tokens_to_decode_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        unmasked_tokens_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        indices = torch.tensor(
            [[6, 7, 8, 4, 5, 0, 1, 2, 3], [7, 8, 3, 4, 5, 6, 0, 1, 2]]
        )

        tokens = Predictor.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        assert torch.equal(
            tokens,
            torch.tensor(
                [[5, 6, 7, 8, 0, 0, 14, 15, 16], [5, 6, 7, 0, 0, 0, 0, 15, 16]]
            ).unsqueeze(-1),
        )

    def test_predictor_config(self, supported_modalities: list[ModalitySpec]) -> None:
        """Tests we can build with default args."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = PredictorConfig(supported_modality_names)
        _ = config.build()

    def test_predictor_config_rope(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Tests we can build a predictor with 2D RoPE."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = PredictorConfig(
            supported_modality_names,
            decoder_embedding_size=16,
            num_heads=2,
            position_encoding="rope",
            rope_base=5000.0,
            rope_coordinate_scale=0.5,
        )
        predictor = config.build()
        assert predictor.position_encoding == "rope"
        assert predictor.rope_base == 5000.0
        assert predictor.rope_coordinate_scale == 0.5
        assert predictor.blocks[0].attn.rope_base == 5000.0

    def test_predictor_config_rope_mixed(
        self, supported_modalities: list[ModalitySpec]
    ) -> None:
        """Tests we can build a predictor with RoPE-Mixed."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = PredictorConfig(
            supported_modality_names,
            decoder_embedding_size=16,
            num_heads=2,
            position_encoding="rope_mixed",
            rope_mixed_base=5.0,
        )
        predictor = config.build()
        assert predictor.position_encoding == "rope_mixed"
        assert predictor.rope_mixed_base == 5.0
        attn = predictor.blocks[0].attn
        assert attn.position_encoding == "rope_mixed"
        assert attn.rope_mixed_freqs.shape == (2, 2, 4)


class TestTokensAndMasks:
    """Test TestTokensAndMasks."""

    def test_flatten_tokens_and_masks(self) -> None:
        """Test TokensAndMasks.flatten_all_tokens_and_masks."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        sentinel_2_mask[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )
        x, mask = t_and_m.flatten_all_tokens_and_masks()

        assert x.shape == (b, h * w * t, d)
        assert mask.shape == (b, h * w * t)
        assert (x[mask.bool()] == 0).all()
        assert (x[(1 - mask).bool()] == 1).all()

    def test_pool_unmasked_tokens(self) -> None:
        """Test TokensAndMasks.pool_unmasked_tokens."""
        b, h, w, t, b_s, d = 2, 4, 4, 3, 1, 128
        # Setup for mean pooling
        sentinel_2_mean = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_mean[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask_mean = torch.zeros((b, h, w, t, b_s)).long()
        sentinel_2_mask_mean[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m_mean = TokensAndMasks(
            sentinel2_l2a=sentinel_2_mean, sentinel2_l2a_mask=sentinel_2_mask_mean
        )
        # Setup for max pooling
        sentinel_2_max = torch.ones((b, h, w, t, b_s, d)) * 2  # set all tokens to 2
        sentinel_2_max[0, 0, 0, 0, :] = 3  # set one "token" to 3s for max pooling
        sentinel_2_mask_max = torch.zeros((b, h, w, t, b_s)).long()
        sentinel_2_mask_max[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m_max = TokensAndMasks(
            sentinel2_l2a=sentinel_2_max, sentinel2_l2a_mask=sentinel_2_mask_max
        )

        # Test max pooling
        pooled_max = pool_unmasked_tokens(
            t_and_m_max, PoolingType.MAX, spatial_pooling=False
        )
        assert pooled_max.shape == (b, d)
        assert (pooled_max == 2).all()  # check the 3 tokens have been ignored

        # Test mean pooling
        pooled_mean = pool_unmasked_tokens(
            t_and_m_mean, PoolingType.MEAN, spatial_pooling=False
        )
        assert pooled_mean.shape == (b, d)
        assert (pooled_mean == 1).all()  # check the 0 tokens have been ignored

    def test_spatial_pool_unmasked_tokens(self) -> None:
        """Test TokensAndMasks.pool_unmasked_tokens."""
        b, h, w, t, b_s, d = 2, 4, 4, 3, 3, 128
        # Setup for mean pooling
        sentinel_2_mean = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_mask_mean = torch.zeros((b, h, w, t, b_s)).long()
        # s1 should be ignored since its masked
        sentinel_1_mean = torch.ones((b, h, w, t, b_s, d)) * 2
        sentinel_1_mask_mean = torch.ones((b, h, w, t, b_s)).long()
        t_and_m_mean = TokensAndMasks(
            sentinel2_l2a=sentinel_2_mean,
            sentinel2_l2a_mask=sentinel_2_mask_mean,
            sentinel1=sentinel_1_mean,
            sentinel1_mask=sentinel_1_mask_mean,
        )

        # Test mean pooling
        pooled_mean = pool_unmasked_tokens(
            t_and_m_mean, PoolingType.MEAN, spatial_pooling=True
        )
        assert pooled_mean.shape == (b, h, w, d)
        assert (pooled_mean == 1).all()  # check the sen1 tokens have been ignored

        # Setup for max pooling
        sentinel_2_max = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_max[:, :, :, 0] = 2  # set one timestep to 2s
        sentinel_2_mask_max = torch.zeros((b, h, w, t, b_s)).long()
        # s1 should be ignored since its masked
        sentinel_1_mean = torch.ones((b, h, w, t, b_s, d)) * 2
        sentinel_1_mask_mean = torch.ones((b, h, w, t, b_s)).long()
        t_and_m_max = TokensAndMasks(
            sentinel2_l2a=sentinel_2_max,
            sentinel2_l2a_mask=sentinel_2_mask_max,
            sentinel1=sentinel_1_mean,
            sentinel1_mask=sentinel_1_mask_mean,
        )

        # Test max pooling
        pooled_max = pool_unmasked_tokens(
            t_and_m_max, PoolingType.MAX, spatial_pooling=True
        )
        assert pooled_max.shape == (b, h, w, d)
        assert (pooled_max == 2).all()  # check the 3 tokens have been ignored

    def test_missing_modalities_ignored(self) -> None:
        """Test TokensAndMasks.modalities does not return missing modalities."""
        b, h, w, t, b_s, d = 2, 4, 4, 3, 3, 128
        # Setup for mean pooling
        sentinel_2_mean = torch.ones((b, h, w, t, b_s, d))
        sentinel_2_mask_mean = torch.zeros((b, h, w, t, b_s)).long()
        # s1 should be ignored since its masked
        sentinel_1_mean = torch.ones((b, h, w, t, b_s, d)) * 2
        sentinel_1_mask_mean = torch.ones((b, h, w, t, b_s)).long()
        t_and_m_mean = TokensAndMasks(
            sentinel2_l2a=sentinel_2_mean,
            sentinel2_l2a_mask=sentinel_2_mask_mean,
            sentinel1=sentinel_1_mean,
            sentinel1_mask=sentinel_1_mask_mean,
        )

        modalities = t_and_m_mean.modalities
        assert len(modalities) == 2  # s2, s1
        assert set(modalities) == set(["sentinel2_l2a", "sentinel1"])


class TestProjectionAndAggregation:
    """Test ProjectAndAggregate."""

    def test_layer_in_all_configs(self) -> None:
        """Test ProjectAndAggregate."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        sentinel_2_mask[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )

        for i in [1, 2, 3]:
            for pre_aggregate in [True, False]:
                layer = ProjectAndAggregate(
                    embedding_size=d, num_layers=i, aggregate_then_project=pre_aggregate
                )
                # for now, lets just check it all runs properly
                _ = layer(t_and_m)

    def test_output_embedding_size(self) -> None:
        """Test output_embedding_size changes output dimension."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        out_d = 64
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )

        for num_layers in [1, 2, 3]:
            for agg_first in [True, False]:
                layer = ProjectAndAggregate(
                    embedding_size=d,
                    num_layers=num_layers,
                    aggregate_then_project=agg_first,
                    output_embedding_size=out_d,
                )
                out = layer(t_and_m)
                assert out.shape == (b, out_d), (
                    f"Expected (b, {out_d}), got {out.shape}"
                )

        # Also test with raw tensor input
        x_tensor = torch.ones((b, h * w * t, d))
        layer = ProjectAndAggregate(
            embedding_size=d,
            num_layers=2,
            aggregate_then_project=True,
            output_embedding_size=out_d,
        )
        out = layer(x_tensor)
        assert out.shape == (b, out_d)

    def test_only_project(self) -> None:
        """Test only_project returns tokens without aggregation."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )

        layer = ProjectAndAggregate(embedding_size=d, num_layers=2, only_project=True)
        out = layer(t_and_m)
        # Should return TokensAndMasks, not aggregated tensor
        assert isinstance(out, TokensAndMasks)
        assert out.sentinel2_l2a is not None
        assert out.sentinel2_l2a.shape == (b, h, w, t, d)

        # Test with raw tensor - should preserve token structure
        x_tensor = torch.ones((b, h * w * t, d))
        out_tensor = layer(x_tensor)
        assert isinstance(out_tensor, torch.Tensor)
        assert out_tensor.shape == (b, h * w * t, d)

    def test_only_project_with_output_embedding_size(self) -> None:
        """Test only_project combined with output_embedding_size."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        out_d = 64
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )

        layer = ProjectAndAggregate(
            embedding_size=d,
            num_layers=2,
            only_project=True,
            output_embedding_size=out_d,
        )
        out = layer(t_and_m)
        assert isinstance(out, TokensAndMasks)
        assert out.sentinel2_l2a is not None
        # Output tokens should have projected dimension
        assert out.sentinel2_l2a.shape == (b, h, w, t, out_d)

        # Test with raw tensor
        x_tensor = torch.ones((b, h * w * t, d))
        out_tensor = layer(x_tensor)
        assert out_tensor.shape == (b, h * w * t, out_d)


class TestBandDropout:
    """Unit tests for band dropout in MultiModalPatchEmbeddings."""

    def test_apply_band_dropout_zeros_some_bands(self) -> None:
        """Test that _apply_band_dropout zeros out some bands."""
        torch.manual_seed(42)
        B, H, W, num_bands = 4, 2, 2, 12
        data = torch.ones(B, H, W, num_bands)
        result = MultiModalPatchEmbeddings._apply_band_dropout(data, rate=0.5)
        # Some bands should be zeroed
        assert (result == 0).any(), "Expected some bands to be dropped"
        # Some bands should be kept
        assert (result == 1).any(), "Expected some bands to be kept"

    def test_apply_band_dropout_at_least_one_band_kept(self) -> None:
        """Test that at least one band is kept per sample even at rate=1.0."""
        torch.manual_seed(0)
        B, num_bands = 8, 6
        data = torch.ones(B, num_bands)
        result = MultiModalPatchEmbeddings._apply_band_dropout(data, rate=1.0)
        # Every sample must have at least one non-zero band
        per_sample_sum = result.sum(dim=-1)
        assert (per_sample_sum > 0).all(), "Each sample must keep at least 1 band"

    def test_apply_band_dropout_rate_zero_no_change(self) -> None:
        """Test that rate=0.0 keeps all bands."""
        B, num_bands = 4, 10
        data = torch.randn(B, num_bands)
        result = MultiModalPatchEmbeddings._apply_band_dropout(data, rate=0.0)
        assert torch.equal(result, data), "rate=0.0 should not modify data"

    def test_band_dropout_not_applied_when_rate_zero(self) -> None:
        """Test that when dropout rate is 0.0, no values are zeroed."""
        B, num_bands = 4, 12
        data = torch.ones(B, num_bands)
        result = MultiModalPatchEmbeddings._apply_band_dropout(data, rate=0.0)
        assert (result != 0).all(), "No values should be zero when dropout rate is 0.0"

    def test_band_dropout_not_applied_in_eval(self) -> None:
        """Test that band dropout is gated by self.training (eval mode skips it)."""
        torch.manual_seed(42)
        B, num_bands = 4, 12
        data = torch.ones(B, num_bands)
        embed = MultiModalPatchEmbeddings(
            supported_modality_names=["sentinel2_l2a"],
            max_patch_size=8,
            embedding_size=16,
            band_dropout_rate=0.5,
            random_band_dropout=False,
        )
        # In train mode, dropout should zero some bands
        embed.train()
        result_train = MultiModalPatchEmbeddings._apply_band_dropout(data, rate=0.5)
        assert (result_train == 0).any(), "Dropout should zero some bands in train mode"
        # In eval mode, the caller gates dropout with self.training
        embed.eval()
        assert not embed.training, "Module should be in eval mode"
        # Since self.training is False, dropout is never called — data stays intact
        assert (data != 0).all(), (
            "No values should be zero when eval mode skips dropout"
        )

    def test_band_dropout_disabled_by_default(self) -> None:
        """Test that Encoder leaves band dropout disabled at construction."""
        encoder = Encoder(
            embedding_size=8,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
            max_sequence_length=12,
            band_dropout_rate=0.5,
            random_band_dropout=True,
        )
        # Configured rate is stored on the encoder but the patch embeddings
        # start with rate 0.0 so band dropout is inactive until enabled.
        assert encoder.band_dropout_rate == 0.5
        assert encoder.patch_embeddings.band_dropout_rate == 0.0

    def test_enable_band_dropout(self) -> None:
        """Test Encoder.enable_band_dropout activates the configured rate."""
        encoder = Encoder(
            embedding_size=8,
            max_patch_size=8,
            min_patch_size=1,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
            max_sequence_length=12,
            band_dropout_rate=0.5,
            random_band_dropout=True,
        )
        encoder.enable_band_dropout()
        assert encoder.patch_embeddings.band_dropout_rate == 0.5
