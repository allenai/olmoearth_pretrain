"""Unit tests for local attention mask building and encoder integration."""

import pytest
import torch
from einops import repeat

from olmoearth_pretrain.data.constants import ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    build_local_attention_mask,
    compute_token_spatial_positions,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def _mask_from_dims(
    dims_dict: dict, window_size: int, device: str = "cpu"
) -> torch.Tensor:
    """Helper: compute positions then build mask (convenience for tests)."""
    rows, cols = compute_token_spatial_positions(dims_dict, device=device)
    return build_local_attention_mask(rows, cols, window_size)


class TestBuildLocalAttentionMask:
    """Tests for the build_local_attention_mask utility."""

    def test_single_modality_4x4_full_window(self) -> None:
        """With window >= grid size, every token should attend to every other."""
        mask = _mask_from_dims({"mod": (1, 4, 4, 1, 1, 8)}, window_size=8)
        assert mask.shape == (16, 16)
        assert mask.all()

    def test_single_modality_4x4_small_window(self) -> None:
        """With window_size=2, only immediate neighbors (±1) attend to each other."""
        mask = _mask_from_dims({"mod": (1, 4, 4, 1, 1, 8)}, window_size=2)
        assert mask.shape == (16, 16)
        assert mask[0, 0]  # self
        assert mask[0, 1]  # right neighbor
        assert mask[0, 4]  # below neighbor
        assert mask[0, 5]  # diagonal neighbor
        assert not mask[0, 2]  # col=2, too far
        assert not mask[0, 8]  # row=2, too far

    def test_window_size_1_is_self_only(self) -> None:
        """Window of 1 means radius=0, each token only attends to tokens at same position."""
        mask = _mask_from_dims({"mod": (1, 3, 3, 1, 1, 8)}, window_size=1)
        assert mask.shape == (9, 9)
        assert mask.diagonal().all()
        off_diag = mask.clone()
        off_diag.fill_diagonal_(False)
        assert not off_diag.any()

    def test_temporal_tokens_same_position_attend(self) -> None:
        """Tokens at same (h,w) but different timesteps should attend to each other."""
        mask = _mask_from_dims({"mod": (1, 2, 2, 3, 1, 8)}, window_size=1)
        assert mask.shape == (12, 12)
        assert mask[0, 1]
        assert mask[0, 2]
        assert mask[1, 2]
        assert not mask[0, 3]

    def test_multi_modality(self) -> None:
        """Tokens from two spatial modalities at same position should attend."""
        dims_dict = {
            "mod_a": (1, 2, 2, 1, 1, 8),
            "mod_b": (1, 2, 2, 1, 1, 8),
        }
        mask = _mask_from_dims(dims_dict, window_size=1)
        assert mask.shape == (8, 8)
        assert mask[0, 4]
        assert not mask[0, 5]

    def test_non_spatial_modality_isolated(self) -> None:
        """Non-spatial tokens (3D) get large negative positions, so they only attend to
        each other.
        """
        dims_dict = {
            "spatial": (1, 2, 2, 1, 1, 8),
            "nonspatial": (1, 2, 8),
        }
        mask = _mask_from_dims(dims_dict, window_size=4)
        assert mask.shape == (6, 6)
        assert mask[4, 5]
        assert mask[5, 4]
        assert not mask[4, 0]
        assert not mask[0, 4]

    def test_5d_modality(self) -> None:
        """5D modality (B, H, W, C, D) should work like spatial without time."""
        mask = _mask_from_dims({"mod": (1, 3, 3, 2, 8)}, window_size=2)
        assert mask.shape == (18, 18)
        assert mask[0, 1]

    def test_symmetry(self) -> None:
        """The mask should be symmetric."""
        mask = _mask_from_dims({"mod": (1, 8, 8, 2, 3, 8)}, window_size=4)
        assert (mask == mask.T).all()

    def test_device(self) -> None:
        """Mask should be created on the specified device."""
        rows, cols = compute_token_spatial_positions(
            {"mod": (1, 2, 2, 1, 1, 4)}, device="cpu"
        )
        mask = build_local_attention_mask(rows, cols, window_size=2)
        assert mask.device == torch.device("cpu")

    def test_batched_positions(self) -> None:
        """2D (B, N) positions should produce a (B, N, N) mask."""
        rows = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        cols = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
        mask = build_local_attention_mask(rows, cols, window_size=2)
        assert mask.shape == (2, 4, 4)


class TestEncoderLocalAttention:
    """Integration tests for Encoder with local_attention_window."""

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
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
        )

    @pytest.fixture
    def modality_band_set_len_and_total_bands(
        self,
        supported_modalities: list[ModalitySpec],
    ) -> dict[str, tuple[int, int]]:
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
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=sentinel2_l2a,
            sentinel2_l2a_mask=sentinel2_l2a_mask,
            latlon=latlon,
            latlon_mask=latlon_mask,
            timestamps=timestamps,
        )
        # H/patch=4, W/patch=4 -> 4x4 token grid. Window of 2 = ±1 neighbors.
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
        ).unsqueeze(0)
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=sentinel2_l2a,
            sentinel2_l2a_mask=sentinel2_l2a_mask,
            latlon=latlon,
            latlon_mask=latlon_mask,
            timestamps=timestamps,
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
        )
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(1, 8, 8, 1, 13),
            sentinel2_l2a_mask=torch.zeros(1, 8, 8, 1, 13, dtype=torch.long),
            latlon=torch.randn(1, 2),
            latlon_mask=torch.zeros(1, 2),
            timestamps=torch.tensor([[[15, 7, 2023]]]),
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
        )
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.randn(1, 8, 8, 1, 13),
            sentinel2_l2a_mask=torch.zeros(1, 8, 8, 1, 13, dtype=torch.long),
            latlon=torch.randn(1, 2),
            latlon_mask=torch.zeros(1, 2),
            timestamps=torch.tensor([[[15, 7, 2023]]]),
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

        sentinel2_l2a = torch.randn(B, H, W, T, sentinel2_l2a_num_bands)
        sentinel2_l2a_mask = torch.full(
            (B, H, W, T, sentinel2_l2a_num_bands),
            MaskValue.ONLINE_ENCODER.value,
            dtype=torch.long,
        )
        # Mark one full timestep as MISSING for the first batch element.
        sentinel2_l2a_mask[0, :, :, 2, :] = MaskValue.MISSING.value

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

        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=sentinel2_l2a,
            sentinel2_l2a_mask=sentinel2_l2a_mask,
            latlon=latlon,
            latlon_mask=latlon_mask,
            timestamps=timestamps,
        )
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


class TestMaskAlignmentWithTokens:
    """Verify the local attention mask aligns with actual token spatial positions.

    Mocks Block.forward to capture the (tokens, attn_mask) that attention sees,
    then verifies the mask entries match expected spatial proximity.
    """

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
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
        )

    @pytest.fixture
    def modality_band_set_len_and_total_bands(
        self,
        supported_modalities: list[ModalitySpec],
    ) -> dict[str, tuple[int, int]]:
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
        """With fast_pass=True, captured mask must equal independently computed mask."""
        _, n_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"]
        _, latlon_num_bands = modality_band_set_len_and_total_bands["latlon"]
        B, H, W, T = 1, 16, 16, 3
        patch_size = 4
        window_size = 2
        sample = _make_sample(B, H, W, T, n_bands, latlon_num_bands)

        captured_positions: dict[str, torch.Tensor] = {}
        orig_compute = compute_token_spatial_positions

        def capturing_compute(*args, **kwargs):
            rows, cols = orig_compute(*args, **kwargs)
            captured_positions["rows"] = rows
            captured_positions["cols"] = cols
            return rows, cols

        monkeypatch.setattr(
            "olmoearth_pretrain.nn.flexi_vit.compute_token_spatial_positions",
            capturing_compute,
        )

        captured_block: dict[str, torch.Tensor] = {}
        orig_forward = encoder.blocks[0].forward

        def capturing_forward(*args, **kwargs):
            captured_block["attn_mask"] = kwargs.get("attn_mask")
            captured_block["x"] = args[0] if args else kwargs.get("x")
            return orig_forward(*args, **kwargs)

        monkeypatch.setattr(encoder.blocks[0], "forward", capturing_forward)

        encoder(
            sample,
            patch_size=patch_size,
            fast_pass=True,
            local_attention_window=window_size,
        )

        mask = captured_block["attn_mask"]
        rows = captured_positions["rows"]
        cols = captured_positions["cols"]

        # fast_pass=True -> (N, N) 2D mask
        N = rows.shape[0]
        assert mask.shape == (N, N)

        # The captured mask should exactly match an independently built mask
        # from the captured positions.
        expected = build_local_attention_mask(rows, cols, window_size)
        assert (mask == expected).all()

        # Spot-check: two tokens at the same (row, col) should attend to each other.
        radius = window_size // 2
        same_pos = (rows.unsqueeze(0) == rows.unsqueeze(1)) & (
            cols.unsqueeze(0) == cols.unsqueeze(1)
        )
        assert mask[same_pos].all(), "Same-position tokens must attend to each other"

        # Spot-check: find a far-apart pair and verify it's blocked.
        far = (rows.unsqueeze(0) - rows.unsqueeze(1)).abs() > radius
        assert not mask[far].any(), "Far-apart tokens must not attend to each other"

    @torch.inference_mode()
    def test_mask_alignment_fast_pass_false(
        self,
        encoder: Encoder,
        modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With missing tokens (fast_pass=False), captured 3D mask must be correct."""
        _, n_bands = modality_band_set_len_and_total_bands["sentinel2_l2a"]
        _, latlon_num_bands = modality_band_set_len_and_total_bands["latlon"]
        B, H, W, T = 2, 16, 16, 3
        patch_size = 4
        window_size = 2
        sample = _make_sample(B, H, W, T, n_bands, latlon_num_bands)

        # Mark timestep 2 as MISSING for batch element 0 only.
        sample.sentinel2_l2a_mask[0, :, :, 2, :] = MaskValue.MISSING.value

        # Capture the (rows, cols) fed to build_local_attention_mask.
        captured_positions: dict[str, torch.Tensor] = {}
        orig_build = build_local_attention_mask

        def capturing_build(rows, cols, window_size):
            captured_positions["rows"] = rows.clone()
            captured_positions["cols"] = cols.clone()
            return orig_build(rows, cols, window_size)

        monkeypatch.setattr(
            "olmoearth_pretrain.nn.flexi_vit.build_local_attention_mask",
            capturing_build,
        )

        # Capture the final attn_mask passed to the first transformer block.
        captured_block: dict[str, torch.Tensor] = {}
        orig_forward = encoder.blocks[0].forward

        def capturing_forward(*args, **kwargs):
            captured_block["attn_mask"] = kwargs.get("attn_mask")
            captured_block["x"] = args[0] if args else kwargs.get("x")
            return orig_forward(*args, **kwargs)

        monkeypatch.setattr(encoder.blocks[0], "forward", capturing_forward)

        encoder(
            sample,
            patch_size=patch_size,
            fast_pass=False,
            local_attention_window=window_size,
        )

        mask = captured_block["attn_mask"]
        rows = captured_positions["rows"]
        cols = captured_positions["cols"]

        # fast_pass=False -> 3D mask (B, N_kept, N_kept)
        assert mask.ndim == 3
        assert mask.shape[0] == B
        N_kept = mask.shape[1]
        assert rows.shape == (B, N_kept)

        # Batch 0 lost one timestep -> fewer valid tokens than batch 1.
        # Self-attention entries along diagonal tell us which positions are valid.
        valid_b0 = mask[0].diagonal().sum().item()
        valid_b1 = mask[1].diagonal().sum().item()
        assert valid_b0 < valid_b1, (
            "Batch 0 (with missing timestep) should have fewer valid tokens"
        )

        # For valid tokens in batch 1 (all valid), verify spatial proximity.
        radius = window_size // 2
        for i in range(0, N_kept, N_kept // 5):
            if not mask[1, i, i]:
                continue
            ri, ci = rows[1, i].item(), cols[1, i].item()
            for j in range(N_kept):
                if not mask[1, j, j]:
                    continue
                rj, cj = rows[1, j].item(), cols[1, j].item()
                spatially_close = abs(ri - rj) <= radius and abs(ci - cj) <= radius
                assert mask[1, i, j] == spatially_close, (
                    f"batch 1, token {i} ({ri},{ci}) vs {j} ({rj},{cj}): "
                    f"mask={mask[1, i, j].item()}, expected={spatially_close}"
                )

        # Padded positions in batch 0 should have all-False columns.
        for j in range(N_kept):
            if not mask[0, j, j]:
                assert not mask[0, :, j].any(), (
                    f"Padded column {j} in batch 0 should be all-False"
                )
