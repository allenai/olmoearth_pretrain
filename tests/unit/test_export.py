"""Unit tests for the static-shape encoder export wrapper."""
from __future__ import annotations

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.export import StaticOlmoEarthEncoder, verify_export
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id


@pytest.fixture(scope="module")
def encoder():
    """Load NANO encoder once for all tests (random weights for speed)."""
    model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
    enc = model.encoder
    if torch.cuda.is_available():
        enc = enc.cuda()
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def static_encoder(encoder):
    """Build static wrapper once for all tests."""
    device = next(encoder.parameters()).device
    static_enc = StaticOlmoEarthEncoder(encoder, patch_size=2, spatial_size=64)
    static_enc = static_enc.to(device).eval()
    return static_enc


def _make_inputs(batch_size=1, device="cpu"):
    """Create dummy inputs for static encoder."""
    x = torch.randn(batch_size, 64, 64, 1, Modality.SENTINEL2_L2A.num_bands, device=device)
    ts = torch.zeros(batch_size, 1, 3, dtype=torch.long, device=device)
    return x, ts


class TestStaticEncoderShapes:
    """Test that static encoder produces correct output shapes."""

    def test_single_sample(self, static_encoder):
        device = next(static_encoder.parameters()).device
        x, ts = _make_inputs(batch_size=1, device=device)
        with torch.no_grad():
            out = static_encoder(x, ts)
        assert out.dim() == 2
        assert out.shape[0] == 1

    def test_batch(self, static_encoder):
        device = next(static_encoder.parameters()).device
        for bs in [2, 4]:
            x, ts = _make_inputs(batch_size=bs, device=device)
            with torch.no_grad():
                out = static_encoder(x, ts)
            assert out.shape[0] == bs
            assert out.dim() == 2

    def test_no_nan_inf(self, static_encoder):
        device = next(static_encoder.parameters()).device
        x, ts = _make_inputs(batch_size=2, device=device)
        with torch.no_grad():
            out = static_encoder(x, ts)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestStaticEncoderMatchesOriginal:
    """Test that static encoder output matches original encoder."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cosine_similarity(self, encoder, static_encoder):
        sim = verify_export(encoder, static_encoder, num_samples=5, device="cuda")
        assert sim > 0.999, f"Cosine similarity too low: {sim:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_deterministic(self, static_encoder):
        """Same input produces same output."""
        device = next(static_encoder.parameters()).device
        torch.manual_seed(123)
        x, ts = _make_inputs(batch_size=1, device=device)
        with torch.no_grad():
            out1 = static_encoder(x, ts)
            out2 = static_encoder(x, ts)
        assert torch.allclose(out1, out2, atol=1e-6)


class TestTorchExport:
    """Test that torch.export works with static encoder."""

    def test_export_succeeds(self, static_encoder):
        device = next(static_encoder.parameters()).device
        x, ts = _make_inputs(batch_size=1, device=device)
        exported = torch.export.export(static_encoder, (x, ts))
        assert exported is not None

    def test_exported_output_matches(self, static_encoder):
        device = next(static_encoder.parameters()).device
        x, ts = _make_inputs(batch_size=1, device=device)
        with torch.no_grad():
            eager_out = static_encoder(x, ts)
        exported = torch.export.export(static_encoder, (x, ts))
        export_out = exported.module()(x, ts)
        sim = torch.nn.functional.cosine_similarity(
            eager_out.flatten(), export_out.flatten(), dim=0
        ).item()
        assert sim > 0.999, f"Exported output diverges: {sim:.6f}"


class TestStateDict:
    """Test that state dict maps correctly to original encoder."""

    def test_has_parameters(self, static_encoder):
        params = dict(static_encoder.named_parameters())
        assert len(params) > 0

    def test_patch_proj_weights_shared(self, encoder, static_encoder):
        """Verify patch projection weights are the same objects (not copies)."""
        orig_key = "sentinel2_l2a__0"
        orig_proj = encoder.patch_embeddings.per_modality_embeddings["sentinel2_l2a"][orig_key].proj
        static_proj = static_encoder.patch_projs[0]
        assert orig_proj is static_proj, "Patch projection should be same object, not a copy"

    def test_blocks_shared(self, encoder, static_encoder):
        """Verify transformer blocks are the same objects."""
        assert encoder.blocks is static_encoder.blocks

    def test_norm_shared(self, encoder, static_encoder):
        """Verify final norm is the same object."""
        assert encoder.norm is static_encoder.norm
