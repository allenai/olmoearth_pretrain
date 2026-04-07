"""Tests for TextEmbeddingTargetGenerator."""

import tempfile

import torch
import torch.nn.functional as F

from olmoearth_pretrain.nn.text_targets import TextEmbeddingTargetGenerator


def _make_embeddings_file(path: str) -> None:
    """Create a minimal embeddings .pt file for testing."""
    wc_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    wc_embs = F.normalize(torch.randn(5, 768), dim=-1)
    osm_embs = F.normalize(torch.randn(30, 768), dim=-1)
    wc_crop_embs = F.normalize(torch.randn(8, 768), dim=-1)
    torch.save(
        {
            "worldcover": {"class_ids": wc_ids, "embeddings": wc_embs},
            "openstreetmap_raster": {"embeddings": osm_embs},
            "worldcereal": {"embeddings": wc_crop_embs},
        },
        path,
    )


def test_discrete_worldcover_targets():
    """Test WorldCover (discrete single-band) target generation."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_embeddings_file(f.name)
        gen = TextEmbeddingTargetGenerator(f.name, ["worldcover"])

        B, H, W, T, C = 2, 16, 16, 1, 1
        patch_size = 8
        # All pixels are class 10 → normalized value = 0.1
        raw = torch.full((B, H, W, T, C), 0.1)

        targets = gen.compute_targets("worldcover", raw, patch_size)
        assert targets.shape == (B, H // patch_size, W // patch_size, 1, 1, 768)

        # All pixels same class → target should be the class embedding
        expected = gen.worldcover_lookup[10]
        actual = targets[0, 0, 0, 0, 0]
        assert torch.allclose(actual.float(), expected.float(), atol=1e-5)


def test_discrete_mixed_patch():
    """Test WorldCover with mixed classes in a patch."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_embeddings_file(f.name)
        gen = TextEmbeddingTargetGenerator(f.name, ["worldcover"])

        B, H, W, T, C = 1, 8, 8, 1, 1
        patch_size = 8
        raw = torch.zeros(B, H, W, T, C)
        # Half class 10 (norm=0.1), half class 20 (norm=0.2)
        raw[0, :4, :, 0, 0] = 0.1
        raw[0, 4:, :, 0, 0] = 0.2

        targets = gen.compute_targets("worldcover", raw, patch_size)
        assert targets.shape == (1, 1, 1, 1, 1, 768)

        expected = F.normalize(
            (gen.worldcover_lookup[10] + gen.worldcover_lookup[20]) / 2, dim=-1
        )
        actual = targets[0, 0, 0, 0, 0]
        assert torch.allclose(actual.float(), expected.float(), atol=1e-4)


def test_multichannel_osm_targets():
    """Test OSM raster (multi-channel binary) target generation."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_embeddings_file(f.name)
        gen = TextEmbeddingTargetGenerator(f.name, ["openstreetmap_raster"])

        B, H, W, T, C = 2, 16, 16, 1, 30
        patch_size = 8
        # All zeros except channel 4 (building) = 1
        raw = torch.zeros(B, H, W, T, C)
        raw[:, :, :, 0, 4] = 1.0

        targets = gen.compute_targets("openstreetmap_raster", raw, patch_size)
        assert targets.shape == (B, H // patch_size, W // patch_size, 1, 1, 768)

        # Only building channel active → target should be the building embedding
        expected = gen.openstreetmap_raster_embeddings[4]
        actual = targets[0, 0, 0, 0, 0]
        cos_sim = F.cosine_similarity(
            actual.float().unsqueeze(0), expected.float().unsqueeze(0)
        )
        assert cos_sim > 0.99


def test_multichannel_worldcereal_targets():
    """Test WorldCereal (multi-channel) target generation."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_embeddings_file(f.name)
        gen = TextEmbeddingTargetGenerator(f.name, ["worldcereal"])

        B, H, W, T, C = 1, 8, 8, 1, 8
        patch_size = 8
        raw = torch.zeros(B, H, W, T, C)
        # Set channel 0 and 2 to 1.0 (fully active)
        raw[:, :, :, 0, 0] = 1.0
        raw[:, :, :, 0, 2] = 1.0

        targets = gen.compute_targets("worldcereal", raw, patch_size)
        assert targets.shape == (1, 1, 1, 1, 1, 768)

        expected = F.normalize(
            gen.worldcereal_embeddings[0] + gen.worldcereal_embeddings[2], dim=-1
        )
        actual = targets[0, 0, 0, 0, 0]
        assert torch.allclose(actual.float(), expected.float(), atol=1e-4)


def test_output_is_unit_normalized():
    """All output embeddings should be unit-normalized."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        _make_embeddings_file(f.name)
        gen = TextEmbeddingTargetGenerator(
            f.name, ["worldcover", "openstreetmap_raster"]
        )

        raw_wc = torch.full((2, 16, 16, 1, 1), 0.3)
        targets_wc = gen.compute_targets("worldcover", raw_wc, 8)
        norms = targets_wc.flatten(0, -2).norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        raw_osm = torch.rand(2, 16, 16, 1, 30)
        targets_osm = gen.compute_targets("openstreetmap_raster", raw_osm, 8)
        norms = targets_osm.flatten(0, -2).norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
