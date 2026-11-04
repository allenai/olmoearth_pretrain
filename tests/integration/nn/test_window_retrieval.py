"""Tests for the window retriever and the window-retrieval loss."""

from pathlib import Path

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import (
    AttentionPoolerConfig,
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig, load_init_weights
from olmoearth_pretrain.nn.location_token_distillation_head import (
    LocationTokenDistillationHeadConfig,
)
from olmoearth_pretrain.nn.window_retriever import WindowFields, WindowRetrieverConfig
from olmoearth_pretrain.train.window_retrieval import (
    slice_window,
    valid_steps,
    window_retrieval_loss,
)

MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
B, H, W, T = 2, 8, 8, 6
STUDENT_DIM = 8


def _config(arch: str = "lowrank", with_retriever: bool = True) -> LatentMIMConfig:
    encoder = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=16,
        num_heads=2,
        depth=2,
        mlp_ratio=4.0,
        max_patch_size=8,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        attention_pooler_config=AttentionPoolerConfig(
            pooler_dim=16, pooler_depth=1, student_dims=[STUDENT_DIM, 4]
        ),
    )
    decoder = PredictorConfig(
        supported_modality_names=MODALITIES,
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=4.0,
        max_sequence_length=12,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_attention_pooler=True,
        pooler_dim=16,
    )
    return LatentMIMConfig(
        encoder_config=encoder,
        decoder_config=decoder,
        location_token_distillation_head_config=LocationTokenDistillationHeadConfig(),
        window_retriever_config=(
            WindowRetrieverConfig(
                dim=STUDENT_DIM, arch=arch, rank=4, code_dim=8, width=16
            )
            if with_retriever
            else None
        ),
    )


def _sample(missing_last: int = 0) -> MaskedOlmoEarthSample:
    """A monthly series starting in a different month per sample."""
    bands = Modality.SENTINEL2_L2A.num_bands
    s2 = torch.randn(B, H, W, T, bands)
    mask = torch.full(
        (B, H, W, T, Modality.SENTINEL2_L2A.num_band_sets),
        MaskValue.ONLINE_ENCODER.value,
    )
    if missing_last:
        mask[1, :, :, T - missing_last :] = MaskValue.MISSING.value
    first = torch.tensor([0, 7])
    month = (first[:, None] + torch.arange(T)) % 12
    year = 2019 + (first[:, None] + torch.arange(T)) // 12
    timestamps = torch.stack([torch.ones_like(month), month, year], dim=-1)
    latlon = torch.randn(B, Modality.LATLON.num_bands)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=mask,
        latlon=latlon,
        latlon_mask=torch.full_like(latlon, MaskValue.ONLINE_ENCODER.value),
        timestamps=timestamps,
    )


def test_valid_steps_and_slice() -> None:
    """Trailing MISSING steps shorten the valid range; slicing is per sample."""
    x = _sample(missing_last=2)
    assert valid_steps(x).tolist() == [T, T - 2]
    starts = torch.tensor([1, 3])
    sub = slice_window(x, starts, 2)
    assert sub.sentinel2_l2a is not None and x.sentinel2_l2a is not None
    assert sub.sentinel2_l2a.shape[3] == 2
    assert torch.equal(sub.sentinel2_l2a[0], x.sentinel2_l2a[0, :, :, 1:3])
    assert torch.equal(sub.sentinel2_l2a[1], x.sentinel2_l2a[1, :, :, 3:5])
    assert sub.timestamps is not None and x.timestamps is not None
    assert torch.equal(sub.timestamps[1], x.timestamps[1, 3:5])
    assert torch.equal(sub.latlon, x.latlon)


@pytest.mark.parametrize("arch", ["lowrank", "film"])
def test_retriever_starts_at_identity(arch: str) -> None:
    """Zero-initialized window terms: the retriever returns z at init."""
    retriever_config = _config(arch).window_retriever_config
    assert retriever_config is not None
    retriever = retriever_config.build()
    z = torch.randn(B, 4, STUDENT_DIM)
    fields = WindowFields(
        *(torch.tensor(v) for v in ([0, 7], [6, 6], [2, 9], [4, 11], [3, 3]))
    )
    assert torch.allclose(retriever(z, fields), z)


@pytest.mark.parametrize("centered_weight", [0.0, 3.0])
def test_loss_trains_pooler_and_retriever(centered_weight: float) -> None:
    """The retrieval loss reaches the retriever, the student and the pooler."""
    model = _config().build()
    x = _sample()
    out = model(x, patch_size=4)
    student_outputs = out[-1]
    loss, metrics = window_retrieval_loss(
        model,
        x,
        4,
        None,
        student_outputs["location_tokens"],
        2,
        4,
        centered_weight=centered_weight,
    )
    assert torch.isfinite(loss) and metrics["window_retrieval/skipped"] == 0
    expected = (
        metrics["window_retrieval/mse"]
        + centered_weight * metrics["window_retrieval/centered_mse"]
    )
    assert torch.allclose(loss.detach(), expected)
    loss.backward()
    assert model.window_retriever is not None
    assert all(p.grad is not None for p in model.window_retriever.parameters())
    pooler = model.encoder.attention_pooler
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in pooler.named_parameters()
        if not n.startswith("student")
    )


def test_loss_skips_with_zero_grads() -> None:
    """Too few steps for a window: zero loss, but every retriever param has a grad."""
    model = _config().build()
    x = _sample()
    out = model(x, patch_size=4)
    loss, metrics = window_retrieval_loss(
        model, x, 4, None, out[-1]["location_tokens"], T, T
    )
    assert loss.item() == 0 and metrics["window_retrieval/skipped"] == 1
    loss.backward()
    assert model.window_retriever is not None
    assert all(
        p.grad is not None and torch.all(p.grad == 0)
        for p in model.window_retriever.parameters()
    )


def test_init_weights_allows_only_missing_retriever(tmp_path: Path) -> None:
    """A checkpoint without the retriever loads; extra keys are refused."""
    source = _config(with_retriever=False).build()
    path = tmp_path / "weights.pth"
    torch.save(source.state_dict(), path)
    model = _config().build()
    load_init_weights(model, str(path))
    assert torch.equal(
        model.encoder.attention_pooler.student[0].weight,
        source.encoder.attention_pooler.student[0].weight,
    )

    state = source.state_dict()
    state["extra.weight"] = torch.zeros(1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="unexpected"):
        load_init_weights(_config().build(), str(path))
