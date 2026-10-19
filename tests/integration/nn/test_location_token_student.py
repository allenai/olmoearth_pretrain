"""Tests for the detached low-dim student projection ("student")."""

import logging

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import (
    AttentionPoolerConfig,
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.location_token_distillation_head import (
    LocationTokenDistillationHead,
    LocationTokenDistillationHeadConfig,
)
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

SUPPORTED_MODALITIES = [Modality.SENTINEL2_L2A, Modality.LATLON, Modality.WORLDCOVER]
POOLER_DIM = 16
PROJECTION_DIMS = [8, 4]


def _encoder_config(with_student: bool) -> EncoderConfig:
    config = EncoderConfig(
        supported_modality_names=[m.name for m in SUPPORTED_MODALITIES],
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
            pooler_dim=POOLER_DIM,
            pooler_depth=2,
        ),
    )
    if with_student:
        assert config.attention_pooler_config is not None
        config.attention_pooler_config.student_dims = list(PROJECTION_DIMS)
    return config


def _latent_mim_config(
    with_student: bool,
    with_supervision: bool = True,
) -> LatentMIMConfig:
    decoder_config = PredictorConfig(
        supported_modality_names=[m.name for m in SUPPORTED_MODALITIES],
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        num_heads=2,
        depth=2,
        mlp_ratio=4.0,
        max_sequence_length=12,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_attention_pooler=True,
        pooler_dim=POOLER_DIM,
    )
    supervision_config = None
    if with_supervision:
        # Regression (not classification) so the random worldcover target is valid.
        supervision_config = SupervisionHeadConfig(
            modality_configs={
                "worldcover": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=1,
                    weight=0.02,
                    regression_loss_type="l1",
                )
            },
        )
    return LatentMIMConfig(
        encoder_config=_encoder_config(with_student),
        decoder_config=decoder_config,
        supervision_head_config=supervision_config,
        location_token_distillation_head_config=(
            LocationTokenDistillationHeadConfig() if with_student else None
        ),
    )


def _assert_student_isolated(model_or_encoder: torch.nn.Module) -> None:
    """No encoder-block or primary-AttentionPooler parameter may carry gradient."""
    encoder = getattr(model_or_encoder, "encoder", model_or_encoder)
    for name, param in encoder.named_parameters():
        if name.startswith("attention_pooler.student"):
            continue
        assert param.grad is None or torch.all(param.grad == 0), (
            f"student gradient leaked into encoder parameter {name}"
        )


def test_encoder_location_token_student_detached(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The student outputs a max(dims)-wide grid and never grads the encoder."""
    encoder = _encoder_config(True).build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    B, H, W = masked_sample_dict["sentinel2_l2a"].shape[:3]
    grid = (H // 4, W // 4)

    output_dict = encoder.forward(x, patch_size=4, input_res=10)
    assert output_dict["location_tokens"].shape == (B, *grid, POOLER_DIM)
    projected = output_dict["student_location_tokens"]
    assert projected.shape == (B, *grid, max(PROJECTION_DIMS))
    assert (
        encoder.attention_pooler is not None
        and encoder.attention_pooler.student is not None
    )
    assert encoder.attention_pooler.student_dims == sorted(
        PROJECTION_DIMS, reverse=True
    )
    # The student lives on the AttentionPooler, not loose on the encoder.
    assert not hasattr(encoder, "register_student")
    # The training-only back-projection heads are not part of the encoder.
    assert not any(
        n.startswith("register_back_projections") for n, _ in encoder.named_parameters()
    )

    encoder.zero_grad()
    projected.sum().backward()
    _assert_student_isolated(encoder)
    assert encoder.attention_pooler.student[0].weight.grad is not None


def test_encoder_registers_grad_without_student_interference(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The teacher path still gets gradients; the student stays untouched by it."""
    encoder = _encoder_config(True).build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    output_dict = encoder.forward(x, patch_size=4, input_res=10)
    encoder.zero_grad()
    output_dict["location_tokens"].sum().backward()
    assert encoder.attention_pooler is not None
    assert encoder.attention_pooler.learned_query.grad is not None
    assert encoder.attention_pooler.student is not None
    assert encoder.attention_pooler.student[0].weight.grad is None


def test_latentmim_supervision_reads_the_location_tokens(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The supervision head runs on the location-token grid next to the student outputs."""
    model: LatentMIM = _latent_mim_config(True).build()
    assert model.supervision_head is not None

    x = MaskedOlmoEarthSample(**masked_sample_dict)
    (_, _, _, _, _, supervision_preds, student_outputs) = model.forward(x, patch_size=4)
    assert supervision_preds is not None and "worldcover" in supervision_preds
    assert student_outputs is not None
    assert student_outputs["student_location_tokens"].shape[-1] == max(PROJECTION_DIMS)
    assert student_outputs["location_tokens"].shape[-1] == POOLER_DIM


def test_latent_mim_owns_the_distillation_head() -> None:
    """One back-projection per Matryoshka prefix, on the head, only when configured."""
    model = _latent_mim_config(True).build()
    head = model.location_token_distillation_head
    assert head is not None
    assert set(head.back_projections.keys()) == {str(d) for d in PROJECTION_DIMS}
    for d in PROJECTION_DIMS:
        assert head.back_projections[str(d)].in_features == d
    assert _latent_mim_config(False).build().location_token_distillation_head is None
    # The encoder itself carries no distillation parameters.
    assert not any(
        n.startswith("register_back_projections")
        for n, _ in model.encoder.named_parameters()
    )


def test_distillation_head_requires_a_student() -> None:
    """A distillation head without a student to distil into is a config error."""
    config = _latent_mim_config(False)
    config.location_token_distillation_head_config = (
        LocationTokenDistillationHeadConfig()
    )
    with pytest.raises(ValueError, match="requires an AttentionPooler with a student"):
        config.validate()


def test_distillation_head_loss_prefixes() -> None:
    """Per-prefix cosine + Gram terms; grads reach the student and heads, never the teacher."""
    torch.manual_seed(0)
    B, N, D = 2, 9, POOLER_DIM
    teacher = torch.randn(B, N, D, requires_grad=True)
    student_source = torch.randn(B, N, max(PROJECTION_DIMS), requires_grad=True)
    student = student_source * 1.0
    head = LocationTokenDistillationHead(
        pooler_dim=D, student_dims=PROJECTION_DIMS, gram_max_tokens=8
    )
    total, metrics = head(teacher, student)
    assert torch.isfinite(total)
    for d in PROJECTION_DIMS:
        assert f"student/distill_cosine_d{d}" in metrics
        assert f"student/distill_gram_d{d}" in metrics
    total.backward()
    assert student_source.grad is not None
    # The teacher is detached inside the head, so no gradient flows back to it.
    assert teacher.grad is None
    for back_projection in head.back_projections.values():
        assert back_projection.weight.grad is not None


def test_attention_pooler_config_rejects_empty_student_dims() -> None:
    """An empty student dim list is a config error, not a silent no-student."""
    config = _encoder_config(True)
    assert config.attention_pooler_config is not None
    config.attention_pooler_config.student_dims = []
    with pytest.raises(ValueError, match="student_dims"):
        config.validate()
