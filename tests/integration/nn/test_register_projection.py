"""Tests for the detached low-dim register projection ("student")."""

import logging

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample
from olmoearth_pretrain.train.train_module.latent_mim import (
    compute_projection_distill_loss,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODALITIES = [Modality.SENTINEL2_L2A, Modality.LATLON, Modality.WORLDCOVER]
REGISTER_DIM = 16
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
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
        register_read_depth=1,
        register_latent_depth=2,
    )
    if with_student:
        config.register_projection_dims = list(PROJECTION_DIMS)
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
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
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
    )


def _assert_student_isolated(model_or_encoder: torch.nn.Module) -> None:
    """No encoder-block or primary-bottleneck parameter may carry gradient."""
    encoder = getattr(model_or_encoder, "encoder", model_or_encoder)
    for name, param in encoder.named_parameters():
        if name.startswith(("register_projection", "register_back_projections")):
            continue
        assert param.grad is None or torch.all(param.grad == 0), (
            f"student gradient leaked into encoder parameter {name}"
        )


def test_encoder_register_projection_detached(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The student outputs a max(dims)-wide grid and never grads the encoder."""
    encoder = _encoder_config(True).build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    B, H, W = masked_sample_dict["sentinel2_l2a"].shape[:3]
    grid = (H // 4, W // 4)

    output_dict = encoder.forward(x, patch_size=4, input_res=10)
    assert output_dict["registers"].shape == (B, *grid, REGISTER_DIM)
    projected = output_dict["projected_registers"]
    assert projected.shape == (B, *grid, max(PROJECTION_DIMS))
    assert encoder.register_projection is not None
    # One back-projection per Matryoshka prefix.
    assert encoder.register_back_projections is not None
    assert set(encoder.register_back_projections.keys()) == {
        str(d) for d in PROJECTION_DIMS
    }

    encoder.zero_grad()
    projected.sum().backward()
    _assert_student_isolated(encoder)
    assert encoder.register_projection.weight.grad is not None


def test_encoder_registers_grad_without_student_interference(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The teacher path still gets gradients; the student stays untouched by it."""
    encoder = _encoder_config(True).build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    output_dict = encoder.forward(x, patch_size=4, input_res=10)
    encoder.zero_grad()
    output_dict["registers"].sum().backward()
    assert encoder.register_bottleneck is not None
    assert encoder.register_bottleneck.register.grad is not None
    assert encoder.register_projection is not None
    assert encoder.register_projection.weight.grad is None


def test_latentmim_supervision_reads_the_registers(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The supervision head runs on the register grid next to the student outputs."""
    model: LatentMIM = _latent_mim_config(True).build()
    assert model.supervision_head is not None

    x = MaskedOlmoEarthSample(**masked_sample_dict)
    (_, _, _, _, _, supervision_preds, projection_outputs) = model.forward(
        x, patch_size=4
    )
    assert supervision_preds is not None and "worldcover" in supervision_preds
    assert projection_outputs is not None
    assert projection_outputs["projected_registers"].shape[-1] == max(PROJECTION_DIMS)
    assert projection_outputs["registers"].shape[-1] == REGISTER_DIM


def test_compute_projection_distill_loss_prefixes() -> None:
    """Per-prefix cosine + Gram terms; grads reach the student, never the teacher."""
    torch.manual_seed(0)
    B, N, D = 2, 9, REGISTER_DIM
    teacher = torch.randn(B, N, D, requires_grad=True)
    student_source = torch.randn(B, N, max(PROJECTION_DIMS), requires_grad=True)
    student = student_source * 1.0
    back_projections: dict[str, torch.nn.Module] = {
        str(d): torch.nn.Linear(d, D) for d in PROJECTION_DIMS
    }
    total, metrics = compute_projection_distill_loss(
        teacher=teacher,
        student=student,
        back_projections=back_projections,
        cosine_weight=1.0,
        gram_weight=1.0,
        gram_max_tokens=8,
    )
    assert torch.isfinite(total)
    for d in PROJECTION_DIMS:
        assert f"projection/distill_cosine_d{d}" in metrics
        assert f"projection/distill_gram_d{d}" in metrics
    total.backward()
    assert student_source.grad is not None
    # The teacher is detached inside the loss, so no gradient flows back to it.
    assert teacher.grad is None
    for back_projection in back_projections.values():
        assert back_projection.weight.grad is not None


def test_encoder_config_projection_requires_bottleneck() -> None:
    """register_projection_dims without the bottleneck is rejected."""
    config = _encoder_config(True)
    config.use_register_bottleneck = False
    with pytest.raises(ValueError, match="use_register_bottleneck"):
        config.validate()
