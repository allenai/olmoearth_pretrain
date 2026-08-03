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
GRID_SIZE = 3
REGISTER_DIM = 16
PROJECTION_DIMS = [8, 4]


def _encoder_config(projection_type: str | None) -> EncoderConfig:
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
        register_grid_size=GRID_SIZE,
        register_dim=REGISTER_DIM,
        register_read_depth=1,
        register_latent_depth=2,
    )
    if projection_type is not None:
        config.register_projection_dims = list(PROJECTION_DIMS)
        config.register_projection_type = projection_type
    return config


def _latent_mim_config(
    projection_type: str | None,
    supervision_source: str = "registers",
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
            register_supervision=True,
        )
    return LatentMIMConfig(
        encoder_config=_encoder_config(projection_type),
        decoder_config=decoder_config,
        supervision_head_config=supervision_config,
        supervision_source=supervision_source,
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


@pytest.mark.parametrize("projection_type", ["linear", "perceiver"])
def test_encoder_register_projection_detached(
    projection_type: str,
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The student outputs a max(dims)-wide grid and never grads the encoder."""
    encoder = _encoder_config(projection_type).build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    B = masked_sample_dict["sentinel2_l2a"].shape[0]

    output_dict = encoder.forward(x, patch_size=4, input_res=10)
    n_reg = GRID_SIZE * GRID_SIZE
    assert output_dict["registers"].shape == (B, n_reg, REGISTER_DIM)
    projected = output_dict["projected_registers"]
    assert projected.shape == (B, n_reg, max(PROJECTION_DIMS))
    if projection_type == "linear":
        assert encoder.register_projection is not None
        assert encoder.register_projection_student is None
    else:
        assert encoder.register_projection is None
        assert encoder.register_projection_student is not None
    # One back-projection per Matryoshka prefix.
    assert encoder.register_back_projections is not None
    assert set(encoder.register_back_projections.keys()) == {
        str(d) for d in PROJECTION_DIMS
    }

    encoder.zero_grad()
    projected.sum().backward()
    _assert_student_isolated(encoder)
    if projection_type == "linear":
        assert encoder.register_projection is not None
        assert encoder.register_projection.weight.grad is not None
    else:
        student = encoder.register_projection_student
        assert student is not None
        assert student.read_blocks[0].attn.q.weight.grad is not None


def test_encoder_registers_grad_without_student_interference(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The teacher path still gets gradients; the student stays untouched by it."""
    encoder = _encoder_config("linear").build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    output_dict = encoder.forward(x, patch_size=4, input_res=10)
    encoder.zero_grad()
    output_dict["registers"].sum().backward()
    assert encoder.register_bottleneck is not None
    assert encoder.register_bottleneck.registers.grad is not None
    assert encoder.register_projection is not None
    assert encoder.register_projection.weight.grad is None


@pytest.mark.parametrize(
    "supervision_source,expect_register_head,expect_projection_heads",
    [
        ("registers", True, False),
        ("both", True, True),
        ("projection", False, True),
    ],
)
def test_latentmim_projection_supervision_sources(
    supervision_source: str,
    expect_register_head: bool,
    expect_projection_heads: bool,
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """supervision_source places heads on the registers, the student, or both."""
    model: LatentMIM = _latent_mim_config("linear", supervision_source).build()
    assert (model.supervision_head is not None) == expect_register_head
    assert (model.projection_supervision_heads is not None) == expect_projection_heads

    x = MaskedOlmoEarthSample(**masked_sample_dict)
    (_, _, _, _, _, supervision_preds, projection_outputs) = model.forward(
        x, patch_size=4
    )
    assert (supervision_preds is not None) == expect_register_head
    assert projection_outputs is not None
    assert projection_outputs["projected_registers"].shape[-1] == max(PROJECTION_DIMS)
    assert projection_outputs["registers"].shape[-1] == REGISTER_DIM
    if expect_projection_heads:
        # One prediction set per Matryoshka prefix.
        preds = projection_outputs["supervision_preds"]
        assert set(preds.keys()) == {str(d) for d in PROJECTION_DIMS}
        for dim_preds in preds.values():
            assert "worldcover" in dim_preds
    else:
        assert projection_outputs["supervision_preds"] is None


def test_latentmim_supervision_source_requires_projection() -> None:
    """supervision_source != registers demands the encoder student."""
    config = _latent_mim_config(None, "projection")
    with pytest.raises(ValueError, match="register_projection_dims"):
        config.build()


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
    config = _encoder_config("linear")
    config.use_register_bottleneck = False
    config.register_grid_size = 0
    with pytest.raises(ValueError, match="use_register_bottleneck"):
        config.validate()


def test_within_scene_gram_is_opt_in_and_per_scene() -> None:
    """The block-diagonal Gram relates cells of one scene only, and defaults off."""
    torch.manual_seed(0)
    B, N, D = 3, 8, REGISTER_DIM
    teacher = torch.randn(B, N, D, requires_grad=True)
    student_source = torch.randn(B, N, max(PROJECTION_DIMS), requires_grad=True)
    back_projections: dict[str, torch.nn.Module] = {
        str(d): torch.nn.Linear(d, D) for d in PROJECTION_DIMS
    }
    kwargs = dict(
        teacher=teacher,
        student=student_source * 1.0,
        back_projections=back_projections,
        cosine_weight=1.0,
        gram_weight=1.0,
        gram_max_tokens=8,
    )

    # Off by default: exactly the loss the flat-only runs were trained with.
    _, baseline = compute_projection_distill_loss(**kwargs)
    for d in PROJECTION_DIMS:
        assert f"projection/distill_gram_within_d{d}" not in baseline

    total, metrics = compute_projection_distill_loss(
        **kwargs, gram_within_weight=1.0, gram_within_max_cells=4
    )
    assert torch.isfinite(total)
    for d in PROJECTION_DIMS:
        assert f"projection/distill_gram_within_d{d}" in metrics
    total.backward()
    assert student_source.grad is not None
    assert teacher.grad is None

    # Scenes whose cells are identical have an all-ones within-scene Gram, so a
    # student with unrelated cells scores ~1.0 -- the term is keyed to structure
    # WITHIN a scene, never across the batch (a foreign cell would score ~0).
    flat_teacher = torch.randn(B, 1, D).expand(B, N, D).contiguous()
    _, degenerate = compute_projection_distill_loss(
        teacher=flat_teacher,
        student=student_source * 1.0,
        back_projections=back_projections,
        cosine_weight=0.0,
        gram_weight=0.0,
        gram_max_tokens=8,
        gram_within_weight=1.0,
        gram_within_max_cells=N,
    )
    for d in PROJECTION_DIMS:
        assert degenerate[f"projection/distill_gram_within_d{d}"] > 0.5

    # A single cell per scene has no within-scene pairs, so the term is skipped.
    _, single_cell = compute_projection_distill_loss(
        teacher=teacher[:, :1],
        student=student_source[:, :1] * 1.0,
        back_projections=back_projections,
        cosine_weight=1.0,
        gram_weight=0.0,
        gram_max_tokens=8,
        gram_within_weight=1.0,
    )
    for d in PROJECTION_DIMS:
        assert f"projection/distill_gram_within_d{d}" not in single_cell


def test_projection_supervision_weight_scale_decouples_the_heads() -> None:
    """The student's heads can run at a fraction of the register head's weight."""
    config = _latent_mim_config("linear", "both")
    config.projection_supervision_weight_scale = 0.1
    model: LatentMIM = config.build()

    assert model.supervision_head is not None
    assert model.projection_supervision_heads is not None
    register_weight = model.supervision_head.modality_configs["worldcover"].weight
    for dim in PROJECTION_DIMS:
        head = model.projection_supervision_heads[str(dim)]
        assert head.modality_configs["worldcover"].weight == pytest.approx(
            register_weight * 0.1
        )

    # Unscaled (the previous behaviour) leaves both heads at the same weight.
    unscaled: LatentMIM = _latent_mim_config("linear", "both").build()
    assert unscaled.projection_supervision_heads is not None
    assert unscaled.projection_supervision_heads[
        str(PROJECTION_DIMS[0])
    ].modality_configs["worldcover"].weight == pytest.approx(register_weight)


def test_projection_supervision_weight_scale_requires_projection_heads() -> None:
    """A scale with no projection heads to scale is a silent no-op, so it raises."""
    config = _latent_mim_config("linear", "registers")
    config.projection_supervision_weight_scale = 0.1
    with pytest.raises(ValueError, match="no projection heads"):
        config.validate()
