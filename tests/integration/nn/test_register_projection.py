"""Tests for the detached low-dim register projection ("student")."""

import logging
from typing import Any

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import (
    DEFAULT_REGISTER_STUDENT_NAME,
    EncoderConfig,
    PredictorConfig,
    RegisterStudentSpec,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample
from olmoearth_pretrain.train.train_module.latent_mim import (
    _namespace_student_metrics,
    build_teacher_gram_state,
    compute_projection_distill_loss,
    validate_distill_overrides,
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
        if name.startswith(("register_students", "register_projection")):
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
    # The scalar register_projection_* fields are shorthand for one student.
    assert set(output_dict["projected_registers"]) == {DEFAULT_REGISTER_STUDENT_NAME}
    projected = output_dict["projected_registers"][DEFAULT_REGISTER_STUDENT_NAME]
    assert projected.shape == (B, n_reg, max(PROJECTION_DIMS))
    assert encoder.register_students is not None
    student = encoder.register_students[DEFAULT_REGISTER_STUDENT_NAME]
    assert student.reads_tokens == (projection_type == "perceiver")
    # One back-projection per Matryoshka prefix.
    assert set(student.back_projections.keys()) == {str(d) for d in PROJECTION_DIMS}

    encoder.zero_grad()
    projected.sum().backward()
    _assert_student_isolated(encoder)
    if projection_type == "linear":
        assert student.projection.weight.grad is not None
    else:
        assert student.projection.read_blocks[0].attn.q.weight.grad is not None


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
    assert encoder.register_students is not None
    student = encoder.register_students[DEFAULT_REGISTER_STUDENT_NAME]
    assert student.projection.weight.grad is None


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
    projected = projection_outputs["projected_registers"]
    assert projected[DEFAULT_REGISTER_STUDENT_NAME].shape[-1] == max(PROJECTION_DIMS)
    assert projection_outputs["registers"].shape[-1] == REGISTER_DIM
    if expect_projection_heads:
        # Keyed by student, then one prediction set per Matryoshka prefix.
        preds = projection_outputs["supervision_preds"][DEFAULT_REGISTER_STUDENT_NAME]
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


def test_compute_projection_distill_loss_within_scene_gram() -> None:
    """The within-scene Gram term is opt-in, per-scene, and reaches the student."""
    torch.manual_seed(0)
    B, N, D = 3, 9, REGISTER_DIM
    teacher = torch.randn(B, N, D, requires_grad=True)
    student_source = torch.randn(B, N, max(PROJECTION_DIMS), requires_grad=True)
    back_projections: dict[str, torch.nn.Module] = {
        str(d): torch.nn.Linear(d, D) for d in PROJECTION_DIMS
    }
    kwargs: dict[str, Any] = dict(
        teacher=teacher,
        student=student_source * 1.0,
        back_projections=back_projections,
        cosine_weight=1.0,
        gram_weight=1.0,
        gram_max_tokens=8,
    )
    # Off by default: the flat-only loss the in-flight proj128 runs were trained with.
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

    # Scenes whose cells are all identical have an all-ones within-scene Gram, so a
    # student with unrelated (near-orthogonal) cells scores ~1.0 -- i.e. the term is
    # keyed to structure WITHIN a scene, not across the batch.
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

    # A single register cell per scene has no within-scene pairs; the term is skipped.
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


def test_encoder_config_projection_requires_bottleneck() -> None:
    """register_projection_dims without the bottleneck is rejected."""
    config = _encoder_config("linear")
    config.use_register_bottleneck = False
    config.register_grid_size = 0
    with pytest.raises(ValueError, match="use_register_bottleneck"):
        config.validate()


def _multi_student_encoder_config() -> EncoderConfig:
    """Two students -- different architectures -- on one shared encoder."""
    config = _encoder_config(None)
    config.register_students = [
        RegisterStudentSpec(
            name="lin_arm", projection_type="linear", dims=list(PROJECTION_DIMS)
        ),
        RegisterStudentSpec(
            name="pcv_arm", projection_type="perceiver", dims=list(PROJECTION_DIMS)
        ),
    ]
    return config


def test_multi_student_encoder_runs_every_arm(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """Each student produces its own grid, and none of them grads the encoder."""
    encoder = _multi_student_encoder_config().build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    B = masked_sample_dict["sentinel2_l2a"].shape[0]

    output_dict = encoder.forward(x, patch_size=4, input_res=10)
    projected = output_dict["projected_registers"]
    assert set(projected) == {"lin_arm", "pcv_arm"}
    n_reg = GRID_SIZE * GRID_SIZE
    for grid in projected.values():
        assert grid.shape == (B, n_reg, max(PROJECTION_DIMS))

    encoder.zero_grad()
    sum(grid.sum() for grid in projected.values()).backward()
    _assert_student_isolated(encoder)


def test_multi_student_arms_are_independent(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """One arm's loss trains only that arm -- the arms never leak into each other."""
    encoder = _multi_student_encoder_config().build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    output_dict = encoder.forward(x, patch_size=4, input_res=10)

    encoder.zero_grad()
    output_dict["projected_registers"]["lin_arm"].sum().backward()
    assert encoder.register_students is not None
    trained = encoder.register_students["lin_arm"]
    untouched = encoder.register_students["pcv_arm"]
    assert trained.projection.weight.grad is not None
    for param in untouched.parameters():
        assert param.grad is None or torch.all(param.grad == 0)


def test_multi_student_names_must_be_unique() -> None:
    """Duplicate names would silently collapse two arms into one."""
    config = _encoder_config(None)
    config.register_students = [
        RegisterStudentSpec(name="dup", dims=[8]),
        RegisterStudentSpec(name="dup", dims=[4]),
    ]
    with pytest.raises(ValueError, match="unique"):
        config.validate()


def test_register_students_and_scalar_fields_are_exclusive() -> None:
    """Declaring students both ways would half-describe the encoder."""
    config = _encoder_config("linear")
    config.register_students = [RegisterStudentSpec(name="extra", dims=[8])]
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()


@pytest.mark.parametrize("projection_type", ["linear", "perceiver"])
def test_legacy_single_student_checkpoint_loads(projection_type: str) -> None:
    """Pre-multi-student checkpoints load into the register_students ModuleDict."""
    encoder = _encoder_config(projection_type).build()
    state_dict = encoder.state_dict()

    # Rewrite the student keys back to their pre-ModuleDict names.
    legacy_projection = (
        "register_projection_student"
        if projection_type == "perceiver"
        else "register_projection"
    )
    prefix = f"register_students.{DEFAULT_REGISTER_STUDENT_NAME}."
    legacy = {}
    for key, value in state_dict.items():
        if key.startswith(prefix + "projection."):
            key = legacy_projection + "." + key[len(prefix + "projection.") :]
        elif key.startswith(prefix + "back_projections."):
            key = (
                "register_back_projections." + key[len(prefix + "back_projections.") :]
            )
        legacy[key] = value
    assert not any(k.startswith("register_students") for k in legacy)

    fresh = _encoder_config(projection_type).build()
    fresh.load_state_dict(legacy, strict=True)
    assert fresh.register_students is not None
    assert encoder.register_students is not None
    student = fresh.register_students[DEFAULT_REGISTER_STUDENT_NAME]
    reference = encoder.register_students[DEFAULT_REGISTER_STUDENT_NAME]
    for (_, loaded), (_, expected) in zip(
        student.named_parameters(), reference.named_parameters()
    ):
        assert torch.equal(loaded, expected)


def test_multi_student_distill_losses_are_per_student(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """Each student is scored against the shared teacher under its own weights."""
    config = _latent_mim_config(None, "registers")
    config.encoder_config.register_students = [
        RegisterStudentSpec(name="flat_gram", dims=list(PROJECTION_DIMS)),
        RegisterStudentSpec(name="within_gram", dims=list(PROJECTION_DIMS)),
    ]
    model: LatentMIM = config.build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    (*_, projection_outputs) = model.forward(x, patch_size=4)
    assert projection_outputs is not None
    assert set(projection_outputs["projected_registers"]) == {
        "flat_gram",
        "within_gram",
    }

    # The within_gram arm turns on the block-diagonal term; the flat arm does not.
    overrides: dict[str, dict[str, Any]] = {"within_gram": {"gram_within_weight": 1.0}}
    validate_distill_overrides(overrides, {"flat_gram", "within_gram"})
    all_metrics: dict[str, torch.Tensor] = {}
    for name, student_out in projection_outputs["projected_registers"].items():
        # Mixed float weights and int sample sizes, so Any is the honest value type.
        weights: dict[str, Any] = {
            "cosine_weight": 1.0,
            "gram_weight": 1.0,
            "gram_max_tokens": 8,
            "gram_within_weight": 0.0,
            "gram_within_max_cells": 4,
        }
        weights.update(overrides.get(name, {}))
        _, metrics = compute_projection_distill_loss(
            teacher=projection_outputs["registers"],
            student=student_out,
            back_projections=dict(
                model.encoder.register_students[name].back_projections
            ),
            **weights,
        )
        all_metrics.update(_namespace_student_metrics(metrics, name))

    # Named students are scoped, so the two arms' curves never collide.
    assert "projection/flat_gram/distill_cosine_d8" in all_metrics
    assert "projection/within_gram/distill_cosine_d8" in all_metrics
    # Only the overridden arm logs the within-scene term.
    assert "projection/within_gram/distill_gram_within_d8" in all_metrics
    assert "projection/flat_gram/distill_gram_within_d8" not in all_metrics


def test_distill_overrides_reject_typos() -> None:
    """A misspelt student or weight silently running the defaults is the failure."""
    with pytest.raises(ValueError, match="unknown students"):
        validate_distill_overrides({"typo": {"gram_weight": 1.0}}, {"real"})
    with pytest.raises(ValueError, match="unknown keys"):
        validate_distill_overrides({"real": {"grahm_weight": 1.0}}, {"real"})


def test_single_student_metrics_keep_legacy_names(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The lone student of a single-student run logs unscoped metric names."""
    metrics = {"projection/distill_cosine_d8": torch.zeros(())}
    assert _namespace_student_metrics(metrics, DEFAULT_REGISTER_STUDENT_NAME) == metrics
    scoped = _namespace_student_metrics(metrics, "arm_a")
    assert set(scoped) == {"projection/arm_a/distill_cosine_d8"}


def _supervision_weight_arms_config() -> LatentMIMConfig:
    """Three arms differing only in how hard their own supervision head pulls."""
    config = _latent_mim_config(None, "registers")
    config.encoder_config.register_students = [
        RegisterStudentSpec(name="sup_w0", dims=list(PROJECTION_DIMS)),
        RegisterStudentSpec(name="sup_w0p1", dims=list(PROJECTION_DIMS)),
        RegisterStudentSpec(name="sup_w1", dims=list(PROJECTION_DIMS)),
    ]
    config.projection_supervision_weight_scales = {
        "sup_w0": 0.0,
        "sup_w0p1": 0.1,
        "sup_w1": 1.0,
    }
    return config


def test_per_student_supervision_weights_scale_the_heads() -> None:
    """Each arm's heads carry its own weight; scale 0 means no heads at all."""
    model: LatentMIM = _supervision_weight_arms_config().build()
    heads = model.projection_supervision_heads
    assert heads is not None
    # The zero-weight arm is the no-supervision control, so it gets no heads.
    assert set(heads.keys()) == {"sup_w0p1", "sup_w1"}

    assert model.supervision_head is not None
    base = model.supervision_head.modality_configs["worldcover"].weight
    for name, scale in (("sup_w0p1", 0.1), ("sup_w1", 1.0)):
        for dim in PROJECTION_DIMS:
            weight = heads[name][str(dim)].modality_configs["worldcover"].weight
            assert weight == pytest.approx(base * scale)


def test_per_student_supervision_is_isolated_per_arm(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """One arm's supervision loss never reaches another arm or the encoder."""
    model: LatentMIM = _supervision_weight_arms_config().build()
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    (_, _, _, _, _, _, projection_outputs) = model.forward(x, patch_size=4)
    assert projection_outputs is not None
    preds = projection_outputs["supervision_preds"]
    assert set(preds) == {"sup_w0p1", "sup_w1"}

    model.zero_grad()
    loss = sum(
        pred.sum()
        for dim_preds in preds["sup_w1"].values()
        for pred in dim_preds.values()
    )
    loss.backward()
    _assert_student_isolated(model)
    assert model.encoder.register_students is not None
    trained = model.encoder.register_students["sup_w1"]
    assert trained.projection.weight.grad is not None
    for other in ("sup_w0", "sup_w0p1"):
        for param in model.encoder.register_students[other].parameters():
            assert param.grad is None or torch.all(param.grad == 0)


def test_multi_student_supervision_requires_explicit_scales() -> None:
    """Without per-student scales, supervision cannot be attributed across arms."""
    config = _latent_mim_config(None, "both")
    config.encoder_config.register_students = [
        RegisterStudentSpec(name="a", dims=list(PROJECTION_DIMS)),
        RegisterStudentSpec(name="b", dims=list(PROJECTION_DIMS)),
    ]
    with pytest.raises(ValueError, match="projection_supervision_weight_scales"):
        config.build()


def test_supervision_weight_scales_reject_unknown_students() -> None:
    """A typo'd arm name would silently leave that arm unsupervised."""
    config = _latent_mim_config(None, "registers")
    config.encoder_config.register_students = [
        RegisterStudentSpec(name="real", dims=list(PROJECTION_DIMS)),
    ]
    config.projection_supervision_weight_scales = {"typo": 0.1}
    with pytest.raises(ValueError, match="unknown students"):
        config.build()


def test_shared_teacher_gram_state_gives_arms_identical_pairs() -> None:
    """Two identical arms must score identically -- no per-arm sampling noise."""
    torch.manual_seed(0)
    B, N, D = 4, 32, REGISTER_DIM
    teacher = torch.randn(B, N, D)
    student = torch.randn(B, N, max(PROJECTION_DIMS))
    back_projections: dict[str, torch.nn.Module] = {
        str(d): torch.nn.Linear(d, D) for d in PROJECTION_DIMS
    }
    weights: dict[str, Any] = dict(
        cosine_weight=0.0,
        gram_weight=1.0,
        gram_max_tokens=16,
        gram_within_weight=1.0,
        gram_within_max_cells=8,
    )

    # Without a shared state each call redraws its own subsample, so two arms with
    # identical inputs disagree -- the failure mode this state exists to remove.
    _, a = compute_projection_distill_loss(
        teacher, student, back_projections, **weights
    )
    _, b = compute_projection_distill_loss(
        teacher, student, back_projections, **weights
    )
    assert a["projection/distill_gram_d8"] != b["projection/distill_gram_d8"]

    state = build_teacher_gram_state(
        teacher.float(),
        gram_max_tokens=16,
        gram_within_max_cells=8,
        build_flat=True,
        build_within=True,
    )
    _, a = compute_projection_distill_loss(
        teacher, student, back_projections, teacher_gram_state=state, **weights
    )
    _, b = compute_projection_distill_loss(
        teacher, student, back_projections, teacher_gram_state=state, **weights
    )
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_teacher_gram_state_narrows_to_nested_samples() -> None:
    """A smaller arm's pairs are a subset of a larger arm's, not a fresh draw."""
    torch.manual_seed(0)
    teacher = torch.randn(4, 32, REGISTER_DIM)
    state = build_teacher_gram_state(
        teacher,
        gram_max_tokens=64,
        gram_within_max_cells=16,
        build_flat=True,
        build_within=True,
    )
    big_idx, big_gram = state.flat_sample(64)
    small_idx, small_gram = state.flat_sample(16)
    assert big_idx is not None and small_idx is not None
    assert big_gram is not None and small_gram is not None
    assert torch.equal(small_idx, big_idx[:16])
    assert torch.equal(small_gram, big_gram[:16, :16])

    big_cells, big_within = state.within_sample(16)
    small_cells, small_within = state.within_sample(4)
    assert big_cells is not None and small_cells is not None
    assert big_within is not None and small_within is not None
    assert torch.equal(small_cells, big_cells[:4])
    assert torch.equal(small_within, big_within[:, :4, :4])


def test_teacher_gram_state_skips_unused_halves() -> None:
    """Arms that disable a term never pay for its teacher matrix."""
    teacher = torch.randn(4, 32, REGISTER_DIM)
    flat_only = build_teacher_gram_state(
        teacher,
        gram_max_tokens=16,
        gram_within_max_cells=8,
        build_flat=True,
        build_within=False,
    )
    assert flat_only.gram is not None
    assert flat_only.within_gram is None
    assert flat_only.within_sample(8) == (None, None)
