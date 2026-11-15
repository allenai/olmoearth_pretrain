"""Tests for the Strata auxiliary losses (recdisc, emainv) and their v1.3 run scripts."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.internal.experiment import SubCmd
from olmoearth_pretrain.nn.attention import DropPath
from olmoearth_pretrain.nn.strata_aux import (
    RecDiscHeadConfig,
    build_ema_teacher,
    ema_invariance_loss,
    ema_update,
    recdisc_loss,
    timestamp_features,
)

ABLATIONS = (
    Path(__file__).resolve().parents[3] / "scripts" / "official" / "v1_3" / "ablations"
)


class _RecordingHead(nn.Module):
    """Returns zeros of the target width and records the registers it was given."""

    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.scale = nn.Parameter(torch.ones(()))
        self.registers: torch.Tensor | None = None

    def forward(
        self, registers: torch.Tensor, time_feats: torch.Tensor
    ) -> torch.Tensor:
        self.registers = registers.detach().clone()
        return self.scale * registers[..., : self.target_dim]


def _ts(batch: int, num_t: int) -> torch.Tensor:
    ts = torch.zeros(batch, num_t, 3, dtype=torch.long)
    ts[..., 0] = 1
    ts[..., 1] = torch.arange(num_t) % 12
    ts[..., 2] = 2024
    return ts


@pytest.mark.parametrize("patch_size,stride", [(4, 1), (4, 2), (4, 4), (1, 1)])
def test_recdisc_picks_the_register_holding_the_drawn_pixel(
    patch_size: int, stride: int
) -> None:
    """Each query's register is the one whose cell contains the cell's drawn pixel."""
    torch.manual_seed(0)
    batch, h, w, num_t, dim = 2, 3, 3, 4, 8
    h_r, w_r = h * patch_size // stride, w * patch_size // stride
    # Register (r, c) holds its own (r, c) in its first two channels.
    grid = torch.zeros(batch, h_r, w_r, dim)
    grid[..., 0] = torch.arange(h_r).view(1, h_r, 1).float()
    grid[..., 1] = torch.arange(w_r).view(1, 1, w_r).float()
    offsets = None
    if patch_size > 1:
        offsets = torch.randint(0, patch_size, (batch, h, w, 2))
    targets = torch.randn(batch, h, w, num_t, 1, dim)
    mask = torch.full((batch, h, w, num_t, 1), MaskValue.TARGET_ENCODER_ONLY.value)
    head = _RecordingHead(dim)
    config = RecDiscHeadConfig(num_queries=h * w * num_t)
    loss, metrics = recdisc_loss(
        head,
        config,
        grid,
        stride,
        patch_size,
        offsets,
        targets,
        mask,
        _ts(batch, num_t),
    )
    assert torch.isfinite(loss)
    assert metrics["queries_per_sample"].item() == h * w * num_t
    regs = head.registers
    assert regs is not None
    # Every (cell, t) is a query; check the set of registers per sample.
    expected = set()
    for b in range(batch):
        for i in range(h):
            for j in range(w):
                dy, dx = (0, 0) if offsets is None else offsets[b, i, j].tolist()
                expected.add(
                    (
                        b,
                        (i * patch_size + dy) // stride,
                        (j * patch_size + dx) // stride,
                    )
                )
    got = {
        (b, int(regs[b, k, 0]), int(regs[b, k, 1]))
        for b in range(batch)
        for k in range(regs.shape[1])
    }
    assert got == expected


def test_recdisc_skips_seen_and_missing_tokens() -> None:
    """Only TARGET_ENCODER_ONLY / DECODER tokens are queried."""
    batch, h, w, num_t, dim = 1, 2, 2, 3, 8
    grid = torch.randn(batch, h, w, dim)
    targets = torch.randn(batch, h, w, num_t, 1, dim)
    mask = torch.full((batch, h, w, num_t, 1), MaskValue.ONLINE_ENCODER.value)
    mask[0, 0, 0, 0] = MaskValue.TARGET_ENCODER_ONLY.value
    mask[0, 1, 1, 2] = MaskValue.DECODER.value
    mask[0, 0, 1, 1] = MaskValue.MISSING.value
    head = _RecordingHead(dim)
    _, metrics = recdisc_loss(
        head,
        RecDiscHeadConfig(num_queries=8),
        grid,
        1,
        1,
        None,
        targets,
        mask,
        _ts(batch, num_t),
    )
    assert metrics["queries_per_sample"].item() == 2


def test_recdisc_centering_drops_constant_samples() -> None:
    """A sample whose queried targets are all identical contributes no queries."""
    batch, h, w, num_t, dim = 2, 2, 2, 2, 8
    grid = torch.randn(batch, h, w, dim)
    targets = torch.randn(batch, h, w, num_t, 1, dim)
    targets[0] = torch.randn(dim)
    mask = torch.full((batch, h, w, num_t, 1), MaskValue.DECODER.value)
    head = _RecordingHead(dim)
    loss, metrics = recdisc_loss(
        head,
        RecDiscHeadConfig(num_queries=8),
        grid,
        1,
        1,
        None,
        targets,
        mask,
        _ts(batch, num_t),
    )
    assert torch.isfinite(loss)
    assert metrics["queries_per_sample"].item() == pytest.approx(4.0)


def test_timestamp_features_shape_and_period() -> None:
    """Day-of-year harmonics repeat after a year; the year offset is centered."""
    ts = torch.tensor([[[1, 0, 2023], [1, 0, 2024]]])
    feats = timestamp_features(ts, num_harmonics=3)
    assert feats.shape == (1, 2, 7)
    assert torch.allclose(feats[0, 0, :6], feats[0, 1, :6], atol=1e-5)
    assert feats[0, :, 6].tolist() == [-0.5, 0.5]


def test_ema_teacher_is_frozen_and_deterministic() -> None:
    """The teacher copy has no stochastic ops and no grads; ema_update moves it."""
    student = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), DropPath(0.3))
    teacher = build_ema_teacher(student)
    assert all(not p.requires_grad for p in teacher.parameters())
    teacher.train()
    x = torch.randn(16, 4)
    assert torch.equal(teacher(x), teacher(x))
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
    before = [p.clone() for p in teacher.parameters()]
    ema_update(student, teacher, 0.9)
    for b, s, t in zip(before, student.parameters(), teacher.parameters()):
        assert torch.allclose(t, 0.9 * b + 0.1 * s)


def test_ema_invariance_loss_grads_only_reach_the_student() -> None:
    """Stop-grad on the teacher grid; zero loss when prediction equals the target."""
    predictor = nn.Identity()
    student = torch.randn(2, 3, 3, 8, requires_grad=True)
    teacher = torch.randn(2, 3, 3, 8, requires_grad=True)
    loss, _ = ema_invariance_loss(predictor, student, teacher, weight=2.0)  # type: ignore[arg-type]
    loss.backward()
    assert student.grad is not None and teacher.grad is None
    same, _ = ema_invariance_loss(predictor, teacher.detach(), teacher, 1.0)  # type: ignore[arg-type]
    assert same.item() == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        ema_invariance_loss(predictor, student, teacher[:, :2], 1.0)  # type: ignore[arg-type]


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ABLATIONS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


@pytest.mark.parametrize(
    "name,field",
    [
        ("rc_ld1_pixtgt_pix512_recdisc", "recdisc_head_config"),
        ("rc_ld1_pixtgt_pix512_emainv", "ema_invariance_config"),
    ],
)
def test_arm_differs_from_release_only_in_its_aux_config(name: str, field: str) -> None:
    """Model config = release + one aux config; train module and evals = release."""
    release = _load("rc_ld1_pixtgt_pix512")
    arm = _load(name)
    common = release.build_common_components(
        f"{name}.py", SubCmd.dry_run, "test", "local", []
    )
    base = _flatten(release.build_model_config(common).as_dict())
    ours = _flatten(arm.build_model_config(common).as_dict())
    differ = {k for k in ours if base.get(k, "<absent>") != ours[k]}
    assert differ and all(k.startswith(field) for k in differ), differ
    assert arm.build_train_module_config(common).as_dict() == (
        release.build_train_module_config(common).as_dict()
    )
    for attr in ("build_common_components", "build_train_module_config"):
        assert getattr(arm, attr) is getattr(release, attr), attr
    evaluator = arm.build_trainer_config(common).callbacks["downstream_evaluator"]
    assert evaluator.beaker_eval_module_path == arm.MODULE_PATH
    release_evaluator = release.build_trainer_config(common).callbacks[
        "downstream_evaluator"
    ]
    assert evaluator.tasks.keys() == release_evaluator.tasks.keys()
    with torch.device("meta"):
        model = arm.build_model_config(common).build()
    if field == "ema_invariance_config":
        assert model.ema_encoder is not None
        assert all(not p.requires_grad for p in model.ema_encoder.parameters())
    else:
        assert model.recdisc_head is not None and model.ema_encoder is None
