"""GroupGradNormCallback: per-group gradient norms, recorded every ``interval`` steps."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from olmoearth_pretrain.train.callbacks.group_grad_norm import GroupGradNormCallback


def _trainer(step: int) -> SimpleNamespace:
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.slot_merge = nn.Module()
    model.encoder.slot_merge.gru = nn.Linear(2, 2, bias=False)
    model.encoder.blocks = nn.Linear(2, 2, bias=False)
    model.encoder.slot_merge.gru.weight.grad = torch.full((2, 2), 3.0)
    model.encoder.blocks.weight.grad = torch.full((2, 2), 1.0)
    recorded: dict[str, float] = {}
    return SimpleNamespace(
        global_step=step,
        train_module=SimpleNamespace(model=model),
        record_metric=lambda k, v, reduce: recorded.__setitem__(k, v),
        recorded=recorded,
    )


def test_records_each_group_norm() -> None:
    """Each group's norm is the L2 norm of its parameters' gradients."""
    callback = GroupGradNormCallback(
        interval=10, groups=("encoder.slot_merge.gru", "encoder.blocks")
    )
    trainer = _trainer(step=20)
    callback.trainer = trainer  # type: ignore[assignment]
    callback.pre_optim_step()
    assert trainer.recorded["grad_norm/slot_merge.gru"] == 6.0  # sqrt(4 * 9)
    assert trainer.recorded["grad_norm/blocks"] == 2.0


def test_skips_off_interval_steps() -> None:
    """Nothing is recorded between intervals."""
    callback = GroupGradNormCallback(interval=10, groups=("encoder.blocks",))
    trainer = _trainer(step=21)
    callback.trainer = trainer  # type: ignore[assignment]
    callback.pre_optim_step()
    assert trainer.recorded == {}
