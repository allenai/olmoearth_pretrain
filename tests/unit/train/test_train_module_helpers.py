"""Tests for shared train module microbatch helpers."""

from types import MethodType
from typing import Any

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig
from torch import nn

from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.modalities import Modality
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)
from olmoearth_pretrain.train.train_module.galileo import GalileoTrainModuleConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig
from olmoearth_pretrain.train.train_module.mae import MAETrainModuleConfig
from olmoearth_pretrain.train.train_module.train_module import (
    MicrobatchTrainOutput,
    OlmoEarthTrainModule,
)


class _TestTrainModule(OlmoEarthTrainModule):
    def train_batch(self, batch: Any, dry_run: bool = False) -> None:
        raise NotImplementedError


class _FakeEncoder(nn.Module):
    def __init__(self, tokenization_config: TokenizationConfig) -> None:
        super().__init__()
        self.tokenization_config = tokenization_config
        self.proj = nn.Linear(1, 1)


class _FakePretrainModel(nn.Module):
    def __init__(self, tokenization_config: TokenizationConfig) -> None:
        super().__init__()
        self.encoder = _FakeEncoder(tokenization_config)
        self.decoder = None


def _sample(batch_size: int) -> MaskedOlmoEarthSample:
    """Create a minimal masked sample with one modality."""
    return MaskedOlmoEarthSample(
        timestamps=torch.ones(batch_size, 1, 3, dtype=torch.long),
        sentinel2_l2a=torch.ones(
            batch_size,
            1,
            1,
            1,
            Modality.SENTINEL2_L2A.num_bands,
        ),
        sentinel2_l2a_mask=torch.zeros(
            batch_size,
            1,
            1,
            1,
            Modality.SENTINEL2_L2A.num_band_sets,
            dtype=torch.long,
        ),
    )


def _module(rank_microbatch_size: int = 2) -> OlmoEarthTrainModule:
    """Build a lightweight train module instance for helper tests."""
    module = object.__new__(_TestTrainModule)
    module.model = nn.Linear(1, 1)
    module._dp_config = None
    module.rank_microbatch_size = rank_microbatch_size
    module.device = torch.device("cpu")
    return module


def _single_bandset_sentinel2_tokenization() -> TokenizationConfig:
    """Build a custom tokenization object that is easy to identify by identity."""
    return TokenizationConfig(
        overrides={
            Modality.SENTINEL2_L2A.name: ModalityTokenization(
                band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
            )
        }
    )


def _optim_config() -> AdamWConfig:
    """Create the minimal optimizer config needed to build train modules."""
    return AdamWConfig(lr=1e-4, weight_decay=0.0)


def _transform_config() -> TransformConfig:
    """Create a no-op transform config for constructor tests."""
    return TransformConfig(transform_type="no_transform")


def _masking_config() -> MaskingConfig:
    """Create masking config without an explicit tokenization override."""
    return MaskingConfig(strategy_config={"type": "random"})


def _patch_discrimination_loss_config() -> LossConfig:
    """Create a lightweight latent-token loss config."""
    return LossConfig(loss_config={"type": "patch_discrimination"})


def _mae_loss_config() -> LossConfig:
    """Create an MAE loss config without an explicit tokenization override."""
    return LossConfig(loss_config={"type": "mae"})


@pytest.mark.parametrize(
    ("config_factory", "mask_attr_names"),
    [
        (
            lambda: ContrastiveLatentMIMTrainModuleConfig(
                optim_config=_optim_config(),
                rank_microbatch_size=2,
                transform_config=_transform_config(),
                loss_config=_patch_discrimination_loss_config(),
                masking_config=_masking_config(),
                mae_loss_config=_mae_loss_config(),
            ),
            ("masking_strategy",),
        ),
        (
            lambda: LatentMIMTrainModuleConfig(
                optim_config=_optim_config(),
                rank_microbatch_size=2,
                transform_config=_transform_config(),
                loss_config=_patch_discrimination_loss_config(),
                masking_config=_masking_config(),
                mae_loss_config=_mae_loss_config(),
            ),
            ("masking_strategy",),
        ),
        (
            lambda: MAETrainModuleConfig(
                optim_config=_optim_config(),
                rank_microbatch_size=2,
                transform_config=_transform_config(),
                mae_loss_config=_mae_loss_config(),
                masking_config=_masking_config(),
            ),
            ("masking_strategy",),
        ),
        (
            lambda: GalileoTrainModuleConfig(
                optim_config=_optim_config(),
                rank_microbatch_size=2,
                transform_config=_transform_config(),
                loss_config_a=_patch_discrimination_loss_config(),
                loss_config_b=_patch_discrimination_loss_config(),
                masking_config_a=_masking_config(),
                masking_config_b=_masking_config(),
                mae_loss_config=_mae_loss_config(),
            ),
            ("masking_strategy_a", "masking_strategy_b"),
        ),
    ],
)
def test_train_modules_propagate_model_tokenization_to_masking_and_mae_loss(
    config_factory: Any,
    mask_attr_names: tuple[str, ...],
) -> None:
    """Runtime masking and MAE losses should share the model tokenization."""
    tokenization_config = _single_bandset_sentinel2_tokenization()
    model = _FakePretrainModel(tokenization_config)

    train_module = config_factory().build(model=model, device=torch.device("cpu"))

    for attr_name in mask_attr_names:
        assert (
            getattr(train_module, attr_name).tokenization_config is tokenization_config
        )
    assert train_module.mae_loss.tokenization_config is tokenization_config


def test_run_masked_microbatches_splits_and_accumulates_loss() -> None:
    """The shared runner should split uneven batches and average backward loss."""
    module = _module(rank_microbatch_size=2)
    param = torch.nn.Parameter(torch.tensor(1.0))
    seen: list[tuple[int, int]] = []

    def step(
        microbatches: tuple[MaskedOlmoEarthSample, ...], microbatch_idx: int
    ) -> MicrobatchTrainOutput:
        seen.append((microbatch_idx, microbatches[0].batch_size))
        return MicrobatchTrainOutput(loss=param * 6.0)

    totals = module._run_masked_microbatches(
        _sample(5),
        step,
        nonfinite_behavior="warn_break",
    )

    assert seen == [(0, 2), (1, 2), (2, 1)]
    assert totals.loss.item() == pytest.approx(6.0)
    assert param.grad is not None
    assert param.grad.item() == pytest.approx(6.0)


def test_run_masked_microbatches_accumulates_regularizer_and_metrics() -> None:
    """Regularizer inputs and metrics should be averaged over microbatches."""
    module = _module(rank_microbatch_size=2)
    param = torch.nn.Parameter(torch.tensor(1.0))

    def compute_regularization(
        self: OlmoEarthTrainModule, regularizer_input: Any
    ) -> torch.Tensor:
        return regularizer_input

    module.compute_regularization = MethodType(compute_regularization, module)

    def step(
        microbatches: tuple[MaskedOlmoEarthSample, ...], microbatch_idx: int
    ) -> MicrobatchTrainOutput:
        return MicrobatchTrainOutput(
            loss=param * 2.0,
            regularizer_inputs=(torch.tensor(3.0), torch.tensor(9.0)),
            metrics={"contrastive": torch.tensor(12.0)},
        )

    totals = module._run_masked_microbatches(
        _sample(4),
        step,
        nonfinite_behavior="warn_break",
    )

    assert totals.loss.item() == pytest.approx(8.0)
    assert totals.regularizer.item() == pytest.approx(6.0)
    assert totals.metrics["contrastive"].item() == pytest.approx(12.0)
    assert param.grad is not None
    assert param.grad.item() == pytest.approx(2.0)


def test_run_masked_microbatches_warn_break_skips_nonfinite_backward() -> None:
    """Warn-break mode should stop before backpropagating a nonfinite loss."""
    module = _module(rank_microbatch_size=2)
    param = torch.nn.Parameter(torch.tensor(1.0))
    seen: list[int] = []

    def step(
        microbatches: tuple[MaskedOlmoEarthSample, ...], microbatch_idx: int
    ) -> MicrobatchTrainOutput:
        seen.append(microbatch_idx)
        if microbatch_idx == 1:
            return MicrobatchTrainOutput(loss=param * torch.tensor(float("nan")))
        return MicrobatchTrainOutput(loss=param * 6.0)

    totals = module._run_masked_microbatches(
        _sample(4),
        step,
        nonfinite_behavior="warn_break",
    )

    assert seen == [0, 1]
    assert torch.isnan(totals.loss)
    assert param.grad is not None
    assert param.grad.item() == pytest.approx(3.0)


def test_run_masked_microbatches_warn_continue_logs_skip_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn-continue mode should skip one backward pass without stopping the batch."""
    module = _module(rank_microbatch_size=2)
    param = torch.nn.Parameter(torch.tensor(1.0))
    seen: list[int] = []

    def step(
        microbatches: tuple[MaskedOlmoEarthSample, ...], microbatch_idx: int
    ) -> MicrobatchTrainOutput:
        seen.append(microbatch_idx)
        if microbatch_idx == 1:
            return MicrobatchTrainOutput(loss=param * torch.tensor(float("nan")))
        return MicrobatchTrainOutput(loss=param * 6.0)

    with caplog.at_level("WARNING"):
        totals = module._run_masked_microbatches(
            _sample(6),
            step,
            nonfinite_behavior="warn_continue",
        )

    assert seen == [0, 1, 2]
    assert torch.isnan(totals.loss)
    assert param.grad is not None
    assert param.grad.item() == pytest.approx(4.0)
    assert "skipping backward for this microbatch" in caplog.text
    assert "stopping training for this batch" not in caplog.text


def test_compute_regularization_for_inputs_rejects_partial_regularizers() -> None:
    """A multi-input regularizer should be consistently enabled or disabled."""
    module = _module()

    def compute_regularization(
        self: OlmoEarthTrainModule, regularizer_input: Any
    ) -> torch.Tensor | None:
        return None if regularizer_input == "missing" else torch.tensor(1.0)

    module.compute_regularization = MethodType(compute_regularization, module)

    with pytest.raises(ValueError, match="Regularization must be computed"):
        module._compute_regularization_for_inputs(("present", "missing"))
