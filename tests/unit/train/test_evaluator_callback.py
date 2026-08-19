"""Unit tests for evaluator-callback helpers."""

import torch.nn as nn

from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    _feature_extraction_model,
)


class _EncoderDecoderModel(nn.Module):
    """Encoder/decoder model without an eval-only EMA copy."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)


class _EncoderDecoderModelWithEma(_EncoderDecoderModel):
    """Encoder/decoder model exposing ``eval_encoder`` (LatentMIM's shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.ema_encoder = nn.Linear(2, 2)

    @property
    def eval_encoder(self) -> nn.Module:
        return self.ema_encoder


def test_feature_extraction_prefers_eval_encoder() -> None:
    """A model exposing eval_encoder has its EMA copy evaluated."""
    model = _EncoderDecoderModelWithEma()
    assert _feature_extraction_model(model) is model.ema_encoder


def test_feature_extraction_falls_back_to_encoder() -> None:
    """Without eval_encoder, the online encoder is evaluated."""
    model = _EncoderDecoderModel()
    assert _feature_extraction_model(model) is model.encoder


def test_feature_extraction_falls_back_to_model() -> None:
    """A model with no encoder attribute is evaluated directly."""
    model = nn.Linear(2, 2)
    assert _feature_extraction_model(model) is model
