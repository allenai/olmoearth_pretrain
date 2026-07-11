"""Integration tests: LatentMIM train step with the Perceiver encoder swap.

Same harness as ``test_latent_mim.py`` with ``EncoderConfig`` replaced by
``PerceiverEncoderConfig`` — verifying the set-latent trunk is a drop-in
encoder for the standard pipeline (masking, frozen teacher, predictor, loss).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.datatypes import TokensAndMasks
from olmoearth_pretrain.nn.flexi_vit import PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.perceiver_encoder import PerceiverEncoderConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

from .helper import check_loss_is_a_reasonable_value
from .test_latent_mim import (  # noqa: F401  (pytest fixtures by name)
    MockTrainer,
    optim_config,
    supported_modality_names,
    train_module_config,
)

torch.set_default_device("cpu")
logger = logging.getLogger(__name__)


@pytest.fixture
def latent_mim_model(
    supported_modality_names: list[str],  # noqa: F811
    set_random_seeds: None,
) -> LatentMIM:
    """LatentMIM with the perceiver encoder (production rope_3d_mixed path)."""
    encoder_config = PerceiverEncoderConfig(
        supported_modality_names=supported_modality_names,
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.1,
        max_sequence_length=12,
        position_encoding="rope_3d_mixed",
        num_latents=8,
        num_input_reads=2,
    )
    predictor_config = PredictorConfig(
        supported_modality_names=supported_modality_names,
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        depth=2,
        mlp_ratio=1.0,
        num_heads=2,
        max_sequence_length=12,
        drop_path=0.0,
        output_embedding_size=None,
        position_encoding="rope_3d_mixed",
    )
    model = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=predictor_config,
    ).build()
    model.to(device="cpu")
    return model


def test_train_batch_with_perceiver_encoder(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    train_module_config: LatentMIMTrainModuleConfig,  # noqa: F811
    set_random_seeds: None,
) -> None:
    """A full train_batch runs through the perceiver encoder and losses."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_single_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
    )
    train_module = train_module_config.build(latent_mim_model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        train_module.on_attach = MagicMock(return_value=None)  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchDisc"])

    # Grads reached the set-latent trunk (read/self/read-out + latent pool).
    trunk = latent_mim_model.encoder.blocks[0]
    assert trunk.latents.grad is not None
    assert any(p.grad is not None for p in trunk.read_block.parameters())
    assert any(p.grad is not None for p in trunk.read_out_block.parameters())


def test_encoder_forward_contract(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    set_random_seeds: None,
) -> None:
    """The encoder honors the standard forward contract used by the evals."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_single_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
    )
    _, sample = batch
    encoder = latent_mim_model.encoder
    out = encoder(sample, patch_size=8)
    assert isinstance(out["tokens_and_masks"], TokensAndMasks)
    assert out["project_aggregated"].ndim == 2  # (B, D) pooled embedding

    # Teacher path with all-zero exits (the mainline config): trunk skipped.
    out_t = latent_mim_model.target_encoder.forward(
        sample.unmask(),
        patch_size=8,
        token_exit_cfg={m: 0 for m in encoder.supported_modality_names},
    )
    assert isinstance(out_t["tokens_and_masks"], TokensAndMasks)
