"""Integration tests for the NAIP GAN train module.

These validate the gradient isolation that defines the objective:
- the discriminator loss updates only the discriminator (not the encoder), and
- the generator loss updates the generator/encoder (not the discriminator).
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.naip_gan import (
    NaipDiscriminatorConfig,
    NaipGanModel,
    NaipGanModelConfig,
    NaipGeneratorConfig,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.naip_gan import (
    NaipGanTrainModule,
    NaipGanTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

torch.set_default_device("cpu")
logger = logging.getLogger(__name__)

SUPPORTED_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.WORLDCOVER.name,
    Modality.LATLON.name,
    Modality.NAIP_10.name,
]
EMBEDDING_SIZE = 16


class MockTrainer:
    """Minimal trainer stub for driving a single train batch."""

    def __init__(self) -> None:
        """Initialize the mock trainer."""
        self._metrics: dict[str, float] = {}
        self.global_step = 0
        self.max_steps = 100

    def record_metric(
        self,
        name: str,
        value: float,
        reduce_type: str,
        namespace: str | None = None,
    ) -> None:
        """Record a metric."""
        self._metrics[name] = value

    def _iter_callbacks(self):  # type: ignore[no-untyped-def]
        """No callbacks in the mock trainer."""
        return iter([])


@pytest.fixture
def naip_samples() -> list[tuple[int, OlmoEarthSample]]:
    """Two samples with a present (decode-only) NAIP modality."""
    s2 = np.random.randn(8, 8, 12, 13).astype(np.float32)
    s1 = np.random.randn(8, 8, 12, 2).astype(np.float32)
    wc = np.random.randn(8, 8, 1, 10).astype(np.float32)
    # naip_10 spatial extent is height * image_tile_size_factor (4) -> 32x32, 4 bands.
    naip = np.random.randn(32, 32, 1, 4).astype(np.float32)
    latlon = np.random.randn(2).astype(np.float32)
    timestamps = np.array(
        [
            [15, 7, 2023],
            [15, 8, 2023],
            [15, 9, 2023],
            [15, 10, 2023],
            [15, 11, 2023],
            [15, 11, 2023],
            [15, 1, 2024],
            [15, 2, 2024],
            [15, 3, 2024],
            [15, 4, 2024],
            [15, 5, 2024],
            [15, 6, 2024],
        ],
        dtype=np.int32,
    )

    def make() -> OlmoEarthSample:
        return OlmoEarthSample(
            sentinel2_l2a=s2,
            sentinel1=s1,
            worldcover=wc,
            naip_10=naip,
            latlon=latlon,
            timestamps=timestamps,
        )

    return [(1, make()), (1, make())]


@pytest.fixture
def naip_gan_model() -> NaipGanModel:
    """A tiny NAIP GAN model on CPU."""
    encoder_config = EncoderConfig(
        supported_modality_names=SUPPORTED_MODALITIES,
        embedding_size=EMBEDDING_SIZE,
        min_patch_size=1,
        max_patch_size=1,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=12,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=SUPPORTED_MODALITIES,
        encoder_embedding_size=EMBEDDING_SIZE,
        decoder_embedding_size=EMBEDDING_SIZE,
        depth=2,
        mlp_ratio=1.0,
        num_heads=2,
        max_sequence_length=12,
        drop_path=0.0,
        output_embedding_size=None,
    )
    generator_config = NaipGeneratorConfig(
        embedding_size=EMBEDDING_SIZE,
        patch_size=1,
        hidden_sizes=[32, 32, 32],
        out_channels=4,
        upsample_factor=4,
        num_res_blocks=1,
    )
    model = NaipGanModelConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        generator_config=generator_config,
    ).build()
    model.to(device="cpu")
    return model


def _build_train_module(
    model: NaipGanModel,
    image_log_interval: int = 0,
    cond_source: str = "online_pooled",
) -> NaipGanTrainModule:
    if cond_source == "raw_sentinel2":
        # Condition on the full Sentinel-2 temporal stack (all 13 sample bands).
        discriminator_config = NaipDiscriminatorConfig(
            embedding_size=EMBEDDING_SIZE,
            in_channels=4,
            image_strided_conv_channels=[16, 32],
            feature_channels=32,
            cond_mode="image",
            cond_in_channels=13,
        )
    else:
        discriminator_config = NaipDiscriminatorConfig(
            embedding_size=EMBEDDING_SIZE,
            in_channels=4,
            image_strided_conv_channels=[16, 32],
            feature_channels=32,
        )
    config = NaipGanTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4, weight_decay=0.0),
        rank_microbatch_size=2,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        masking_config=MaskingConfig(
            strategy_config={
                "type": "random_time_with_decode",
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
                "random_ratio": 0.5,
                "only_decode_modalities": [Modality.NAIP_10.name],
            }
        ),
        discriminator_config=discriminator_config,
        disc_optim_config=AdamWConfig(lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0),
        lambda_adv=1.0,
        lambda_l1=1.0,
        gan_warmup_steps=0,
        image_log_interval=image_log_interval,
        discriminator_cond_source=cond_source,
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        max_grad_norm=1.0,
        transform_config=TransformConfig(transform_type="no_transform"),
    )
    tm = config.build(model, device="cpu")
    tm.on_attach = MagicMock(return_value=None)  # type: ignore
    tm._attach_trainer(MockTrainer())
    return tm


def _collate(
    naip_samples: list[tuple[int, OlmoEarthSample]],
) -> tuple[int, MaskedOlmoEarthSample]:
    masking_strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": [Modality.NAIP_10.name],
        }
    ).build()
    return collate_single_masked_batched(
        naip_samples, transform=None, masking_strategy=masking_strategy
    )


def _has_any_grad(module: torch.nn.Module) -> bool:
    return any(
        p.grad is not None and torch.any(p.grad != 0) for p in module.parameters()
    )


def _all_grads_none(module: torch.nn.Module) -> bool:
    return all(p.grad is None for p in module.parameters())


def test_train_batch_runs_and_records_metrics(
    naip_samples: list[tuple[int, OlmoEarthSample]],
    naip_gan_model: NaipGanModel,
    set_random_seeds: None,
) -> None:
    """A full train batch runs end-to-end and records the GAN metrics."""
    batch = _collate(naip_samples)
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        tm = _build_train_module(naip_gan_model)
        tm.train_batch(batch)
        assert "train/G_l1" in tm.trainer._metrics
        assert "train/D_loss" in tm.trainer._metrics
        # Generator (in the model) received gradients from the GAN branch.
        assert _has_any_grad(tm.model.generator)


def test_gradient_isolation(
    naip_samples: list[tuple[int, OlmoEarthSample]],
    naip_gan_model: NaipGanModel,
    set_random_seeds: None,
) -> None:
    """D loss updates only D; G loss updates only the generator/encoder."""
    batch = _collate(naip_samples)
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        tm = _build_train_module(naip_gan_model)
        patch_size, batch_data = batch
        microbatch = split_masked_batch(batch_data, tm.rank_microbatch_size)[0]
        microbatch = microbatch.to_device(torch.device("cpu"))

        _, _, _, target_output, pooled, fake_naip = tm.model_forward(
            microbatch, patch_size, tm.token_exit_cfg
        )
        cond, cond_valid, cond_time_mask = tm._extract_discriminator_cond(
            microbatch, pooled, target_output
        )

        # --- Discriminator loss: only the discriminator gets gradients. ---
        tm.zero_grads()
        d_loss = tm._maybe_discriminator_loss(
            microbatch,
            cond,
            cond_valid,
            fake_naip,
            num_microbatches=1,
            adversarial_active=True,
            patch_size=patch_size,
            cond_time_mask=cond_time_mask,
        )
        assert d_loss is not None
        d_loss.backward()
        assert _has_any_grad(tm.discriminator)
        assert _all_grads_none(tm.model.encoder)
        assert _all_grads_none(tm.model.generator)

        # --- Generator loss: only the generator/encoder get gradients. ---
        tm.zero_grads()
        gen_loss, _, _ = tm._generator_loss(
            microbatch,
            cond,
            cond_valid,
            fake_naip,
            adversarial_active=True,
            patch_size=patch_size,
            cond_time_mask=cond_time_mask,
        )
        gen_loss.backward()
        assert _all_grads_none(tm.discriminator)
        assert _has_any_grad(tm.model.generator)
        assert _has_any_grad(tm.model.encoder)


@pytest.mark.parametrize(
    "cond_source", ["online_pooled", "target_pooled", "raw_sentinel2"]
)
def test_train_batch_runs_for_cond_sources(
    naip_samples: list[tuple[int, OlmoEarthSample]],
    naip_gan_model: NaipGanModel,
    set_random_seeds: None,
    cond_source: str,
) -> None:
    """Each discriminator conditioning source runs a full batch end-to-end."""
    batch = _collate(naip_samples)
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        tm = _build_train_module(naip_gan_model, cond_source=cond_source)
        tm.train_batch(batch)
        assert "train/G_l1" in tm.trainer._metrics
        assert "train/D_loss" in tm.trainer._metrics
        # The discriminator was trained (D loss active from step 0 here).
        assert _has_any_grad(tm.discriminator)
        # Generator (in the model) received gradients from the GAN branch.
        assert _has_any_grad(tm.model.generator)


class FakeWandb:
    """Minimal wandb stand-in capturing Image/log calls."""

    def __init__(self) -> None:
        """Initialize the fake wandb module."""
        self.logged: list[tuple[dict, int | None]] = []

    def Image(self, arr: np.ndarray, caption: str | None = None) -> np.ndarray:  # noqa: N802
        """Return the array unchanged (records nothing extra)."""
        return arr

    def log(self, data: dict, step: int | None = None) -> None:
        """Capture a log call."""
        self.logged.append((data, step))


def test_naip_images_logged_at_interval(
    naip_samples: list[tuple[int, OlmoEarthSample]],
    naip_gan_model: NaipGanModel,
    set_random_seeds: None,
) -> None:
    """Paired real/generated NAIP images are logged to W&B at the interval."""
    batch = _collate(naip_samples)
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        tm = _build_train_module(naip_gan_model, image_log_interval=1)
        fake_wandb = FakeWandb()
        tm._get_wandb = lambda: fake_wandb  # type: ignore[method-assign]
        tm.train_batch(batch)

        assert len(fake_wandb.logged) == 1
        data, step = fake_wandb.logged[0]
        assert step == tm.trainer.global_step
        images = data["gan/naip_real_vs_fake"]
        # Two valid NAIP samples in the batch -> two paired images.
        assert len(images) == 2
        for pair in images:
            # Side-by-side [H, 2W, 3] RGB in [0, 1].
            assert pair.ndim == 3
            assert pair.shape[2] == 3
            assert float(pair.min()) >= 0.0 and float(pair.max()) <= 1.0


def test_naip_images_not_logged_when_disabled(
    naip_samples: list[tuple[int, OlmoEarthSample]],
    naip_gan_model: NaipGanModel,
    set_random_seeds: None,
) -> None:
    """No images are logged when image_log_interval is 0."""
    batch = _collate(naip_samples)
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        tm = _build_train_module(naip_gan_model, image_log_interval=0)
        fake_wandb = FakeWandb()
        tm._get_wandb = lambda: fake_wandb  # type: ignore[method-assign]
        tm.train_batch(batch)
        assert len(fake_wandb.logged) == 0
