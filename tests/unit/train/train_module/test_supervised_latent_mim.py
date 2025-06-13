"""Unit tests for the supervised latent MIM Training Module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from torch.nn.functional import one_hot
from einops import repeat
import torch
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train.config import TrainerConfig

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample, collate_helios
from helios.data.transform import TransformConfig
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.latent_mim import LatentMIM, LatentMIMConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.supervised_latent_mim import SupervisedLatentMIMTrainModule


class TestSupervisedLatentMIMUnit:
    def test_supervised_loss(self) -> None:
        max_patch_size = 12
        batch_patch_size = 3
        b, h, w, = 1, 2, 2
        supervisory_modalities = {
            # 95 will be assigned class 11
            "worldcover": repeat(torch.tensor([[95, 95], [95, 70]]), "h w -> b (h p1) (w p2) t d", b=b, p1=batch_patch_size, p2=batch_patch_size, t=1, d=1)
        }
        # 1s where the value is present
        probe_outputs = {
            "mask": repeat(torch.tensor([[1, 0], [0, 1]]), "h w -> b (h p1) (w p2)", b=b, p1=max_patch_size, p2=max_patch_size),
            "worldcover_0": repeat(one_hot(torch.tensor([[4, 11], [11, 7]]), num_classes=12).float(), "h w d -> b (p1 h) (p2 w) d", b=b, p1=max_patch_size, p2=max_patch_size)
        }
        org_loss = torch.tensor(0).float()
        loss = SupervisedLatentMIMTrainModule.supervisory_losses(supervisory_modalities, probe_outputs, org_loss)
        print(loss)
        assert loss == 0
