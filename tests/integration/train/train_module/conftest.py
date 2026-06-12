"""Shared fixtures for train module integration tests."""

import pytest
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.modalities import Modality


@pytest.fixture
def supported_modality_names() -> list[str]:
    """Return the modalities used by the train module integration tests."""
    return [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.LATLON.name,
    ]


@pytest.fixture
def optim_config() -> AdamWConfig:
    """Create a small AdamW config for tests."""
    return AdamWConfig(
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
