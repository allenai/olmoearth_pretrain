"""Unit tests for launching experiments."""

import pytest
from olmo_core.config import DType
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train import TrainerConfig

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthBeakerLaunchConfig,
    OlmoEarthExperimentConfig,
    OlmoEarthVisualizeConfig,
    build_config,
)
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

MAX_PATCH_SIZE = 8  # NOTE: actual patch_size <= max_patch_size


def minimal_common_components() -> CommonComponents:
    """Return a minimal CommonComponents object."""
    return CommonComponents(
        run_name="test_run",
        save_folder="test_save_folder",
        training_modalities=["sentinel2", "sentinel1", "worldcover", "naip"],
        launch=OlmoEarthBeakerLaunchConfig(
            name="test_run",
            cmd=["dummy_cmd"],
            clusters=["dummy_cluster"],
            budget="dummy_budget",
        ),
    )


def minimal_model_config_builder(common: CommonComponents) -> LatentMIMConfig:
    """Return a minimal LatentMIMConfig."""
    ENCODER_EMBEDDING_SIZE = 16
    DECODER_EMBEDDING_SIZE = 16
    ENCODER_DEPTH = 2
    DECODER_DEPTH = 2
    ENCODER_NUM_HEADS = 2
    DECODER_NUM_HEADS = 8
    MLP_RATIO = 4.0
    encoder_config = EncoderConfig(
        supported_modality_names=common.training_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        num_heads=ENCODER_NUM_HEADS,
        depth=ENCODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        drop_path=0.1,
        max_sequence_length=12,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DECODER_DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=DECODER_NUM_HEADS,
        max_sequence_length=12,
        supported_modality_names=common.training_modalities,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def minimal_dataset_config_builder(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Return a minimal OlmoEarthDatasetConfig."""
    h5py_dir = "test_tile_path"
    return OlmoEarthDatasetConfig(
        h5py_dir=h5py_dir,
        training_modalities=common.training_modalities,
        dtype=DType.float32,
    )


def minimal_dataloader_config_builder(
    common: CommonComponents,
) -> OlmoEarthDataLoaderConfig:
    """Return a minimal OlmoEarthDataLoaderConfig."""
    GLOBAL_BATCH_SIZE = 16
    dataloader_config = OlmoEarthDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=3622,
        work_dir=common.save_folder,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[256],
        token_budget=1000000,
    )
    return dataloader_config


def minimal_trainer_config_builder(common: CommonComponents) -> TrainerConfig:
    """Return a minimal TrainerConfig."""
    METRICS_COLLECT_INTERVAL = 1
    CANCEL_CHECK_INTERVAL = 1
    # Let us not use garbage collector fallback
    trainer_config = TrainerConfig(
        work_dir=common.save_folder,
        save_folder=common.save_folder,
        cancel_check_interval=CANCEL_CHECK_INTERVAL,
        metrics_collect_interval=METRICS_COLLECT_INTERVAL,
    )
    return trainer_config


def minimal_train_module_config_builder(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """Return a minimal LatentMIMTrainModuleConfig."""
    LR = 0.002
    WD = 0.02
    RANK_MICRO_BATCH_SIZE = 16
    ENCODE_RATIO = 0.1
    DECODE_RATIO = 0.75

    optim_config = AdamWConfig(lr=LR, weight_decay=WD)
    masking_config = MaskingConfig(
        strategy_config={
            "type": "random",
            "encode_ratio": ENCODE_RATIO,
            "decode_ratio": DECODE_RATIO,
        }
    )
    loss_config = LossConfig(
        loss_config={
            "type": "patch_discrimination",
        }
    )
    token_exit_cfg = {modality: 0 for modality in common.training_modalities}
    transform_config = TransformConfig(transform_type="flip_and_rotate")
    train_module_config = LatentMIMTrainModuleConfig(
        optim_config=optim_config,
        masking_config=masking_config,
        loss_config=loss_config,
        token_exit_cfg=token_exit_cfg,
        rank_microbatch_size=RANK_MICRO_BATCH_SIZE,
        max_grad_norm=1.0,
        transform_config=transform_config,
    )
    return train_module_config


def minimal_visualize_config_builder(
    common: CommonComponents,
) -> OlmoEarthVisualizeConfig:
    """Return a minimal OlmoEarthVisualizeConfig."""
    return OlmoEarthVisualizeConfig(output_dir="dummy_visuals")


def test_build_config_no_overrides() -> None:
    """Test that build_config produces a valid OlmoEarthExperimentConfig."""
    common = minimal_common_components()
    config = build_config(
        common=common,
        model_config_builder=minimal_model_config_builder,
        dataset_config_builder=minimal_dataset_config_builder,
        dataloader_config_builder=minimal_dataloader_config_builder,
        trainer_config_builder=minimal_trainer_config_builder,
        train_module_config_builder=minimal_train_module_config_builder,
        visualize_config_builder=minimal_visualize_config_builder,
        overrides=[],
    )

    assert isinstance(config, OlmoEarthExperimentConfig)
    assert config.run_name == "test_run"
    assert config.data_loader.global_batch_size == 16
    assert config.visualize is not None
    assert config.visualize.output_dir == "dummy_visuals"


@pytest.mark.parametrize(
    "overrides,expected_cancel_check,expected_metrics_collect,expected_run_name",
    [
        # override trainer fields: cancel_check_interval & metrics_collect_interval
        # plus the top-level run_name
        (
            [
                "trainer.cancel_check_interval=2",
                "--trainer.metrics_collect_interval=5",
                "run_name=override_run",
            ],
            2,
            5,
            "override_run",
        ),
        (
            [
                "--trainer.cancel_check_interval=10",
                "trainer.metrics_collect_interval=13",
                "run_name=special_expt",
            ],
            10,
            13,
            "special_expt",
        ),
    ],
)
def test_build_config_with_trainer_overrides(
    overrides: list[str],
    expected_cancel_check: int,
    expected_metrics_collect: int,
    expected_run_name: str,
) -> None:
    """Test applying multiple overrides to trainer-related fields."""
    common = minimal_common_components()

    config = build_config(
        common=common,
        model_config_builder=minimal_model_config_builder,
        dataset_config_builder=minimal_dataset_config_builder,
        dataloader_config_builder=minimal_dataloader_config_builder,
        trainer_config_builder=minimal_trainer_config_builder,
        train_module_config_builder=minimal_train_module_config_builder,
        visualize_config_builder=None,
        overrides=overrides,
    )

    assert isinstance(config, OlmoEarthExperimentConfig)
    # Confirm that the overrides took effect
    assert config.trainer.cancel_check_interval == expected_cancel_check
    assert config.trainer.metrics_collect_interval == expected_metrics_collect
    assert config.run_name == expected_run_name


def test_overrides_with_common_prefix() -> None:
    """Test that overrides with the common prefix are processed correctly."""
    common = minimal_common_components()
    config = build_config(
        common=common,
        model_config_builder=minimal_model_config_builder,
        dataset_config_builder=minimal_dataset_config_builder,
        dataloader_config_builder=minimal_dataloader_config_builder,
        trainer_config_builder=minimal_trainer_config_builder,
        train_module_config_builder=minimal_train_module_config_builder,
        visualize_config_builder=None,
        overrides=["common.training_modalities=[sentinel2, sentinel1]"],
    )

    assert isinstance(config, OlmoEarthExperimentConfig)
    assert config.dataset.training_modalities == ["sentinel2", "sentinel1"]


def test_build_config_invalid_override_raises() -> None:
    """Example test to confirm that an invalid override raises an exception."""
    common = minimal_common_components()
    invalid_overrides = ["trainer.this_field_does_not_exist=999"]

    # Depending on how config.merge is implemented, it may raise a KeyError, AttributeError, or another exception.
    # Adjust the expected exception type as necessary for your config merging system.
    with pytest.raises(Exception):
        build_config(
            common=common,
            model_config_builder=minimal_model_config_builder,
            dataset_config_builder=minimal_dataset_config_builder,
            dataloader_config_builder=minimal_dataloader_config_builder,
            trainer_config_builder=minimal_trainer_config_builder,
            train_module_config_builder=minimal_train_module_config_builder,
            visualize_config_builder=None,
            overrides=invalid_overrides,
        )
