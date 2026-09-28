"""Unit tests for launching experiments."""

from typing import Any

import pytest
from gantry.api import GitRepoState
from gantry.api import Recipe as GantryRecipe
from olmo_core.config import DType
from olmo_core.launch.beaker import BeakerLaunchConfig
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


def stub_git_repo_state() -> GitRepoState:
    """Return a GitRepoState stub that doesn't touch git or the GitHub API.

    The launch config's ``git`` field defaults to ``GitRepoState.from_env()``,
    which requires running from the root of a pushed git checkout, so tests
    pass an explicit stub instead.
    """
    return GitRepoState(
        repo="dummy/dummy",
        repo_url="https://github.com/dummy/dummy",
        ref="0" * 40,
        _is_remote=True,
    )


def minimal_launch_config(**kwargs: Any) -> OlmoEarthBeakerLaunchConfig:
    """Return a minimal OlmoEarthBeakerLaunchConfig."""
    return OlmoEarthBeakerLaunchConfig(
        name="test_run",
        cmd=["dummy_cmd"],
        clusters=["dummy_cluster"],
        budget="dummy_budget",
        git=stub_git_repo_state(),
        **kwargs,
    )


def minimal_common_components() -> CommonComponents:
    """Return a minimal CommonComponents object."""
    return CommonComponents(
        run_name="test_run",
        save_folder="test_save_folder",
        training_modalities=["sentinel2", "sentinel1", "worldcover", "naip"],
        launch=minimal_launch_config(),
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
        num_masked_views=1,
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
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


@pytest.fixture
def stub_base_recipe(monkeypatch: pytest.MonkeyPatch) -> GantryRecipe:
    """Stub the olmo-core recipe builder so no Beaker/network access happens.

    Returns the recipe object that the stubbed base ``_build_recipe`` will
    hand back, so tests can assert on the mutations our subclass applies.
    """
    recipe = GantryRecipe(args=["echo", "hello"])

    def fake_build_recipe(
        self: BeakerLaunchConfig, beaker: Any, **kwargs: Any
    ) -> tuple[GantryRecipe, dict[str, Any]]:
        return recipe, {"show_logs": False}

    monkeypatch.setattr(BeakerLaunchConfig, "_build_recipe", fake_build_recipe)
    return recipe


def test_min_runtime_defaults_to_8h(stub_base_recipe: GantryRecipe) -> None:
    """The built recipe should carry an 8h min runtime by default."""
    launch_config = minimal_launch_config()
    recipe, _ = launch_config._build_recipe(beaker=None)
    assert recipe is stub_base_recipe
    assert recipe.min_runtime == "8h"


def test_min_runtime_disabled(stub_base_recipe: GantryRecipe) -> None:
    """Setting min_runtime=None should leave the recipe untouched."""
    launch_config = minimal_launch_config(min_runtime=None)
    recipe, _ = launch_config._build_recipe(beaker=None)
    assert recipe.min_runtime is None


def test_min_runtime_override(stub_base_recipe: GantryRecipe) -> None:
    """A custom min_runtime value should be applied to the recipe."""
    launch_config = minimal_launch_config(min_runtime="30m")
    recipe, _ = launch_config._build_recipe(beaker=None)
    assert recipe.min_runtime == "30m"


def test_install_and_gh_token_secret_applied(stub_base_recipe: GantryRecipe) -> None:
    """The install command and GitHub token secret should be applied when set."""
    launch_config = minimal_launch_config(
        install="uv sync --locked --all-extras",
        gh_token_secret="someuser_GITHUB_TOKEN",
    )
    recipe, _ = launch_config._build_recipe(beaker=None)
    assert recipe.install == "uv sync --locked --all-extras"
    assert recipe.gh_token_secret == "someuser_GITHUB_TOKEN"


def test_install_and_gh_token_secret_defaults(stub_base_recipe: GantryRecipe) -> None:
    """When unset, the recipe's install/gh_token_secret defaults are kept."""
    launch_config = minimal_launch_config()
    recipe, _ = launch_config._build_recipe(beaker=None)
    assert recipe.install is None
    assert recipe.gh_token_secret == "GITHUB_TOKEN"


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
