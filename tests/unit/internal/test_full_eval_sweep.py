"""Comprehensive test suite for the full eval sweep command builder."""

import argparse
from typing import Any
from unittest.mock import Mock, patch

import pytest

from olmoearth_pretrain.evals.models import BaselineModelName
from olmoearth_pretrain.internal.all_evals import EVAL_TASKS
from olmoearth_pretrain.internal.full_eval_sweep import (
    LP_LRs,
    Normalization_MODES,
    build_commands,
    create_linear_probe_arg,
    get_dino_v3_args,
    get_galileo_args,
    get_panopticon_args,
    loop_through_params,
    pooling_types,
)


# Fixtures for reusable test data
@pytest.fixture
def base_args() -> argparse.Namespace:
    """Base arguments for testing."""
    return argparse.Namespace(
        cluster="test-cluster",
        checkpoint_path="/path/to/checkpoint",
        module_path="test_module.py",
        project_name="test_project",
        defaults_only=False,
        dry_run=True,
        model_name=None,
        model=None,
        all_sizes=False,
        lr_only=False,
        select_best_val=False,
        model_skip_names=None,
        size=None,
    )


@pytest.fixture
def minimal_args() -> argparse.Namespace:
    """Minimal required arguments."""
    return argparse.Namespace(
        cluster="local",
        checkpoint_path=None,
        module_path="test_module.py",
        project_name=None,
        defaults_only=True,
        dry_run=True,
        model_name=None,
        model=None,
        all_sizes=False,
        lr_only=False,
        select_best_val=False,
        model_skip_names=None,
        size=None,
    )


# Unit tests for helper functions
class TestCreateLinearProbeArg:
    """Test create_linear_probe_arg function."""

    def test_basic_functionality(self) -> None:
        """Test basic linear probe argument creation."""
        result: str = create_linear_probe_arg("eurosat", "probe_lr")
        expected: str = (
            "--trainer.callbacks.downstream_evaluator.tasks.eurosat.probe_lr={arg}"
        )
        assert result == expected

    def test_different_task_names(self) -> None:
        """Test with different task names."""
        result: str = create_linear_probe_arg("cropharvest", "pooling_type")
        expected: str = "--trainer.callbacks.downstream_evaluator.tasks.cropharvest.pooling_type={arg}"
        assert result == expected

    def test_special_characters_in_names(self) -> None:
        """Test with special characters in task names."""
        result: str = create_linear_probe_arg("m-eurosat", "embedding_batch_size")
        expected: str = "--trainer.callbacks.downstream_evaluator.tasks.m-eurosat.embedding_batch_size={arg}"
        assert result == expected


class TestLoopThroughParams:
    """Test loop_through_params function."""

    def test_generates_all_combinations(self) -> None:
        """Test that all parameter combinations are generated."""
        params_list: list[dict[str, float | str]] = list(loop_through_params())

        # Should generate len(LP_LRs) * len(Normalization_MODES) * len(pooling_types) combinations
        expected_count: int = (
            len(LP_LRs) * len(Normalization_MODES) * len(pooling_types)
        )
        assert len(params_list) == expected_count

    def test_parameter_structure(self) -> None:
        """Test that each parameter dict has the correct structure."""
        params_list: list[dict[str, float | str]] = list(loop_through_params())

        for params in params_list:
            assert "lr" in params
            assert "norm_mode" in params
            assert "pooling_type" in params
            assert params["lr"] in LP_LRs
            assert params["norm_mode"] in Normalization_MODES
            assert params["pooling_type"] in pooling_types

    def test_all_learning_rates_included(self) -> None:
        """Test that all learning rates are included in the sweep."""
        params_list: list[dict[str, float | str]] = list(loop_through_params())
        lrs_found = {params["lr"] for params in params_list}
        assert lrs_found == set(LP_LRs)


class TestLoopThroughParamsNoNorm:
    """Test loop_through_params function with no_norm=True."""

    def test_generates_correct_combinations(self) -> None:
        """Test that loop_through_params(no_norm=True) generates the right combinations."""
        params_list: list[dict[str, float | str]] = list(
            loop_through_params(no_norm=True)
        )

        # Should generate len(pooling_types) * len(LP_LRs) combinations (only dataset norm mode)
        expected_count: int = len(pooling_types) * len(LP_LRs)
        assert len(params_list) == expected_count

    def test_parameter_structure(self) -> None:
        """Test parameter structure for loop_through_params(no_norm=True)."""
        params_list: list[dict[str, float | str]] = list(
            loop_through_params(no_norm=True)
        )

        for params in params_list:
            assert "lr" in params
            assert "pooling_type" in params
            assert "norm_mode" in params
            assert params["norm_mode"] == "dataset"  # Should only be dataset mode
            assert params["lr"] in LP_LRs
            assert params["pooling_type"] in pooling_types


class TestModelSpecificArgs:
    """Test model-specific argument generation functions."""

    def test_get_dino_v3_args(self) -> None:
        """Test DinoV3 argument generation."""
        result: str = get_dino_v3_args()

        # Should contain dataset args and norm method args
        assert "norm_stats_from_pretrained=False" in result
        assert "norm_method=NormMethod.NORM_YES_CLIP_MIN_MAX_INT" in result

        # Check that some real task names are included
        assert any(task_name in result for task_name in EVAL_TASKS.keys())

    def test_get_panopticon_args(self) -> None:
        """Test Panopticon argument generation."""
        result: str = get_panopticon_args()

        # Should contain dataset args and standardization
        assert "norm_stats_from_pretrained=False" in result
        assert "norm_method=NormMethod.STANDARDIZE" in result

        # Check that some real task names are included
        assert any(task_name in result for task_name in EVAL_TASKS.keys())

    def test_get_galileo_args_pretrained_normalizer(self) -> None:
        """Test Galileo args with pretrained normalizer."""
        result: str = get_galileo_args(pretrained_normalizer=True)

        assert "norm_stats_from_pretrained=False" in result
        assert "norm_method=NormMethod.NO_NORM" in result
        assert "use_pretrained_normalizer=True" in result
        assert "embedding_batch_size=8" in result

    def test_get_galileo_args_dataset_normalizer(self) -> None:
        """Test Galileo args with dataset normalizer."""
        result: str = get_galileo_args(pretrained_normalizer=False)

        assert "norm_stats_from_pretrained=False" in result
        assert "use_pretrained_normalizer=False" in result
        assert "embedding_batch_size=8" in result


class TestBuildCommandsBasic:
    """Test basic build_commands functionality."""

    def test_build_commands_defaults_only(self, base_args: argparse.Namespace) -> None:
        """Test build_commands with defaults_only=True."""
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command: str = commands[0]
        assert "TRAIN_SCRIPT_PATH=test_module.py" in command
        assert "dry_run" in command
        assert "test-cluster" in command
        assert "/path/to/checkpoint" in command
        assert "_df" in command

    def test_build_commands_no_checkpoint_path(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test build_commands without checkpoint path."""
        base_args.checkpoint_path = None
        base_args.defaults_only = True

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "test1234"
            commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command: str = commands[0]
        assert "--trainer.load_path=" not in command
        assert "TRAIN_SCRIPT_PATH=test_module.py" in command

    def test_build_commands_with_model_name(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test build_commands with specified model name."""
        base_args.checkpoint_path = None
        base_args.model_name = "my_custom_model"
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command: str = commands[0]
        assert "my_custom_model_df" in command

    def test_model_name_and_checkpoint_path(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test build_commands with specified model name and checkpoint path."""
        base_args.checkpoint_path = "/path/to/checkpoint"
        base_args.model_name = "my_custom_model"
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command: str = commands[0]
        assert "my_custom_model" in command


class TestBuildCommandsModelTypes:
    """Test build_commands with different model types."""

    def test_build_commands_dino_v3(self, base_args: argparse.Namespace) -> None:
        """Test build_commands with DinoV3 model."""
        base_args.model = BaselineModelName.DINO_V3
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command: str = commands[0]
        assert "norm_method=NormMethod.NORM_YES_CLIP_MIN_MAX_INT" in command

    def test_build_commands_panopticon(self, base_args: argparse.Namespace) -> None:
        """Test build_commands with Panopticon model."""
        base_args.model = BaselineModelName.PANOPTICON
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command: str = commands[0]
        assert "norm_method=NormMethod.STANDARDIZE" in command

    def test_build_commands_galileo(self, base_args: argparse.Namespace) -> None:
        """Test build_commands with Galileo model."""
        base_args.model = BaselineModelName.GALILEO
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command: str = commands[0]
        assert "use_pretrained_normalizer=True" in command
        assert "embedding_batch_size=8" in command


class TestBuildCommandsSweep:
    """Test build_commands with full parameter sweep."""

    def test_build_commands_full_sweep_default_model(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test build_commands with full sweep for default model."""
        base_args.defaults_only = False

        with patch(
            "olmoearth_pretrain.evals.datasets.configs.get_eval_mode"
        ) as mock_get_eval_mode:
            mock_get_eval_mode.return_value = "linear_probe"
            with patch(
                "olmoearth_pretrain.evals.datasets.configs.dataset_to_config"
            ) as mock_dataset_to_config:
                mock_config = Mock()
                mock_config.task_type = "classification"
                mock_dataset_to_config.return_value = mock_config

                commands: list[str] = build_commands(base_args, [])

        # Should generate multiple commands for parameter sweep
        expected_count: int = (
            len(LP_LRs) * len(Normalization_MODES) * len(pooling_types)
        )
        assert len(commands) == expected_count

        # Check that different parameters are included
        command_text: str = " ".join(commands)
        assert "dataset" in command_text
        assert "pre_trained" in command_text

    def test_build_commands_sweep_dino_v3(self, base_args: argparse.Namespace) -> None:
        """Test build_commands sweep with DinoV3 (dataset norm only)."""
        base_args.model = BaselineModelName.DINO_V3
        base_args.defaults_only = False

        commands: list[str] = build_commands(base_args, [])

        # Should use loop_through_params(no_norm=True), so fewer combinations (only dataset norm mode)
        expected_count: int = len(pooling_types) * len(LP_LRs)
        assert len(commands) == expected_count

        # All commands should have DinoV3 args
        for command in commands:
            assert "norm_method=NormMethod.NORM_YES_CLIP_MIN_MAX_INT" in command


class TestBuildCommandsExecution:
    """Test build_commands execution modes."""

    def test_build_commands_dry_run(self, base_args: argparse.Namespace) -> None:
        """Test build_commands in dry run mode."""
        base_args.dry_run = True
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        assert "dry_run" in commands[0]

    def test_build_commands_local_cluster(self, base_args: argparse.Namespace) -> None:
        """Test build_commands with local cluster."""
        base_args.cluster = "local"
        base_args.dry_run = False
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        # Local should use torchrun instead of python3
        assert "torchrun" in commands[0]

    def test_build_commands_remote_cluster(self, base_args: argparse.Namespace) -> None:
        """Test build_commands with remote cluster."""
        base_args.cluster = "ai2/saturn"
        base_args.dry_run = False
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        # Remote should use python3 and launch
        assert "python3" in commands[0]
        assert "launch" in commands[0]

    def test_build_commands_with_extra_cli(self, base_args: argparse.Namespace) -> None:
        """Test build_commands with extra CLI arguments."""
        base_args.defaults_only = True
        extra_cli: list[str] = ["--custom_arg=value", "--another_flag"]

        commands: list[str] = build_commands(base_args, extra_cli)

        assert len(commands) == 1
        command: str = commands[0]
        assert "--custom_arg=value" in command
        assert "--another_flag" in command

    def test_build_commands_no_project_name(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test build_commands without project name (uses default)."""
        base_args.project_name = None
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        assert "helios_in_loop_evals" in commands[0]


class TestParametrizedTests:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "model_type,expected_args",
        [
            (
                BaselineModelName.DINO_V3,
                "norm_method=NormMethod.NORM_YES_CLIP_MIN_MAX_INT",
            ),
            (BaselineModelName.PANOPTICON, "norm_method=NormMethod.STANDARDIZE"),
            (BaselineModelName.GALILEO, "use_pretrained_normalizer=True"),
        ],
    )
    def test_model_specific_args_parametrized(
        self,
        model_type: BaselineModelName,
        expected_args: str,
        base_args: argparse.Namespace,
    ) -> None:
        """Test different model types parametrically."""
        base_args.defaults_only = True
        base_args.model = model_type

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        assert expected_args in commands[0]

    @pytest.mark.parametrize("lr", LP_LRs)
    @pytest.mark.parametrize("norm_mode", Normalization_MODES)
    @pytest.mark.parametrize("pooling_type", pooling_types)
    def test_parameter_combinations(
        self, lr: float, norm_mode: str, pooling_type: Any
    ) -> None:
        """Test that all parameter combinations are valid."""
        params_list: list[dict[str, float | str]] = list(loop_through_params())

        # Check that this specific combination exists
        target_combo: dict[str, float | str] = {
            "lr": lr,
            "norm_mode": norm_mode,
            "pooling_type": pooling_type,
        }
        assert target_combo in params_list


class TestIntegration:
    """Integration tests that test the full workflow."""

    def test_full_workflow_minimal_args(self, minimal_args: argparse.Namespace) -> None:
        """Test the complete workflow with minimal arguments."""
        commands: list[str] = build_commands(minimal_args, [])

        assert len(commands) == 1
        command: str = commands[0]

        # Check basic structure
        assert "TRAIN_SCRIPT_PATH=test_module.py" in command
        assert "dry_run" in command
        assert "local" in command
        assert "helios_in_loop_evals" in command  # default project name

    def test_complex_workflow_with_all_options(
        self, base_args: argparse.Namespace
    ) -> None:
        """Test complex workflow with multiple options enabled."""
        base_args.defaults_only = False
        base_args.model = BaselineModelName.GALILEO
        base_args.project_name = "complex_test"

        with patch(
            "olmoearth_pretrain.evals.datasets.configs.get_eval_mode"
        ) as mock_get_eval_mode:
            mock_get_eval_mode.return_value = "linear_probe"
            with patch(
                "olmoearth_pretrain.evals.datasets.configs.dataset_to_config"
            ) as mock_dataset_to_config:
                mock_config = Mock()
                mock_config.task_type = "classification"
                mock_dataset_to_config.return_value = mock_config

                commands: list[str] = build_commands(
                    base_args,
                    [
                        "--extra",
                        "--trainer.callbacks.downstream_evaluator.tasks_to_run=[m_eurosat,sickle_landsat]",
                    ],
                )

        # Should generate full sweep for Galileo model
        expected_count: int = (
            len(LP_LRs) * len(Normalization_MODES) * len(pooling_types)
        )
        assert len(commands) == expected_count

        # All commands should have Galileo-specific args
        for command in commands:
            assert "embedding_batch_size=8" in command
            assert "complex_test" in command
            assert "--extra" in command
            assert "pre_trained" in command or "dataset" in command
            assert (
                "--trainer.callbacks.downstream_evaluator.tasks_to_run=[m_eurosat,sickle_landsat]"
                in command
            )
