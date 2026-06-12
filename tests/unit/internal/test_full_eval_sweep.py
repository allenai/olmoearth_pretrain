"""Comprehensive test suite for the full eval sweep command builder."""

import argparse
from unittest.mock import Mock, patch

import pytest

from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.evals.models import BaselineModelName
from olmoearth_pretrain.internal.all_evals import (
    EVAL_TASKS,
    _eval_task,
    _studio_linear_probe_task,
)
from olmoearth_pretrain.internal.full_eval_sweep import (
    _MODEL_SWEEP_SPECS,
    LP_LRs,
    Normalization_MODES,
    _get_model_sizes,
    _get_normalization_args,
    _model_uses_dataset_norm_only,
    build_commands,
    create_linear_probe_arg,
    get_dino_v3_args,
    get_galileo_args,
    get_panopticon_args,
    loop_through_params,
    pooling_types,
)
from olmoearth_pretrain.modalities import Modality
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.callbacks.evaluator_callback import EvalMode


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
        task_skip_names=None,
        task_names=None,
        size=None,
        load_eval_settings_from_json=False,
        quantize_embeddings=False,
        embedding_dim=None,
        embedding_diagnostics_only=False,
        checkpoint_dir=None,
        steps=None,
        label_fraction=1.0,
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
        task_skip_names=None,
        task_names=None,
        size=None,
        load_eval_settings_from_json=False,
        quantize_embeddings=False,
        embedding_dim=None,
        embedding_diagnostics_only=False,
        checkpoint_dir=None,
        steps=None,
        label_fraction=1.0,
    )


# Unit tests for helper functions
def test_eval_task_helper_derives_single_modality_metadata() -> None:
    """Eval task helper should materialize stable dataset metadata."""
    task = _eval_task(dataset="m-eurosat")

    assert task.input_modalities == [Modality.SENTINEL2_L2A.name]
    assert task.eval_mode == EvalMode.KNN


def test_eval_task_helper_requires_explicit_multimodal_input() -> None:
    """Ambiguous multimodal datasets should keep explicit task modalities."""
    with pytest.raises(ValueError, match="input_modalities must be set explicitly"):
        _eval_task(dataset="pastis")


def test_eval_task_helper_preserves_manual_overrides() -> None:
    """Manual task choices should remain explicit over derived metadata."""
    task = _eval_task(
        dataset="pastis",
        input_modalities=[Modality.SENTINEL1.name],
        primary_metric=EvalMetric.MIOU,
        pooling_type=PoolingType.MAX,
    )

    assert task.input_modalities == [Modality.SENTINEL1.name]
    assert task.eval_mode == EvalMode.LINEAR_PROBE
    assert task.primary_metric == EvalMetric.MIOU
    assert task.pooling_type == PoolingType.MAX


def test_studio_linear_probe_task_derives_registry_defaults() -> None:
    """Studio helper should derive modalities while applying LP defaults."""
    task = _studio_linear_probe_task(
        dataset="tolbi_crop",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
    )

    assert task.input_modalities == [Modality.SENTINEL2_L2A.name]
    assert task.eval_mode == EvalMode.LINEAR_PROBE
    assert task.pooling_type == PoolingType.MEAN
    assert task.norm_stats_from_pretrained is True


def test_eval_tasks_materialize_derived_fields() -> None:
    """Public eval tasks should still expose concrete mode and modality fields."""
    assert EVAL_TASKS["m_eurosat"].eval_mode == EvalMode.KNN
    assert EVAL_TASKS["m_eurosat"].input_modalities == [Modality.SENTINEL2_L2A.name]
    assert EVAL_TASKS["m_bigearthnet"].primary_metric == EvalMetric.MACRO_F1
    assert EVAL_TASKS["pastis_sentinel1"].eval_mode == EvalMode.LINEAR_PROBE
    assert EVAL_TASKS["pastis_sentinel1"].input_modalities == [Modality.SENTINEL1.name]
    assert EVAL_TASKS["forest_loss_driver"].eval_mode == EvalMode.LINEAR_PROBE
    assert EVAL_TASKS["forest_loss_driver"].input_modalities == [
        Modality.SENTINEL2_L2A.name
    ]


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
        result: str = create_linear_probe_arg("nandi", "pooling_type")
        expected: str = (
            "--trainer.callbacks.downstream_evaluator.tasks.nandi.pooling_type={arg}"
        )
        assert result == expected

    def test_special_characters_in_names(self) -> None:
        """Test with special characters in task names."""
        result: str = create_linear_probe_arg("m-eurosat", "embedding_batch_size")
        expected: str = "--trainer.callbacks.downstream_evaluator.tasks.m-eurosat.embedding_batch_size={arg}"
        assert result == expected


def test_loop_through_params_generates_expected_grid() -> None:
    """The default sweep should cover every lr, normalization, and pooling pair."""
    params = {
        (item["lr"], item["norm_mode"], item["pooling_type"])
        for item in loop_through_params()
    }
    expected = {
        (lr, norm_mode, pooling_type)
        for lr in LP_LRs
        for norm_mode in Normalization_MODES
        for pooling_type in pooling_types
    }

    assert params == expected


def test_loop_through_params_can_skip_pretrained_norm() -> None:
    """Dataset-norm-only sweeps should keep the lr/pooling grid."""
    params = {
        (item["lr"], item["norm_mode"], item["pooling_type"])
        for item in loop_through_params(no_norm=True)
    }
    expected = {
        (lr, "dataset", pooling_type) for lr in LP_LRs for pooling_type in pooling_types
    }

    assert params == expected


class TestModelSpecificArgs:
    """Test model-specific argument generation functions."""

    def test_model_sweep_specs_cover_registered_baselines(self) -> None:
        """Every registered baseline should have explicit full-sweep policy."""
        assert set(_MODEL_SWEEP_SPECS) == set(BaselineModelName)

    def test_dataset_norm_only_models_are_explicit(self) -> None:
        """The reduced normalization sweep should come from model sweep specs."""
        dataset_norm_only_models = {
            model for model in BaselineModelName if _model_uses_dataset_norm_only(model)
        }

        assert dataset_norm_only_models == {
            BaselineModelName.DINO_V3,
            BaselineModelName.PANOPTICON,
            BaselineModelName.TESSERA,
        }

    def test_get_normalization_args_uses_model_specific_builder(self) -> None:
        """Models with pretrained normalizers should route through their builder."""
        pre_trained_args = _get_normalization_args(
            BaselineModelName.GALILEO,
            "pre_trained",
        )
        dataset_args = _get_normalization_args(BaselineModelName.GALILEO, "dataset")

        assert "norm_method=NormMethod.NO_NORM" in pre_trained_args
        assert "use_pretrained_normalizer=True" in pre_trained_args
        assert "norm_method=NormMethod.NORM_NO_CLIP_2_STD" in dataset_args
        assert "use_pretrained_normalizer=False" in dataset_args

    def test_get_model_sizes_prefers_requested_size_then_all_sizes(self) -> None:
        """Model-size selection should be explicit and type-stable."""
        assert _get_model_sizes(BaselineModelName.CROMA, "large", True) == ["large"]
        assert _get_model_sizes(BaselineModelName.CROMA, None, True) == [
            "base",
            "large",
        ]
        assert _get_model_sizes(BaselineModelName.CROMA, None, False) == [None]

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

    def test_checkpoint_sweep_embedding_diagnostics_only(
        self, base_args: argparse.Namespace
    ) -> None:
        """Checkpoint sweep diagnostics-only mode propagates env vars."""
        base_args.checkpoint_dir = "/path/to/checkpoints/run"
        base_args.checkpoint_path = None
        base_args.embedding_diagnostics_only = True
        base_args.defaults_only = True

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command = commands[0]
        assert "CHECKPOINT_DIR=/path/to/checkpoints/run" in command
        assert "EMBEDDING_DIAGNOSTICS_ONLY=1" in command
        assert "checkpoint_sweep_evals.py" in command

    def test_checkpoint_sweep_with_steps_and_size(
        self, base_args: argparse.Namespace
    ) -> None:
        """Checkpoint sweep includes selected steps and size override."""
        base_args.checkpoint_dir = "/path/to/checkpoints/run"
        base_args.checkpoint_path = None
        base_args.steps = "50000,100000"
        base_args.model = BaselineModelName.GALILEO
        base_args.size = "large"
        base_args.model_name = "official_large"

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command = commands[0]
        assert "CHECKPOINT_STEPS=50000,100000" in command
        assert "official_large" in command
        assert "--model.size=large" in command

    def test_checkpoint_sweep_with_task_names(
        self, base_args: argparse.Namespace
    ) -> None:
        """Checkpoint sweep applies task include filters."""
        base_args.checkpoint_dir = "/path/to/checkpoints/run"
        base_args.checkpoint_path = None
        base_args.task_names = "m_eurosat,m_bigearthnet"

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        command = commands[0]
        assert "checkpoint_sweep_evals.py" in command
        assert (
            "--trainer.callbacks.downstream_evaluator.tasks_to_run="
            '\'["m_eurosat", "m_bigearthnet"]\''
        ) in command

    def test_label_fraction_overrides(self, base_args: argparse.Namespace) -> None:
        """Label fraction emits one simple per-task low-label override."""
        base_args.defaults_only = True
        base_args.label_fraction = 0.1

        commands: list[str] = build_commands(base_args, [])

        assert len(commands) == 1
        assert "_label0.1x" in commands[0]
        assert "label_fraction=0.1" in commands[0]
        assert "partition=" not in commands[0]


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
                        "--trainer.callbacks.downstream_evaluator.tasks_to_run=[m_eurosat]",
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
                "--trainer.callbacks.downstream_evaluator.tasks_to_run=[m_eurosat]"
                in command
            )
