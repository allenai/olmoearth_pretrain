"""Unit tests for the centralized config module."""

import pytest

from olmoearth_pretrain.config import (
    OLMO_CORE_AVAILABLE,
    Config,
    _StandaloneConfig,
    require_olmo_core,
)


class TestOlmoCoreAvailability:
    """Tests for olmo-core availability detection."""

    def test_olmo_core_available_flag_is_bool(self) -> None:
        """Test that OLMO_CORE_AVAILABLE is a boolean."""
        assert isinstance(OLMO_CORE_AVAILABLE, bool)

    def test_config_is_type(self) -> None:
        """Test that Config is a type/class."""
        assert isinstance(Config, type)


class TestRequireOlmoCore:
    """Tests for the require_olmo_core guard function."""

    def test_require_olmo_core_does_not_raise_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that require_olmo_core doesn't raise when olmo-core is available."""
        # Ensure OLMO_CORE_AVAILABLE is True
        monkeypatch.setattr("olmoearth_pretrain.config.OLMO_CORE_AVAILABLE", True)
        # Should not raise
        require_olmo_core("Test operation")

    def test_require_olmo_core_raises_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that require_olmo_core raises ImportError when olmo-core unavailable."""
        monkeypatch.setattr("olmoearth_pretrain.config.OLMO_CORE_AVAILABLE", False)
        with pytest.raises(ImportError, match="requires olmo-core"):
            require_olmo_core("Training")

    def test_require_olmo_core_includes_operation_in_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the operation name is included in the error message."""
        monkeypatch.setattr("olmoearth_pretrain.config.OLMO_CORE_AVAILABLE", False)
        with pytest.raises(ImportError, match="My custom operation"):
            require_olmo_core("My custom operation")


class TestStandaloneConfig:
    """Tests for the standalone config implementation."""

    def test_standalone_config_from_dict_simple(self) -> None:
        """Test that _StandaloneConfig.from_dict works with simple data."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_StandaloneConfig):
            value: int
            name: str

        data = {"value": 42, "name": "test"}
        config = SimpleConfig.from_dict(data)

        assert config.value == 42
        assert config.name == "test"

    def test_standalone_config_as_dict(self) -> None:
        """Test that _StandaloneConfig.as_dict works correctly."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_StandaloneConfig):
            value: int
            name: str

        config = SimpleConfig(value=42, name="test")
        result = config.as_dict()

        assert result == {"value": 42, "name": "test"}

    def test_standalone_config_as_config_dict(self) -> None:
        """Test that as_config_dict includes class name."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_StandaloneConfig):
            value: int

        config = SimpleConfig(value=42)
        result = config.as_config_dict()

        assert "_CLASS_" in result
        assert "SimpleConfig" in result["_CLASS_"]
        assert result["value"] == 42

    def test_standalone_config_build_not_implemented(self) -> None:
        """Test that build() raises NotImplementedError by default."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig(_StandaloneConfig):
            value: int

        config = SimpleConfig(value=42)
        with pytest.raises(NotImplementedError, match="must implement build"):
            config.build()

    def test_standalone_config_resolve_class(self) -> None:
        """Test that _resolve_class can resolve fully-qualified class names."""
        # Test resolving a known class
        resolved = _StandaloneConfig._resolve_class(
            "olmoearth_pretrain.datatypes.MaskValue"
        )
        assert resolved is not None

        from olmoearth_pretrain.datatypes import MaskValue

        assert resolved is MaskValue

    def test_standalone_config_resolve_class_raises_for_invalid(self) -> None:
        """Test that _resolve_class raises for invalid class names."""
        # No dot in name
        with pytest.raises(ValueError, match="must be fully qualified"):
            _StandaloneConfig._resolve_class("InvalidName")

        # Non-existent module
        with pytest.raises(ModuleNotFoundError):
            _StandaloneConfig._resolve_class("nonexistent.module.Class")


class TestConfigCompatibility:
    """Tests for compatibility between olmo-core Config and standalone Config."""

    @pytest.mark.skipif(not OLMO_CORE_AVAILABLE, reason="olmo-core not installed")
    def test_config_is_olmo_core_when_available(self) -> None:
        """Test that Config is olmo-core's Config when available."""
        from olmo_core.config import Config as OlmoCoreConfig

        # At runtime (not TYPE_CHECKING), Config should be olmo-core's
        assert Config is OlmoCoreConfig

    def test_standalone_config_has_required_methods(self) -> None:
        """Test that _StandaloneConfig has all required methods."""
        assert hasattr(_StandaloneConfig, "from_dict")
        assert hasattr(_StandaloneConfig, "as_dict")
        assert hasattr(_StandaloneConfig, "as_config_dict")
        assert hasattr(_StandaloneConfig, "build")
        assert hasattr(_StandaloneConfig, "validate")
