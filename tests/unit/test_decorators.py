"""Tests for decorators module."""

import warnings

from olmoearth_pretrain.decorators import experimental


class TestExperimentalDecorator:
    """Tests for @experimental decorator."""

    def test_experimental_function(self) -> None:
        """Test that experimental decorator works on functions."""

        @experimental()
        def test_func() -> int:
            """Test function."""
            return 42

        # Check marker attribute
        assert hasattr(test_func, "__experimental__")
        assert test_func.__experimental__ is True

        # Check docstring updated
        assert test_func.__doc__ is not None
        assert "EXPERIMENTAL" in test_func.__doc__
        assert "Test function" in test_func.__doc__

        # Check warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "experimental" in str(w[0].message).lower()

    def test_experimental_function_with_reason(self) -> None:
        """Test experimental decorator with reason."""

        @experimental("Still testing performance")
        def test_func() -> int:
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()
            assert len(w) == 1
            assert "Still testing performance" in str(w[0].message)

    def test_experimental_class(self) -> None:
        """Test that experimental decorator works on classes."""

        @experimental()
        class TestClass:
            """Test class."""

            def __init__(self, value: int) -> None:
                self.value = value

        # Check marker attribute
        assert hasattr(TestClass, "__experimental__")
        assert TestClass.__experimental__ is True

        # Check docstring updated
        assert TestClass.__doc__ is not None
        assert "EXPERIMENTAL" in TestClass.__doc__
        assert "Test class" in TestClass.__doc__

        # Check warning is raised on instantiation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = TestClass(42)
            assert obj.value == 42
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)

    def test_experimental_class_with_reason(self) -> None:
        """Test experimental decorator on class with reason."""

        @experimental("API may change")
        class TestClass:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TestClass()
            assert len(w) == 1
            assert "API may change" in str(w[0].message)

    def test_experimental_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function metadata."""

        @experimental()
        def my_function(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        assert my_function.__name__ == "my_function"
        assert "my_function" in my_function.__qualname__

        # Function should still work
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert my_function(1, 2) == 3
