"""Tests for decorators module."""

import warnings

from olmoearth_pretrain.decorators import deprecated, experimental, internal


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


class TestDeprecatedDecorator:
    """Tests for @deprecated decorator."""

    def test_deprecated_function(self) -> None:
        """Test that deprecated decorator works on functions."""

        @deprecated()
        def test_func() -> int:
            """Test function."""
            return 42

        assert hasattr(test_func, "__deprecated__")
        assert test_func.__deprecated__ is True

        assert test_func.__doc__ is not None
        assert "DEPRECATED" in test_func.__doc__

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_deprecated_with_reason(self) -> None:
        """Test deprecated decorator with reason."""

        @deprecated(reason="No longer needed")
        def test_func() -> int:
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()
            assert len(w) == 1
            assert "No longer needed" in str(w[0].message)

    def test_deprecated_with_alternative(self) -> None:
        """Test deprecated decorator with alternative."""

        @deprecated(alternative="new_func")
        def test_func() -> int:
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()
            assert len(w) == 1
            assert "new_func" in str(w[0].message)
            assert "Use" in str(w[0].message)

    def test_deprecated_with_reason_and_alternative(self) -> None:
        """Test deprecated decorator with both reason and alternative."""

        @deprecated(reason="Old API", alternative="new_func")
        def test_func() -> int:
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()
            assert len(w) == 1
            msg = str(w[0].message)
            assert "Old API" in msg
            assert "new_func" in msg

    def test_deprecated_class(self) -> None:
        """Test that deprecated decorator works on classes."""

        @deprecated()
        class TestClass:
            """Test class."""

            def __init__(self, value: int) -> None:
                self.value = value

        assert hasattr(TestClass, "__deprecated__")
        assert TestClass.__deprecated__ is True

        assert TestClass.__doc__ is not None
        assert "DEPRECATED" in TestClass.__doc__

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = TestClass(42)
            assert obj.value == 42
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


class TestInternalDecorator:
    """Tests for @internal decorator."""

    def test_internal_function(self) -> None:
        """Test that internal decorator works on functions."""

        @internal
        def test_func() -> int:
            """Test function."""
            return 42

        assert hasattr(test_func, "__internal__")
        assert test_func.__internal__ is True

        assert test_func.__doc__ is not None
        assert "INTERNAL API" in test_func.__doc__
        assert "Test function" in test_func.__doc__

        # Internal decorator doesn't emit warnings, just updates docs
        result = test_func()
        assert result == 42

    def test_internal_class(self) -> None:
        """Test that internal decorator works on classes."""

        @internal
        class TestClass:
            """Test class."""

            def __init__(self, value: int) -> None:
                self.value = value

        assert hasattr(TestClass, "__internal__")
        assert TestClass.__internal__ is True

        assert TestClass.__doc__ is not None
        assert "INTERNAL API" in TestClass.__doc__
        assert "Test class" in TestClass.__doc__

        obj = TestClass(42)
        assert obj.value == 42

    def test_internal_no_docstring(self) -> None:
        """Test internal decorator on function without docstring."""

        @internal
        def test_func() -> int:
            return 42

        assert test_func.__doc__ is not None
        assert "INTERNAL API" in test_func.__doc__


class TestDecoratorCombinations:
    """Test combining multiple decorators."""

    def test_experimental_and_internal(self) -> None:
        """Test combining experimental and internal decorators."""

        @experimental()
        @internal
        def test_func() -> int:
            return 42

        assert hasattr(test_func, "__experimental__")
        assert hasattr(test_func, "__internal__")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_func()
            assert result == 42
            # Should get warning from experimental
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
