"""Test AnySat."""

from olmoearth_pretrain.evals.models import AnySat


class TestAnySat:
    """Test AnySat."""

    def test_ps_correctly_computed(self) -> None:
        """test_ps_correctly_computed."""
        height = 1
        ps = AnySat._calculate_patch_size(height)
        assert ps == 1
