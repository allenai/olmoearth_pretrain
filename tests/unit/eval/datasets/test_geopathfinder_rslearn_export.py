"""Tests for the GeoPathfinder rslearn export's dot labels."""

import numpy as np
import pytest

from olmoearth_pretrain.evals.datasets.geopathfinder_rslearn_export import (
    IGNORE_LABEL,
    dot_label,
)


@pytest.mark.parametrize("cls", [0, 1])
def test_dot_label_marks_only_the_dots(cls: int) -> None:
    """The two dot pixels carry the class; everything else is ignored."""
    label = dot_label(((3, 200), (255, 0)), cls)
    assert label.shape == (1, 256, 256)
    assert label.dtype == np.uint8
    assert label[0, 3, 200] == cls and label[0, 255, 0] == cls
    assert (label != IGNORE_LABEL).sum() == 2


@pytest.mark.parametrize(
    "dots", [((5, 5), (5, 5)), ((256, 0), (1, 1)), ((-1, 0), (1, 1))]
)
def test_dot_label_rejects_bad_dots(
    dots: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    """Coincident or out-of-patch dots cannot make a valid sample."""
    with pytest.raises(ValueError):
        dot_label(dots, 1)
