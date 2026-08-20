"""Tests for zero_bands, the atmospheric-band control's mechanism.

The point of the feature is that dropping a band from the model.yaml read leaves
it zero after normalization via the loader's band scatter, WITHOUT changing the
channel count the model tokenizes. These cover the config rewrite and its
guards; that the zeroing reaches the tensor is covered by the loader's own
band-scatter tests.
"""

import pytest

from olmoearth_pretrain.evals.datasets.rslearn_dataset import _apply_zero_bands


def _config(inputs: dict[str, list[str]]) -> dict:
    """A model.yaml skeleton carrying the given input -> bands mapping."""
    return {
        "data": {
            "init_args": {
                "inputs": {
                    name: {"bands": list(bands), "layers": [name]}
                    for name, bands in inputs.items()
                }
            }
        }
    }


def _bands(config: dict, name: str) -> list[str]:
    return config["data"]["init_args"]["inputs"][name]["bands"]


def test_drops_named_bands_and_keeps_order() -> None:
    """Named bands go; the survivors keep their declared order."""
    config = _config(
        {
            "sentinel2_l2a": ["B02", "B03", "B04", "B01", "B09"],
            "landsat": ["B8", "B1", "B2", "B9"],
        }
    )
    out = _apply_zero_bands(
        config, {"sentinel2_l2a": ["B01", "B09"], "landsat": ["B1", "B9"]}
    )
    assert _bands(out, "sentinel2_l2a") == ["B02", "B03", "B04"]
    assert _bands(out, "landsat") == ["B8", "B2"]


def test_does_not_mutate_caller_config() -> None:
    """The parsed model.yaml is reused across splits, so it must survive intact."""
    config = _config({"sentinel2_l2a": ["B02", "B01"]})
    _apply_zero_bands(config, {"sentinel2_l2a": ["B01"]})
    assert _bands(config, "sentinel2_l2a") == ["B02", "B01"]


def test_untouched_inputs_pass_through() -> None:
    """An input absent from zero_bands keeps every band."""
    config = _config({"sentinel2_l2a": ["B02", "B01"], "sentinel1": ["vv", "vh"]})
    out = _apply_zero_bands(config, {"sentinel2_l2a": ["B01"]})
    assert _bands(out, "sentinel1") == ["vv", "vh"]


def test_unknown_input_raises() -> None:
    """A typo must fail loudly: silently scoring an unzeroed run as zeroed is worse."""
    config = _config({"sentinel2_l2a": ["B02", "B01"]})
    with pytest.raises(ValueError, match="does not declare"):
        _apply_zero_bands(config, {"sentinel2": ["B01"]})


def test_unknown_band_raises() -> None:
    """Naming a band the input does not declare is a typo, not a no-op."""
    config = _config({"sentinel2_l2a": ["B02", "B01"]})
    with pytest.raises(ValueError, match="B10"):
        _apply_zero_bands(config, {"sentinel2_l2a": ["B10"]})


def test_blanking_every_band_raises() -> None:
    """Zero bands left means an empty read, not an all-zero modality."""
    config = _config({"sentinel2_l2a": ["B02", "B01"]})
    with pytest.raises(ValueError, match="every band"):
        _apply_zero_bands(config, {"sentinel2_l2a": ["B02", "B01"]})


def test_registered_tasks_zero_the_atmospheric_pairs() -> None:
    """The registered arm blanks S2 B01/B09 and their Landsat analogues."""
    from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS

    task = EMBEDDING_EVAL_TASKS[
        "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_nob0109_knn"
    ]
    assert task.zero_bands == {
        "sentinel2_l2a": ["B01", "B09"],
        "landsat": ["B1", "B9"],
    }
    # The unzeroed baseline it is compared against must be the same task
    # otherwise, so the delta isolates the bands.
    baseline = EMBEDDING_EVAL_TASKS[
        "ethiopia_crops_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn"
    ]
    assert task.input_modalities == baseline.input_modalities
    assert task.window_size == baseline.window_size
    assert task.embedding_batch_size == baseline.embedding_batch_size
    assert baseline.zero_bands is None
