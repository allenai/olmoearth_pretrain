"""Tests for promoting an all-optional input stack to required.

Landsat is optional in every year-aligned model.yaml so that windows the
Landsat materialize has not finished still run. rslearn drops an optional
``load_all_layers`` input entirely when *any* one of its twelve monthly layers
is unmaterialized, and the loader then represents the absent modality as
all-MISSING. That is sound only while a required input (the S2/S1 monthlies) is
present to carry the sample.

A Landsat-only stack has no carrier, so a window with a partial Landsat year
reaches ``_transform_sample`` with nothing in it and the eval dies mid-run with
"No input modalities present in sample" -- which is how the Landsat-only KNN
tasks first failed, on the ~0.4% of AEF windows with an incomplete Landsat year.
``require_stack_inputs`` moves that decision to rslearn's window resolution,
which skips those windows up front.

The promotion has to stay surgical: it must not touch a stack that already has
a required input (that would silently change the window set of every existing
task).
"""

import copy
from typing import Any

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.rslearn_builder import require_stack_inputs


def _model_config() -> dict[str, Any]:
    """A year-aligned model.yaml shape: required S2/S1, optional Landsat."""
    return {
        "data": {
            "init_args": {
                "inputs": {
                    "sentinel2_l2a": {
                        "layers": ["sentinel2_l2a_mo01", "sentinel2_l2a_mo02"],
                        "load_all_layers": True,
                    },
                    "sentinel1": {
                        "layers": ["sentinel1_mo01"],
                        "load_all_layers": True,
                    },
                    "landsat": {
                        "layers": ["landsat_mo01", "landsat_mo02"],
                        "load_all_layers": True,
                        "required": False,
                    },
                    "targets": {"layers": ["label"], "is_target": True},
                }
            }
        }
    }


def _required(config: dict[str, Any], name: str) -> bool:
    return config["data"]["init_args"]["inputs"][name].get("required", True)


def test_landsat_only_stack_is_promoted_to_required() -> None:
    """The carrier-less stack gets its one input required."""
    config = _model_config()

    patched = require_stack_inputs(config, [Modality.LANDSAT.name])

    assert _required(patched, "landsat") is True


def test_stack_with_a_required_input_is_returned_untouched() -> None:
    """Mixed stacks keep their carrier, so their window set must not move."""
    config = _model_config()

    for stack in (
        [Modality.SENTINEL2_L2A.name],
        [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        [Modality.SENTINEL2_L2A.name, Modality.LANDSAT.name],
    ):
        assert require_stack_inputs(config, stack) is config


def test_source_config_is_not_mutated() -> None:
    """The caller's parsed model.yaml is shared; promotion works on a copy."""
    config = _model_config()
    before = copy.deepcopy(config)

    require_stack_inputs(config, [Modality.LANDSAT.name])

    assert config == before


def test_stack_matching_no_input_is_returned_untouched() -> None:
    """A stack naming nothing in the config has nothing to promote."""
    config = _model_config()

    assert require_stack_inputs(config, ["NOT_A_MODALITY"]) is config
