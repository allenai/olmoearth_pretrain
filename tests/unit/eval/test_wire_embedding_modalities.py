"""Unit tests for scripts/tools/wire_embedding_modalities.py.

The model.yaml edits are textual (to preserve comments and layout that a
parse/re-dump round trip would destroy), so they are worth pinning down: a
bad insert produces a file that still parses but wires the wrong thing.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from olmoearth_pretrain.data.constants import Modality

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tools" / "wire_embedding_modalities.py"
PASTIS_CONFIG = (
    REPO_ROOT / "data" / "rslearn_dataset_configs" / "config_pastis_rslearn.json"
)

MODEL_YAML = """# A supplemental eval dataset config.
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: /weka/dfive-default/olmoearth/eval_datasets/lcmap_lu
    inputs:
      sentinel2_l2a:
        data_type: raster
        dtype: FLOAT32
        layers:
          - sentinel2_l2a_mo01
        load_all_layers: true
        bands:
          - B02
        passthrough: true
      targets:
        data_type: raster
        dtype: INT32
        layers:
          - label
        bands:
          - label
        is_target: true
    batch_size: 8
"""


def _load_script() -> types.ModuleType:
    """Import the script by path — scripts/ is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "wire_embedding_modalities", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wire() -> types.ModuleType:
    """The loaded wiring script module."""
    return _load_script()


def _inputs(text: str) -> dict:
    """Return the parsed data.init_args.inputs mapping of a model.yaml."""
    return yaml.safe_load(text)["data"]["init_args"]["inputs"]


def test_insert_adds_input_without_disturbing_siblings(wire: types.ModuleType) -> None:
    """A new input lands under inputs/ and nothing else moves."""
    result = wire.insert_model_yaml_input(
        MODEL_YAML, Modality.TESSERA_V2, required=False
    )
    assert result is not None
    assert result.startswith("# A supplemental eval dataset config.")
    assert _inputs(result)["tessera_v2"] == {
        "data_type": "raster",
        "dtype": "FLOAT32",
        "layers": ["tessera_v2"],
        "required": False,
        "use_all_bands_in_order_of_band_set_idx": 0,
        "passthrough": True,
    }
    # Siblings and later keys at other nesting levels survive the insert.
    assert set(_inputs(result)) == {"sentinel2_l2a", "targets", "tessera_v2"}
    assert yaml.safe_load(result)["data"]["init_args"]["batch_size"] == 8


def test_insert_is_idempotent(wire: types.ModuleType) -> None:
    """Re-inserting an existing input reports "already there" instead of duplicating."""
    once = wire.insert_model_yaml_input(MODEL_YAML, Modality.GSE, required=True)
    assert once is not None
    assert wire.insert_model_yaml_input(once, Modality.GSE, required=True) is None


def test_insert_rejects_a_file_with_no_inputs_mapping(wire: types.ModuleType) -> None:
    """A model.yaml we do not understand is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="no data.init_args.inputs"):
        wire.insert_model_yaml_input(
            "data:\n  init_args: {}\n", Modality.TESSERA_V2, False
        )


def test_set_required_flips_only_the_named_input(wire: types.ModuleType) -> None:
    """Flipping one input's required field leaves the others alone."""
    text = wire.insert_model_yaml_input(MODEL_YAML, Modality.TESSERA_V2, required=False)
    text = wire.insert_model_yaml_input(text, Modality.GSE, required=False)
    flipped = wire.set_model_yaml_required(text, Modality.TESSERA_V2, required=True)
    assert _inputs(flipped)["tessera_v2"]["required"] is True
    assert _inputs(flipped)["gse"]["required"] is False
    assert flipped.count("\n") == text.count("\n")


def test_set_required_needs_an_existing_required_line(wire: types.ModuleType) -> None:
    """sentinel2_l2a has no required: field, so there is nothing to flip."""
    with pytest.raises(ValueError, match="no 'required:' line"):
        wire.set_model_yaml_required(MODEL_YAML, Modality.SENTINEL2_L2A, required=True)


@pytest.mark.parametrize("modality_name", ["gse", "tessera_v2"])
def test_generated_layer_matches_the_committed_pastis_block(
    wire: types.ModuleType, modality_name: str
) -> None:
    """The layer we synthesize equals the one PASTIS already evaluates against.

    PASTIS is the only dataset with these layers committed, so it is the
    reference for what rslearn expects to read.
    """
    with PASTIS_CONFIG.open() as f:
        expected = json.load(f)["layers"][modality_name]
    modality = next(
        m for m in wire.PRODUCT_TO_MODALITY.values() if m.name == modality_name
    )
    config: dict = {"layers": {}}
    assert wire.add_config_layer(config, modality) is True
    assert config["layers"][modality_name] == expected
    assert wire.add_config_layer(config, modality) is False


def test_product_map_covers_the_materializer_products(wire: types.ModuleType) -> None:
    """Every product the materializer can bake can also be wired.

    Plus tessera_v2, which our own inference bakes (tessera_v2_export.py) in
    the materializer's manifest shape rather than via a published product.
    """
    from olmoearth_pretrain.evals.embedding_materializer.__main__ import PRODUCT_NAMES

    assert set(wire.PRODUCT_TO_MODALITY) == {"tessera_v2", *PRODUCT_NAMES}


def _write_manifest(wire: types.ModuleType, path: Path, **fields: int) -> None:
    """Write a materializer manifest for the tessera product under *path*."""
    with wire.manifest_path(str(path), "tessera_v2").open("w") as f:
        json.dump(fields, f)


def test_bake_is_complete_gates_on_the_manifest(
    wire: types.ModuleType, tmp_path: Path
) -> None:
    """Only a finished, failure-free, well-covered bake is allowed to go live."""
    ready = lambda: wire.bake_is_complete(str(tmp_path), "tessera_v2", 0.99)[0]  # noqa: E731

    # No manifest at all: the materializer has not finished.
    assert ready() is False

    _write_manifest(
        wire,
        tmp_path,
        num_windows_written=1000,
        num_windows_skipped_existing=0,
        num_coverage_gaps=2,
        num_windows_failed=0,
    )
    assert ready() is True

    # Failures mean some windows could have been baked but were not; they are
    # recoverable by re-running, so do not go live on a partial result.
    _write_manifest(
        wire,
        tmp_path,
        num_windows_written=1000,
        num_windows_skipped_existing=0,
        num_coverage_gaps=0,
        num_windows_failed=3,
    )
    assert ready() is False

    _write_manifest(
        wire,
        tmp_path,
        num_windows_written=0,
        num_windows_skipped_existing=0,
        num_coverage_gaps=0,
        num_windows_failed=0,
    )
    assert ready() is False


def test_bake_is_complete_rejects_partial_coverage(
    wire: types.ModuleType, tmp_path: Path
) -> None:
    """A finished bake covering only part of the dataset must not go live.

    Real case: Tessera covers 8% of ethiopia_crops. Requiring that layer would
    drop the other 92% of windows from every eval on the dataset, OlmoEarth
    baselines included. See docs/PrecomputedEmbeddingCoverage.md.
    """
    _write_manifest(
        wire,
        tmp_path,
        num_windows_written=206,
        num_windows_skipped_existing=0,
        num_coverage_gaps=2324,
        num_windows_failed=0,
    )
    ready, reason = wire.bake_is_complete(str(tmp_path), "tessera_v2", 0.99)
    assert ready is False
    assert "coverage=8.1%" in reason
    assert "below --min_coverage" in reason

    # Explicitly lowering the bar lets it through, for a deliberate partial run.
    assert wire.bake_is_complete(str(tmp_path), "tessera_v2", 0.05)[0] is True
