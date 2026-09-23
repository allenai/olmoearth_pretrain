"""Unit tests for scripts/tools/register_embedding_products.py."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.embedding_materializer.materialize import PRODUCTS
from olmoearth_pretrain.evals.studio_ingest.schema import (
    EmbeddingProductRecord,
    EvalDatasetEntry,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tools" / "register_embedding_products.py"
PASTIS_CONFIG = (
    REPO_ROOT / "data" / "rslearn_dataset_configs" / "config_pastis_rslearn.json"
)

# A model.yaml as the old wiring script left it: a gse input between S2 and
# the targets.
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
      gse:
        data_type: raster
        dtype: FLOAT32
        layers:
          - gse
        required: true
        use_all_bands_in_order_of_band_set_idx: 0
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
        "register_embedding_products", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reg() -> types.ModuleType:
    """The loaded registration script module."""
    return _load_script()


def _inputs(text: str) -> dict:
    """Return the parsed data.init_args.inputs mapping of a model.yaml."""
    return yaml.safe_load(text)["data"]["init_args"]["inputs"]


def _manifest(**fields: object) -> dict:
    """A bake manifest with zeroed counts, overridden by fields."""
    base: dict = {
        "num_windows_written": 0,
        "num_windows_skipped_existing": 0,
        "num_coverage_gaps": 0,
        "num_windows_failed": 0,
        "num_windows_without_year": 0,
    }
    return {**base, **fields}


def test_product_map_covers_the_materializer_products(reg: types.ModuleType) -> None:
    """Every product the materializer can bake can be registered, plus tessera_v2."""
    assert set(reg.PRODUCT_TO_MODALITY) == {"tessera_v2", *PRODUCTS}


def test_default_products_are_all_known(reg: types.ModuleType) -> None:
    """The --products default parses to known products only."""
    args = reg.parse_args([])
    assert set(args.products.split(",")) == set(reg.PRODUCT_TO_MODALITY)


def test_bake_is_complete_gates_on_the_manifest(reg: types.ModuleType) -> None:
    """Only a finished, failure-free, well-covered bake is registered."""
    assert reg.bake_is_complete(None, 0.99)[0] is False
    manifest = _manifest(num_windows_written=1000, num_coverage_gaps=2)
    assert reg.bake_is_complete(manifest, 0.99)[0] is True
    # Failures are recoverable by re-running; do not register a partial result.
    manifest = _manifest(num_windows_written=1000, num_windows_failed=3)
    assert reg.bake_is_complete(manifest, 0.99)[0] is False
    assert reg.bake_is_complete(_manifest(), 0.99)[0] is False


def test_bake_is_complete_rejects_partial_coverage(reg: types.ModuleType) -> None:
    """Tessera covers 8% of ethiopia_crops: too little to share a window set."""
    manifest = _manifest(num_windows_written=206, num_coverage_gaps=2324)
    ready, reason = reg.bake_is_complete(manifest, 0.99)
    assert ready is False
    assert "coverage=8.1%" in reason
    assert reg.bake_is_complete(manifest, 0.05)[0] is True


@pytest.mark.parametrize("modality_name", ["gse", "tessera_v2"])
def test_generated_layer_matches_the_committed_pastis_block(
    reg: types.ModuleType, modality_name: str
) -> None:
    """A directly-written layer is declared the way PASTIS already reads it."""
    with PASTIS_CONFIG.open() as f:
        expected = json.load(f)["layers"][modality_name]
    config: dict = {"layers": {}}
    assert reg.add_config_layer(config, Modality.get(modality_name)) is True
    assert config["layers"][modality_name] == expected
    assert reg.add_config_layer(config, Modality.get(modality_name)) is False


def test_remove_input_leaves_siblings_and_comments(reg: types.ModuleType) -> None:
    """Deleting gse keeps the other inputs, later keys, and the header comment."""
    result = reg.remove_model_yaml_input(MODEL_YAML, Modality.GSE)
    assert result is not None
    assert result.startswith("# A supplemental eval dataset config.")
    assert set(_inputs(result)) == {"sentinel2_l2a", "targets"}
    assert _inputs(result)["targets"] == _inputs(MODEL_YAML)["targets"]
    assert yaml.safe_load(result)["data"]["init_args"]["batch_size"] == 8


def test_remove_input_is_a_noop_without_the_input(reg: types.ModuleType) -> None:
    """A model.yaml that never declared the product is left alone."""
    assert reg.remove_model_yaml_input(MODEL_YAML, Modality.TESSERA_V2) is None


def _entry(tmp_path: Path, modalities: list[str]) -> EvalDatasetEntry:
    """A registry entry whose dataset folder and model.yaml live in tmp_path."""
    (tmp_path / "config.json").write_text(json.dumps({"layers": {}}))
    (tmp_path / "model.yaml").write_text(MODEL_YAML)
    return EvalDatasetEntry(
        name="lcmap_lu",
        source_path=str(tmp_path),
        weka_path=str(tmp_path),
        task_type="classification",
        num_classes=2,
        modalities=modalities,
    )


def test_register_migrates_a_legacy_entry(
    reg: types.ModuleType, tmp_path: Path
) -> None:
    """A wired-the-old-way dataset ends up with a record and no model.yaml input."""
    entry = _entry(tmp_path, ["gse", "sentinel2_l2a"])
    reg.register_product(entry, "aef", dry_run=False)

    assert entry.modalities == ["sentinel2_l2a"]
    assert entry.embedding_products == {"gse": EmbeddingProductRecord(product="aef")}
    assert entry.supported_modalities == ["sentinel2_l2a", "gse"]
    assert "gse" not in _inputs((tmp_path / "model.yaml").read_text())
    # The layer is declared if the bake did not already declare it.
    assert "gse" in json.loads((tmp_path / "config.json").read_text())["layers"]

    reg.restamp_config_json(entry, dry_run=False)
    assert entry.config_json_sha256 is not None


def test_register_dry_run_changes_nothing(
    reg: types.ModuleType, tmp_path: Path
) -> None:
    """--dry_run reports but leaves the entry and the files untouched."""
    entry = _entry(tmp_path, ["gse", "sentinel2_l2a"])
    reg.register_product(entry, "aef", dry_run=True)
    assert entry.modalities == ["gse", "sentinel2_l2a"]
    assert entry.embedding_products == {}
    assert (tmp_path / "model.yaml").read_text() == MODEL_YAML
    assert json.loads((tmp_path / "config.json").read_text()) == {"layers": {}}


def test_registered_products_are_supported_modalities() -> None:
    """Registered products gate tasks on alongside the imagery modalities."""
    entry = EvalDatasetEntry(
        name="d",
        source_path="/x",
        weka_path="/x",
        task_type="classification",
        num_classes=2,
        modalities=["sentinel2_l2a"],
        embedding_products={
            "gse": EmbeddingProductRecord(product="aef"),
            "tessera_v2": EmbeddingProductRecord(product="tessera_v2"),
        },
    )
    assert entry.to_eval_config().supported_modalities == [
        "sentinel2_l2a",
        "gse",
        "tessera_v2",
    ]
