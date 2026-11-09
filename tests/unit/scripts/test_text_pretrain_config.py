"""The text-pretraining experiment scripts: v1.3 as released, plus only the text head."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "scripts" / "vnext" / "text_pretrain"
RELEASE = (
    ROOT
    / "tests/fixtures/checkpoint_configs"
    / "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1.json"
)


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"text_pretrain_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v1_3() -> ModuleType:
    """The v1.3 baseline script."""
    return _load("v1_3")


@pytest.fixture(scope="module")
def text() -> ModuleType:
    """The text-head script."""
    return _load("text")


@pytest.fixture(scope="module")
def common(v1_3: ModuleType) -> CommonComponents:
    """Common components as a dry run builds them."""
    return v1_3.build_common_components(
        str(SCRIPTS / "v1_3.py"), SubCmd.dry_run, "test", "ai2/jupiter", []
    )


def test_v1_3_model_is_the_release_model(
    v1_3: ModuleType, common: CommonComponents
) -> None:
    """The port builds exactly the released v1.3 model config."""
    release = json.loads(RELEASE.read_text())["model"]
    assert v1_3.build_model_config(common).as_config_dict() == release


def test_text_arm_adds_only_the_text_head_and_captions(
    v1_3: ModuleType, text: ModuleType, common: CommonComponents
) -> None:
    """Same model, data and training as v1.3 apart from the head and caption_cell."""
    base_model = v1_3.build_model_config(common).as_config_dict()
    text_model = text.build_model_config(common).as_config_dict()
    head = text_model.pop("text_head_config")
    assert head["embeddings_dir"] == text.CAPTION_EMBEDDINGS_DIR
    assert text_model == base_model
    assert Modality.CAPTION_CELL.name not in common.training_modalities

    base_data = v1_3.build_dataset_config(common).as_config_dict()
    text_data = text.build_dataset_config(common).as_config_dict()
    assert text_data.pop("caption_dir") == text.CAPTION_DIR
    assert text_data.pop("training_modalities") == [
        *base_data.pop("training_modalities"),
        Modality.CAPTION_CELL.name,
    ]
    assert "caption_dir" not in base_data
    assert text_data == base_data

    for builder in ("build_train_module_config", "build_dataloader_config"):
        assert (
            getattr(text, builder)(common).as_config_dict()
            == getattr(v1_3, builder)(common).as_config_dict()
        )


def test_text_model_builds_with_its_head(
    text: ModuleType, common: CommonComponents, tmp_path: Path
) -> None:
    """The full model builds, the head reading the 768-d teacher location tokens."""
    np.save(tmp_path / "embeddings.npy", np.zeros((1, 17, 768), dtype=np.float16))
    (tmp_path / "meta.json").write_text(json.dumps({"dim": 768}))
    config = text.build_model_config(common)
    config.text_head_config.embeddings_dir = str(tmp_path)
    with torch.device("meta"):
        model = config.build()
    assert model.text_head is not None
    assert model.text_head.proj.in_features == 768
    assert model.text_head.proj.out_features == 768
