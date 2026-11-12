"""The text-pretraining experiment scripts: v1.3 as published, plus only the text head."""

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
# The published v1.3: its HF model config, and its run's training config.
PUBLISHED = ROOT / "tests/fixtures/release_training_configs/v1_3_published.json"


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


def test_v1_3_model_is_the_published_model(
    v1_3: ModuleType, common: CommonComponents
) -> None:
    """The port builds exactly the published v1.3 model config."""
    published = json.loads(PUBLISHED.read_text())["model"]
    assert v1_3.build_model_config(common).as_config_dict() == published


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


# Published-run fields that no longer exist, at values equal to what the current code does.
PUBLISHED_FIELDS_NOW_BUILT_IN: dict[tuple[str, ...], object] = {
    # Main keeps only the shared draw: one pixel per token cell for all its tokens.
    ("train_module", "pixel_target_draw"): "shared",
}
# Current fields at their off/default values that the published config predates.
CURRENT_FIELDS_AT_DEFAULT: dict[tuple[str, ...], object] = {
    ("train_module", "subpatch_offset_loss_weight"): 0.0,
    ("trainer", "callbacks", "checkpointer", "max_checkpoints"): None,
    # LossConfig.build pops "type", and the run's config was saved after building; its
    # v1.2 base sets this same loss.
    ("train_module", "loss_config", "loss_config", "type"): (
        "modality_patch_discrimination_masked_negatives_vec"
    ),
}


def _drop(config: dict, path: tuple[str, ...]) -> object:
    for key in path[:-1]:
        config = config[key]
    return config.pop(path[-1], None)


def test_text_arm_trains_like_the_published_run(
    text: ModuleType, common: CommonComponents
) -> None:
    """Data, sampler, optimization, duration and seed match the published v1.3 run."""
    from olmoearth_pretrain.internal.experiment import build_config

    built = build_config(
        common,
        text.build_model_config,
        text.build_dataset_config,
        text.build_dataloader_config,
        text.build_trainer_config,
        text.build_train_module_config,
        [],
        text.build_visualize_config,
    ).as_config_dict()
    release = json.loads(PUBLISHED.read_text())

    for path, value in PUBLISHED_FIELDS_NOW_BUILT_IN.items():
        assert _drop(release, path) == value, path
    # The run's per-pixel Perceiver latents at a random stride under a 512 budget are
    # now the dataloader's location-query budget.
    queries = release["location_queries"]
    assert queries["pixel_latents"] and queries["random_latent_stride"]
    assert built["data_loader"].pop("max_location_queries") == queries["max_latents"]
    for path, value in CURRENT_FIELDS_AT_DEFAULT.items():
        assert _drop(built, path) == value, path
    assert built["dataset"].pop("caption_dir") == text.CAPTION_DIR
    assert built["dataset"]["training_modalities"].pop() == Modality.CAPTION_CELL.name
    built["data_loader"].pop("work_dir")
    trainer = built["trainer"]
    for key in ("save_folder", "work_dir"):
        trainer.pop(key)
    trainer["checkpointer"].pop("work_dir")
    assert trainer["callbacks"].pop("wandb")["project"] == "2026_11_09_text_pretrain"
    assert trainer["callbacks"].pop("downstream_evaluator")["enabled"] is False
    # The olmo-core release config spelled out the BeakerCallback's default.
    assert release["trainer"]["callbacks"]["beaker"].pop("enabled") is True

    for section in ("dataset", "data_loader", "train_module", "trainer", "init_seed"):
        assert built[section] == release[section], section


@pytest.fixture(scope="module")
def text_mix() -> ModuleType:
    """The captions + labels script."""
    return _load("text_mix")


def test_mix_arm_adds_labels_and_open_set_windows(
    text: ModuleType, text_mix: ModuleType, common: CommonComponents
) -> None:
    """text.py plus the label term, the open-set windows, and v1.3's step count."""
    head = text_mix.build_model_config(common).as_config_dict()["text_head_config"]
    assert head.pop("label_vocab_dir") == text_mix.LABEL_VOCAB_DIR
    text_head = text.build_model_config(common).as_config_dict()["text_head_config"]
    assert head == text_head
    model = text_mix.build_model_config(common).as_config_dict()
    model.pop("text_head_config")
    base = text.build_model_config(common).as_config_dict()
    base.pop("text_head_config")
    assert model == base

    pretrain, open_set = text_mix.build_dataset_config(common).dataset_configs
    text_data = text.build_dataset_config(common)
    modalities = [*text_data.training_modalities, Modality.TEXT_LABEL.name]
    assert pretrain.training_modalities == modalities
    assert pretrain.label_vocab_dir == text_mix.LABEL_VOCAB_DIR
    assert pretrain.caption_dir == text_data.caption_dir
    assert pretrain.h5py_dir == text_data.h5py_dir
    assert open_set.training_modalities == modalities
    assert open_set.caption_dir is None
    assert open_set.h5py_dir.endswith("/103112")

    # 300 epochs of the pretraining set alone, as the release run trained.
    assert text_mix.MAX_STEPS == 300 * (1_138_828 // 512)
    trainer = text_mix.build_trainer_config(common).as_config_dict()
    assert trainer.pop("max_duration") == {
        "value": text_mix.MAX_STEPS,
        "unit": "steps",
        "_CLASS_": "olmo_core.train.common.Duration",
    }
    base_trainer = text.build_trainer_config(common).as_config_dict()
    base_trainer.pop("max_duration")
    assert trainer == base_trainer
    for builder in ("build_train_module_config", "build_dataloader_config"):
        assert (
            getattr(text_mix, builder)(common).as_config_dict()
            == getattr(text, builder)(common).as_config_dict()
        )


@pytest.mark.parametrize(
    ("frozen_name", "joint_name"),
    [("text_frozen", "text"), ("text_mix_frozen", "text_mix")],
)
def test_frozen_arm_differs_from_its_joint_arm_only_in_freezing(
    frozen_name: str, joint_name: str, common: CommonComponents
) -> None:
    """Same model, data, masking and batch; frozen at the published weights, short schedule."""
    frozen, joint = _load(frozen_name), _load(joint_name)
    weights = frozen.freeze.__globals__["PUBLISHED_WEIGHTS"]
    model = frozen.build_model_config(common).as_config_dict()
    assert model.pop("frozen_backbone_weights") == weights
    base_model = joint.build_model_config(common).as_config_dict()
    base_model.pop("frozen_backbone_weights", None)
    assert model == base_model

    for builder in ("build_dataset_config", "build_dataloader_config"):
        assert (
            getattr(frozen, builder)(common).as_config_dict()
            == getattr(joint, builder)(common).as_config_dict()
        )

    module = frozen.build_train_module_config(common).as_config_dict()
    base_module = joint.build_train_module_config(common).as_config_dict()
    steps = frozen.freeze.__globals__["STEPS"]
    assert module.pop("optim_config")["lr"] == frozen.freeze.__globals__["LR"]
    assert module.pop("scheduler")["warmup"] == frozen.freeze.__globals__["WARMUP"]
    base_module.pop("optim_config")
    base_module.pop("scheduler")
    assert module == base_module

    trainer = frozen.build_trainer_config(common).as_config_dict()
    assert trainer.pop("max_duration")["value"] == steps
    base_trainer = joint.build_trainer_config(common).as_config_dict()
    base_trainer.pop("max_duration")
    assert trainer == base_trainer


@pytest.mark.parametrize(
    ("finetune_name", "joint_name"),
    [("text_finetune", "text"), ("text_mix_finetune", "text_mix")],
)
def test_finetune_arm_differs_from_its_joint_arm_only_in_init_and_schedule(
    finetune_name: str, joint_name: str, common: CommonComponents
) -> None:
    """Same model, data, masking and batch; started from the published weights."""
    finetune, joint = _load(finetune_name), _load(joint_name)
    g = finetune.from_published.__globals__
    model = finetune.build_model_config(common).as_config_dict()
    assert model.pop("init_weights") == g["PUBLISHED_WEIGHTS"]
    base_model = joint.build_model_config(common).as_config_dict()
    base_model.pop("init_weights", None)
    assert model == base_model

    for builder in ("build_dataset_config", "build_dataloader_config"):
        assert (
            getattr(finetune, builder)(common).as_config_dict()
            == getattr(joint, builder)(common).as_config_dict()
        )

    module = finetune.build_train_module_config(common).as_config_dict()
    base_module = joint.build_train_module_config(common).as_config_dict()
    optim = module.pop("optim_config")
    assert optim["lr"] == g["BACKBONE_LR"]
    assert [(o["params"], o["opts"]) for o in optim["group_overrides"]] == [
        (["text_head.*"], {"lr": g["HEAD_LR"]})
    ]
    assert module.pop("scheduler")["warmup"] == g["WARMUP"]
    base_module.pop("optim_config")
    base_module.pop("scheduler")
    assert module == base_module

    trainer = finetune.build_trainer_config(common).as_config_dict()
    assert trainer.pop("max_duration")["value"] == g["STEPS"]
    base_trainer = joint.build_trainer_config(common).as_config_dict()
    base_trainer.pop("max_duration")
    assert trainer == base_trainer
