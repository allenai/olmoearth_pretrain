"""Unit tests for the supervised open-set probe."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue, TokensAndMasks
from olmoearth_pretrain.train.open_set_probe import (
    OPEN_SET_NODATA,
    OpenSetProbe,
    OpenSetProbeConfig,
)

_CLASS_MAPPING_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "open_set_segmentation_data"
    / "class_mapping.json"
)


@pytest.fixture(scope="module")
def class_mapping() -> dict:
    """Load the frozen open-set class mapping."""
    with _CLASS_MAPPING_PATH.open() as f:
        return json.load(f)


@pytest.fixture()
def probe(class_mapping: dict) -> OpenSetProbe:
    """Build a deterministic probe for unit tests."""
    torch.manual_seed(0)
    return OpenSetProbe(embedding_size=8, class_mapping=class_mapping)


def _tiny_mapping(groups: list[dict], num_classes: int = 4) -> dict:
    return {
        "open_set": {
            "num_classes": num_classes,
            "training_datasets": groups,
        },
        "open_set_regression": {"datasets": []},
    }


def _make_latent(
    b: int, p: int, t: int, d: int, visible: torch.Tensor
) -> tuple[TokensAndMasks, torch.Tensor]:
    """Build a TokensAndMasks with a single sentinel2_l2a modality.

    Args:
        b: Batch size.
        p: Number of patches along each spatial dimension.
        t: Number of timesteps.
        d: Token embedding dimension.
        visible: (b, p, p, t, 1) bool mask of which tokens the online encoder saw.
    """
    tokens = torch.randn(b, p, p, t, 1, d, requires_grad=True)
    mask = torch.where(
        visible,
        torch.full_like(visible, MaskValue.ONLINE_ENCODER.value, dtype=torch.long),
        torch.full_like(visible, MaskValue.DECODER.value, dtype=torch.long),
    )
    return TokensAndMasks(sentinel2_l2a=tokens, sentinel2_l2a_mask=mask), tokens


def test_lookup_buffers_cover_all_classes(probe: OpenSetProbe) -> None:
    """Lookup buffers map every global class into a valid group position."""
    assert probe.group_of_global_id.shape == (probe.num_classes,)
    assert (probe.group_of_global_id >= 0).all()
    for global_id in [0, 5, 12, probe.num_classes - 1]:
        group = probe.group_of_global_id[global_id]
        local_idx = probe.local_index_of_global_id[global_id]
        assert probe.group_global_ids[group, local_idx] == global_id
        assert probe.target_allowed_positions[global_id, local_idx]


def test_config_rejects_changed_frozen_mapping(tmp_path: Path) -> None:
    """Training refuses a mapping whose bytes differ from the frozen fingerprint."""
    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(
        json.dumps(_tiny_mapping([{"name": "all", "global_ids": [0, 1, 2, 3]}]))
    )
    config = OpenSetProbeConfig(
        class_mapping_path=str(mapping_path),
        expected_class_mapping_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="class mapping hash mismatch"):
        config.build(embedding_size=1)


def test_pool_patches_averages_visible_tokens(probe: OpenSetProbe) -> None:
    """Patch pooling averages visible tokens and rejects invisible patches."""
    b, p, t, d = 2, 4, 3, 8
    visible = torch.ones(b, p, p, t, 1, dtype=torch.bool)
    # Make one patch fully invisible (all decoder) -> invalid.
    visible[0, 0, 0] = False
    latent, _ = _make_latent(b, p, t, d, visible)
    pooled, valid = probe.pool_patches(latent)
    assert pooled.shape == (b, p, p, d)
    assert valid.shape == (b, p, p)
    assert not valid[0, 0, 0]
    assert valid[1, 1, 1]


def test_classification_label_pooling_majority_and_nodata(
    probe: OpenSetProbe,
) -> None:
    """Classification pooling uses the majority label and ignores nodata."""
    b, p = 1, 2
    h = w = 4  # block size 2x2
    open_set = torch.full((b, h, w, 1, 1), float(OPEN_SET_NODATA))
    # Patch (0,0): mostly class 5, one nodata pixel -> majority 5.
    open_set[0, 0, 0, 0, 0] = 5
    open_set[0, 0, 1, 0, 0] = 5
    open_set[0, 1, 0, 0, 0] = 5
    # (0,1) pixel remains nodata within that block.
    # Patch (1,1): all nodata -> invalid.
    target, valid = probe.pool_classification_labels(open_set, p, p)
    assert target.shape == (b, p, p)
    assert valid[0, 0, 0]
    assert target[0, 0, 0] == 5
    assert not valid[0, 1, 1]


def test_classification_label_pooling_tie_uses_lowest_id(
    probe: OpenSetProbe,
) -> None:
    """Sparse majority pooling resolves equal counts to the lowest global id."""
    open_set = torch.full((1, 2, 4, 1, 1), float(OPEN_SET_NODATA))
    # The first patch ties classes 7 and 3 at two pixels each. The second is nodata.
    open_set[0, :, :2, 0, 0] = torch.tensor([[7, 3], [3, 7]])

    target, valid = probe.pool_classification_labels(open_set, 1, 2)

    assert valid[0, 0, 0]
    assert target[0, 0, 0] == 3
    assert not valid[0, 0, 1]


def test_classification_loss_backprops(probe: OpenSetProbe) -> None:
    """Classification loss backpropagates through tokens and probe weights."""
    b, p, t, d = 2, 2, 2, 8
    visible = torch.ones(b, p, p, t, 1, dtype=torch.bool)
    latent, tokens = _make_latent(b, p, t, d, visible)
    pooled, repr_valid = probe.pool_patches(latent)

    h = w = p * 2
    open_set = torch.full((b, h, w, 1, 1), float(OPEN_SET_NODATA))
    open_set[:, :, :, 0, 0] = 5  # every pixel class 5 (dataset agrifieldnet_india)

    loss, n = probe.classification_loss(pooled, repr_valid, open_set)
    assert n == b * p * p
    assert torch.isfinite(loss)
    loss.backward()
    # Gradient flows into both the encoder tokens and the probe weights.
    assert tokens.grad is not None
    assert probe.cls_head.weight.grad is not None


def test_classification_loss_only_uses_target_group_vectors() -> None:
    """Classes from inactive source groups cannot affect the exact softmax."""
    probe = OpenSetProbe(
        embedding_size=1,
        class_mapping=_tiny_mapping(
            [
                {"name": "first", "global_ids": [0, 1]},
                {"name": "second", "global_ids": [2, 3]},
            ]
        ),
    )
    with torch.no_grad():
        probe.cls_head.weight.copy_(torch.tensor([[0.0], [1.0], [100.0], [100.0]]))
        probe.cls_head.bias.zero_()
    pooled = torch.ones(1, 1, 1, 1, requires_grad=True)
    repr_valid = torch.ones(1, 1, 1, dtype=torch.bool)
    open_set = torch.zeros(1, 1, 1, 1, 1)

    loss, n = probe.classification_loss(pooled, repr_valid, open_set)

    assert n == 1
    assert loss.detach().item() == pytest.approx(
        torch.log(torch.tensor(1.0 + torch.e)).item()
    )
    loss.backward()
    assert torch.count_nonzero(probe.cls_head.weight.grad[2:]) == 0


def test_classification_loss_excludes_target_conflicts() -> None:
    """Declared overlapping concepts are excluded as target-specific negatives."""
    probe = OpenSetProbe(
        embedding_size=1,
        class_mapping=_tiny_mapping(
            [
                {
                    "name": "presence",
                    "global_ids": [0, 1, 2],
                    "conflicts": {"0": [1]},
                },
                {"name": "other", "global_ids": [3]},
            ]
        ),
    )
    with torch.no_grad():
        probe.cls_head.weight.copy_(torch.tensor([[0.0], [100.0], [1.0], [100.0]]))
        probe.cls_head.bias.zero_()
    pooled = torch.ones(1, 1, 1, 1)
    repr_valid = torch.ones(1, 1, 1, dtype=torch.bool)
    open_set = torch.zeros(1, 1, 1, 1, 1)

    loss, n = probe.classification_loss(pooled, repr_valid, open_set)

    assert n == 1
    assert loss.detach().item() == pytest.approx(
        torch.log(torch.tensor(1.0 + torch.e)).item()
    )


def test_regression_loss_and_scaling(probe: OpenSetProbe) -> None:
    """Regression labels are scaled and contribute a finite loss."""
    b, p, t, d = 1, 2, 1, 8
    visible = torch.ones(b, p, p, t, 1, dtype=torch.bool)
    latent, _ = _make_latent(b, p, t, d, visible)
    pooled, repr_valid = probe.pool_patches(latent)

    h = w = p * 2
    reg = torch.zeros(b, h, w, 1, 2)
    # dataset id 1 (0-based idx 0), value = max_out -> target 1.0.
    reg[..., 0] = 1
    reg[..., 1] = probe.reg_value_max_out

    dataset_idx, target, valid = probe.pool_regression_labels(reg, p, p)
    assert valid.all()
    assert (dataset_idx == 0).all()
    assert torch.allclose(target, torch.ones_like(target))

    loss, n = probe.regression_loss(pooled, repr_valid, reg)
    assert n == b * p * p
    assert torch.isfinite(loss)


def test_regression_pooling_ignores_degenerate_frozen_range() -> None:
    """Invalid ranges already frozen into a build contribute no regression patches."""
    mapping = _tiny_mapping([{"name": "all", "global_ids": [0, 1, 2, 3]}])
    mapping["open_set_regression"]["datasets"] = [
        {"slug": "constant", "value_range": [0.0, 0.0]}
    ]
    probe = OpenSetProbe(embedding_size=1, class_mapping=mapping)
    reg = torch.zeros(1, 2, 2, 1, 2)
    reg[..., 0] = 1
    reg[..., 1] = 1

    _, _, valid = probe.pool_regression_labels(reg, 1, 1)

    assert not valid.any()


def test_forward_zero_touch_when_no_labels(probe: OpenSetProbe) -> None:
    """With all-missing labels the loss must still connect to probe params."""
    b, p, t, d = 2, 2, 2, 8
    visible = torch.ones(b, p, p, t, 1, dtype=torch.bool)
    latent, _ = _make_latent(b, p, t, d, visible)

    h = w = p * 2
    open_set = torch.full((b, h, w, 1, 1), float(OPEN_SET_NODATA))
    reg = torch.zeros(b, h, w, 1, 2)  # dataset id 0 everywhere -> no labels
    batch = SimpleNamespace(
        **{
            Modality.OPEN_SET.name: open_set,
            Modality.OPEN_SET_REGRESSION.name: reg,
        }
    )

    loss, metrics = probe(latent, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert probe.cls_head.weight.grad is not None
    assert probe.reg_head.weight.grad is not None
    assert metrics["open_set_ce_patches"] == 0.0
    assert metrics["open_set_mse_patches"] == 0.0
