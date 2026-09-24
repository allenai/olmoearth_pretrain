"""Unit tests for the supervised open-set probe."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
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


def _cls(probe: OpenSetProbe) -> torch.nn.Linear:
    """The learned class head (present for the linear / mlp variants)."""
    assert probe.cls_head is not None
    return probe.cls_head


def _make_spatial_latent(b: int, p: int, d: int) -> torch.Tensor:
    """Build a random spatial latent grid (B, P, P, D) with gradients enabled.

    This mimics the encoder's register/Perceiver bottleneck output: one embedding
    per spatial cell, time/modality already collapsed.
    """
    return torch.randn(b, p, p, d, requires_grad=True)


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


def test_forward_rejects_non_grid_latent(probe: OpenSetProbe) -> None:
    """The probe requires a (B, P_H, P_W, D) spatial latent grid."""
    batch = SimpleNamespace(
        **{
            Modality.OPEN_SET.name: None,
            Modality.OPEN_SET_REGRESSION.name: None,
        }
    )
    with pytest.raises(ValueError, match="spatial_latent must have shape"):
        probe(torch.randn(2, 4, 8), batch)


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
    """Classification loss backpropagates through the latent and probe weights."""
    b, p, d = 2, 2, 8
    pooled = _make_spatial_latent(b, p, d)
    repr_valid = torch.ones(b, p, p, dtype=torch.bool)

    h = w = p * 2
    open_set = torch.full((b, h, w, 1, 1), float(OPEN_SET_NODATA))
    open_set[:, :, :, 0, 0] = 5  # every pixel class 5 (dataset agrifieldnet_india)

    loss, n_samples, n_patches = probe.classification_loss(pooled, repr_valid, open_set)
    assert n_samples == b
    assert n_patches == b * p * p
    assert torch.isfinite(loss)
    loss.backward()
    # Gradient flows into both the spatial latent and the probe weights.
    assert pooled.grad is not None
    assert _cls(probe).weight.grad is not None


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
        _cls(probe).weight.copy_(torch.tensor([[0.0], [1.0], [100.0], [100.0]]))
        _cls(probe).bias.zero_()
    pooled = torch.ones(1, 1, 1, 1, requires_grad=True)
    repr_valid = torch.ones(1, 1, 1, dtype=torch.bool)
    open_set = torch.zeros(1, 1, 1, 1, 1)

    loss, n_samples, n_patches = probe.classification_loss(pooled, repr_valid, open_set)

    assert n_samples == 1
    assert n_patches == 1
    assert loss.detach().item() == pytest.approx(
        torch.log(torch.tensor(1.0 + torch.e)).item()
    )
    loss.backward()
    assert torch.count_nonzero(_cls(probe).weight.grad[2:]) == 0


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
        _cls(probe).weight.copy_(torch.tensor([[0.0], [100.0], [1.0], [100.0]]))
        _cls(probe).bias.zero_()
    pooled = torch.ones(1, 1, 1, 1)
    repr_valid = torch.ones(1, 1, 1, dtype=torch.bool)
    open_set = torch.zeros(1, 1, 1, 1, 1)

    loss, n_samples, n_patches = probe.classification_loss(pooled, repr_valid, open_set)

    assert n_samples == 1
    assert n_patches == 1
    assert loss.detach().item() == pytest.approx(
        torch.log(torch.tensor(1.0 + torch.e)).item()
    )


def test_classification_loss_balances_samples() -> None:
    """Loss is the mean of per-sample means, not the mean over all patches."""
    probe = OpenSetProbe(
        embedding_size=1,
        class_mapping=_tiny_mapping(
            [{"name": "all", "global_ids": [0, 1]}], num_classes=2
        ),
    )
    with torch.no_grad():
        _cls(probe).weight.copy_(torch.tensor([[0.0], [1.0]]))
        _cls(probe).bias.zero_()
    pooled = torch.ones(2, 2, 2, 1)
    repr_valid = torch.ones(2, 2, 2, dtype=torch.bool)
    # Sample 0: all four patches labeled class 0. Sample 1: one patch labeled class 1.
    open_set = torch.full((2, 2, 2, 1, 1), float(OPEN_SET_NODATA))
    open_set[0, :, :, 0, 0] = 0
    open_set[1, 0, 0, 0, 0] = 1

    loss, n_samples, n_patches = probe.classification_loss(pooled, repr_valid, open_set)

    assert n_samples == 2
    assert n_patches == 5
    # Per-patch CE: class 0 -> log(1 + e); class 1 -> log(1 + e) - 1. The dense
    # sample must not outweigh the sparse one: mean of the two per-sample means.
    ce0 = torch.log(torch.tensor(1.0 + torch.e)).item()
    assert loss.detach().item() == pytest.approx((ce0 + (ce0 - 1.0)) / 2, rel=1e-5)


def test_regression_loss_and_scaling(probe: OpenSetProbe) -> None:
    """Regression labels are scaled and contribute a finite loss."""
    b, p, d = 1, 2, 8
    pooled = _make_spatial_latent(b, p, d)
    repr_valid = torch.ones(b, p, p, dtype=torch.bool)

    h = w = p * 2
    reg = torch.zeros(b, h, w, 1, 2)
    # dataset id 1 (0-based idx 0), value = max_out -> target 1.0.
    reg[..., 0] = 1
    reg[..., 1] = probe.reg_value_max_out

    dataset_idx, target, valid = probe.pool_regression_labels(reg, p, p)
    assert valid.all()
    assert (dataset_idx == 0).all()
    assert torch.allclose(target, torch.ones_like(target))

    loss, n_samples, n_patches = probe.regression_loss(pooled, repr_valid, reg)
    assert n_samples == b
    assert n_patches == b * p * p
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


def _change_batch(visible_timesteps: list[int]) -> MaskedOlmoEarthSample:
    """One change sample (boundary July 1st 2021) plus one non-change sample.

    Only ``visible_timesteps`` of the S2 series are marked ONLINE_ENCODER; both
    samples carry a dense classification label (class 5).
    """
    b, h, w, t = 2, 4, 4, 12
    s2_mask = torch.full((b, h, w, t, 3), MaskValue.DECODER.value)
    s2_mask[..., visible_timesteps, :] = MaskValue.ONLINE_ENCODER.value
    timestamps = torch.tensor([[1, m, 2021] for m in range(t)]).unsqueeze(0)
    boundary = torch.tensor([[1, 6, 2021], [MISSING_VALUE] * 3])
    open_set = torch.full((b, h, w, 1, 1), 5.0)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(b, h, w, t, 12),
        sentinel2_l2a_mask=s2_mask,
        open_set=open_set,
        open_set_mask=torch.full((b, h, w, 1, 1), MaskValue.DECODER.value),
        open_set_change_boundary=boundary,
        open_set_change_boundary_mask=torch.full((b, 1), MaskValue.DECODER.value),
        timestamps=timestamps.expand(b, t, 3).clone(),
    )


@pytest.mark.parametrize(
    "visible_timesteps,change_supervised",
    [
        ([0, 1, 2], False),  # pre only
        ([8, 9], False),  # post only
        ([2, 9], True),  # both sides
    ],
)
def test_change_sample_needs_both_sides_visible(
    probe: OpenSetProbe, visible_timesteps: list[int], change_supervised: bool
) -> None:
    """Change labels are dropped unless the encoder saw a pre and a post timestep."""
    batch = _change_batch(visible_timesteps)
    valid = OpenSetProbe.change_samples_with_both_sides_visible(batch)
    assert valid is not None
    assert bool(valid[0]) is change_supervised
    assert bool(valid[1])  # the non-change sample is always supervisable

    latent = _make_spatial_latent(2, 2, 8)
    _, metrics = probe(latent, batch)
    assert metrics["open_set_ce_samples"] == (2.0 if change_supervised else 1.0)


def test_forward_zero_touch_when_no_labels(probe: OpenSetProbe) -> None:
    """With all-missing labels the loss must still connect to probe params."""
    b, p, d = 2, 2, 8
    latent = _make_spatial_latent(b, p, d)

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
    assert _cls(probe).weight.grad is not None
    assert probe.reg_head.weight.grad is not None
    assert "open_set_ce" not in metrics and "open_set_mse" not in metrics
    assert metrics["open_set_ce_samples"] == 0.0
    assert metrics["open_set_ce_patches"] == 0.0
    assert metrics["open_set_mse_samples"] == 0.0
    assert metrics["open_set_mse_patches"] == 0.0


# ----------------------------------------------------------------------------
# Probe variants: MLP trunk, text-embedding targets, dataset balancing
# ----------------------------------------------------------------------------


def _variant_batch(b: int = 2, p: int = 2) -> SimpleNamespace:
    """Dense class-5 labels plus a regression label on dataset id 1."""
    h = w = p * 2
    open_set = torch.full((b, h, w, 1, 1), 5.0)
    reg = torch.zeros(b, h, w, 1, 2)
    reg[..., 0] = 1
    reg[..., 1] = 30000
    return SimpleNamespace(
        **{Modality.OPEN_SET.name: open_set, Modality.OPEN_SET_REGRESSION.name: reg}
    )


@pytest.mark.parametrize("head_type", ["linear", "mlp", "text"])
def test_head_variants_build_and_backprop(class_mapping: dict, head_type: str) -> None:
    """Every head type runs end to end and reaches the latent and its own params."""
    torch.manual_seed(0)
    num_classes = class_mapping["open_set"]["num_classes"]
    text = torch.randn(num_classes, 6) if head_type == "text" else None
    probe = OpenSetProbe(
        embedding_size=8,
        class_mapping=class_mapping,
        head_type=head_type,
        mlp_hidden_size=5,
        text_embeddings=text,
    )
    if head_type == "linear":
        assert isinstance(probe.trunk, torch.nn.Identity)
    else:
        assert isinstance(probe.trunk[0], torch.nn.Linear)
        assert probe.trunk[0].out_features == 5
    if head_type == "text":
        assert probe.cls_head is None and probe.logit_scale is not None
        assert probe.trunk[-1].out_features == 6
        assert probe.reg_head.in_features == 6
    else:
        assert probe.cls_head is not None and probe.logit_scale is None

    latent = _make_spatial_latent(2, 2, 8)
    loss, metrics = probe(latent, _variant_batch())
    assert torch.isfinite(loss)
    assert metrics["open_set_ce_samples"] == 2.0
    assert metrics["open_set_mse_samples"] == 2.0
    loss.backward()
    assert latent.grad is not None
    for name, param in probe.named_parameters():
        assert param.grad is not None, name


def test_text_head_scores_only_target_group_embeddings() -> None:
    """Text logits are scaled cosines against the group's rows; conflicts masked."""
    mapping = _tiny_mapping(
        [
            {"name": "first", "global_ids": [0, 1, 2], "conflicts": {"0": [2]}},
            {"name": "second", "global_ids": [3]},
        ]
    )
    # Orthogonal text embeddings; class 3 (other group) is identical to class 0's
    # target so it would dominate if it leaked into the softmax.
    text = torch.eye(4)
    text[3] = text[0]
    probe = OpenSetProbe(
        embedding_size=4,
        class_mapping=mapping,
        head_type="text",
        text_embeddings=text,
        text_logit_scale_init=1.0,
    )
    # Make the trunk the identity so features == latent.
    with torch.no_grad():
        for layer in (probe.trunk[0], probe.trunk[2]):
            layer.weight.copy_(torch.eye(4))
            layer.bias.zero_()
    latent = torch.eye(4)[0].view(1, 1, 1, 4) * 10.0  # GELU(10) ~ 10 -> normalized
    repr_valid = torch.ones(1, 1, 1, dtype=torch.bool)
    open_set = torch.zeros(1, 1, 1, 1, 1)  # target class 0

    loss, n_samples, _ = probe.classification_loss(latent, repr_valid, open_set)

    # Group "first" restricted to classes {0, 1}, class 2 excluded as a conflict:
    # logits are [1, 0] -> CE = log(1 + e^-1).
    assert n_samples == 1
    assert loss.item() == pytest.approx(math.log(1.0 + math.exp(-1.0)), rel=1e-4)


def test_text_head_requires_matching_embeddings() -> None:
    """Text embeddings must exist and have one row per global class."""
    mapping = _tiny_mapping([{"name": "all", "global_ids": [0, 1, 2, 3]}])
    with pytest.raises(ValueError, match="requires text_embeddings"):
        OpenSetProbe(embedding_size=4, class_mapping=mapping, head_type="text")
    with pytest.raises(ValueError, match="rows"):
        OpenSetProbe(
            embedding_size=4,
            class_mapping=mapping,
            head_type="text",
            text_embeddings=torch.randn(3, 4),
        )
    with pytest.raises(ValueError, match="head_type"):
        OpenSetProbe(embedding_size=4, class_mapping=mapping, head_type="conv")


def _balance_mapping() -> dict:
    """Two plain datasets, one merged presence-only class, one regression dataset."""
    return {
        "open_set": {
            "num_classes": 4,
            "classes": [
                {"global_id": 0, "slug": "big", "local_id": 0, "name": "a"},
                {"global_id": 1, "slug": "big", "local_id": 1, "name": "b"},
                {"global_id": 2, "slug": "small", "local_id": 0, "name": "c"},
                {
                    "global_id": 3,
                    "slug": None,
                    "local_id": None,
                    "name": "merged",
                    "members": [
                        {"slug": "small", "local_id": 1, "name": "x"},
                        {"slug": "tiny", "local_id": 0, "name": "y"},
                    ],
                },
            ],
            "training_datasets": [
                {"name": "big", "global_ids": [0, 1]},
                {"name": "small", "global_ids": [2]},
                {"name": "__presence_only__", "global_ids": [3]},
            ],
        },
        "open_set_regression": {
            "datasets": [{"slug": "reg", "dataset_id": 1, "value_range": [0.0, 1.0]}]
        },
    }


def test_dataset_balance_weights() -> None:
    """Weights follow n^(tau-1), merged classes sum members, mean weight is one."""
    counts = {"big": 900, "small": 100, "tiny": 25, "reg": 400}
    tau = 0.5
    probe = OpenSetProbe(
        embedding_size=2,
        class_mapping=_balance_mapping(),
        dataset_counts=counts,
        balance_temperature=tau,
    )
    w = probe.class_balance_weight
    # Same dataset -> same weight; ratio between datasets is (n_a / n_b)^(tau - 1).
    assert w[0] == pytest.approx(w[1])
    assert (w[2] / w[0]).item() == pytest.approx((100 / 900) ** (tau - 1), rel=1e-5)
    # Merged class counts every member dataset (100 + 25).
    assert (w[3] / w[0]).item() == pytest.approx((125 / 900) ** (tau - 1), rel=1e-5)
    assert (probe.reg_balance_weight[0] / w[0]).item() == pytest.approx(
        (400 / 900) ** (tau - 1), rel=1e-5
    )
    # Expected weight under natural sampling over the datasets is one: w_d = Z n_d^(tau-1)
    # with Z = sum(n) / sum(n^tau), so sum(n_d w_d) / sum(n_d) == 1.
    n = torch.tensor([900.0, 100.0, 25.0, 400.0], dtype=torch.float64)
    normalizer = (w[0] / 900 ** (tau - 1)).item()
    per_dataset = normalizer * n ** (tau - 1)
    assert (n * per_dataset).sum().item() / n.sum().item() == pytest.approx(
        1.0, rel=1e-5
    )


def test_dataset_balance_default_is_all_ones_and_matches_unbalanced_loss() -> None:
    """Without counts the weights are one and the loss equals the plain probe's."""
    torch.manual_seed(0)
    mapping = _balance_mapping()
    plain = OpenSetProbe(embedding_size=3, class_mapping=mapping)
    assert torch.all(plain.class_balance_weight == 1)
    assert torch.all(plain.reg_balance_weight == 1)

    balanced = OpenSetProbe(
        embedding_size=3,
        class_mapping=mapping,
        dataset_counts={"big": 10, "small": 10, "tiny": 5, "reg": 10},
        balance_temperature=0.5,
    )
    balanced.load_state_dict(plain.state_dict())
    pooled = torch.randn(2, 1, 1, 3)
    repr_valid = torch.ones(2, 1, 1, dtype=torch.bool)
    open_set = torch.zeros(2, 1, 1, 1, 1)  # both samples class 0 (dataset "big")
    loss_plain, _, _ = plain.classification_loss(pooled, repr_valid, open_set)
    loss_bal, _, _ = balanced.classification_loss(pooled, repr_valid, open_set)
    # All labeled samples come from one dataset, so the balanced loss is the plain
    # loss times that dataset's weight.
    assert loss_bal.item() == pytest.approx(
        loss_plain.item() * balanced.class_balance_weight[0].item(), rel=1e-5
    )


def test_dataset_balance_requires_counts_for_every_slug() -> None:
    """A dataset without a registry count is an error, not a silent weight of one."""
    with pytest.raises(ValueError, match="no sample count"):
        OpenSetProbe(
            embedding_size=2,
            class_mapping=_balance_mapping(),
            dataset_counts={"big": 1, "small": 1, "tiny": 1},
        )


def test_config_builds_text_probe_from_files(
    class_mapping: dict, tmp_path: Path
) -> None:
    """The config loads the .npy + sidecar and rejects a stale sidecar hash."""
    import numpy as np

    from olmoearth_pretrain.train.open_set_probe import load_dataset_counts

    mapping_sha = hashlib.sha256(_CLASS_MAPPING_PATH.read_bytes()).hexdigest()
    num_classes = class_mapping["open_set"]["num_classes"]
    npy = tmp_path / "class_text_embeddings.npy"
    np.save(npy, np.random.randn(num_classes, 6).astype(np.float16))
    (tmp_path / "class_text_embeddings.json").write_text(
        json.dumps({"class_mapping_sha256": mapping_sha})
    )
    config = OpenSetProbeConfig(
        class_mapping_path=str(_CLASS_MAPPING_PATH),
        head_type="text",
        text_embeddings_path=str(npy),
        dataset_balance="dataset_temperature",
    )
    probe = config.build(embedding_size=8)
    assert probe.text_embeddings.shape == (num_classes, 6)
    # Unit-normalized rows.
    assert torch.allclose(
        probe.text_embeddings.norm(dim=-1), torch.ones(num_classes), atol=1e-5
    )
    # Balancing from the checked-in registry covers every slug in the mapping.
    assert (probe.class_balance_weight != 1).any()
    counts = load_dataset_counts(_CLASS_MAPPING_PATH.with_name("registry.json"))
    assert counts["agrifieldnet_india"] > 0

    (tmp_path / "class_text_embeddings.json").write_text(
        json.dumps({"class_mapping_sha256": "0" * 64})
    )
    with pytest.raises(ValueError, match="generated for class mapping"):
        config.build(embedding_size=8)
