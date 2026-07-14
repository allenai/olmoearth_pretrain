"""Unit tests for ModalityPatchDiscriminationCentroidVec."""

import math

import pytest
import torch
import torch.nn.functional as F

from olmoearth_pretrain.datatypes import MaskValue, TokensAndMasks
from olmoearth_pretrain.train.loss import (
    ModalityPatchDiscriminationCentroidVec,
    ModalityPatchDiscriminationMaskedNegativesVec,
)


def _tam(tokens: torch.Tensor, masks: torch.Tensor) -> TokensAndMasks:
    """Wrap [B,T,D] tokens as a single-bandset S2 TokensAndMasks."""
    b, t, d = tokens.shape
    return TokensAndMasks(
        sentinel2_l2a=tokens.view(b, 1, 1, t, 1, d),
        sentinel2_l2a_mask=masks.view(b, 1, 1, t, 1),
    )


def _make_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """4 decoder tokens: two near-identical (group A), two distinct."""
    d = 16
    base = F.normalize(torch.randn(4, d), dim=-1)
    targets = base.clone()
    # tokens 0 and 1 near-duplicates
    targets[1] = F.normalize(base[0] + 0.01 * torch.randn(d), dim=-1)
    preds = F.normalize(torch.randn(4, d), dim=-1)
    masks = torch.full((1, 4), MaskValue.DECODER.value, dtype=torch.long)
    return preds.unsqueeze(0), targets.unsqueeze(0), masks, d


def test_duplicates_share_group_and_centroid_math() -> None:
    """Loss equals hand-computed CE against 3 centroids (dup pair merged)."""
    torch.manual_seed(0)
    preds, targets, masks, d = _make_case()
    loss_fn = ModalityPatchDiscriminationCentroidVec(
        tau=0.1, group_threshold=0.95, center_targets=False
    )
    loss = loss_fn.compute(_tam(preds, masks), _tam(targets, masks))

    # Hand computation: groups {0,1}, {2}, {3} -> 3 centroids.
    t = targets[0]
    centroids = torch.stack([F.normalize((t[0] + t[1]) / 2, dim=-1), t[2], t[3]])
    labels = torch.tensor([0, 0, 1, 2])
    scores = preds[0] @ centroids.T / 0.1
    expected = F.cross_entropy(scores, labels) * 0.2
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-5)


def test_no_duplicates_matches_parent() -> None:
    """With orthogonal targets every group is a singleton -> parent loss."""
    torch.manual_seed(1)
    d = 32
    targets = F.normalize(torch.linalg.qr(torch.randn(d, d))[0][:6], dim=-1)
    preds = F.normalize(torch.randn(6, d), dim=-1)
    masks = torch.full((1, 6), MaskValue.DECODER.value, dtype=torch.long)
    p, t = preds.unsqueeze(0), targets.unsqueeze(0)

    centroid = ModalityPatchDiscriminationCentroidVec(
        tau=0.1, group_threshold=0.95, center_targets=False
    )
    parent = ModalityPatchDiscriminationMaskedNegativesVec(
        tau=0.1, mask_negatives_for_modalities=[]
    )
    loss_c = centroid.compute(_tam(p, masks), _tam(t, masks))
    loss_p = parent.compute(_tam(p, masks), _tam(t, masks))
    assert math.isclose(loss_c.item(), loss_p.item(), rel_tol=1e-5)


def test_single_group_sample_skipped() -> None:
    """All-duplicate targets (one group) contribute zero loss, no NaN."""
    torch.manual_seed(2)
    d = 16
    base = F.normalize(torch.randn(d), dim=-1)
    targets = base.unsqueeze(0).repeat(5, 1).unsqueeze(0)
    preds = F.normalize(torch.randn(1, 5, d), dim=-1)
    masks = torch.full((1, 5), MaskValue.DECODER.value, dtype=torch.long)
    loss_fn = ModalityPatchDiscriminationCentroidVec(
        tau=0.1, group_threshold=0.95, center_targets=False
    )
    loss = loss_fn.compute(_tam(preds, masks), _tam(targets, masks))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_chained_components_merge() -> None:
    """A~B and B~C merge into one group even if A!~C (connected components)."""
    d = 64
    a = F.normalize(torch.randn(d), dim=-1)
    # b close to a; c close to b but farther from a
    b = F.normalize(a + 0.15 * torch.randn(d), dim=-1)
    c = F.normalize(b + 0.15 * torch.randn(d), dim=-1)
    far = F.normalize(torch.randn(d), dim=-1)
    targets = torch.stack([a, b, c, far]).unsqueeze(0)
    sim = targets[0] @ targets[0].T
    thr = float(min(sim[0, 1], sim[1, 2])) - 1e-4
    if sim[0, 2] >= thr:  # ensure the chain condition actually holds
        pytest.skip("random draw did not produce a chain")
    preds = F.normalize(torch.randn(1, 4, d), dim=-1)
    masks = torch.full((1, 4), MaskValue.DECODER.value, dtype=torch.long)
    loss_fn = ModalityPatchDiscriminationCentroidVec(
        tau=0.1, group_threshold=thr, center_targets=False
    )
    loss = loss_fn.compute(_tam(preds, masks), _tam(targets, masks))
    # Expected: groups {a,b,c} and {far} -> 2 centroids.
    centroids = torch.stack([F.normalize((a + b + c) / 3, dim=-1), far])
    labels = torch.tensor([0, 0, 0, 1])
    scores = preds[0] @ centroids.T / 0.1
    expected = F.cross_entropy(scores, labels) * 0.2
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-4)


def test_gradients_flow_to_predictions() -> None:
    """Backward reaches predictions; targets need no grad."""
    torch.manual_seed(3)
    preds, targets, masks, _ = _make_case()
    preds = preds.clone().requires_grad_(True)
    loss_fn = ModalityPatchDiscriminationCentroidVec(
        tau=0.1, group_threshold=0.95, center_targets=False
    )
    loss = loss_fn.compute(_tam(preds, masks), _tam(targets, masks))
    loss.backward()
    assert preds.grad is not None and preds.grad.abs().sum() > 0


def test_centering_separates_shared_mean_component() -> None:
    """Targets sharing a dominant mean split into groups once centered."""
    torch.manual_seed(4)
    d = 64
    mu = 10.0 * F.normalize(torch.randn(d), dim=-1)
    d1 = F.normalize(torch.randn(d), dim=-1)
    d2 = F.normalize(torch.randn(d) - d1 * (torch.randn(d) @ d1), dim=-1)
    # four targets = mu +/- small distinct deviations: raw cosines ~1
    targets = torch.stack([mu + d1, mu + d1 * 0.9, mu + d2, mu + d2 * 1.1])
    raw = F.normalize(targets, dim=-1)
    assert (raw @ raw.T).min() > 0.98  # raw space: all near-duplicates
    preds = F.normalize(torch.randn(1, 4, d), dim=-1)
    masks = torch.full((1, 4), MaskValue.DECODER.value, dtype=torch.long)

    uncentered = ModalityPatchDiscriminationCentroidVec(
        tau=0.1, group_threshold=0.9, center_targets=False
    )
    centered = ModalityPatchDiscriminationCentroidVec(
        tau=0.1, group_threshold=0.9, center_targets=True
    )
    t = targets.unsqueeze(0)
    # uncentered: one group -> degenerate -> zero loss
    loss_u = uncentered.compute(_tam(preds, masks), _tam(t, masks))
    assert loss_u.item() == pytest.approx(0.0)
    # centered: two groups ({d1-ish}, {d2-ish}) -> real contrastive signal
    loss_c = centered.compute(_tam(preds, masks), _tam(t, masks))
    assert torch.isfinite(loss_c) and loss_c.item() > 0.0
