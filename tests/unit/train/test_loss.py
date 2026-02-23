"""Test losses."""

import logging

import torch

from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.train.loss import (
    AdjustedPatchDiscriminationLoss,
    CrossEntropyLoss,
    InfoNCELoss,
    L1Loss,
    L2Loss,
    ModalityPatchDiscriminationLossNew,
    ModalityPatchDiscriminationLossVec,
    PatchDiscriminationLoss,
    PatchDiscriminationLossNew,
)
from olmoearth_pretrain.train.masking import MaskValue

logger = logging.getLogger(__name__)


def test_patch_disc_loss() -> None:
    """Just test that it runs as expected."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchDiscriminationLoss()
    loss_value = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 0.5


def test_adjusted_patch_disc_loss_comparison() -> None:
    """Compare loss under different mu/sigma configs."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )

    # Loss hard is very sharp focus on the hard negatives, expect higher loss
    loss_easy = AdjustedPatchDiscriminationLoss(mu=0.3, sigma=1.0).compute(
        preds, targets
    )
    loss_hard = AdjustedPatchDiscriminationLoss(mu=0.9, sigma=0.1).compute(
        preds, targets
    )

    assert loss_hard >= loss_easy or abs(loss_hard - loss_easy) < 1e-3


def test_if_old_and_new_loss_are_the_same() -> None:
    """Test that the old and new patch discrimination loss are the same."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 2
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    loss_old = PatchDiscriminationLoss()
    loss_new = PatchDiscriminationLossNew()
    old_loss = loss_old.compute(preds, targets)
    new_loss = loss_new.compute(preds, targets)
    logger.info(f"old_loss: {old_loss}, new_loss: {new_loss}")
    assert torch.isclose(old_loss, new_loss)


def test_if_old_and_new_loss_are_the_same_uneven_number_of_decoder_tokens() -> None:
    """Test that the old and new patch discrimination loss are the same."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 2

    s2_preds_mask = torch.randint(0, 3, (b, t_h, t_w, t))

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_preds_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss_old = PatchDiscriminationLoss()
    loss_new = PatchDiscriminationLossNew()
    old_loss = loss_old.compute(preds, targets)
    new_loss = loss_new.compute(preds, targets)
    logger.info(f"old_loss: {old_loss}, new_loss: {new_loss}")
    assert torch.isclose(old_loss, new_loss)


def test_patch_disc_loss_averaged_over_batch_size() -> None:
    """Test it doesn't scale with batch size."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchDiscriminationLoss()
    loss_value = loss.compute(preds, targets)

    # now, use a larger batch size
    b, t_h, t_w, t, d = 8, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss_value_8 = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert torch.isclose(loss_value, loss_value_8)


def test_l1_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = L1Loss()
    loss_value = loss.compute(preds, targets)
    # MAE should be 1 since preds are 1, targets are 0
    assert loss_value == 1


def test_l2_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2_l2a=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = L2Loss()
    loss_value = loss.compute(preds, targets)
    # MSE should be 4 since preds are 2, targets are 0
    assert loss_value == 4


def test_cross_entropy_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2_l2a=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, 1), dtype=torch.long),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, 1), dtype=torch.long),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = CrossEntropyLoss()
    loss_value = loss.compute(preds, targets)
    # loss for BCE, prediction of .5 for both classes
    assert torch.isclose(loss_value, -torch.log(torch.tensor(0.5)), 0.0001)


def test_infonce_loss() -> None:
    """Just test that it runs as expected."""
    b, d = 16, 128

    loss = InfoNCELoss()
    loss_value = loss.compute(torch.ones((b, d)), torch.zeros((b, d)))
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 0.5
    # check the weight
    loss = InfoNCELoss(weight=0.1)
    w_loss_value = loss.compute(torch.ones((b, d)), torch.zeros((b, d)))
    assert 0.1 * loss_value == w_loss_value


def test_modality_patch_disc_parallelized_matches_sequential() -> None:
    """Test that parallelized modality patch disc loss matches sequential."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(42)
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )

    vec_loss = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    new_loss = ModalityPatchDiscriminationLossNew().compute(preds, targets)

    assert torch.isclose(vec_loss, new_loss, rtol=1e-4, atol=1e-6)


def test_modality_patch_disc_parallelized_uneven_tokens() -> None:
    """Test parallelized loss with uneven number of decoder tokens per sample."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(123)
    s2_mask = torch.randint(0, 3, (b, t_h, t_w, t))
    latlon_mask = torch.randint(0, 3, (b, 1))

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=latlon_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=latlon_mask,
    )

    vec_loss = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    new_loss = ModalityPatchDiscriminationLossNew().compute(preds, targets)

    assert torch.isclose(vec_loss, new_loss, rtol=1e-4, atol=1e-6)


def test_modality_patch_disc_parallelized_with_missing_samples() -> None:
    """Test parallelized loss when some samples have no decoder tokens."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(456)
    s2_mask = torch.randint(0, 3, (b, t_h, t_w, t))
    s2_mask[0] = MaskValue.ONLINE_ENCODER.value
    s2_mask[2] = MaskValue.MISSING.value

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )

    vec_loss = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    new_loss = ModalityPatchDiscriminationLossNew().compute(preds, targets)

    assert torch.isclose(vec_loss, new_loss, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Extended vec loss tests: gradients, dtypes, edge cases, scale
# ---------------------------------------------------------------------------


def test_vec_gradient_matches_new() -> None:
    """Verify gradients (not just loss) are equivalent between Vec and New."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 16

    for seed in [0, 7, 42, 999]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))

        s2_data = torch.randn((b, t_h, t_w, t, d))
        ll_data = torch.randn((b, 1, d))
        s2_tgt = torch.randn((b, t_h, t_w, t, d))
        ll_tgt = torch.randn((b, 1, d))

        # Vec path
        s2_pred_v = s2_data.clone().requires_grad_(True)
        ll_pred_v = ll_data.clone().requires_grad_(True)
        preds_v = TokensAndMasks(
            sentinel2_l2a=s2_pred_v, sentinel2_l2a_mask=s2_mask,
            latlon=ll_pred_v, latlon_mask=ll_mask,
        )
        targets_v = TokensAndMasks(
            sentinel2_l2a=s2_tgt.clone(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.clone(), latlon_mask=ll_mask,
        )
        loss_v = ModalityPatchDiscriminationLossVec().compute(preds_v, targets_v)
        loss_v.backward()

        # New path
        s2_pred_n = s2_data.clone().requires_grad_(True)
        ll_pred_n = ll_data.clone().requires_grad_(True)
        preds_n = TokensAndMasks(
            sentinel2_l2a=s2_pred_n, sentinel2_l2a_mask=s2_mask,
            latlon=ll_pred_n, latlon_mask=ll_mask,
        )
        targets_n = TokensAndMasks(
            sentinel2_l2a=s2_tgt.clone(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.clone(), latlon_mask=ll_mask,
        )
        loss_n = ModalityPatchDiscriminationLossNew().compute(preds_n, targets_n)
        loss_n.backward()

        assert torch.isclose(loss_v, loss_n, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: loss mismatch {loss_v.item()} vs {loss_n.item()}"
        )
        assert torch.allclose(s2_pred_v.grad, s2_pred_n.grad, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: s2 grad mismatch, "
            f"max diff={(s2_pred_v.grad - s2_pred_n.grad).abs().max().item()}"
        )
        grad_v = ll_pred_v.grad if ll_pred_v.grad is not None else torch.zeros_like(ll_pred_v)
        grad_n = ll_pred_n.grad if ll_pred_n.grad is not None else torch.zeros_like(ll_pred_n)
        assert torch.allclose(grad_v, grad_n, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: latlon grad mismatch, "
            f"max diff={(grad_v - grad_n).abs().max().item()}"
        )


def test_vec_gradient_matches_new_bfloat16() -> None:
    """Same gradient test but in bfloat16 to match training autocast."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 16

    for seed in [0, 42]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))

        s2_data = torch.randn((b, t_h, t_w, t, d))
        ll_data = torch.randn((b, 1, d))
        s2_tgt = torch.randn((b, t_h, t_w, t, d))
        ll_tgt = torch.randn((b, 1, d))

        # Vec
        s2_v = s2_data.bfloat16().requires_grad_(True)
        ll_v = ll_data.bfloat16().requires_grad_(True)
        preds_v = TokensAndMasks(
            sentinel2_l2a=s2_v, sentinel2_l2a_mask=s2_mask,
            latlon=ll_v, latlon_mask=ll_mask,
        )
        targets_v = TokensAndMasks(
            sentinel2_l2a=s2_tgt.bfloat16(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.bfloat16(), latlon_mask=ll_mask,
        )
        loss_v = ModalityPatchDiscriminationLossVec().compute(preds_v, targets_v)
        loss_v.backward()

        # New
        s2_n = s2_data.bfloat16().requires_grad_(True)
        ll_n = ll_data.bfloat16().requires_grad_(True)
        preds_n = TokensAndMasks(
            sentinel2_l2a=s2_n, sentinel2_l2a_mask=s2_mask,
            latlon=ll_n, latlon_mask=ll_mask,
        )
        targets_n = TokensAndMasks(
            sentinel2_l2a=s2_tgt.bfloat16(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.bfloat16(), latlon_mask=ll_mask,
        )
        loss_n = ModalityPatchDiscriminationLossNew().compute(preds_n, targets_n)
        loss_n.backward()

        assert torch.isclose(loss_v.float(), loss_n.float(), rtol=5e-3, atol=1e-4), (
            f"seed={seed} bf16 loss mismatch {loss_v.item()} vs {loss_n.item()}"
        )
        assert torch.allclose(
            s2_v.grad.float(), s2_n.grad.float(), rtol=5e-3, atol=1e-4
        ), (
            f"seed={seed} bf16 s2 grad max diff="
            f"{(s2_v.grad - s2_n.grad).float().abs().max().item()}"
        )


def test_vec_multiple_seeds_forward() -> None:
    """Sweep 20 random seeds to catch any seed-dependent mismatch."""
    b, t_h, t_w, t, d = 6, 4, 4, 2, 8
    for seed in range(20):
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))
        preds = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        targets = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
        new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
        assert torch.isclose(vec, new, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: {vec.item()} vs {new.item()}"
        )


def test_vec_single_decoder_token_per_sample() -> None:
    """Edge case: every sample has exactly 1 decoder token."""
    b, d = 8, 16
    torch.manual_seed(77)
    # 1 spatial position, 1 timestep → 1 token per sample
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, 1, 1, 1, d)),
        sentinel2_l2a_mask=torch.ones((b, 1, 1, 1), dtype=torch.long)
        * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1), dtype=torch.long)
        * MaskValue.ONLINE_ENCODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, 1, 1, 1, d)),
        sentinel2_l2a_mask=preds.sentinel2_l2a_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=preds.latlon_mask,
    )
    vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    assert torch.isclose(vec, new, rtol=1e-4, atol=1e-6), (
        f"single-token: {vec.item()} vs {new.item()}"
    )


def test_vec_all_samples_zero_decoder_tokens() -> None:
    """Edge case: no decoder tokens in any sample → loss should be 0."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 8
    torch.manual_seed(88)
    zero_mask = torch.zeros((b, t_h, t_w, t), dtype=torch.long)  # all ONLINE_ENCODER
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=zero_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.zeros((b, 1), dtype=torch.long),
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=zero_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.zeros((b, 1), dtype=torch.long),
    )
    par = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    assert par.item() == 0.0, f"expected 0.0 loss, got {par.item()}"


def test_vec_large_batch() -> None:
    """Larger batch closer to training microbatch size."""
    b, t_h, t_w, t, d = 32, 4, 4, 2, 64
    torch.manual_seed(2024)
    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    ll_mask = torch.randint(0, 4, (b, 1))
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=ll_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=ll_mask,
    )
    vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    assert torch.isclose(vec, new, rtol=1e-4, atol=1e-6), (
        f"large batch: {vec.item()} vs {new.item()}"
    )


def test_vec_multiple_modalities() -> None:
    """Test with 3 modalities having different mask patterns."""
    b, t_h, t_w, t, d = 6, 3, 3, 2, 16
    torch.manual_seed(555)

    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    s1_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    # worldcover: only decode modality — all decoder
    wc_mask = (
        torch.ones((b, t_h, t_w, 1), dtype=torch.long) * MaskValue.DECODER.value
    )

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn((b, t_h, t_w, t, d)),
        sentinel1_mask=s1_mask,
        worldcover=torch.randn((b, t_h, t_w, 1, d)),
        worldcover_mask=wc_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn((b, t_h, t_w, t, d)),
        sentinel1_mask=s1_mask,
        worldcover=torch.randn((b, t_h, t_w, 1, d)),
        worldcover_mask=wc_mask,
    )
    vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    assert torch.isclose(vec, new, rtol=1e-4, atol=1e-6), (
        f"multi-modality: {vec.item()} vs {new.item()}"
    )


def test_vec_modality_weights() -> None:
    """Test that modality_weights are applied identically."""
    b, t_h, t_w, t, d = 5, 3, 3, 2, 16
    weights = {"sentinel2_l2a": 2.0, "latlon": 0.5}

    for seed in [0, 42, 99]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))
        preds = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        targets = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        vec = ModalityPatchDiscriminationLossVec(
            modality_weights=weights
        ).compute(preds, targets)
        new = ModalityPatchDiscriminationLossNew(
            modality_weights=weights
        ).compute(preds, targets)
        assert torch.isclose(vec, new, rtol=1e-4, atol=1e-6), (
            f"seed={seed} weighted: {vec.item()} vs {new.item()}"
        )


def test_vec_high_dim_large_tokens() -> None:
    """Higher dimension and larger spatial size closer to real training."""
    b, t_h, t_w, t, d = 8, 8, 8, 3, 128
    torch.manual_seed(314)
    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
    )
    vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    assert torch.isclose(vec, new, rtol=1e-3, atol=1e-5), (
        f"high-dim: {vec.item()} vs {new.item()}, "
        f"diff={abs(vec.item() - new.item())}"
    )


def test_vec_gradient_no_leak_to_non_decoder() -> None:
    """Non-decoder tokens should receive zero gradient from the loss."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 16
    torch.manual_seed(12)

    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    # Ensure at least some non-decoder tokens exist
    s2_mask[0, 0, 0, 0] = MaskValue.ONLINE_ENCODER.value
    s2_mask[1, 0, 0, 0] = MaskValue.MISSING.value

    s2_pred = torch.randn((b, t_h, t_w, t, d), requires_grad=True)
    preds = TokensAndMasks(
        sentinel2_l2a=s2_pred,
        sentinel2_l2a_mask=s2_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
    )
    loss = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    loss.backward()

    non_decoder = s2_mask != MaskValue.DECODER.value
    non_decoder_grad = s2_pred.grad[non_decoder]
    assert (non_decoder_grad == 0).all(), (
        f"non-decoder tokens got non-zero gradients: "
        f"max={non_decoder_grad.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Direct comparison: ModalityPatchDiscriminationLossNew vs Vec
# These use the real production loss classes, no reference implementations.
# ---------------------------------------------------------------------------


def test_new_vs_vec_uniform_masks() -> None:
    """Vec loss matches New loss when all tokens are decoder tokens."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(42)
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )

    loss_new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    loss_vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)

    assert torch.isclose(loss_new, loss_vec, rtol=1e-4, atol=1e-6), (
        f"new={loss_new.item()}, vec={loss_vec.item()}"
    )


def test_new_vs_vec_uneven_tokens() -> None:
    """Vec loss matches New loss with uneven decoder token counts per sample."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(123)
    s2_mask = torch.randint(0, 3, (b, t_h, t_w, t))
    latlon_mask = torch.randint(0, 3, (b, 1))

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=latlon_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=latlon_mask,
    )

    loss_new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    loss_vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)

    assert torch.isclose(loss_new, loss_vec, rtol=1e-4, atol=1e-6), (
        f"new={loss_new.item()}, vec={loss_vec.item()}"
    )


def test_new_vs_vec_missing_samples() -> None:
    """Vec loss matches New loss when some samples have no decoder tokens."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(456)
    s2_mask = torch.randint(0, 3, (b, t_h, t_w, t))
    s2_mask[0] = MaskValue.ONLINE_ENCODER.value
    s2_mask[2] = MaskValue.MISSING.value

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )

    loss_new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    loss_vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)

    assert torch.isclose(loss_new, loss_vec, rtol=1e-4, atol=1e-6), (
        f"new={loss_new.item()}, vec={loss_vec.item()}"
    )


def test_new_vs_vec_multiple_seeds() -> None:
    """Sweep seeds to catch any seed-dependent mismatch between New and Vec."""
    b, t_h, t_w, t, d = 6, 4, 4, 2, 8
    for seed in range(20):
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))
        preds = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        targets = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        loss_new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
        loss_vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
        assert torch.isclose(loss_new, loss_vec, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: new={loss_new.item()}, vec={loss_vec.item()}"
        )


def test_new_vs_vec_gradients() -> None:
    """Gradients from Vec match gradients from New for the same inputs."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 16

    for seed in [0, 7, 42]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))

        s2_data = torch.randn((b, t_h, t_w, t, d))
        ll_data = torch.randn((b, 1, d))
        s2_tgt = torch.randn((b, t_h, t_w, t, d))
        ll_tgt = torch.randn((b, 1, d))

        # New path
        s2_new = s2_data.clone().requires_grad_(True)
        ll_new = ll_data.clone().requires_grad_(True)
        preds_new = TokensAndMasks(
            sentinel2_l2a=s2_new, sentinel2_l2a_mask=s2_mask,
            latlon=ll_new, latlon_mask=ll_mask,
        )
        targets_new = TokensAndMasks(
            sentinel2_l2a=s2_tgt.clone(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.clone(), latlon_mask=ll_mask,
        )
        loss_n = ModalityPatchDiscriminationLossNew().compute(preds_new, targets_new)
        loss_n.backward()

        # Vec path
        s2_vec = s2_data.clone().requires_grad_(True)
        ll_vec = ll_data.clone().requires_grad_(True)
        preds_vec = TokensAndMasks(
            sentinel2_l2a=s2_vec, sentinel2_l2a_mask=s2_mask,
            latlon=ll_vec, latlon_mask=ll_mask,
        )
        targets_vec = TokensAndMasks(
            sentinel2_l2a=s2_tgt.clone(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.clone(), latlon_mask=ll_mask,
        )
        loss_v = ModalityPatchDiscriminationLossVec().compute(preds_vec, targets_vec)
        loss_v.backward()

        assert torch.isclose(loss_n, loss_v, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: loss mismatch new={loss_n.item()} vs vec={loss_v.item()}"
        )
        assert torch.allclose(s2_new.grad, s2_vec.grad, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: s2 grad mismatch, "
            f"max diff={(s2_new.grad - s2_vec.grad).abs().max().item()}"
        )
        grad_n = ll_new.grad if ll_new.grad is not None else torch.zeros_like(ll_new)
        grad_v = ll_vec.grad if ll_vec.grad is not None else torch.zeros_like(ll_vec)
        assert torch.allclose(grad_n, grad_v, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: latlon grad mismatch, "
            f"max diff={(grad_n - grad_v).abs().max().item()}"
        )


def test_new_vs_vec_modality_weights() -> None:
    """Modality weights applied identically between New and Vec."""
    b, t_h, t_w, t, d = 5, 3, 3, 2, 16
    weights = {"sentinel2_l2a": 2.0, "latlon": 0.5}

    for seed in [0, 42, 99]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))
        preds = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        targets = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        loss_new = ModalityPatchDiscriminationLossNew(modality_weights=weights).compute(
            preds, targets
        )
        loss_vec = ModalityPatchDiscriminationLossVec(modality_weights=weights).compute(
            preds, targets
        )
        assert torch.isclose(loss_new, loss_vec, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: new={loss_new.item()}, vec={loss_vec.item()}"
        )


def test_new_vs_vec_large_batch() -> None:
    """New vs Vec at training-like microbatch size."""
    b, t_h, t_w, t, d = 32, 4, 4, 2, 64
    torch.manual_seed(2024)
    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    ll_mask = torch.randint(0, 4, (b, 1))
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=ll_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=ll_mask,
    )
    loss_new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    loss_vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    assert torch.isclose(loss_new, loss_vec, rtol=1e-4, atol=1e-6), (
        f"large batch: new={loss_new.item()}, vec={loss_vec.item()}"
    )


def test_new_vs_vec_multiple_modalities() -> None:
    """New vs Vec with 3 modalities having different mask patterns."""
    b, t_h, t_w, t, d = 6, 3, 3, 2, 16
    torch.manual_seed(555)

    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    s1_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    wc_mask = (
        torch.ones((b, t_h, t_w, 1), dtype=torch.long) * MaskValue.DECODER.value
    )

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn((b, t_h, t_w, t, d)),
        sentinel1_mask=s1_mask,
        worldcover=torch.randn((b, t_h, t_w, 1, d)),
        worldcover_mask=wc_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn((b, t_h, t_w, t, d)),
        sentinel1_mask=s1_mask,
        worldcover=torch.randn((b, t_h, t_w, 1, d)),
        worldcover_mask=wc_mask,
    )
    loss_new = ModalityPatchDiscriminationLossNew().compute(preds, targets)
    loss_vec = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    assert torch.isclose(loss_new, loss_vec, rtol=1e-4, atol=1e-6), (
        f"multi-modality: new={loss_new.item()}, vec={loss_vec.item()}"
    )
