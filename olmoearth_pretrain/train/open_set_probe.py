"""Supervised open-set probe head for pretraining.

This module adds *supervised* segmentation + regression signal on top of the
self-supervised latent-MIM objective, driven by the ``open_set`` /
``open_set_regression`` label layers (built by
``olmoearth_pretrain.open_set_segmentation_data``).

The probe is a linear map from the pooled per-spatial-patch encoder
representation to per-class logits (classification) and per-dataset scalars
(regression):

* One learned vector per global class id (a ``num_classes x D`` weight matrix).
* One learned vector per regression dataset (a ``num_reg_datasets x D`` matrix).

Per spatial patch we pool the encoder tokens over modality / timestep / band-set
(only the tokens actually seen by the online encoder), run the linear probe, and
compute:

* **cross-entropy** for classification, with a *masked softmax* restricted to the
  source dataset's class subset (each open-set window comes from a single source
  dataset, so negatives only come from within that dataset / the merged
  presence-only group);
* **mean-squared error** for regression, against the stored value mapped to
  ``[0, 1]``.

The probe parameters are meant to live *inside* the model (see
``olmoearth_pretrain.nn.open_set_latent_mim``) so the DDP gradient all-reduce and
the optimizer, which both iterate ``self.model.parameters()``, cover them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue, TokensAndMasks

logger = logging.getLogger(__name__)

# Sentinels from the open-set label layers (see
# open_set_segmentation_data.pretrain_constants). Duplicated here to avoid a train-time
# dependency on the dataset-creation package.
OPEN_SET_NODATA = 65535
REGRESSION_DATASET_ID_NODATA = 0
REGRESSION_VALUE_NODATA = 0
REGRESSION_VALUE_MIN_OUT = 1
REGRESSION_VALUE_MAX_OUT = 65535


@dataclass
class OpenSetProbeConfig(Config):
    """Configuration for :class:`OpenSetProbe`.

    Args:
        class_mapping_path: Path to ``class_mapping.json`` (produced by
            ``open_set_segmentation_data.assemble_classes``).
        seg_loss_weight: Relative weight of the classification (CE) term.
        reg_loss_weight: Relative weight of the regression (MSE) term.
    """

    class_mapping_path: str
    expected_class_mapping_sha256: str | None = None
    seg_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0

    def build(self, embedding_size: int) -> OpenSetProbe:
        """Build the probe for an encoder with the given token embedding size."""
        mapping_bytes = Path(self.class_mapping_path).read_bytes()
        if self.expected_class_mapping_sha256 is not None:
            actual_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
            if actual_sha256 != self.expected_class_mapping_sha256:
                raise ValueError(
                    "class mapping hash mismatch: expected "
                    f"{self.expected_class_mapping_sha256}, got {actual_sha256}"
                )
        class_mapping = json.loads(mapping_bytes)
        return OpenSetProbe(
            embedding_size=embedding_size,
            class_mapping=class_mapping,
            seg_loss_weight=self.seg_loss_weight,
            reg_loss_weight=self.reg_loss_weight,
        )


class OpenSetProbe(nn.Module):
    """Linear probe over pooled per-patch encoder representations.

    Args:
        embedding_size: The token embedding size ``D`` emitted by the encoder.
        class_mapping: Parsed ``class_mapping.json`` dict.
        seg_loss_weight: Relative weight of the classification (CE) term.
        reg_loss_weight: Relative weight of the regression (MSE) term.
    """

    def __init__(
        self,
        embedding_size: int,
        class_mapping: dict[str, Any],
        seg_loss_weight: float = 1.0,
        reg_loss_weight: float = 1.0,
    ):
        """Initialize the probe and the class-subset lookup buffers."""
        super().__init__()
        self.embedding_size = embedding_size
        self.seg_loss_weight = seg_loss_weight
        self.reg_loss_weight = reg_loss_weight

        open_set = class_mapping["open_set"]
        self.num_classes: int = int(open_set["num_classes"])
        training_datasets = open_set["training_datasets"]
        self.num_groups: int = len(training_datasets)

        regression = class_mapping["open_set_regression"]
        regression_datasets = regression["datasets"]
        self.num_reg_datasets: int = len(regression_datasets)
        value_out_range = regression.get(
            "value_out_range",
            [REGRESSION_VALUE_MIN_OUT, REGRESSION_VALUE_MAX_OUT],
        )
        self.reg_value_min_out: float = float(value_out_range[0])
        self.reg_value_max_out: float = float(value_out_range[1])

        # Linear probes: one weight vector per class / per regression dataset.
        self.cls_head = nn.Linear(embedding_size, self.num_classes)
        self.reg_head = nn.Linear(embedding_size, max(self.num_reg_datasets, 1))

        valid_regression_datasets = torch.zeros(self.num_reg_datasets, dtype=torch.bool)
        invalid_regression_slugs = []
        for dataset_idx, dataset in enumerate(regression_datasets):
            value_range = dataset.get("value_range")
            is_valid = (
                isinstance(value_range, list)
                and len(value_range) == 2
                and all(math.isfinite(float(value)) for value in value_range)
                and float(value_range[1]) > float(value_range[0])
            )
            valid_regression_datasets[dataset_idx] = is_valid
            if not is_valid:
                invalid_regression_slugs.append(
                    dataset.get("slug", str(dataset_idx + 1))
                )
        if invalid_regression_slugs:
            logger.warning(
                "Ignoring open-set regression labels with invalid frozen value "
                "ranges: %s",
                ", ".join(invalid_regression_slugs),
            )
        self.register_buffer(
            "valid_regression_datasets",
            valid_regression_datasets,
            persistent=False,
        )

        # Compact lookup tables for exact, group-local softmaxes. The learned
        # classifier still has one row per global class, but each patch is projected
        # only against its source dataset's rows rather than all global classes.
        max_group_size = max(len(td["global_ids"]) for td in training_datasets)
        group_of_global_id = torch.full((self.num_classes,), -1, dtype=torch.long)
        local_index_of_global_id = torch.full((self.num_classes,), -1, dtype=torch.long)
        group_global_ids = torch.full(
            (self.num_groups, max_group_size), -1, dtype=torch.long
        )
        group_sizes = torch.zeros(self.num_groups, dtype=torch.long)
        target_allowed_positions = torch.zeros(
            (self.num_classes, max_group_size), dtype=torch.bool
        )
        for group_idx, td in enumerate(training_datasets):
            global_ids = [int(global_id) for global_id in td["global_ids"]]
            group_size = len(global_ids)
            group_sizes[group_idx] = group_size
            group_global_ids[group_idx, :group_size] = torch.tensor(global_ids)
            for local_idx, global_id in enumerate(global_ids):
                if group_of_global_id[global_id] >= 0:
                    raise ValueError(
                        f"global class id {global_id} belongs to multiple "
                        "training groups"
                    )
                group_of_global_id[global_id] = group_idx
                local_index_of_global_id[global_id] = local_idx
                target_allowed_positions[global_id, :group_size] = True

            local_by_global = {
                global_id: local_idx for local_idx, global_id in enumerate(global_ids)
            }
            for target_str, conflict_ids in td.get("conflicts", {}).items():
                target_id = int(target_str)
                if target_id not in local_by_global:
                    raise ValueError(
                        f"conflict target {target_id} is outside training group "
                        f"{td['name']}"
                    )
                for conflict_id in conflict_ids:
                    conflict_id = int(conflict_id)
                    if conflict_id not in local_by_global:
                        raise ValueError(
                            f"conflict class {conflict_id} is outside training group "
                            f"{td['name']}"
                        )
                    target_allowed_positions[
                        target_id, local_by_global[conflict_id]
                    ] = False
        if (group_of_global_id < 0).any():
            missing = int((group_of_global_id < 0).sum())
            raise ValueError(
                f"{missing} global class ids are not covered by any training dataset "
                "group in class_mapping.json"
            )
        if (local_index_of_global_id < 0).any():
            raise ValueError("some global class ids have no group-local target index")

        self.register_buffer("group_of_global_id", group_of_global_id, persistent=False)
        self.register_buffer(
            "local_index_of_global_id", local_index_of_global_id, persistent=False
        )
        self.register_buffer("group_global_ids", group_global_ids, persistent=False)
        self.register_buffer("group_sizes", group_sizes, persistent=False)
        self.register_buffer(
            "target_allowed_positions", target_allowed_positions, persistent=False
        )

    # ------------------------------------------------------------------
    # Per-patch pooling of encoder tokens
    # ------------------------------------------------------------------
    def pool_patches(self, latent: TokensAndMasks) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the online-encoder tokens to one vector per spatial patch.

        Averages the tokens that were actually seen by the online encoder
        (``MaskValue.ONLINE_ENCODER``) over modality, timestep and band-set for each
        ``(batch, patch_row, patch_col)``.

        Returns:
            pooled: ``(B, P_H, P_W, D)`` pooled representation.
            valid: ``(B, P_H, P_W)`` bool mask, ``True`` where at least one token was
                pooled.
        """
        p_h, p_w = self._reference_grid(latent)

        pooled_sum: torch.Tensor | None = None
        pooled_cnt: torch.Tensor | None = None
        for modality in latent.modalities:
            tokens = getattr(latent, modality)
            mask = getattr(latent, latent.get_masked_modality_name(modality))
            if tokens is None or mask is None:
                continue
            # Spatial modalities have shape (B, P_H, P_W, T, BandSets, D). Skip
            # non-spatial (era5, latlon) and modalities on a different token grid.
            if tokens.dim() != 6:
                continue
            if tokens.shape[1] != p_h or tokens.shape[2] != p_w:
                continue
            visible = (mask == MaskValue.ONLINE_ENCODER.value).to(tokens.dtype)
            # sum over T and BandSets -> (B, P_H, P_W, D) and (B, P_H, P_W)
            weighted = tokens * visible.unsqueeze(-1)
            mod_sum = weighted.sum(dim=(3, 4))
            mod_cnt = visible.sum(dim=(3, 4))
            if pooled_sum is None:
                pooled_sum = mod_sum
                pooled_cnt = mod_cnt
            else:
                pooled_sum = pooled_sum + mod_sum
                pooled_cnt = pooled_cnt + mod_cnt

        if pooled_sum is None or pooled_cnt is None:
            raise ValueError("No spatial modality found in encoder output for pooling")

        valid = pooled_cnt > 0
        pooled = pooled_sum / pooled_cnt.clamp(min=1.0).unsqueeze(-1)
        return pooled, valid

    @staticmethod
    def _reference_grid(latent: TokensAndMasks) -> tuple[int, int]:
        """Return the (P_H, P_W) token grid of the first spatial modality."""
        for modality in latent.modalities:
            tokens = getattr(latent, modality)
            if tokens is not None and tokens.dim() == 6:
                return int(tokens.shape[1]), int(tokens.shape[2])
        raise ValueError("No spatial modality found in encoder output")

    # ------------------------------------------------------------------
    # Label pooling (pixels -> patches)
    # ------------------------------------------------------------------
    @staticmethod
    def _blockify(label: torch.Tensor, p_h: int, p_w: int) -> torch.Tensor:
        """Reshape a per-pixel label ``(B, H, W)`` into ``(B, P_H, P_W, block)``."""
        b, h, w = label.shape
        if h % p_h != 0 or w % p_w != 0:
            raise ValueError(
                f"label spatial size ({h}, {w}) not divisible by token grid "
                f"({p_h}, {p_w})"
            )
        block_h, block_w = h // p_h, w // p_w
        blocks = rearrange(
            label,
            "b (ph bh) (pw bw) -> b ph pw (bh bw)",
            ph=p_h,
            pw=p_w,
            bh=block_h,
            bw=block_w,
        )
        return blocks

    def pool_classification_labels(
        self, open_set: torch.Tensor, p_h: int, p_w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the ``open_set`` label to per-patch class ids by majority vote.

        Args:
            open_set: ``(B, H, W, 1, 1)`` per-pixel global class ids (float tensor;
                nodata ``65535`` / missing ``-99999`` are ignored).
            p_h: Target patch-grid height.
            p_w: Target patch-grid width.

        Returns:
            target: ``(B, P_H, P_W)`` long tensor of the majority global class id
                (0 where the patch has no valid pixel; use ``valid`` to mask).
            valid: ``(B, P_H, P_W)`` bool mask, ``True`` where the patch has >=1 valid
                pixel.
        """
        label = open_set.squeeze(-1).squeeze(-1)  # (B, H, W)
        blocks = self._blockify(label, p_h, p_w)  # (B, P_H, P_W, block)
        b, _, _, block = blocks.shape
        n = b * p_h * p_w
        flat = blocks.reshape(n, block)

        ids = flat.round().to(torch.long)
        valid_pix = (ids >= 0) & (ids < self.num_classes)
        patch_idx = torch.arange(n, device=flat.device).unsqueeze(1).expand(-1, block)
        valid_ids = ids[valid_pix]
        valid_patch_idx = patch_idx[valid_pix]

        target = torch.zeros(n, dtype=torch.long, device=flat.device)
        valid = torch.zeros(n, dtype=torch.bool, device=flat.device)
        if valid_ids.numel() > 0:
            # Count only observed (patch, class) pairs. This avoids allocating a dense
            # num_patches x num_global_classes histogram for sparse open-set labels.
            pair_keys = valid_patch_idx * self.num_classes + valid_ids
            unique_keys, pair_counts = torch.unique(
                pair_keys, sorted=True, return_counts=True
            )
            pair_patch_idx = torch.div(
                unique_keys, self.num_classes, rounding_mode="floor"
            )
            pair_class_id = unique_keys.remainder(self.num_classes)

            max_counts = torch.zeros(n, dtype=pair_counts.dtype, device=flat.device)
            max_counts.scatter_reduce_(
                0, pair_patch_idx, pair_counts, reduce="amax", include_self=False
            )
            winners = pair_counts == max_counts[pair_patch_idx]
            winner_ids = torch.where(
                winners,
                pair_class_id,
                torch.full_like(pair_class_id, self.num_classes),
            )
            target.fill_(self.num_classes)
            target.scatter_reduce_(
                0, pair_patch_idx, winner_ids, reduce="amin", include_self=True
            )
            valid = max_counts > 0
            target.masked_fill_(~valid, 0)

        target = target.reshape(b, p_h, p_w)
        valid = valid.reshape(b, p_h, p_w)
        return target, valid

    def pool_regression_labels(
        self, open_set_regression: torch.Tensor, p_h: int, p_w: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool the ``open_set_regression`` label to per-patch targets.

        Args:
            open_set_regression: ``(B, H, W, 1, 2)``; band 0 = 1-based dataset id
                (0 = nodata), band 1 = value remapped to ``[1, 65535]`` (0 = nodata).
            p_h: Target patch-grid height.
            p_w: Target patch-grid width.

        Returns:
            dataset_idx: ``(B, P_H, P_W)`` long tensor, 0-based regression dataset
                index (0 where invalid; use ``valid`` to mask).
            target: ``(B, P_H, P_W)`` float tensor, value mapped to ``[0, 1]``.
            valid: ``(B, P_H, P_W)`` bool mask.
        """
        reg = open_set_regression.squeeze(-2)  # (B, H, W, 2)
        dataset_id = reg[..., 0]  # (B, H, W)
        value = reg[..., 1]  # (B, H, W)

        id_blocks = self._blockify(dataset_id, p_h, p_w)  # (B,P_H,P_W,block)
        val_blocks = self._blockify(value, p_h, p_w)

        id_round = id_blocks.round().to(torch.long)
        valid_pix = (id_round >= 1) & (id_round <= self.num_reg_datasets)
        if self.num_reg_datasets > 0:
            safe_dataset_idx = (id_round - 1).clamp(
                min=0, max=self.num_reg_datasets - 1
            )
            valid_pix = valid_pix & self.valid_regression_datasets[safe_dataset_idx]
        valid_pix = valid_pix & (val_blocks >= REGRESSION_VALUE_MIN_OUT)

        pix_count = valid_pix.sum(dim=-1)  # (B,P_H,P_W)
        valid = pix_count > 0

        # Each open-set window comes from a single dataset, so any valid pixel's id is
        # the patch id. Take the max id over valid pixels (0 elsewhere).
        masked_id = torch.where(valid_pix, id_round, torch.zeros_like(id_round))
        patch_id = masked_id.amax(dim=-1)  # (B,P_H,P_W), 1-based (0 = invalid)
        dataset_idx = (patch_id - 1).clamp(min=0)

        masked_val = torch.where(valid_pix, val_blocks, torch.zeros_like(val_blocks))
        value_sum = masked_val.sum(dim=-1)
        value_mean = value_sum / pix_count.clamp(min=1).to(value_sum.dtype)
        # Map [min_out, max_out] -> [0, 1].
        span = max(self.reg_value_max_out - self.reg_value_min_out, 1.0)
        target = (value_mean - self.reg_value_min_out) / span
        return dataset_idx, target, valid

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def classification_loss(
        self,
        pooled: torch.Tensor,
        repr_valid: torch.Tensor,
        open_set: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Masked-softmax cross-entropy over each patch's source-dataset classes.

        Returns the (unweighted) mean CE loss and the number of contributing patches.
        """
        p_h, p_w = pooled.shape[1], pooled.shape[2]
        target, label_valid = self.pool_classification_labels(open_set, p_h, p_w)
        keep = repr_valid & label_valid  # (B,P_H,P_W)
        n_keep = int(keep.sum())
        if n_keep == 0:
            return pooled.new_zeros(()), 0

        pooled_keep = pooled[keep]  # (n_keep, D)
        target_keep = target[keep]  # (n_keep,)
        groups = self.group_of_global_id[target_keep]

        loss_sum = pooled.new_zeros(())
        for group_idx in torch.unique(groups).tolist():
            group_keep = groups == group_idx
            group_targets = target_keep[group_keep]
            group_size = int(self.group_sizes[group_idx])
            class_ids = self.group_global_ids[group_idx, :group_size]
            logits = F.linear(
                pooled_keep[group_keep],
                self.cls_head.weight[class_ids],
                self.cls_head.bias[class_ids],
            )
            allowed = self.target_allowed_positions[group_targets, :group_size]
            logits = logits.masked_fill(~allowed, float("-inf"))
            local_targets = self.local_index_of_global_id[group_targets]
            loss_sum = loss_sum + F.cross_entropy(
                logits, local_targets, reduction="sum"
            )
        return loss_sum / n_keep, n_keep

    def regression_loss(
        self,
        pooled: torch.Tensor,
        repr_valid: torch.Tensor,
        open_set_regression: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Per-dataset MSE against the value mapped to ``[0, 1]``.

        Returns the (unweighted) mean MSE loss and the number of contributing patches.
        """
        p_h, p_w = pooled.shape[1], pooled.shape[2]
        dataset_idx, target, label_valid = self.pool_regression_labels(
            open_set_regression, p_h, p_w
        )
        keep = repr_valid & label_valid
        n_keep = int(keep.sum())
        if n_keep == 0:
            return pooled.new_zeros(()), 0

        pooled_keep = pooled[keep]  # (n_keep, D)
        idx_keep = dataset_idx[keep]  # (n_keep,)
        target_keep = target[keep]  # (n_keep,)
        preds = self.reg_head(pooled_keep)  # (n_keep, num_reg_datasets)
        pred = preds.gather(1, idx_keep.unsqueeze(1)).squeeze(1)  # (n_keep,)
        loss = F.mse_loss(pred, target_keep)
        return loss, n_keep

    def zero_touch(self) -> torch.Tensor:
        """A ``0 * sum(params)`` term to keep probe params in the autograd graph.

        Under the DDP path the per-step gradient all-reduce flattens only params whose
        ``.grad`` is not ``None``; if some ranks have no labeled patches this term
        guarantees every probe param still receives a (zero) gradient, keeping the
        flattened buffers identical across ranks.
        """
        total = self.cls_head.weight.sum() + self.cls_head.bias.sum()
        total = total + self.reg_head.weight.sum() + self.reg_head.bias.sum()
        return 0.0 * total

    def forward(
        self, latent: TokensAndMasks, batch: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined weighted supervised loss for one encoder output.

        Args:
            latent: Online-encoder ``TokensAndMasks`` for this view.
            batch: The ``MaskedOlmoEarthSample`` for this view (carries the label
                fields ``open_set`` / ``open_set_regression``).

        Returns:
            loss: Weighted CE + MSE (+ zero-touch), always connected to the probe
                params so gradients are well-defined on every rank.
            metrics: Detached scalar metrics for logging.
        """
        pooled, repr_valid = self.pool_patches(latent)
        loss = self.zero_touch()
        metrics: dict[str, float] = {}

        open_set = getattr(batch, Modality.OPEN_SET.name, None)
        if open_set is not None:
            ce, n_ce = self.classification_loss(pooled, repr_valid, open_set)
            loss = loss + self.seg_loss_weight * ce
            metrics["open_set_ce"] = float(ce.detach())
            metrics["open_set_ce_patches"] = float(n_ce)

        open_set_regression = getattr(batch, Modality.OPEN_SET_REGRESSION.name, None)
        if open_set_regression is not None:
            mse, n_mse = self.regression_loss(pooled, repr_valid, open_set_regression)
            loss = loss + self.reg_loss_weight * mse
            metrics["open_set_mse"] = float(mse.detach())
            metrics["open_set_mse_patches"] = float(n_mse)

        return loss, metrics
