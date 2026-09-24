"""Supervised open-set probe head for pretraining.

This module adds *supervised* segmentation + regression signal on top of the
self-supervised latent-MIM objective, driven by the ``open_set`` /
``open_set_regression`` label layers (built by
``olmoearth_pretrain.open_set_segmentation_data``).

The probe maps the encoder's *spatial latent grid* (the Perceiver/register
bottleneck output, where time / modality / band-set are already collapsed into one
embedding per spatial cell) to per-class logits (classification) and per-dataset
scalars (regression). Three head variants (``head_type``):

* ``linear``: one learned vector per global class id (``num_classes x D``) and per
  regression dataset (``num_reg_datasets x D``) directly on the grid;
* ``mlp``: a shared ``Linear -> GELU`` trunk (first layer shared across all classes
  and datasets) followed by the same linear heads;
* ``text``: a shared ``Linear -> GELU -> Linear`` trunk whose output is scored by
  scaled cosine similarity against frozen class *text embeddings* (CLIP-style), so
  classes that share a concept across datasets share a target; regression keeps a
  linear head on the trunk output.

Per spatial latent cell we run the head against the pooled label block and compute:

* **cross-entropy** for classification, with a *masked softmax* restricted to the
  source dataset's class subset (each open-set window comes from a single source
  dataset, so negatives only come from within that dataset / the merged
  presence-only group);
* **mean-squared error** for regression, against the stored value mapped to
  ``[0, 1]``.

Per-patch losses are averaged *within each sample* first and then across labeled
samples, so densely labeled samples (e.g. wall-to-wall land-cover maps) carry the
same weight as sparsely labeled ones (e.g. point/polygon datasets). Optionally
(``dataset_balance="dataset_temperature"``) each sample is further weighted by its
source dataset's size ``n_d ** (tau - 1)``, emulating temperature sampling over
datasets without changing the data loader.

Paired pre/post **change** samples (those with an ``open_set_change_boundary``) are
only supervised when the online encoder actually saw at least one timestep on each
side of the boundary; otherwise the register grid cannot encode the change and the
sample's labels are dropped for that step.

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.change_boundary import (
    has_change_boundary,
    timestep_is_post,
)
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue

logger = logging.getLogger(__name__)

# Sentinels from the open-set label layers (see
# open_set_segmentation_data.pretrain_constants). Duplicated here to avoid a train-time
# dependency on the dataset-creation package.
OPEN_SET_NODATA = 65535
REGRESSION_DATASET_ID_NODATA = 0
REGRESSION_VALUE_NODATA = 0
REGRESSION_VALUE_MIN_OUT = 1
REGRESSION_VALUE_MAX_OUT = 65535


HEAD_TYPES = ("linear", "mlp", "text")
DATASET_BALANCE_MODES = ("sample", "dataset_temperature")

# Default per-dataset sample counts: the label-bank registry (``num_samples`` per slug).
DEFAULT_DATASET_COUNTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "open_set_segmentation_data"
    / "registry.json"
)


def load_dataset_counts(path: str | Path) -> dict[str, int]:
    """Read ``{slug: num_samples}`` from the label-bank ``registry.json``."""
    with Path(path).open() as f:
        registry = json.load(f)
    counts: dict[str, int] = {}
    for entry in registry["datasets"]:
        num_samples = entry.get("num_samples")
        if num_samples is None or not str(num_samples).isdigit():
            continue
        counts[entry["slug"]] = int(num_samples)
    return counts


def load_text_embeddings(
    path: str | Path, expected_mapping_sha256: str, num_classes: int
) -> torch.Tensor:
    """Load the frozen class text embeddings written by ``embed_class_names``.

    ``path`` is the ``.npy``; its ``.json`` sidecar must record the sha256 of the
    class mapping the rows were generated for, which must match the mapping being
    trained against (row = global class id).
    """
    npy_path = Path(path)
    sidecar_path = npy_path.with_suffix(".json")
    with sidecar_path.open() as f:
        sidecar = json.load(f)
    if sidecar.get("class_mapping_sha256") != expected_mapping_sha256:
        raise ValueError(
            f"text embeddings {npy_path} were generated for class mapping "
            f"{sidecar.get('class_mapping_sha256')}, expected {expected_mapping_sha256}"
        )
    embeddings = torch.from_numpy(np.load(npy_path).astype(np.float32))
    if embeddings.ndim != 2 or embeddings.shape[0] != num_classes:
        raise ValueError(
            f"text embeddings {npy_path} have shape {tuple(embeddings.shape)}, expected "
            f"({num_classes}, dim)"
        )
    return embeddings


@dataclass
class OpenSetProbeConfig(Config):
    """Configuration for :class:`OpenSetProbe`.

    Args:
        class_mapping_path: Path to ``class_mapping.json`` (produced by
            ``open_set_segmentation_data.assemble_classes``).
        seg_loss_weight: Relative weight of the classification (CE) term.
        reg_loss_weight: Relative weight of the regression (MSE) term.
        head_type: ``"linear"`` (per-class weight rows on the register grid),
            ``"mlp"`` (a shared ``Linear -> GELU`` trunk before the linear heads) or
            ``"text"`` (a shared ``Linear -> GELU -> Linear`` trunk whose output is
            matched by cosine similarity against frozen class text embeddings,
            CLIP-style; regression keeps a linear head on the trunk output).
        mlp_hidden_size: Hidden width of the shared trunk (``mlp`` / ``text``);
            None uses the register dim.
        text_embeddings_path: ``.npy`` of L2-normalizable class text embeddings
            (row = global id) with a ``.json`` sidecar; required for ``text``.
        text_logit_scale_init: Initial (learnable) logit scale of the cosine
            logits, as in CLIP (``1 / 0.07``).
        dataset_balance: ``"sample"`` weights every labeled sample equally;
            ``"dataset_temperature"`` reweights each sample by its source dataset
            size ``n_d ** (tau - 1)`` (normalized to mean 1 under natural sampling),
            which emulates temperature sampling ``p_d ~ n_d ** tau``.
        balance_temperature: The ``tau`` above.
        dataset_counts_path: ``registry.json`` with per-slug ``num_samples``;
            None uses the checked-in label-bank registry.
    """

    class_mapping_path: str
    expected_class_mapping_sha256: str | None = None
    seg_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    head_type: str = "linear"
    mlp_hidden_size: int | None = None
    text_embeddings_path: str | None = None
    text_logit_scale_init: float = 1.0 / 0.07
    dataset_balance: str = "sample"
    balance_temperature: float = 0.5
    dataset_counts_path: str | None = None

    def build(self, embedding_size: int) -> OpenSetProbe:
        """Build the probe for an encoder with the given token embedding size."""
        mapping_bytes = Path(self.class_mapping_path).read_bytes()
        actual_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
        if (
            self.expected_class_mapping_sha256 is not None
            and actual_sha256 != self.expected_class_mapping_sha256
        ):
            raise ValueError(
                "class mapping hash mismatch: expected "
                f"{self.expected_class_mapping_sha256}, got {actual_sha256}"
            )
        class_mapping = json.loads(mapping_bytes)

        text_embeddings = None
        if self.head_type == "text":
            if self.text_embeddings_path is None:
                raise ValueError("head_type='text' requires text_embeddings_path")
            text_embeddings = load_text_embeddings(
                self.text_embeddings_path,
                actual_sha256,
                int(class_mapping["open_set"]["num_classes"]),
            )

        dataset_counts = None
        if self.dataset_balance == "dataset_temperature":
            dataset_counts = load_dataset_counts(
                self.dataset_counts_path or DEFAULT_DATASET_COUNTS_PATH
            )

        return OpenSetProbe(
            embedding_size=embedding_size,
            class_mapping=class_mapping,
            seg_loss_weight=self.seg_loss_weight,
            reg_loss_weight=self.reg_loss_weight,
            head_type=self.head_type,
            mlp_hidden_size=self.mlp_hidden_size,
            text_embeddings=text_embeddings,
            text_logit_scale_init=self.text_logit_scale_init,
            dataset_counts=dataset_counts,
            balance_temperature=self.balance_temperature,
        )


class OpenSetProbe(nn.Module):
    """Supervised probe over the encoder's spatial latent grid.

    Args:
        embedding_size: The embedding size ``D`` of the spatial latent grid (the
            encoder register/bottleneck dim, e.g. 768 or 128).
        class_mapping: Parsed ``class_mapping.json`` dict.
        seg_loss_weight: Relative weight of the classification (CE) term.
        reg_loss_weight: Relative weight of the regression (MSE) term.
        head_type: See :class:`OpenSetProbeConfig`.
        mlp_hidden_size: Hidden width of the shared trunk (``mlp`` / ``text``).
        text_embeddings: ``(num_classes, E)`` frozen class text embeddings
            (``text`` head only).
        text_logit_scale_init: Initial learnable logit scale of the cosine logits.
        dataset_counts: ``{slug: num_samples}``; if given, samples are reweighted by
            their source dataset size (see :class:`OpenSetProbeConfig`).
        balance_temperature: The ``tau`` of the dataset reweighting.
    """

    def __init__(
        self,
        embedding_size: int,
        class_mapping: dict[str, Any],
        seg_loss_weight: float = 1.0,
        reg_loss_weight: float = 1.0,
        head_type: str = "linear",
        mlp_hidden_size: int | None = None,
        text_embeddings: torch.Tensor | None = None,
        text_logit_scale_init: float = 1.0 / 0.07,
        dataset_counts: dict[str, int] | None = None,
        balance_temperature: float = 0.5,
    ):
        """Initialize the probe and the class-subset lookup buffers."""
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type}")
        self.embedding_size = embedding_size
        self.seg_loss_weight = seg_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.head_type = head_type

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

        # Shared trunk (identity for the linear probe), then one head per task. The
        # trunk's first layer is shared across all classes and datasets.
        hidden = mlp_hidden_size or embedding_size
        self.cls_head: nn.Linear | None
        self.logit_scale: nn.Parameter | None
        if head_type == "linear":
            self.trunk: nn.Module = nn.Identity()
            feature_size = embedding_size
        elif head_type == "mlp":
            self.trunk = nn.Sequential(nn.Linear(embedding_size, hidden), nn.GELU())
            feature_size = hidden
        else:
            if text_embeddings is None:
                raise ValueError("head_type='text' requires text_embeddings")
            if text_embeddings.shape[0] != self.num_classes:
                raise ValueError(
                    f"text_embeddings has {text_embeddings.shape[0]} rows, expected "
                    f"{self.num_classes}"
                )
            feature_size = int(text_embeddings.shape[1])
            self.trunk = nn.Sequential(
                nn.Linear(embedding_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, feature_size),
            )
            self.register_buffer(
                "text_embeddings",
                F.normalize(text_embeddings.float(), dim=-1),
                persistent=False,
            )
        if head_type == "text":
            # CLIP-style cosine logits against the frozen text embeddings; the class
            # "weights" are the embeddings, so there is no per-class parameter.
            self.cls_head = None
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(text_logit_scale_init))
            )
        else:
            self.cls_head = nn.Linear(feature_size, self.num_classes)
            self.logit_scale = None
        self.reg_head = nn.Linear(feature_size, max(self.num_reg_datasets, 1))

        # Per-class / per-regression-dataset sample weights for dataset balancing
        # (all ones when dataset_counts is None).
        class_balance_weight, reg_balance_weight = self._dataset_balance_weights(
            class_mapping, dataset_counts, balance_temperature
        )
        self.register_buffer(
            "class_balance_weight", class_balance_weight, persistent=False
        )
        self.register_buffer("reg_balance_weight", reg_balance_weight, persistent=False)

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
    # Dataset balancing
    # ------------------------------------------------------------------
    @staticmethod
    def _dataset_balance_weights(
        class_mapping: dict[str, Any],
        dataset_counts: dict[str, int] | None,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-class and per-regression-dataset weights ``w_d = Z * n_d ** (tau - 1)``.

        ``n_d`` is the source dataset's sample count (a merged presence-only class
        counts every member dataset). ``Z`` normalizes so the expected weight under
        natural sampling is one: ``sum_d n_d * w_d / sum_d n_d == 1``. Weighting each
        sample's loss this way is equivalent, in expectation, to drawing datasets
        with probability ``p_d ~ n_d ** tau`` (``tau = 1`` natural, ``tau = 0``
        uniform over datasets). Returns all ones when ``dataset_counts`` is None.
        """
        num_classes = int(class_mapping["open_set"]["num_classes"])
        reg_datasets = class_mapping["open_set_regression"]["datasets"]
        class_weight = torch.ones(num_classes)
        reg_weight = torch.ones(max(len(reg_datasets), 1))
        if dataset_counts is None:
            return class_weight, reg_weight
        counts_by_slug = dataset_counts
        classes = class_mapping["open_set"]["classes"]

        def _count(slug: str) -> float:
            if slug not in counts_by_slug:
                raise ValueError(f"no sample count for dataset {slug!r}")
            return float(counts_by_slug[slug])

        class_slugs = {
            m["slug"]
            for c in classes
            for m in (c.get("members") or [{"slug": c["slug"]}])
        }
        reg_slugs = {d["slug"] for d in reg_datasets}
        slugs = sorted(class_slugs | reg_slugs)
        counts = torch.tensor([_count(s) for s in slugs], dtype=torch.float64)
        # Z = N / sum_d n_d^tau  =>  sum_d (n_d / N) * Z * n_d^(tau-1) = 1.
        normalizer = counts.sum() / counts.pow(temperature).sum()

        def _weight(n: float) -> float:
            return float(normalizer * n ** (temperature - 1.0))

        for c in classes:
            members = c.get("members") or [{"slug": c["slug"]}]
            n = sum(_count(m["slug"]) for m in members)
            class_weight[int(c["global_id"])] = _weight(n)
        for idx, d in enumerate(reg_datasets):
            reg_weight[idx] = _weight(_count(d["slug"]))
        return class_weight, reg_weight

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
    @staticmethod
    def _sample_mean(
        per_patch: torch.Tensor, keep: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Average per-patch losses within each sample, then across labeled samples.

        Args:
            per_patch: ``(n_keep,)`` per-patch losses, ordered like ``keep.nonzero()``.
            keep: ``(B, P_H, P_W)`` bool mask of the contributing patches.

        Returns:
            The mean-of-per-sample-means loss and the number of labeled samples.
        """
        b = keep.shape[0]
        sample_idx = (
            torch.arange(b, device=keep.device).view(b, 1, 1).expand_as(keep)[keep]
        )
        loss_per_sample = per_patch.new_zeros(b).index_add(0, sample_idx, per_patch)
        count_per_sample = torch.bincount(sample_idx, minlength=b).to(per_patch.dtype)
        labeled = count_per_sample > 0
        n_samples = int(labeled.sum())
        loss = (loss_per_sample[labeled] / count_per_sample[labeled]).mean()
        return loss, n_samples

    def classification_loss(
        self,
        pooled: torch.Tensor,
        repr_valid: torch.Tensor,
        open_set: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Masked-softmax cross-entropy over each patch's source-dataset classes.

        Per-patch CE is averaged within each sample and then across labeled samples,
        so densely and sparsely labeled samples contribute equally.

        Returns the (unweighted) loss, the number of contributing samples, and the
        number of contributing patches.
        """
        p_h, p_w = pooled.shape[1], pooled.shape[2]
        target, label_valid = self.pool_classification_labels(open_set, p_h, p_w)
        keep = repr_valid & label_valid  # (B,P_H,P_W)
        n_keep = int(keep.sum())
        if n_keep == 0:
            return pooled.new_zeros(()), 0, 0

        pooled_keep = pooled[keep]  # (n_keep, D)
        target_keep = target[keep]  # (n_keep,)
        groups = self.group_of_global_id[target_keep]

        per_patch = torch.zeros(n_keep, dtype=torch.float32, device=pooled.device)
        for group_idx in torch.unique(groups).tolist():
            group_keep = groups == group_idx
            group_targets = target_keep[group_keep]
            group_size = int(self.group_sizes[group_idx])
            class_ids = self.group_global_ids[group_idx, :group_size]
            logits = self._class_logits(pooled_keep[group_keep], class_ids)
            allowed = self.target_allowed_positions[group_targets, :group_size]
            logits = logits.masked_fill(~allowed, float("-inf"))
            local_targets = self.local_index_of_global_id[group_targets]
            per_patch[group_keep] = F.cross_entropy(
                logits, local_targets, reduction="none"
            ).float()
        per_patch = per_patch * self.class_balance_weight[target_keep]
        loss, n_samples = self._sample_mean(per_patch, keep)
        return loss, n_samples, n_keep

    def _class_logits(
        self, features: torch.Tensor, class_ids: torch.Tensor
    ) -> torch.Tensor:
        """Logits of ``features`` ``(N, F)`` against the given global classes.

        Linear / MLP heads project against the learned rows of ``cls_head``; the text
        head uses scaled cosine similarity against the frozen text embeddings.
        """
        if self.head_type == "text":
            assert self.logit_scale is not None
            text = self.text_embeddings[class_ids]  # (C, E), unit norm
            return self.logit_scale.exp() * F.normalize(features, dim=-1) @ text.T
        assert self.cls_head is not None
        return F.linear(
            features, self.cls_head.weight[class_ids], self.cls_head.bias[class_ids]
        )

    def regression_loss(
        self,
        pooled: torch.Tensor,
        repr_valid: torch.Tensor,
        open_set_regression: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Per-dataset MSE against the value mapped to ``[0, 1]``.

        Per-patch MSE is averaged within each sample and then across labeled samples,
        so densely and sparsely labeled samples contribute equally.

        Returns the (unweighted) loss, the number of contributing samples, and the
        number of contributing patches.
        """
        p_h, p_w = pooled.shape[1], pooled.shape[2]
        dataset_idx, target, label_valid = self.pool_regression_labels(
            open_set_regression, p_h, p_w
        )
        keep = repr_valid & label_valid
        n_keep = int(keep.sum())
        if n_keep == 0:
            return pooled.new_zeros(()), 0, 0

        pooled_keep = pooled[keep]  # (n_keep, D)
        idx_keep = dataset_idx[keep]  # (n_keep,)
        target_keep = target[keep]  # (n_keep,)
        preds = self.reg_head(pooled_keep)  # (n_keep, num_reg_datasets)
        pred = preds.gather(1, idx_keep.unsqueeze(1)).squeeze(1)  # (n_keep,)
        per_patch = F.mse_loss(pred.float(), target_keep.float(), reduction="none")
        per_patch = per_patch * self.reg_balance_weight[idx_keep]
        loss, n_samples = self._sample_mean(per_patch, keep)
        return loss, n_samples, n_keep

    def zero_touch(self) -> torch.Tensor:
        """A ``0 * sum(params)`` term to keep probe params in the autograd graph.

        Under the DDP path the per-step gradient all-reduce flattens only params whose
        ``.grad`` is not ``None``; if some ranks have no labeled patches this term
        guarantees every probe param still receives a (zero) gradient, keeping the
        flattened buffers identical across ranks.
        """
        total = sum(p.sum() for p in self.parameters())
        return 0.0 * total

    @staticmethod
    def change_samples_with_both_sides_visible(batch: Any) -> torch.Tensor | None:
        """Per-sample flag: is the sample supervisable given what the encoder saw?

        Non-change samples (no ``open_set_change_boundary``, or missing-filled) are
        always supervisable. A change sample is supervisable only if at least one
        pre-boundary AND one post-boundary timestep of some spatial modality was
        visible to the online encoder (``MaskValue.ONLINE_ENCODER``); otherwise the
        register grid has no way to encode the change.

        Returns:
            ``(B,)`` bool tensor, or None when the batch carries no boundary field.
        """
        boundary = getattr(batch, Modality.OPEN_SET_CHANGE_BOUNDARY.name, None)
        timestamps = getattr(batch, "timestamps", None)
        if boundary is None or timestamps is None:
            return None
        batch_size = timestamps.shape[0]
        is_change = has_change_boundary(boundary).to(timestamps.device)
        if not bool(is_change.any()):
            return torch.ones(batch_size, dtype=torch.bool, device=timestamps.device)

        # Per-timestep visibility over all spacetime modalities the encoder consumed.
        visible_t = torch.zeros(
            (batch_size, timestamps.shape[1]),
            dtype=torch.bool,
            device=timestamps.device,
        )
        for modality in batch.modalities:
            spec = Modality.get(modality)
            if not spec.is_spacetime_varying:
                continue
            mask = getattr(batch, batch.get_masked_modality_name(modality), None)
            if mask is None:
                continue
            # (B, H, W, T, BandSets) -> (B, T)
            visible_t |= (mask == MaskValue.ONLINE_ENCODER.value).any(dim=(1, 2, 4))

        is_post = timestep_is_post(timestamps, boundary)
        saw_pre = (visible_t & ~is_post).any(dim=1)
        saw_post = (visible_t & is_post).any(dim=1)
        return ~is_change | (saw_pre & saw_post)

    def forward(
        self, spatial_latent: torch.Tensor, batch: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the supervised loss from the encoder's spatial latent grid.

        Args:
            spatial_latent: ``(B, P_H, P_W, D)`` spatial latent embeddings (the
                encoder's register/Perceiver bottleneck grid, with time / modality
                already collapsed).
            batch: The ``MaskedOlmoEarthSample`` for this view (carries the label
                fields ``open_set`` / ``open_set_regression`` and, for paired change
                samples, ``open_set_change_boundary``).

        Returns:
            loss: Weighted CE + MSE (each a per-sample mean over this rank's labeled
                samples, like the map supervision heads: DP averaging then yields the
                mean of per-rank means) plus a zero-touch term that keeps probe
                gradients well-defined on every rank.
            metrics: Detached scalar metrics for logging (``open_set_ce`` /
                ``open_set_mse`` only when at least one sample contributed, plus the
                ``*_samples`` / ``*_patches`` counts).
        """
        if spatial_latent.dim() != 4:
            raise ValueError(
                "spatial_latent must have shape (B, P_H, P_W, D), got "
                f"{tuple(spatial_latent.shape)}"
            )
        # Shared trunk (identity for the linear probe), then per-task heads.
        pooled = self.trunk(spatial_latent)
        # Every spatial latent cell attends over the full input, so all cells carry
        # a valid representation; validity is governed by the labels, except for
        # change samples whose pre or post side the encoder never saw.
        repr_valid = torch.ones(
            pooled.shape[:3], dtype=torch.bool, device=pooled.device
        )
        sample_valid = self.change_samples_with_both_sides_visible(batch)
        if sample_valid is not None:
            repr_valid &= sample_valid.to(pooled.device).view(-1, 1, 1)
        loss = self.zero_touch()
        metrics: dict[str, float] = {}

        open_set = getattr(batch, Modality.OPEN_SET.name, None)
        if open_set is not None:
            ce, ce_samples, ce_patches = self.classification_loss(
                pooled, repr_valid, open_set
            )
            loss = loss + self.seg_loss_weight * ce
            # Only report the value when something contributed, so the per-step
            # average over microbatches is not diluted by unlabeled ones.
            if ce_samples > 0:
                metrics["open_set_ce"] = float(ce.detach())
            metrics["open_set_ce_samples"] = float(ce_samples)
            metrics["open_set_ce_patches"] = float(ce_patches)

        open_set_regression = getattr(batch, Modality.OPEN_SET_REGRESSION.name, None)
        if open_set_regression is not None:
            mse, mse_samples, mse_patches = self.regression_loss(
                pooled, repr_valid, open_set_regression
            )
            loss = loss + self.reg_loss_weight * mse
            if mse_samples > 0:
                metrics["open_set_mse"] = float(mse.detach())
            metrics["open_set_mse_samples"] = float(mse_samples)
            metrics["open_set_mse_patches"] = float(mse_patches)

        return loss, metrics
