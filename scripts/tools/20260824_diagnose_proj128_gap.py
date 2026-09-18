#!/usr/bin/env python3
"""Decompose the d768-teacher vs 128d-student PASTIS gap of the proj128lin runs.

The proj128lin runs ship a d768 register grid (teacher) and a DETACHED
``Linear(768, 128)`` student trained by distillation alone. On the frozen ps=1
PASTIS probe the student trails the teacher by ~8 mIoU, and the W&B curves show
that gap is FLAT from 40k to 640k steps -- so it is not an optimization lag.
This script asks which of four candidate causes actually holds, using ONE
embedding extraction:

1. **int8 quantization.** The ps=1 tasks probe AEF-power-quantized embeddings.
   Run every rung with and without the round-trip to price it per width.
2. **The 128-dim budget.** PASTIS has ~20 classes and the probe is linear, so at
   most ~19 directions of the embedding can matter -- 128 dims should be far more
   than enough. ``pca{k}`` sweeps the width of a variance-optimal linear
   compression of the SAME teacher to measure what width actually costs.
3. **Which directions the student keeps.** ``student`` vs ``pca128`` vs
   ``fisher128`` puts three different 128-d linear compressions of one teacher
   through the same probe: distillation-optimal, variance-optimal, and
   class-optimal. ``fisher128`` is the constructive ceiling -- if it matches the
   teacher, the width is innocent and the objective is the whole story.
4. **Geometry.** ``geometry`` reports, with no probe at all, how much of the
   teacher's between-class scatter each compression's row space retains.

Because the student is exactly ``X W^T + b`` on the teacher registers, every rung
is a linear map of one cached tensor: extract once, then probe as many
compressions as you like.

Usage (needs /weka mounted and one GPU)::

    python scripts/tools/20260824_diagnose_proj128_gap.py extract
    python scripts/tools/20260824_diagnose_proj128_gap.py geometry
    python scripts/tools/20260824_diagnose_proj128_gap.py analyze --rungs teacher,student,pca128

Reading the output:

* ``student`` << ``pca128`` ~ ``teacher``  -> distillation keeps the wrong
  directions; fix the objective (supervise the student, whiten the target, or
  distill a decorrelated teacher basis).
* ``student`` ~ ``pca128`` << ``teacher``  -> variance-ranked compression is the
  problem: the class signal lives in the teacher's low-variance tail, which
  BOTH cosine and Gram distillation weight by variance. ``fisher128`` says how
  much is recoverable at 128 dims.
* ``pca128`` ~ ``pca32`` ~ ``teacher``     -> width is irrelevant; anything that
  costs the student mIoU is a property of its particular map.
* ``*_f32`` >> quantized                   -> the int8 round-trip is eating the
  score at that width (the teacher clips ~1.6% of coordinates, the student
  ~0.9%), and the Tessera per-vector scheme in ``embedding_transforms`` is the
  cheaper fix than anything about the student.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olmo_core.config import Config  # noqa: E402
from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402

from olmoearth_pretrain.evals.balanced_trial import (  # noqa: E402
    run_balanced_trials,
)
from olmoearth_pretrain.evals.embedding_transforms import (  # noqa: E402
    dequantize_embeddings,
    quantize_embeddings,
)
from olmoearth_pretrain.internal.all_evals import (  # noqa: E402
    EMBEDDING_EVAL_TASKS,
    EVAL_TASKS,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (  # noqa: E402
    DownstreamEvaluator,
    DownstreamTaskConfig,
    EvalMode,
)

logger = logging.getLogger(__name__)

# The in-flight new-maps proj128lin run: d768 teacher + detached linear [128, 64]
# student, w1 register supervision, Landsat-reflectance h5, 3-band DSM maps.
DEFAULT_CHECKPOINT = (
    "/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/"
    "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_psuniform_newmaps_refl_dsm3/step440000"
)
# The probe the gap is measured on. S1+S2 rather than S2-only because it is the
# arm quoted in the run comparisons; both behave the same here.
DEFAULT_TASK = "pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export"
DEFAULT_CACHE = "/tmp/proj128_gap_cache"  # nosec B108

# Rows drawn from the train split to fit the PCA / Fisher bases. The bases are
# 768x768 covariances, so a few hundred thousand cells is far past enough and
# keeps the SVD in GPU memory.
BASIS_ROWS = 200_000
BASIS_SEED = 0
# Rows per chunk in the transform/quantize pass; 200k x 768 floats is ~0.6 GiB.
CHUNK_ROWS = 200_000
# Tiles kept from PASTIS's 92,672-tile train split. The full split is 68 GiB in
# float32 at the teacher's width, and every rung re-materializes it; a quarter of
# it costs a little absolute mIoU but the ladder is a comparison between rungs
# that all see exactly the same rows, and the val split stays whole (its score is
# what the run reports). A multiple of the 64-tile embedding batch.
DEFAULT_TRAIN_TILES = 24576
# Set from --transfer-cache in analyze(); read by the fisherfrom rung, which
# is built inside _build_transforms and has no access to the parsed args.
_TRANSFER_CACHE = "/var/tmp/fc_cache"  # nosec B108
# The probe steps 3k times an epoch on batches of 8 tiles, so each op is small.
# Left at the machine's core count, torch's intra-op pool spin-waits on every one
# of them -- measured at ~207 busy cores with the GPU at 0%. Cap it.
PROBE_THREADS = 8
# Headroom kept free on the GPU when deciding whether the probe's tensors fit in
# device memory; the probe itself needs almost none (batch of 8 tiles).
DEVICE_TENSOR_HEADROOM_GIB = 8.0
# Ridge added to the within-class scatter before whitening. The scatter is
# estimated from cells, which are heavily correlated within a scene, so its small
# eigenvalues are noise -- inverting them unregularized would hand the Fisher
# basis directions that do not generalize.
FISHER_RIDGE = 1e-3


def _task(name: str) -> DownstreamTaskConfig:
    """Look a task up in the embedding catalog, then the general one.

    The cross-task transfer rungs fit a basis on fifty_cities, which lives in
    EVAL_TASKS rather than EMBEDDING_EVAL_TASKS.
    """
    if name in EMBEDDING_EVAL_TASKS:
        return EMBEDDING_EVAL_TASKS[name]
    return EVAL_TASKS[name]


def _build_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Rebuild the run's model from its saved config and load the checkpoint.

    The supervision head is dropped before building: it is dead weight for
    embedding extraction, and the in-flight dsm3 runs were launched from a
    checkout whose ``glo30_aspect`` supervision modality does not exist here.
    The distributed-checkpoint load tolerates the extra tensors in the file.
    """
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    model_config = dict(config["model"])
    model_config["supervision_head_config"] = None
    # Dropping the head means the config can no longer claim to supervise the
    # projection: the supstu arms set supervision_source="both", and the
    # validator rejects that without a head. Neither field affects the encoder
    # or the projection weights we extract, so neutralize them.
    model_config["supervision_source"] = "registers"
    model_config["projection_supervision_weight_scale"] = None
    model = Config.from_dict(model_config).build().to(device)
    load_model_and_optim_state(str(Path(checkpoint) / "model_and_optim"), model)
    model.eval()
    return model


def _evaluator(
    task_name: str,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int | None = None,
) -> DownstreamEvaluator:
    """A DownstreamEvaluator wired to a stub trainer.

    Only ``trainer.train_module.model`` is touched on the paths used here
    (``_get_data_loader`` / ``_get_embeddings``), so the real Trainer -- with its
    callbacks, checkpointer and W&B run -- is not needed to reproduce the eval.
    """
    task = _task(task_name)
    if patch_size is not None:
        # Cross-task transfer needs the DONOR dataset embedded in the same
        # space as the target: fifty_cities defaults to patch size 4, so a
        # basis fitted on it describes 4x4-pixel patch embeddings while
        # PASTIS ps=1 cells are single pixels. Same encoder, different input
        # distribution -- enough to sink a transferred basis on its own.
        task = replace(task, patch_size=patch_size)
    trainer = SimpleNamespace(
        train_module=SimpleNamespace(model=model), device=device, save_folder="."
    )
    return DownstreamEvaluator(task_name, task, trainer=trainer, device=device)


def extract(args: argparse.Namespace) -> None:
    """Cache the teacher's raw float embeddings for both splits, plus the student map.

    Only the TEACHER is embedded: the student is a per-cell linear map of exactly
    these vectors, so caching ``W``/``b`` alongside makes every student rung (and
    every alternative 128-d compression) a matmul instead of another forward pass.
    """
    device = torch.device(args.device)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    model = _build_model(args.checkpoint, device)
    projection = model.encoder.register_projection
    torch.save(
        {
            "weight": projection.weight.detach().cpu(),
            "bias": None if projection.bias is None else projection.bias.detach().cpu(),
            "checkpoint": args.checkpoint,
        },
        cache / "student_projection.pt",
    )

    evaluator = _evaluator(args.task, model, device, patch_size=args.patch_size)
    for split, is_train in (("train", True), ("valid", False)):
        out = cache / f"{split}.pt"
        if out.exists() and not args.overwrite:
            logger.info("%s exists, skipping (use --overwrite)", out)
            continue
        # Both splits need a cap at ps=1 on 64x64 imagery: each tile yields 4096
        # cells there instead of the 256 a ps=4 tile gives, so an uncapped
        # fifty_cities val split is ~81 GiB in fp16 alone.
        loader = evaluator._get_data_loader(
            split,
            evaluator.embedding_batch_size,
            max_samples=args.max_train_samples if is_train else args.max_valid_samples,
        )
        # quantize=False / normalizer=None: cache what the model emitted, so the
        # quantization step can be switched on and off per rung downstream.
        embeddings, labels = evaluator._get_embeddings(
            loader, is_train=is_train, normalizer=None, quantize=False
        )
        logger.info(
            "%s: embeddings %s %s, labels %s",
            split,
            tuple(embeddings.shape),
            embeddings.dtype,
            tuple(labels.shape),
        )
        torch.save({"embeddings": embeddings.half(), "labels": labels}, out)
        del embeddings, labels
        torch.cuda.empty_cache()


def _load_split(cache: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    blob = torch.load(cache / f"{split}.pt", map_location="cpu")
    return blob["embeddings"], blob["labels"]


def _rows(embeddings: torch.Tensor) -> torch.Tensor:
    """Flatten a ``[N, ...,  D]`` embedding tensor to ``[rows, D]``."""
    return embeddings.reshape(-1, embeddings.shape[-1])


def _basis_rows(train: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A fixed random subsample of train cells, float32 on ``device``."""
    rows = _rows(train)
    if rows.shape[0] > BASIS_ROWS:
        generator = torch.Generator().manual_seed(BASIS_SEED)
        idx = torch.randperm(rows.shape[0], generator=generator)[:BASIS_ROWS]
        rows = rows[idx]
    return rows.to(device=device, dtype=torch.float32)


def _cell_labels(labels: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Per-cell labels for the register grid, by nearest-neighbour subsampling.

    The probe upsamples cell logits to the label resolution; for the scatter
    analytics we need the reverse. Nearest-neighbour rather than a majority vote
    because at ps=1 the grid already matches the label map on these tasks, so the
    two agree and this stays exact in the common case.
    """
    n_h, n_w = grid
    labels = labels.long()
    if labels.dim() == 3 and labels.shape[1:] == (n_h, n_w):
        return labels.reshape(-1)
    if labels.dim() != 3:
        raise ValueError(f"expected [N, H, W] labels, got {tuple(labels.shape)}")
    rows = torch.linspace(0, labels.shape[1] - 1, n_h).round().long()
    cols = torch.linspace(0, labels.shape[2] - 1, n_w).round().long()
    return labels[:, rows][:, :, cols].reshape(-1)


def _row_labels(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """One label per row of ``_rows(embeddings)``, for either cache shape.

    The segmentation caches hold ``[N, H, W, D]`` embeddings against ``[N, H, W]``
    label maps, so a row is a register cell. The AEF supplemental caches
    (ethiopia_crops and its siblings) hold ONE mean-pooled ``[N, D]`` vector per
    window against a single ``[N]`` label, so a row is a whole window. Every
    basis fitter downstream wants the same thing -- labels aligned with the
    flattened rows -- so branch here rather than at each call site.
    """
    if embeddings.dim() == 2:
        if labels.dim() != 1:
            raise ValueError(
                f"[N, D] embeddings need [N] labels, got {tuple(labels.shape)}"
            )
        return labels.long().reshape(-1)
    return _cell_labels(labels, (embeddings.shape[1], embeddings.shape[2]))


def _pca_basis(rows: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k principal directions of the (centered) teacher cells, as ``[k, D]``."""
    centered = rows - rows.mean(dim=0, keepdim=True)
    _, _, V = torch.linalg.svd(centered, full_matrices=False)
    return V[:k]


def _scatter_matrices(
    rows: torch.Tensor, labels: torch.Tensor, ignore_label: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Between- and within-class scatter of the teacher cells."""
    labels = labels.to(rows.device)
    valid = labels >= 0 if ignore_label < 0 else labels != ignore_label
    rows, labels = rows[valid], labels[valid]
    mean = rows.mean(dim=0, keepdim=True)
    dim = rows.shape[1]
    between = torch.zeros(dim, dim, device=rows.device, dtype=rows.dtype)
    within = torch.zeros_like(between)
    for cls in labels.unique():
        members = rows[labels == cls]
        if members.shape[0] < 2:
            continue
        class_mean = members.mean(dim=0, keepdim=True)
        delta = (class_mean - mean).squeeze(0)
        between += members.shape[0] * torch.outer(delta, delta)
        centered = members - class_mean
        within += centered.T @ centered
    return between, within


def _fisher_basis(
    rows: torch.Tensor, labels: torch.Tensor, k: int, ridge: float = FISHER_RIDGE
) -> torch.Tensor:
    """A class-optimal ``[k, D]`` linear compression: Fisher directions, then PCA.

    The discriminative subspace of a C-class problem has rank <= C-1, so the
    leading directions of ``within^-1 between`` exhaust what a linear probe can
    use; the remaining k - (C-1) rows are filled with the top principal
    directions of the residual so the compression still has width k (and so a
    probe with batchnorm sees a comparably conditioned input).
    """
    between, within = _scatter_matrices(rows, labels)
    trace = torch.diagonal(within).mean()
    whitener = torch.linalg.cholesky(
        within / rows.shape[0]
        + ridge
        * trace
        / rows.shape[0]
        * torch.eye(within.shape[0], device=within.device, dtype=within.dtype)
    )
    whitened = torch.cholesky_solve(between, whitener)
    # The whitened scatter is non-symmetric, and torch's non-symmetric eig is
    # CPU-only in this build; float64 there also keeps the small eigenvalues of a
    # near-singular scatter from turning into noise directions.
    eigenvalues, eigenvectors = torch.linalg.eig(whitened.double().cpu())
    order = eigenvalues.real.argsort(descending=True)
    fisher = eigenvectors.real[:, order].T.to(device=rows.device, dtype=rows.dtype)
    fisher = torch.nn.functional.normalize(fisher, dim=1)
    n_classes = int(labels.max().item()) + 1
    keep = min(k, max(n_classes - 1, 1))
    basis = fisher[:keep]
    if keep < k:
        centered = rows - rows.mean(dim=0, keepdim=True)
        residual = centered - (centered @ basis.T) @ basis
        _, _, V = torch.linalg.svd(residual, full_matrices=False)
        basis = torch.cat([basis, V[: k - keep]], dim=0)
    return basis


def _scene_scatters(
    train: torch.Tensor, device: torch.device, max_tiles: int = 2048
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Within-tile and between-tile scatter of the teacher cells, LABEL-FREE.

    The nuisance the class probe wants suppressed -- illumination, phenology,
    sensor geometry, region -- is shared by every cell of a tile, while the
    structure it wants (which crop is at this pixel) varies WITHIN the tile. So
    tile identity is a usable stand-in for "nuisance class" that needs no labels:
    the between-tile scatter estimates the nuisance subspace and the within-tile
    scatter estimates everything else. This is the unsupervised analogue of the
    Fisher basis, which used real PASTIS labels.

    Returns (within, between, rows) with rows the cells the scatters were built
    from, so a caller can reuse the same sample for a PCA.
    """
    generator = torch.Generator().manual_seed(BASIS_SEED)
    n_tiles = train.shape[0]
    idx = torch.randperm(n_tiles, generator=generator)[:max_tiles]
    tiles = train[idx].to(device=device, dtype=torch.float32)
    tiles = tiles.reshape(tiles.shape[0], -1, tiles.shape[-1])  # [T, cells, D]
    tile_means = tiles.mean(dim=1, keepdim=True)
    residual = (tiles - tile_means).reshape(-1, tiles.shape[-1])
    within = residual.T @ residual
    centers = tile_means.squeeze(1)
    centered_centers = centers - centers.mean(dim=0, keepdim=True)
    between = tiles.shape[1] * (centered_centers.T @ centered_centers)
    return within, between, tiles.reshape(-1, tiles.shape[-1])


def _generalized_basis(
    numerator: torch.Tensor, denominator: torch.Tensor, k: int, n: int
) -> torch.Tensor:
    """Top-k directions maximizing ``x'Nx / x'Dx`` (ridged, CPU float64 eig)."""
    ridge = FISHER_RIDGE * torch.diagonal(denominator).mean()
    eye = torch.eye(
        denominator.shape[0], device=denominator.device, dtype=denominator.dtype
    )
    chol = torch.linalg.cholesky((denominator + ridge * eye) / n)
    whitened = torch.cholesky_solve(numerator / n, chol)
    values, vectors = torch.linalg.eig(whitened.double().cpu())
    order = values.real.argsort(descending=True)
    basis = vectors.real[:, order].T.to(device=numerator.device, dtype=numerator.dtype)
    return torch.nn.functional.normalize(basis[:k], dim=1)


def _donor_rows(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows and labels from the ``--transfer-cache`` (donor) dataset.

    Shared by ``fisherfrom{k}`` and ``donoraug{k}``: both fit on a different
    task's cached embeddings, and a second copy of the subsample-alignment
    dance is a second place for the labels to silently stop matching the rows.
    """
    other = Path(_TRANSFER_CACHE)
    blob = torch.load(other / "train.pt", map_location="cpu")
    other_emb, other_labels = blob["embeddings"], blob["labels"]
    other_rows = _basis_rows(other_emb, device)
    other_cells = _row_labels(other_emb, other_labels)
    if other_rows.shape[0] < _rows(other_emb).shape[0]:
        generator = torch.Generator().manual_seed(BASIS_SEED)
        idx = torch.randperm(_rows(other_emb).shape[0], generator=generator)[
            : other_rows.shape[0]
        ]
        other_cells = other_cells[idx]
    return other_rows, other_cells


def _apply(
    embeddings: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
    device: torch.device,
    quantize: bool,
) -> torch.Tensor:
    """Compress and (optionally) int8 round-trip a cached ``[N, ..., D]`` tensor.

    Both stages run in one chunked pass on the GPU. The teacher split is ~68 GiB
    in float32, so doing them separately -- transform the whole split, then
    quantize the whole split -- would hold two copies of it at once for no
    benefit.
    """
    out = []
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    for start in range(0, flat.shape[0], CHUNK_ROWS):
        chunk = flat[start : start + CHUNK_ROWS].to(device=device, dtype=torch.float32)
        if transform is not None:
            chunk = transform(chunk)
        if quantize:
            chunk = _quantize_roundtrip(chunk)
        out.append(chunk.cpu())
    mapped = torch.cat(out, dim=0)
    return mapped.reshape(*embeddings.shape[:-1], mapped.shape[-1])


def _quantize_roundtrip(embeddings: torch.Tensor) -> torch.Tensor:
    """The eval pipeline's int8 stage: AEF power quantize, then dequantize."""
    return dequantize_embeddings(quantize_embeddings(embeddings))


def _linear(
    weight: torch.Tensor, bias: torch.Tensor | None
) -> Callable[[torch.Tensor], torch.Tensor]:
    def transform(chunk: torch.Tensor) -> torch.Tensor:
        out = chunk @ weight.T.to(chunk)
        if bias is not None:
            out = out + bias.to(chunk)
        return out

    return transform


def _build_transforms(
    rung: str,
    cache: Path,
    train: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, bool]:
    """Map a rung name to (per-cell transform, quantize?)."""
    quantize = not rung.endswith("_f32")
    name = rung[:-4] if rung.endswith("_f32") else rung
    if name == "teacher":
        return None, quantize
    if name.startswith("student"):
        # "student" is the full 128-d map; "student64" is its Matryoshka prefix,
        # i.e. exactly the d64 product the run also ships.
        blob = torch.load(cache / "student_projection.pt", map_location=device)
        weight, bias = blob["weight"].to(device), blob["bias"]
        bias = None if bias is None else bias.to(device)
        if name != "student":
            keep = int(name[len("student") :])
            weight = weight[:keep]
            bias = None if bias is None else bias[:keep]
        return _linear(weight, bias), quantize
    rows = _basis_rows(train, device)
    if name.startswith("wscener"):
        # REGULARIZED within/between-tile basis. The unregularized "wscene"
        # variant scored 0.06 -- maximizing within/between finds directions where
        # between-tile variance is ~0, i.e. the near-null space of a rank-limited
        # scatter, which is numerical noise. Standard fix: solve inside a
        # well-conditioned subspace (top PCs) and use a real ridge.
        pre_dim, ridge = 256, 0.1
        rows_all = _basis_rows(train, device)
        pcs = _pca_basis(rows_all, pre_dim)
        within, between, scene_rows = _scene_scatters(train, device)
        within = pcs @ within @ pcs.T
        between = pcs @ between @ pcs.T
        eye = torch.eye(pre_dim, device=device, dtype=within.dtype)
        n = scene_rows.shape[0]
        chol = torch.linalg.cholesky(
            between / n + ridge * torch.diagonal(between).mean() / n * eye
        )
        whitened = torch.cholesky_solve(within / n, chol)
        values, vectors = torch.linalg.eig(whitened.double().cpu())
        order = values.real.argsort(descending=True)
        sub = vectors.real[:, order].T.to(device=device, dtype=within.dtype)
        basis = torch.nn.functional.normalize(
            sub[: int(name[len("wscener") :])] @ pcs, dim=1
        )
    elif name.startswith("wscene"):
        # LABEL-FREE alternatives to the Fisher oracle. "wscene{k}" maximizes
        # within-tile over between-tile variance (suppress what a whole tile
        # shares, keep what varies inside it); "wscenepca{k}" is the cheaper
        # cousin -- plain PCA of the tile-centered residuals.
        within, between, scene_rows = _scene_scatters(train, device)
        if name.startswith("wscenepca"):
            centered = scene_rows - scene_rows.mean(dim=0, keepdim=True)
            tiles = centered.reshape(
                -1, train.shape[1] * train.shape[2], centered.shape[-1]
            )
            residual = (tiles - tiles.mean(dim=1, keepdim=True)).reshape(
                -1, centered.shape[-1]
            )
            _, _, V = torch.linalg.svd(residual, full_matrices=False)
            basis = V[: int(name[len("wscenepca") :])]
        else:
            basis = _generalized_basis(
                within, between, int(name[len("wscene") :]), scene_rows.shape[0]
            )
    elif name.startswith("pcarange"):
        # An arbitrary BAND of the target's variance spectrum: PCs
        # [start, start + k). The control for "a transferred Fisher basis only
        # helps because it hands the target low-variance TAIL directions that
        # PCA never selects" -- 76% of the PASTIS Fisher core's energy sits above
        # PC rank 256 (notes entry, 2026-08-25). If some tail band matches the
        # transferred basis at the same rank, the win is about WHERE in the
        # spectrum the directions sit, not about the donor's labels.
        # Name is pcarange{start}x{k}, e.g. pcarange256x18.
        start, width = (int(v) for v in name[len("pcarange") :].split("x"))
        basis = _pca_basis(rows, start + width)[start:]
    elif name.startswith("pcaskip"):
        # Drop the leading PCs before keeping 128: the top-32 hold 69% of the
        # variance and a fisher ratio of only 0.40, i.e. mostly within-class
        # nuisance. If skipping them helps, it is a one-line shippable win.
        skip = int(name[len("pcaskip") :])
        basis = _pca_basis(rows, skip + 128)[skip:]
    elif name.startswith("randw"):
        # Random directions in the WHITENED space: whitening equalizes the
        # variance ordering, so a random draw no longer concentrates on the
        # nuisance-heavy leading directions.
        centered = rows - rows.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / centered.shape[0]
        ridge = FISHER_RIDGE * torch.diagonal(cov).mean()
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        whitener = torch.linalg.inv(torch.linalg.cholesky(cov + ridge * eye))
        gen = torch.Generator(device="cpu").manual_seed(BASIS_SEED)
        gaussian = torch.randn(int(name[len("randw") :]), cov.shape[0], generator=gen)
        basis = torch.nn.functional.normalize(gaussian.to(device) @ whitener, dim=1)
    elif name.startswith("pca"):
        basis = _pca_basis(rows, int(name[3:]))
    elif name.startswith("rand"):
        k = int(name[4:])
        generator = torch.Generator(device="cpu").manual_seed(BASIS_SEED)
        gaussian = torch.randn(k, rows.shape[1], generator=generator).to(device)
        basis = torch.linalg.qr(gaussian.T)[0].T
    elif name.startswith("kmfisher"):
        # LABEL-FREE Fisher: pseudo-classes from k-means over the teacher's own
        # cells. Fisher needs only a notion of "these cells belong together";
        # labels are one source, clusters are another. Unlike the tile-identity
        # attempt (which failed because a ws16 tile is SMALLER than a parcel, so
        # between-tile variance carried the class signal), clusters group cells
        # by what they look like, so within-cluster spread is closer to nuisance.
        # Name is kmfisher{clusters}x{dims}, e.g. kmfisher64x128.
        n_clusters, dim = (int(v) for v in name[len("kmfisher") :].split("x"))
        centroids = rows[torch.randperm(rows.shape[0], device=rows.device)[:n_clusters]]
        for _ in range(15):  # Lloyd iterations; plenty for a pseudo-label proxy
            assign = torch.cdist(rows, centroids).argmin(dim=1)
            for c in range(n_clusters):
                members = rows[assign == c]
                if members.shape[0]:
                    centroids[c] = members.mean(dim=0)
        counts = torch.bincount(assign, minlength=n_clusters)
        logger.info(
            "kmeans: %d clusters, sizes min=%d median=%d max=%d",
            n_clusters,
            int(counts.min()),
            int(counts.median()),
            int(counts.max()),
        )
        basis = _fisher_basis(rows, assign, dim)
    elif name.startswith("fisherfromshuf"):
        # The donor's rows and the donor's FITTER, but its labels randomly
        # permuted. The control that separates the donor's label CONTENT from
        # the geometry of the Fisher construction itself: the basis is still
        # built by whitening the donor's within-class scatter and taking leading
        # eigenvectors, so it inherits the same conditioning, the same whitened
        # per-dim scales and the same rank -- everything except a real class
        # signal. If a transferred basis beats PCA at matched rank and this one
        # does too, the win is the fitter, not the supervision.
        other_rows, other_cells = _donor_rows(device)
        generator = torch.Generator().manual_seed(BASIS_SEED)
        permutation = torch.randperm(other_cells.shape[0], generator=generator)
        shuffled = other_cells[permutation.to(other_cells.device)]
        basis = _fisher_basis(other_rows, shuffled, int(name[len("fisherfromshuf") :]))
    elif name.startswith("donoraug"):
        # MARGINAL value of a donor task's supervision, on top of the best
        # label-free basis. A C-class donor supplies only C-1 real Fisher
        # directions -- three for ethiopia_crops -- so a plain fisherfrom128 rung
        # is 125 dims of PCA filler and its score is dominated by the filler
        # rather than by the donor. donoraug{k} instead spends k - (C-1) dims on
        # the target's own top PCs and the rest on the donor's real directions,
        # then orthonormalizes. Against pca{k} it answers the shippable
        # question: does adding foreign crop supervision to a variance-ranked
        # basis buy anything at all?
        k = int(name[len("donoraug") :])
        other_rows, other_cells = _donor_rows(device)
        n_classes = int(other_cells[other_cells >= 0].max().item()) + 1
        core = _fisher_basis(other_rows, other_cells, max(n_classes - 1, 1))
        logger.info(
            "donoraug: %d donor Fisher directions + %d target PCs",
            core.shape[0],
            k - core.shape[0],
        )
        filler = _pca_basis(rows, k - core.shape[0])
        basis = torch.linalg.qr(torch.cat([core, filler], dim=0).T)[0].T
    elif name.startswith("fisherfrom"):
        # Fit the class-optimal basis on a DIFFERENT dataset's cached embeddings
        # (--transfer-cache) and apply it here. `fisherhalf` only held out
        # CLASSES of the same imagery; this holds out the whole task, which is
        # what a single shipped projection actually faces.
        other_rows, other_cells = _donor_rows(device)
        basis = _fisher_basis(other_rows, other_cells, int(name[len("fisherfrom") :]))
    elif name.startswith("fisherhalf"):
        # Fit the oracle on HALF the classes, probe on all of them. If the
        # directions found from one label set carry to classes it never saw, a
        # supervised basis generalizes beyond its supervision -- which is the
        # question that decides whether ONE shipped projection can serve tasks
        # it was not fitted on. Cheap stand-in for a full cross-task transfer.
        cells = _row_labels(train, train_labels)
        if rows.shape[0] < _rows(train).shape[0]:
            generator = torch.Generator().manual_seed(BASIS_SEED)
            idx = torch.randperm(_rows(train).shape[0], generator=generator)[
                : rows.shape[0]
            ]
            cells = cells[idx]
        held_out = cells.clone()
        n_classes = int(cells.max().item()) + 1
        held_out[cells >= n_classes // 2] = -1  # -1 is ignored by the scatters
        basis = _fisher_basis(rows, held_out, int(name[len("fisherhalf") :]))
    elif name.startswith("fisher"):
        cells = _row_labels(train, train_labels)
        if rows.shape[0] < _rows(train).shape[0]:
            generator = torch.Generator().manual_seed(BASIS_SEED)
            idx = torch.randperm(_rows(train).shape[0], generator=generator)[
                : rows.shape[0]
            ]
            cells = cells[idx]
        basis = _fisher_basis(rows, cells, int(name[6:]))
    else:
        raise ValueError(f"unknown rung {rung!r}")
    return _linear(basis, None), quantize


def _to_probe_device(
    tensors: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    """Park the probe's splits in device memory when they fit.

    The probe's DataLoader indexes these tensors 3k times an epoch; from CPU
    that is a 6 MB gather plus a host-to-device copy per step, which pins the
    whole run to memory bandwidth while the GPU idles. Parking both splits on
    the device turns each step into a device-side gather. Falls back to CPU
    (i.e. the original behaviour) when the splits do not fit.
    """
    if device.type != "cuda":
        return tensors
    free, _ = torch.cuda.mem_get_info(device)
    needed = sum(t.numel() * t.element_size() for t in tensors)
    if needed > free - DEVICE_TENSOR_HEADROOM_GIB * 2**30:
        logger.info(
            "probe splits need %.1f GiB, %.1f GiB free: keeping them on CPU",
            needed / 2**30,
            free / 2**30,
        )
        return tensors
    logger.info("moving %.1f GiB of probe splits to %s", needed / 2**30, device)
    return [t.to(device) for t in tensors]


def analyze(args: argparse.Namespace) -> None:
    """Probe each rung through the eval's own probe and print the score table."""
    global _TRANSFER_CACHE
    _TRANSFER_CACHE = args.transfer_cache
    device = torch.device(args.device)
    cache = Path(args.cache_dir)
    train, train_labels = _load_split(cache, "train")
    val, val_labels = _load_split(cache, "valid")
    logger.info("train %s | valid %s", tuple(train.shape), tuple(val.shape))

    # The evaluator is rebuilt only for its config + probe partial; no model is
    # needed now that the embeddings are cached, so no checkpoint is loaded.
    task = _task(args.task)
    evaluator = DownstreamEvaluator(
        args.task, task, trainer=SimpleNamespace(save_folder="."), device=device
    )

    # Merge into any earlier run's results rather than replacing them: rungs are
    # launched in batches as questions come up, and a later batch overwriting the
    # file would silently drop the numbers the earlier one paid for.
    results_path = cache / "analyze_results.json"
    # Most rungs map to a flat {metric: float} dict, but the "{rung}__folds"
    # entries carry the raw per-fold lists (list[float]) for paired comparisons,
    # so the inner value type is intentionally heterogeneous.
    results: dict[str, dict[str, Any]] = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    for rung in args.rungs.split(","):
        # Skip what a previous invocation already measured. Weka stalls have
        # wedged this process mid-sweep more than once; restarting should cost
        # only the rungs that never finished.
        if rung in results and not args.force:
            logger.info("rung %s already in results, skipping", rung)
            continue
        transform, quantize = _build_transforms(
            rung, cache, train, train_labels, device
        )
        train_e = _apply(train, transform, device, quantize)
        val_e = _apply(val, transform, device, quantize)
        logger.info("rung %s: dim %d, quantize=%s", rung, train_e.shape[-1], quantize)
        train_e, val_e, train_y, val_y = _to_probe_device(
            [train_e, val_e, train_labels, val_labels], device
        )
        started = time.time()
        # eval_interval is a linear-probe knob; run_knn does not take it, and the
        # KNN tasks (the AEF balanced-trial ones) go through the same call.
        extra = (
            {"eval_interval": args.probe_eval_interval}
            if evaluator.eval_mode == EvalMode.LINEAR_PROBE
            else {}
        )
        result = evaluator.eval_function(  # type: ignore[misc]
            config=evaluator.config,
            train_embeddings=train_e,
            train_labels=train_y,
            val_embeddings=val_e,
            val_labels=val_y,
            test_embeddings=None,
            test_labels=None,
            device=device,
            **extra,
            n_bootstrap=0,
            bootstrap_seed=42,
        )
        logger.info("rung %s took %.1f min", rung, (time.time() - started) / 60)
        metrics = dict(result.val_result.metrics) if result.val_result else {}
        # AEF balanced trials (the aeftrial_* / bt_* numbers, kNN k=5 and k=20 on
        # class-balanced draws) live outside the eval function -- the production
        # callback runs them separately on the same embeddings, so do the same
        # here rather than reporting a plain kNN and calling it the AEF metric.
        trial_config = getattr(evaluator, "balanced_trial", None)
        if trial_config is not None:
            trial = run_balanced_trials(
                config=evaluator.config,
                embeddings_by_split={"train": train_e, "val": val_e},
                labels_by_split={"train": train_y, "val": val_y},
                trial_config=trial_config,
                device=device,
            )
            # BalancedTrialResult carries one EvalResult PER PREDICTOR (ridge,
            # knn5, knn20) plus the per-fold values behind each, because the
            # production callback files each predictor as its own task. Flatten
            # to bt_{predictor}_{metric} so one rung is one row here, and keep
            # the across-fold std: on a 4-class task with 9 val rows in its
            # rarest class, the spread across draws is the only thing that says
            # whether a rung difference is real.
            for predictor, result in trial.results.items():
                for key, value in result.metrics.items():
                    if isinstance(value, int | float):
                        metrics[f"bt_{predictor}_{key}"] = float(value)
                folds = trial.per_fold.get(predictor, {})
                primary = getattr(
                    result.primary_metric, "value", str(result.primary_metric)
                )
                values = folds.get(primary)
                if values:
                    # Keep the raw per-fold list, not just its spread. Every rung
                    # is scored on the SAME draws, so a rung-vs-rung difference
                    # is a PAIRED comparison and the across-draw std badly
                    # overstates its error bar; without these lists that test
                    # cannot be done after the fact.
                    results.setdefault(f"{rung}__folds", {})[
                        f"bt_{predictor}_{primary}"
                    ] = values
                    tensor = torch.tensor(values, dtype=torch.float64)
                    metrics[f"bt_{predictor}_{primary}_std"] = float(tensor.std())
                    metrics[f"bt_{predictor}_{primary}_sem"] = float(
                        tensor.std() / max(len(values), 1) ** 0.5
                    )
        results[rung] = metrics
        print(
            f"  {rung:<16} dim={train_e.shape[-1]:<4} "
            + "  ".join(
                f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)
            )
        )
        del train_e, val_e
        torch.cuda.empty_cache()
        results_path.write_text(json.dumps(results, indent=2))

    # Segmentation tasks report mIoU; the AEF classification tasks (ethiopia and
    # siblings) report balanced_accuracy and have no mIoU at all, so summarize on
    # whatever the task itself calls primary rather than a hardcoded key.
    primary = getattr(task.primary_metric, "value", str(task.primary_metric))
    print(f"\n=== {primary} by rung ===")
    for rung, metrics in results.items():
        value = metrics.get(primary, metrics.get("miou", float("nan")))
        print(f"  {rung:<16} {value:.4f}")


def geometry(args: argparse.Namespace) -> None:
    """Probe-free comparison of the three 128-d compressions' row spaces.

    Reports, for the student / PCA / Fisher / random maps: the share of the
    teacher's total (centered) variance their row space keeps, and the share of
    its BETWEEN-CLASS scatter -- the part a linear probe can actually use. A map
    can score well on the first and badly on the second, and that dissociation is
    the whole hypothesis about why distillation underperforms at fixed width.
    """
    global _TRANSFER_CACHE
    _TRANSFER_CACHE = args.transfer_cache
    device = torch.device(args.device)
    cache = Path(args.cache_dir)
    train, train_labels = _load_split(cache, "train")
    rows = _basis_rows(train, device)
    cells = _row_labels(train, train_labels)
    generator = torch.Generator().manual_seed(BASIS_SEED)
    total_rows = _rows(train).shape[0]
    if rows.shape[0] < total_rows:
        idx = torch.randperm(total_rows, generator=generator)[: rows.shape[0]]
        cells = cells[idx]

    blob = torch.load(cache / "student_projection.pt", map_location=device)
    student = blob["weight"].to(device=device, dtype=torch.float32)
    print(f"student W: {tuple(student.shape)}")
    singular = torch.linalg.svdvals(student)
    print(
        f"  W singular values: max={singular.max():.4f} min={singular.min():.4f} "
        f"ratio={singular.max() / singular.min():.1f}"
    )

    centered = rows - rows.mean(dim=0, keepdim=True)
    total_variance = centered.pow(2).sum()
    between, within = _scatter_matrices(rows, cells)
    total_between = torch.diagonal(between).sum()

    bases = {
        "student128": student,
        "pca128": _pca_basis(rows, 128),
        "fisher128": _fisher_basis(rows, cells, 128),
        "rand128": torch.linalg.qr(
            torch.randn(
                768, 128, generator=torch.Generator().manual_seed(BASIS_SEED)
            ).to(device)
        )[0].T,
    }
    # Fisher ratio, not raw scatter energy. Between-class ENERGY retention turned
    # out to be a bad predictor of probe accuracy (pca128 keeps 99.7% of it and
    # still loses 6.1 mIoU): energy ignores how much WITHIN-class spread sits
    # along the same directions, which is what actually decides separability.
    # tr(S_w^-1 S_b) restricted to each subspace is the quantity a linear probe
    # can exploit.
    whitener = torch.linalg.cholesky(
        within / rows.shape[0]
        + FISHER_RIDGE
        * torch.diagonal(within).mean()
        / rows.shape[0]
        * torch.eye(within.shape[0], device=within.device, dtype=within.dtype)
    )
    total_fisher = torch.diagonal(torch.cholesky_solve(between, whitener)).sum()

    header = f"{'basis':<12} {'variance':>10} {'between-cls':>12} {'fisher':>10}"
    print(f"\n{header}")
    for name, basis in bases.items():
        orthonormal = torch.linalg.qr(basis.T)[0]  # [D, k]
        variance_kept = (centered @ orthonormal).pow(2).sum() / total_variance
        between_kept = (
            torch.diagonal(orthonormal.T @ between @ orthonormal).sum() / total_between
        )
        sub_between = orthonormal.T @ between @ orthonormal
        sub_within = orthonormal.T @ within @ orthonormal
        sub_whitener = torch.linalg.cholesky(
            sub_within / rows.shape[0]
            + FISHER_RIDGE
            * torch.diagonal(sub_within).mean()
            / rows.shape[0]
            * torch.eye(
                sub_within.shape[0], device=sub_within.device, dtype=sub_within.dtype
            )
        )
        fisher_kept = (
            torch.diagonal(torch.cholesky_solve(sub_between, sub_whitener)).sum()
            / total_fisher
        )
        print(
            f"{name:<12} {variance_kept.item():>10.4f} "
            f"{between_kept.item():>12.4f} {fisher_kept.item():>10.4f}"
        )

    # How much the learned map's subspace actually agrees with the alternatives,
    # via principal angles: mean cos^2 of the angles between row spaces, which is
    # the fraction of one subspace lying inside the other.
    print("\nsubspace overlap (mean cos^2 of principal angles)")
    student_q = torch.linalg.qr(bases["student128"].T)[0]
    for name in ("pca128", "fisher128", "rand128"):
        other_q = torch.linalg.qr(bases[name].T)[0]
        singular = torch.linalg.svdvals(student_q.T @ other_q)
        print(f"  student128 vs {name:<10} {singular.pow(2).mean().item():.4f}")

    # WHERE the class signal lives relative to the variance ordering. Two very
    # different things make a variance-ranked basis the wrong one, and the probe
    # results alone cannot separate them:
    #   (a) discriminative directions sit BELOW the cutoff (low-variance signal
    #       that PCA truncates), or
    #   (b) the budget above the cutoff is spent on high-variance WITHIN-class
    #       nuisance (illumination, phenology, sensor geometry) that a class
    #       probe wants suppressed.
    # Projecting the cells onto each principal direction and splitting their
    # variance into between- and within-class parts shows both at once: the
    # fisher ratio per rank band says how much discriminative value a band
    # carries, and the within column says how much of what PCA bought is
    # nuisance.
    pcs_full = _pca_basis(rows, 768)
    coords = (rows - rows.mean(dim=0, keepdim=True)) @ pcs_full.T
    labels_valid = cells.to(rows.device)
    mask = labels_valid >= 0
    coords, labels_valid = coords[mask], labels_valid[mask]
    grand = coords.mean(dim=0)
    between_per_pc = torch.zeros(768, device=rows.device)
    within_per_pc = torch.zeros(768, device=rows.device)
    for cls in labels_valid.unique():
        members = coords[labels_valid == cls]
        if members.shape[0] < 2:
            continue
        class_mean = members.mean(dim=0)
        between_per_pc += members.shape[0] * (class_mean - grand).pow(2)
        within_per_pc += (members - class_mean).pow(2).sum(dim=0)
    print("\nclass signal by teacher-PC rank band")
    print(
        f"  {'PC band':<12} {'var share':>10} {'between share':>14} {'fisher ratio':>13}"
    )
    total_var = (between_per_pc + within_per_pc).sum()
    total_between = between_per_pc.sum()
    for start, end in (
        (0, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 768),
    ):
        band_between = between_per_pc[start:end].sum()
        band_within = within_per_pc[start:end].sum()
        print(
            f"  {f'{start}-{end}':<12} "
            f"{((band_between + band_within) / total_var).item():>10.4f} "
            f"{(band_between / total_between).item():>14.4f} "
            f"{(band_between / band_within.clamp_min(1e-12)).item():>13.4f}"
        )

    # Where the ORACLE's discriminative directions sit in that same ordering: the
    # first (num_classes - 1) rows of the fisher basis are the ones carrying the
    # class structure, the rest is PCA filler.
    n_classes = int(cells.max().item()) + 1
    fisher_core = bases["fisher128"][: max(n_classes - 1, 1)]
    core_coords = (fisher_core @ pcs_full.T).pow(2)
    print(
        f"\nwhere the {fisher_core.shape[0]} fisher directions live (energy by PC band)"
    )
    for start, end in (
        (0, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 768),
    ):
        print(
            f"  PCs {start:>3}-{end:>3}: {core_coords[:, start:end].sum().item() / core_coords.sum().item():.4f}"
        )

    # Where the student's subspace sits in the teacher's variance ordering: if
    # distillation is behaving like a noisy PCA it should load on the leading
    # bins and taper; anything else says it is selecting on a different axis.
    pcs = _pca_basis(rows, 768)
    print("\nstudent row-space energy by teacher-PC decile")
    for start in range(0, 768, 77):
        block = pcs[start : start + 77]
        energy = (student_q.T @ block.T).pow(2).sum() / block.shape[0]
        print(f"  PCs {start:>3}-{min(start + 77, 768):>3}: {energy.item():.4f}")


def main() -> None:
    """Parse arguments and dispatch to the requested stage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(min(PROBE_THREADS, os.cpu_count() or PROBE_THREADS))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "analyze", "geometry"])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_TRAIN_TILES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--probe-eval-interval", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--transfer-cache", default="/var/tmp/fc_cache")  # nosec B108
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument(
        "--rungs",
        default="teacher,student,pca128,fisher128,teacher_f32,student_f32",
        help="comma-separated: teacher, student, pca{k}, fisher{k}, rand{k}; "
        "suffix _f32 to skip the int8 round-trip",
    )
    args = parser.parse_args()
    {"extract": extract, "analyze": analyze, "geometry": geometry}[args.stage](args)


if __name__ == "__main__":
    main()
