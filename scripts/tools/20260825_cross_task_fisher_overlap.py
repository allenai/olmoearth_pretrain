#!/usr/bin/env python3
r"""Does supervision from ANOTHER task point at the directions THIS task needs?

The proj128 gap decomposition (``20260824_diagnose_proj128_gap.py``, notes entry
2026-08-25) established that ~4.5 of the 7.5-point PASTIS gap is the student
keeping the wrong 128 directions, that nothing built from the teacher's own
second-order statistics finds the right ones, and that a PASTIS-fitted Fisher
basis does NOT transfer to fifty_cities. That left one live question for the
whole supervision program: fifty_cities is COARSE land cover, so its failure to
share directions with PASTIS may just be a coarse-vs-fine mismatch. If a
genuinely FINE-GRAINED crop task in another region shares PASTIS's
discriminative directions, then broad crop supervision (CDL / worldcereal) is
worth GPUs; if it does not, the supervision route is task-specific by nature and
one shipped projection cannot be made crop-aware for everybody.

A probe alone cannot answer this at these ranks. A C-class donor supplies only
C-1 real Fisher directions -- three for ethiopia_crops's four classes -- and a
128-d rung built from it is then 125 dims of PCA filler, whose score swamps
whatever the three directions contribute. So compare SUBSPACES at matched rank
instead, measured on the target's own scatter matrices:

* **overlap** -- mean cos^2 of the principal angles between the candidate's row
  space and the target's full Fisher core (its C-1 oracle directions). The
  fraction of the candidate lying inside the subspace the target needs.
* **fisher kept** -- ``tr(S_w^-1 S_b)`` retained inside the candidate's row
  space, as a share of the target's total. The notes entry established this as
  the ONLY cheap proxy that orders bases the way the probe does; retained
  variance and between-class energy both fail.

Every candidate is reported at the SAME rank the donor supplies, against
references that bracket the answer:

* ``self`` -- the target's own Fisher core truncated to that rank. The ceiling.
* ``pca`` -- the target's top principal directions. What supervision must beat
  to be worth anything, since PCA needs no labels.
* ``rand`` -- random orthonormal directions. The floor.
* ``self_small`` -- the target's OWN labels fitted on only as many rows as the
  donor had. THE SAMPLE-SIZE CONTROL, and the one that decides whether a weak
  donor number means anything: ethiopia_crops has 574 train windows, so if a
  574-row fit of the target's own labels already fails to recover the target's
  directions, a weak donor result says nothing about transfer.

Usage::

    python scripts/tools/20260825_cross_task_fisher_overlap.py \\
        --target-cache /var/tmp/proj128_gap_cache \\
        --donor-cache /var/tmp/ethiopia_cache

Both caches are written by ``20260824_diagnose_proj128_gap.py extract``; the
Fisher / PCA fitters are imported from that script rather than reimplemented, so
every number here is built the same way as the ones in the notes entry. Swap the
two flags for the reverse direction.

Caveat on every cross-task number here: the AEF classification caches hold ONE
MEAN-POOLED vector per 16x16 window, while the PASTIS ps=1 cache holds one per
register cell. The directions live in the same 768-d space so the angles are
meaningful, but a Fisher basis fitted on window means whitens by the within-class
scatter OF MEANS, which is smaller and differently shaped than the cell-level
scatter. ``--target-pool-mean`` fits the target side on tile means too, which
removes the mismatch at the cost of far fewer target rows -- run it as a check on
any result that looks decisive.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

# The gap script's filename starts with a date, so it is not importable by name.
# Load it by path and reuse its fitters: a second copy of _fisher_basis here
# would silently drift from the one every number in the notes entry came from.
_spec = importlib.util.spec_from_file_location(
    "diagnose_proj128_gap", _HERE.parent / "20260824_diagnose_proj128_gap.py"
)
assert _spec is not None and _spec.loader is not None
_gap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gap)

logger = logging.getLogger(__name__)


def _load(cache: Path, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
    blob = torch.load(cache / f"{split}.pt", map_location="cpu")
    return blob["embeddings"], blob["labels"]


def _rows_and_labels(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    max_rows: int,
    device: torch.device,
    pool_mean: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a cache to ``(rows, labels)`` on ``device``, subsampled to max_rows.

    Handles both cache shapes via the gap script's ``_row_labels``. ``pool_mean``
    collapses an ``[N, H, W, D]`` cache to one mean vector per tile carrying the
    tile's majority label, so it can be compared against a mean-pooled AEF donor
    on equal footing.
    """
    cells = _gap._row_labels(embeddings, labels)
    if pool_mean:
        if embeddings.dim() != 4:
            raise ValueError("--target-pool-mean needs an [N, H, W, D] cache")
        flat = embeddings.reshape(embeddings.shape[0], -1, embeddings.shape[-1])
        rows = flat.float().mean(dim=1)
        per_tile = cells.reshape(embeddings.shape[0], -1)
        valid = per_tile.clamp_min(0)
        counts = torch.nn.functional.one_hot(valid, int(per_tile.max().item()) + 1) * (
            per_tile >= 0
        ).unsqueeze(-1)
        cells = counts.sum(dim=1).argmax(dim=1)
        cells[(per_tile >= 0).sum(dim=1) == 0] = -1
    else:
        rows = embeddings.reshape(-1, embeddings.shape[-1])
    if rows.shape[0] > max_rows:
        generator = torch.Generator().manual_seed(_gap.BASIS_SEED)
        idx = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
        rows, cells = rows[idx], cells[idx]
    return rows.to(device=device, dtype=torch.float32), cells.to(device)


def _fisher_core(rows: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """The real Fisher directions only -- ``C-1`` rows, no PCA filler.

    ``_fisher_basis(rows, cells, k)`` pads to width k with principal directions
    of the residual once it runs out of class structure. Asking for exactly
    ``C-1`` returns the discriminative part alone, which is what a subspace
    comparison at matched rank has to use.
    """
    n_classes = int(cells[cells >= 0].max().item()) + 1
    keep = max(n_classes - 1, 1)
    return _gap._fisher_basis(rows, cells, keep)


def _overlap(basis: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean cos^2 of the principal angles between two row spaces.

    Normalized by the SMALLER rank, so a rank-3 candidate sitting entirely
    inside a rank-18 reference scores 1.0 rather than 3/18. The question is what
    fraction of the candidate is useful, not how much of the reference it covers.
    """
    q_basis = torch.linalg.qr(basis.T)[0]
    q_reference = torch.linalg.qr(reference.T)[0]
    singular = torch.linalg.svdvals(q_basis.T @ q_reference)
    return float(singular.pow(2).sum() / min(q_basis.shape[1], q_reference.shape[1]))


def _target_geometry(
    rows: torch.Tensor, cells: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return ``(between, within, total_fisher, n)`` for the target.

    Ridged exactly as the gap script's geometry stage does -- same ridge, same
    normalization -- so a "fisher kept" here is on the same scale as the ones in
    the notes entry.
    """
    between, within = _gap._scatter_matrices(rows, cells)
    n = rows.shape[0]
    whitener = torch.linalg.cholesky(
        within / n
        + _gap.FISHER_RIDGE
        * torch.diagonal(within).mean()
        / n
        * torch.eye(within.shape[0], device=within.device, dtype=within.dtype)
    )
    total = torch.diagonal(torch.cholesky_solve(between, whitener)).sum()
    return between, within, total, n


def _fisher_kept(
    basis: torch.Tensor,
    geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
) -> float:
    """Share of the target's ``tr(S_w^-1 S_b)`` living inside ``basis``'s row space."""
    between, within, total, n = geometry
    orthonormal = torch.linalg.qr(basis.T)[0]
    sub_between = orthonormal.T @ between @ orthonormal
    sub_within = orthonormal.T @ within @ orthonormal
    whitener = torch.linalg.cholesky(
        sub_within / n
        + _gap.FISHER_RIDGE
        * torch.diagonal(sub_within).mean()
        / n
        * torch.eye(
            sub_within.shape[0], device=sub_within.device, dtype=sub_within.dtype
        )
    )
    kept = torch.diagonal(torch.cholesky_solve(sub_between, whitener)).sum()
    return float(kept / total)


def compare(
    target_rows: torch.Tensor,
    target_cells: torch.Tensor,
    donor_rows: torch.Tensor,
    donor_cells: torch.Tensor,
    device: torch.device,
) -> list[dict]:
    """The candidate table: overlap with, and fisher retention on, the target."""
    geometry = _target_geometry(target_rows, target_cells)
    target_core = _fisher_core(target_rows, target_cells)
    donor_core = _fisher_core(donor_rows, donor_cells)
    rank = donor_core.shape[0]
    logger.info(
        "donor supplies %d real Fisher directions; the target's own core is %d",
        rank,
        target_core.shape[0],
    )

    generator = torch.Generator().manual_seed(_gap.BASIS_SEED)
    candidates: dict[str, torch.Tensor] = {
        "self": target_core[:rank],
        "donor": donor_core,
        "pca": _gap._pca_basis(target_rows, rank),
        "rand": torch.linalg.qr(
            torch.randn(target_rows.shape[1], rank, generator=generator).to(device)
        )[0].T,
    }
    n_donor = donor_rows.shape[0]
    if n_donor < target_rows.shape[0]:
        idx = torch.randperm(target_rows.shape[0], generator=generator)[:n_donor].to(
            device
        )
        candidates["self_small"] = _fisher_core(target_rows[idx], target_cells[idx])[
            :rank
        ]

    table = []
    for name, basis in candidates.items():
        table.append(
            {
                "candidate": name,
                "rank": int(basis.shape[0]),
                "overlap": _overlap(basis, target_core),
                "fisher_kept": _fisher_kept(basis, geometry),
            }
        )
    return table


def main() -> None:
    """Parse args and print the cross-task overlap / fisher-kept table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--donor-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=_gap.BASIS_ROWS)
    parser.add_argument(
        "--target-pool-mean",
        action="store_true",
        help="fit the target side on per-tile MEAN embeddings, matching the "
        "mean-pooled AEF caches (removes the pooling mismatch, costs rows)",
    )
    parser.add_argument("--out", default=None, help="write the table as JSON here")
    args = parser.parse_args()

    device = torch.device(args.device)
    target_emb, target_labels = _load(Path(args.target_cache))
    donor_emb, donor_labels = _load(Path(args.donor_cache))

    target_rows, target_cells = _rows_and_labels(
        target_emb, target_labels, args.rows, device, pool_mean=args.target_pool_mean
    )
    del target_emb
    donor_rows, donor_cells = _rows_and_labels(
        donor_emb, donor_labels, args.rows, device
    )
    logger.info(
        "target %s, %d classes | donor %s, %d classes",
        tuple(target_rows.shape),
        int(target_cells[target_cells >= 0].max()) + 1,
        tuple(donor_rows.shape),
        int(donor_cells[donor_cells >= 0].max()) + 1,
    )

    table = compare(target_rows, target_cells, donor_rows, donor_cells, device)
    print(f"\n{'candidate':<12} {'rank':>5} {'overlap':>9} {'fisher kept':>12}")
    for row in table:
        print(
            f"{row['candidate']:<12} {row['rank']:>5} "
            f"{row['overlap']:>9.4f} {row['fisher_kept']:>12.4f}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
