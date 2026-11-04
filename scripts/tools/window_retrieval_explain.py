"""What lets a retriever recover window embeddings from the annual one?

Two analyses on the calendar-year cache (``--cache_dir``, same train / val split
as ``window_retrieval_train.py``), scored on the val rows with the trainer's two
metrics: gap closed (cosine, relative to the annual vector) and within-tile
closed (MSE after subtracting each (sample, window)'s mean):

- **Nearest neighbours (how much is predictable from z at all).** For each val
  (pixel, window), the k most similar annual embeddings among training pixels
  with the same window; the prediction is the mean of their fresh window
  embeddings. k-NN is a real predictor, so its score is achievable. Half its
  1-NN squared error estimates the noise no function of z can remove
  (Cover-Hart), i.e. roughly the ceiling. ``knn shift`` instead predicts z plus
  the neighbours' mean (fresh - z), so neighbours only need to share the change.
- **Cluster prior (typical trajectory vs this pixel).** k-means on annual
  embeddings with ``--clusters``; per (cluster, window): ``prior`` predicts the
  mean fresh embedding, ``prior shift`` predicts z plus the mean (fresh - z).
  If a coarse prior recovers most of what the retriever does, retrieval is
  mostly "what this kind of place does in that window".

A trained retriever (``--retriever_dir``) is scored on the same rows for reference.

Usage::

    python scripts/tools/window_retrieval_explain.py --cache_dir $CAL20K --retriever_dir $R --out $OUT
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_retrieval_train import (  # noqa: E402
    MONTHS,
    build,
    load_cache,
    retrieve,
    split,
)

logger = logging.getLogger(__name__)


def rows(d: dict, max_samples: int | None = None) -> dict[str, torch.Tensor]:
    """Every valid (sample, window, pixel) as flat rows, with window and block ids."""
    n = (
        d["z_T"].shape[0]
        if max_samples is None
        else min(max_samples, d["z_T"].shape[0])
    )
    P = d["z_T"].shape[1]
    s, k = torch.nonzero(d["valid"][:n], as_tuple=True)
    keep = d["length"][s, k] < MONTHS
    s, k = s[keep], k[keep]
    rep = lambda t: t[:, None].expand(-1, P).reshape(-1)  # noqa: E731
    z = d["z_T"][s].reshape(-1, d["z_T"].shape[-1])
    f = d["z_w"][s, k].reshape(-1, d["z_w"].shape[-1])
    st, ln = d["start"][s, k], d["length"][s, k]
    return {
        "z": z,
        "f": f,
        "window": rep(st * 16 + ln),  # unique id per (start, length)
        "start": rep(st),
        "length": rep(ln),
        "start_month": rep(d["start_month"][s]),
        "block": rep(
            torch.arange(len(s))
        ),  # one (sample, window) = one block of P pixels
    }


def centered(x: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Each row minus the mean of its block."""
    nb = int(block.max()) + 1
    sums = torch.zeros(nb, x.shape[1], dtype=x.dtype, device=x.device).index_add_(
        0, block, x
    )
    counts = torch.bincount(block, minlength=nb).clamp_min(1)[:, None].to(x.dtype)
    return x - (sums / counts)[block]


def score(pred: torch.Tensor, v: dict) -> dict[str, float]:
    """Gap closed (cosine) and within-tile closed (block-centred MSE) vs the annual floor."""
    z, f, b = v["z"], v["f"], v["block"]
    cos_floor = F.cosine_similarity(z, f, dim=-1).mean().item()
    cos = F.cosine_similarity(pred, f, dim=-1).mean().item()
    fc = centered(f, b)
    local_floor = ((centered(z, b) - fc) ** 2).mean().item()
    local = ((centered(pred, b) - fc) ** 2).mean().item()
    mse_floor = ((z - f) ** 2).mean().item()
    mse = ((pred - f) ** 2).mean().item()
    return {
        "gap_closed": (cos - cos_floor) / max(1 - cos_floor, 1e-8),
        "within_tile_closed": 1 - local / max(local_floor, 1e-12),
        "mse_closed": 1 - mse / max(mse_floor, 1e-12),
    }


@torch.no_grad()
def knn(tr: dict, v: dict, ks: list[int], device: str, chunk: int = 4096) -> dict:
    """k-NN predictions per window id, plus the 1-NN squared errors (overall and within-tile)."""
    D = tr["f"].shape[1]
    preds = {k: torch.zeros(len(v["z"]), D) for k in ks}
    shifts = {k: torch.zeros(len(v["z"]), D) for k in ks}
    nn1_err = torch.zeros(len(v["z"]))
    for w in torch.unique(v["window"]):
        ti = torch.nonzero(tr["window"] == w).squeeze(1)
        vi = torch.nonzero(v["window"] == w).squeeze(1)
        if len(ti) < max(ks):
            continue
        zt = F.normalize(tr["z"][ti].to(device), dim=-1).half()
        ft = tr["f"][ti].to(device)
        dt = ft - tr["z"][ti].to(device)
        for c in vi.split(chunk):
            zq = F.normalize(v["z"][c].to(device), dim=-1).half()
            top = (zq @ zt.T).topk(max(ks), dim=-1).indices  # [C, kmax]
            for k in ks:
                preds[k][c] = ft[top[:, :k]].mean(1).cpu()
                shifts[k][c] = v["z"][c] + dt[top[:, :k]].mean(1).cpu()
            nn1_err[c] = ((ft[top[:, 0]] - v["f"][c].to(device)) ** 2).mean(-1).cpu()
    return preds, shifts, nn1_err


def kmeans(x: torch.Tensor, k: int, iters: int, device: str, seed: int) -> torch.Tensor:
    """Spherical-ish k-means on rows of x (cosine assignment); returns centroids."""
    g = torch.Generator().manual_seed(seed)
    xs = F.normalize(x.to(device).float(), dim=-1)
    cent = xs[torch.randperm(len(xs), generator=g)[:k].to(device)].clone()
    for _ in range(iters):
        a = torch.cat([(c @ cent.T).argmax(-1) for c in xs.split(65536)])
        sums = torch.zeros_like(cent).index_add_(0, a, xs)
        counts = torch.bincount(a, minlength=k)
        empty = counts == 0
        cent = torch.where(empty[:, None], cent, sums / counts.clamp_min(1)[:, None])
        cent = F.normalize(cent, dim=-1)
    return cent


def assign(x: torch.Tensor, cent: torch.Tensor, device: str) -> torch.Tensor:
    """Nearest centroid (cosine) of each row."""
    xs = F.normalize(x.to(device).float(), dim=-1)
    return torch.cat([(c @ cent.T).argmax(-1) for c in xs.split(65536)]).cpu()


def cluster_prior(
    tr: dict, v: dict, k: int, device: str, seed: int
) -> dict[str, torch.Tensor]:
    """``prior`` and ``prior shift`` predictions from (cluster, window) means; global window means as fallback."""
    cent = kmeans(tr["z"], k, iters=20, device=device, seed=seed) if k > 1 else None
    ct = (
        assign(tr["z"], cent, device)
        if cent is not None
        else torch.zeros(len(tr["z"]), dtype=torch.long)
    )
    cv = (
        assign(v["z"], cent, device)
        if cent is not None
        else torch.zeros(len(v["z"]), dtype=torch.long)
    )
    n_w = int(max(tr["window"].max(), v["window"].max())) + 1
    key_t, key_v = ct * n_w + tr["window"], cv * n_w + v["window"]
    n_key = k * n_w
    D = tr["f"].shape[1]
    sum_f = torch.zeros(n_key, D).index_add_(0, key_t, tr["f"].float())
    sum_d = torch.zeros(n_key, D).index_add_(0, key_t, (tr["f"] - tr["z"]).float())
    cnt = torch.bincount(key_t, minlength=n_key).float()[:, None]
    # Fallback for (cluster, window) pairs unseen in training: the window's global mean.
    wsum_f = torch.zeros(n_w, D).index_add_(0, tr["window"], tr["f"].float())
    wsum_d = torch.zeros(n_w, D).index_add_(
        0, tr["window"], (tr["f"] - tr["z"]).float()
    )
    wcnt = torch.bincount(tr["window"], minlength=n_w).float()[:, None].clamp_min(1)
    seen = cnt[key_v].squeeze(1) > 0
    mean_f = torch.where(
        seen[:, None],
        sum_f[key_v] / cnt[key_v].clamp_min(1),
        wsum_f[v["window"]] / wcnt[v["window"]],
    )
    mean_d = torch.where(
        seen[:, None],
        sum_d[key_v] / cnt[key_v].clamp_min(1),
        wsum_d[v["window"]] / wcnt[v["window"]],
    )
    return {
        "prior": mean_f,
        "prior shift": v["z"].float() + mean_d,
        "unseen_frac": 1 - seen.float().mean(),
    }


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache_dir", nargs="+", required=True)
    parser.add_argument("--retriever_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument(
        "--clusters", type=int, nargs="*", default=[1, 10, 100, 1000, 10000]
    )
    parser.add_argument("--val_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val, _ = split(
        load_cache(args.cache_dir, quantize=True), 0.1, 0.1, args.seed
    )
    tr = {k: x.float() if x.is_floating_point() else x for k, x in rows(train).items()}
    v = {
        k: x.float() if x.is_floating_point() else x
        for k, x in rows(val, args.val_samples).items()
    }
    logger.info(
        f"train rows {len(tr['z'])}, val rows {len(v['z'])}, windows {len(torch.unique(v['window']))}"
    )
    results: dict = {"floor": score(v["z"], v)}

    # Reference retriever on the same rows.
    meta = json.loads((Path(args.retriever_dir) / "metrics.json").read_text())
    model = build(argparse.Namespace(**meta["args"]), dim=v["z"].shape[1])
    model.load_state_dict(
        torch.load(Path(args.retriever_dir) / "model.pt", map_location="cpu")
    )
    model = model.to(device).eval()
    with torch.no_grad():
        pred = torch.cat(
            [
                retrieve(
                    model,
                    v["z"][c].to(device),
                    v["start_month"][c].to(device),
                    v["start"][c].to(device),
                    v["length"][c].to(device),
                ).cpu()
                for c in torch.arange(len(v["z"])).split(65536)
            ]
        )
    results["retriever"] = score(pred, v)
    logger.info(f"retriever: {results['retriever']}")

    # A: nearest neighbours in annual space, same window, other (training) tiles.
    preds, shifts, nn1 = knn(tr, v, args.ks, device)
    for k in args.ks:
        results[f"knn{k}"] = score(preds[k], v)
        results[f"knn shift{k}"] = score(shifts[k], v)
        logger.info(
            f"knn k={k}: {results[f'knn{k}']} | shift {results[f'knn shift{k}']}"
        )
    floor_mse = ((v["z"] - v["f"]) ** 2).mean().item()
    noise = nn1.mean().item() / 2
    results["cover_hart"] = {
        "noise_mse": noise,
        "floor_mse": floor_mse,
        "ceiling_mse_closed": 1 - noise / floor_mse,
    }
    logger.info(
        f"Cover-Hart: noise MSE ~{noise:.5f} of floor {floor_mse:.5f} -> ceiling ~{100 * (1 - noise / floor_mse):.1f}% of MSE closable"
    )

    # B: cluster prior.
    for k in args.clusters:
        out = cluster_prior(tr, v, k, device, args.seed)
        for name in ("prior", "prior shift"):
            results[f"{name} K={k}"] = score(out[name], v)
        results[f"unseen K={k}"] = float(out["unseen_frac"])
        logger.info(
            f"K={k}: prior {results[f'prior K={k}']} | prior shift {results[f'prior shift K={k}']} | unseen {float(out['unseen_frac']):.3f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("summary (gap closed / within-tile closed / MSE closed):")
    for name, r in results.items():
        if isinstance(r, dict) and "gap_closed" in r:
            logger.info(
                f"  {name:22s} {100 * r['gap_closed']:6.1f}% {100 * r['within_tile_closed']:6.1f}% {100 * r['mse_closed']:6.1f}%"
            )
    logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
