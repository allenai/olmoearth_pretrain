"""Train a window retriever that reads neighbouring pixels (spatial context).

Like ``window_retrieval_train.py`` (same caches, splits, window code, metrics),
but each example is a whole cached patch: the cache must be written with
``window_retrieval_cache.py --patch_px P`` (one contiguous P x P square per
sample). The retriever is FiLM with ``--context_layers`` 3x3 convolutional
blocks in front of the per-pixel blocks, each modulated by the window, so a
pixel's retrieval sees a (2 * context_layers + 1)-pixel square around it.

Loss and metrics use only the inner pixels whose whole neighbourhood lies
inside the patch, so no pixel is scored on zero padding. ``--context_layers 0``
is per-pixel FiLM scored on exactly those pixels: the matched baseline.

Usage::

    python scripts/tools/window_retrieval_train_spatial.py --cache_dir $PATCHES --out_dir $OUT \
        --context_layers 2
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_retrieval_train import (  # noqa: E402
    FiLMBlock,
    WindowCode,
    concat,
    examples,
    load_cache,
    log_metrics,
    split,
    subset,
)

logger = logging.getLogger(__name__)


class ConvFiLMBlock(nn.Module):
    """Residual 3x3 conv over the patch, scaled/shifted/gated by the window (AdaLN-Zero)."""

    def __init__(self, width: int, code_dim: int) -> None:
        """Modulation starts at zero, so the block starts as the identity."""
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.conv = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.mod = nn.Linear(code_dim, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """h: [B, P, P, W]; c: [B, code_dim]."""
        shift, scale, gate = (t[:, None, None, :] for t in self.mod(c).chunk(3, dim=-1))
        x = F.gelu(self.norm(h) * (1 + scale) + shift)
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return h + gate * x


class SpatialFiLM(nn.Module):
    """FiLM retriever with convolutional context; maps [B, P, P, D] patches."""

    def __init__(
        self, dim: int, width: int, depth: int, code_dim: int, context_layers: int
    ) -> None:
        """Initialize at the floor: every modulation and the output gate start at zero."""
        super().__init__()
        self.code = WindowCode(code_dim)
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.context = nn.ModuleList(
            ConvFiLMBlock(width, code_dim) for _ in range(context_layers)
        )
        self.blocks = nn.ModuleList(FiLMBlock(width, code_dim) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
        self.alpha = nn.Linear(code_dim, dim)
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

    def forward(
        self, z: Tensor, start_month: Tensor, start: Tensor, length: Tensor
    ) -> Tensor:
        """z: [B, P, P, D]; window fields [B]."""
        c = self.code(start_month, start, length)
        h = self.inp(z)
        for block in self.context:
            h = block(h, c)
        cp = c[:, None, None, :]
        for block in self.blocks:
            h = block(h, cp)
        return z + self.alpha(cp) * self.out(h)


def patch_side(d: dict[str, Tensor]) -> int:
    """Side of the square patch the cache stores per sample."""
    n = d["z_T"].shape[1]
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(
            f"{n} pixels per sample is not a square patch; cache with --patch_px"
        )
    return side


def gather_patches(
    d: dict[str, Tensor], pairs: Tensor, side: int
) -> tuple[Tensor, ...]:
    """Patches, targets and window fields for (sample, window) pairs."""
    s, k = pairs[:, 0], pairs[:, 1]
    shape = (len(pairs), side, side, -1)
    return (
        d["z_T"][s].float().reshape(shape),
        d["z_w"][s, k].float().reshape(shape),
        d["start_month"][s],
        d["start"][s, k],
        d["length"][s, k],
    )


def inner(x: Tensor, margin: int) -> Tensor:
    """The pixels at least ``margin`` from every patch edge, flattened: [N, D]."""
    if margin:
        x = x[:, margin:-margin, margin:-margin]
    return x.reshape(-1, x.shape[-1])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    d: dict[str, Tensor],
    side: int,
    margin: int,
    device: str,
    seed: int,
) -> dict:
    """Per-window-length metrics on the inner pixels of every valid (sample, window)."""
    model.eval()
    pairs = examples(d)
    g = torch.Generator(device="cpu").manual_seed(seed)
    stats: dict[int, dict[str, list]] = {}
    for chunk in pairs.split(64):
        z, tgt, sm, st, ln = (t.to(device) for t in gather_patches(d, chunk, side))
        pred = model(z, sm, st, ln)
        perm = torch.randperm(len(st), generator=g).to(device)
        wrong = model(z, sm, st[perm], ln[perm])
        z_i, t_i = inner(z, margin), inner(tgt, margin)
        p_i, w_i = inner(pred, margin), inner(wrong, margin)
        per_pix = t_i.shape[0] // len(chunk)
        ln_pix = ln.repeat_interleave(per_pix)
        rows = {
            "cos_floor": F.cosine_similarity(z_i, t_i, dim=-1),
            "cos_ret": F.cosine_similarity(p_i, t_i, dim=-1),
            "cos_wrong": F.cosine_similarity(w_i, t_i, dim=-1),
            "mse_floor": ((z_i - t_i) ** 2).mean(-1),
            "mse_ret": ((p_i - t_i) ** 2).mean(-1),
        }
        for L in ln.unique().tolist():
            sel = ln_pix == L
            entry = stats.setdefault(L, {key: [] for key in rows})
            for key, v in rows.items():
                entry[key].append(v[sel].float().cpu())
    model.train()
    out: dict = {}
    for L in sorted(stats):
        m = {key: torch.cat(v).mean().item() for key, v in stats[L].items()}
        m["gap_closed"] = (m["cos_ret"] - m["cos_floor"]) / max(
            1 - m["cos_floor"], 1e-8
        )
        m["n"] = int(sum(len(x) for x in stats[L]["cos_ret"]))
        out[L] = m
    total = sum(m["n"] for m in out.values())
    out["all"] = {
        key: sum(m[key] * m["n"] for m in out.values()) / total
        for key in (
            "cos_floor",
            "cos_ret",
            "cos_wrong",
            "mse_floor",
            "mse_ret",
            "gap_closed",
        )
    }
    return out


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
    parser.add_argument(
        "--cache_dir", nargs="+", required=True, help="calendar-year patch caches"
    )
    parser.add_argument(
        "--extra_train_cache_dir",
        nargs="*",
        default=[],
        help="other start months, train only",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--context_layers",
        type=int,
        default=2,
        help="3x3 conv blocks; 0 = per-pixel FiLM",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--code_dim", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--min_steps", type=int, default=0)
    parser.add_argument("--patches_per_batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantize = not args.no_quantize
    train, val, test = split(
        load_cache(args.cache_dir, quantize), args.val_frac, args.test_frac, args.seed
    )
    if args.max_train_samples:
        train = subset(
            train, torch.arange(min(args.max_train_samples, train["z_T"].shape[0]))
        )
    if args.extra_train_cache_dir:
        train = concat(train, load_cache(args.extra_train_cache_dir, quantize))
    side = patch_side(train)
    margin = args.context_layers  # each 3x3 block reaches one pixel further
    train = {k: v.to(device) for k, v in train.items()}
    pairs = examples(train).to(device)
    steps = max(
        args.min_steps, math.ceil(args.epochs * len(pairs) / args.patches_per_batch)
    )
    logger.info(
        f"train: {train['z_T'].shape[0]} samples, {len(pairs)} windows of {side}x{side} patches "
        f"(scored on the inner {side - 2 * margin}x{side - 2 * margin}); {steps} steps"
    )

    model = SpatialFiLM(
        train["z_T"].shape[-1],
        args.width,
        args.depth,
        args.code_dim,
        args.context_layers,
    ).to(device)
    name = args.run_name or f"spatial_c{args.context_layers}_{int(time.time())}"
    run_dir = os.path.join(args.out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(
        f"SpatialFiLM, {args.context_layers} context layers: {sum(p.numel() for p in model.parameters())} parameters -> {run_dir}"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.05
    )
    best_step, best_cos, best_state, stale, history = (
        0,
        -1.0,
        copy.deepcopy(model.state_dict()),
        0,
        [],
    )
    for step in range(1, steps + 1):
        batch = pairs[
            torch.randint(len(pairs), (args.patches_per_batch,), device=device)
        ]
        z, tgt, sm, st, ln = gather_patches(train, batch, side)
        loss = F.mse_loss(inner(model(z, sm, st, ln), margin), inner(tgt, margin))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            logger.info(f"step {step}/{steps} loss {loss.item():.5f}")
        if step % args.eval_every == 0 or step == steps:
            m = evaluate(model, val, side, margin, device, args.seed)
            history.append({"step": step, "val_all": m["all"]})
            cos = m["all"]["cos_ret"]
            logger.info(
                f"step {step}: val cos {cos:.4f} (best {max(best_cos, cos):.4f})"
            )
            if cos > best_cos:
                best_step, best_cos, best_state, stale = (
                    step,
                    cos,
                    copy.deepcopy(model.state_dict()),
                    0,
                )
            else:
                stale += 1
                if stale >= args.patience:
                    logger.info(f"early stop at step {step}; best step {best_step}")
                    break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "model.pt"))
    val_m = evaluate(model, val, side, margin, device, args.seed)
    test_m = evaluate(model, test, side, margin, device, args.seed)
    log_metrics(f"VAL (best step {best_step})", val_m)
    log_metrics(f"TEST (best step {best_step})", test_m)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "best_step": best_step,
                "steps": steps,
                "val": val_m,
                "test": test_m,
                "history": history,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
