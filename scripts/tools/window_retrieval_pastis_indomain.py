"""In-domain retrievers for PASTIS: how much of fresh can be recovered from annual on PASTIS itself?

The global retriever (trained on pretraining samples) closes the PASTIS cosine
gap mostly with generic corrections and leaves the top principal components of
fresh untouched. These retrievers are trained on PASTIS's own embeddings, from
the per-window int8 codes ``window_retrieval_pastis_bake.py embed`` wrote, on
train-split windows only and early-stopped on embedding MSE over val-split
windows. Labels are never read, so the PASTIS eval stays clean.

- Mar-Aug (input calendar 2019, target fresh Mar-Aug):
  * ``ind_maraug_ft``: the global FiLM retriever fine-tuned at the fixed window;
  * ``ind_maraug_scratch``: the same architecture from random init.
- Sep-Aug (inputs calendar 2018 and 2019, target fresh Sep-Aug):
  * ``ind_sepaug_compft``: the bake's composition, 4/12 h(z_2018, Sep-Dec) +
    8/12 h(z_2019, Jan-Aug), with h fine-tuned through it;
  * ``ind_sepaug_mlp``: a residual MLP on [z_2018, z_2019] over the naive
    4/12 + 8/12 average (a learned cross-year composition).

``train`` writes ``<out_dir>/<name>/model.pt`` and ``metrics.json``;
``window_retrieval_pastis_diag.py --indomain_dir`` scores them next to the
global retriever.

Usage::

    python scripts/tools/window_retrieval_pastis_indomain.py train --work_dir $W --retriever_dir $R --out_dir $OUT
"""

import argparse
import copy
import glob
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from window_retrieval_pastis_bake import (  # noqa: E402
    SOURCES,
    TEMPLATE_DS,
    dequantize_embeddings,
    key_to_file,
    load_retriever,
)
from window_retrieval_train import build, retrieve  # noqa: E402

logger = logging.getLogger(__name__)

MARAUG = (2, 6)
SEP_DEC, JAN_AUG = (8, 4), (0, 8)
NAMES = ("ind_maraug_ft", "ind_maraug_scratch", "ind_sepaug_compft", "ind_sepaug_mlp")


def apply_window(
    h: nn.Module, z: torch.Tensor, window: tuple[int, int]
) -> torch.Tensor:
    """Retrieve one fixed window (start, length) for a batch of calendar-year embeddings [N, D]."""
    n = z.shape[0]
    full = lambda v: torch.full((n,), v, dtype=torch.long, device=z.device)  # noqa: E731
    return retrieve(h, z, full(0), full(window[0]), full(window[1]))


class Composition(nn.Module):
    """4/12 h(z_2018, Sep-Dec) + 8/12 h(z_2019, Jan-Aug) with a trainable h."""

    def __init__(self, h: nn.Module) -> None:
        """Wrap a retriever."""
        super().__init__()
        self.h = h

    def forward(self, z18: torch.Tensor, z19: torch.Tensor) -> torch.Tensor:
        """Composed Sep-Aug embedding."""
        return (
            4 * apply_window(self.h, z18, SEP_DEC)
            + 8 * apply_window(self.h, z19, JAN_AUG)
        ) / 12


class ResidualComposition(nn.Module):
    """Naive 4/12 + 8/12 average plus an MLP correction from both years."""

    def __init__(self, dim: int, width: int = 1024, depth: int = 3) -> None:
        """Build the MLP; its last layer starts at zero, so training starts from the average."""
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2 * dim, width), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, dim))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)

    def forward(self, z18: torch.Tensor, z19: torch.Tensor) -> torch.Tensor:
        """Composed Sep-Aug embedding."""
        return (4 * z18 + 8 * z19) / 12 + self.mlp(torch.cat([z18, z19], dim=-1))


def predict(name: str, model: nn.Module, z: dict[str, torch.Tensor]) -> torch.Tensor:
    """One in-domain variant from dequantized source embeddings [N, D]."""
    if name.startswith("ind_maraug"):
        return apply_window(model, z["cal2019"], MARAUG)
    return model(z["cal2018"], z["cal2019"])


def target_of(name: str) -> str:
    """The fresh source each variant is trained against."""
    return "maraug" if name.startswith("ind_maraug") else "sepaug"


def build_model(name: str, retriever_dir: str) -> nn.Module:
    """Initial model for one variant."""
    if name == "ind_maraug_ft":
        return load_retriever(retriever_dir)
    if name == "ind_maraug_scratch":
        # Same architecture at its own initialization.
        meta = json.loads((Path(retriever_dir) / "metrics.json").read_text())
        return build(argparse.Namespace(**meta["args"]), dim=128)
    if name == "ind_sepaug_compft":
        return Composition(load_retriever(retriever_dir))
    return ResidualComposition(dim=128)


def load_models(
    indomain_dir: str, retriever_dir: str, device: str
) -> dict[str, nn.Module]:
    """Trained in-domain models from ``train``'s output."""
    models = {}
    for name in NAMES:
        model = build_model(name, retriever_dir)
        model.load_state_dict(
            torch.load(Path(indomain_dir) / name / "model.pt", map_location="cpu")
        )
        models[name] = model.to(device).eval()
    return models


def window_split(window_dir: str) -> str | None:
    """The window's eval_split option."""
    with open(os.path.join(window_dir, "metadata.json")) as f:
        return json.load(f).get("options", {}).get("eval_split")


def load_pixels(
    work_dir: str, split: str, pixels: int, seed: int, workers: int
) -> dict[str, torch.Tensor]:
    """Random pixels of every source, from the windows of one split, dequantized [N, D]."""
    work = Path(work_dir)
    dirs = [
        d
        for d in sorted(glob.glob(os.path.join(TEMPLATE_DS, "windows", "*", "*")))
        if window_split(d) == split
    ]
    seeds = np.random.default_rng(seed).integers(0, 2**31, len(dirs))

    def handle(i: int) -> dict[str, np.ndarray] | None:
        key = "/".join(Path(dirs[i]).parts[-2:])
        files = {s: work / s / key_to_file(key) for s in SOURCES}
        if not all(f.exists() for f in files.values()):
            return None
        codes = {s: np.load(f) for s, f in files.items()}
        flat = {s: c.reshape(-1, c.shape[-1]) for s, c in codes.items()}
        n = flat["cal2019"].shape[0]
        pick = np.random.default_rng(seeds[i]).choice(n, min(pixels, n), replace=False)
        return {s: f[pick] for s, f in flat.items()}

    with ThreadPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(handle, range(len(dirs))) if r is not None]
    logger.info(f"{split}: {len(rows)} windows")
    return {
        s: dequantize_embeddings(
            torch.from_numpy(np.concatenate([r[s] for r in rows]))
        ).float()
        for s in SOURCES
    }


def fit(
    name: str,
    model: nn.Module,
    train: dict,
    val: dict,
    args: argparse.Namespace,
    device: str,
) -> tuple[nn.Module, dict]:
    """MSE to the fresh target, early-stopped on val MSE."""
    lr = (
        args.lr_scratch
        if name in ("ind_maraug_scratch", "ind_sepaug_mlp")
        else args.lr_ft
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    tgt = target_of(name)
    n = len(train["cal2019"])

    def val_mse() -> float:
        model.eval()
        with torch.no_grad():
            err = [
                (
                    (
                        predict(
                            name, model, {s: v[i].to(device) for s, v in val.items()}
                        )
                        - val[tgt][i].to(device)
                    )
                    ** 2
                )
                .sum(-1)
                .cpu()
                for i in torch.arange(len(val["cal2019"])).split(16384)
            ]
        model.train()
        return float(torch.cat(err).mean())

    best, best_state, bad, history = val_mse(), copy.deepcopy(model.state_dict()), 0, []
    logger.info(f"{name}: lr {lr}, start val MSE {best:.5f}")
    model.train()
    for step in range(1, args.max_steps + 1):
        idx = torch.randint(n, (args.batch,))
        z = {s: v[idx].to(device, non_blocking=True) for s, v in train.items()}
        loss = ((predict(name, model, z) - z[tgt]) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            score = val_mse()
            history.append((step, float(loss), score))
            logger.info(f"{name} step {step}: train {float(loss):.5f} val {score:.5f}")
            if score < best:
                best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
    model.load_state_dict(best_state)
    return model.eval(), {"best_val_mse": best, "lr": lr, "history": history}


def train_all(args: argparse.Namespace) -> None:
    """Train every in-domain variant and write it out."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = load_pixels(
        args.work_dir, "train", args.train_pixels, args.seed, args.workers
    )
    val = load_pixels(
        args.work_dir, "val", args.val_pixels, args.seed + 1, args.workers
    )
    # Reference points: the naive inputs and the global retriever, on the same val pixels.
    with torch.no_grad():
        h = load_retriever(args.retriever_dir).to(device)
        z = {s: v[:200000].to(device) for s, v in val.items()}
        ref = {
            "annual_vs_maraug": float(
                ((z["cal2019"] - z["maraug"]) ** 2).sum(-1).mean()
            ),
            "global_ret_vs_maraug": float(
                ((apply_window(h, z["cal2019"], MARAUG) - z["maraug"]) ** 2)
                .sum(-1)
                .mean()
            ),
            "annual_vs_sepaug": float(
                ((z["cal2019"] - z["sepaug"]) ** 2).sum(-1).mean()
            ),
            "avg_vs_sepaug": float(
                (((4 * z["cal2018"] + 8 * z["cal2019"]) / 12 - z["sepaug"]) ** 2)
                .sum(-1)
                .mean()
            ),
            "global_ret_vs_sepaug": float(
                ((Composition(h)(z["cal2018"], z["cal2019"]) - z["sepaug"]) ** 2)
                .sum(-1)
                .mean()
            ),
        }
    logger.info(f"val MSE references: {ref}")
    for name in args.only or NAMES:
        model = build_model(name, args.retriever_dir).to(device)
        model, meta = fit(name, model, train, val, args, device)
        out = Path(args.out_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "metrics.json").write_text(
            json.dumps({**meta, "references": ref, "args": vars(args)}, indent=2)
        )
        logger.info(f"{name}: best val MSE {meta['best_val_mse']:.5f} -> {out}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--retriever_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--only", nargs="*", choices=NAMES)
    parser.add_argument(
        "--train_pixels", type=int, default=1024, help="pixels sampled per train window"
    )
    parser.add_argument(
        "--val_pixels", type=int, default=256, help="pixels sampled per val window"
    )
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr_ft", type=float, default=1e-4)
    parser.add_argument("--lr_scratch", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_all(args)


if __name__ == "__main__":
    main()
