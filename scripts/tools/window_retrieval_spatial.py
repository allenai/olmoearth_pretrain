r"""Window retriever with spatial context, trained on the dense cache.

A frozen per-pixel retriever (``window_retrieval_train.py --dense_dirs``, e.g.
``dense_mse``) is the trunk. A dilated-convolution branch reads a crop of the
annual embedding map plus the trunk's per-pixel retrieval and adds a residual
correction, so each pixel's window embedding can use its neighbourhood (which
field it is in, what surrounds it). The branch output starts at zero, so
training starts exactly at the per-pixel model.

Branch: 1x1 conv to ``--width``, ``len(--dilations)`` residual blocks of
GroupNorm -> window-modulated (FiLM) 3x3 dilated conv -> GELU -> 1x1 conv, then a
zero-initialized 1x1 conv back to the embedding size. With dilations 1, 2, 4, 8
the receptive field is 31 px (~300 m). The window code is its own WindowCode
(optionally with lat/lon).

Data: random ``--crop`` x ``--crop`` crops of random dense training tiles, read
in place through memory-mapped chunks in DataLoader workers; ``--windows`` of each
tile's 8 stored windows, valid first. Loss: MSE on the central ``--core`` x
``--core`` pixels, so every scored pixel has its whole receptive field.

Evaluation: ``--eval_tiles`` dense val tiles (the held-out samples of
``--cache_dir``, never trained on), centre crop, every valid window; annual
(floor), the trunk alone, and trunk + branch on exactly the same pixels: cosine
gap closed and within-crop gap closed (each (tile, window)'s mean subtracted).

Usage::

    python scripts/tools/window_retrieval_spatial.py --cache_dir $CAL20K --dense_dirs $D1 $D2 \\
        --trunk_dir $W/lr_dense/dense_mse --out_dir $OUT --run_name spatial
"""

import argparse
import copy
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_retrieval_train import (  # noqa: E402
    MONTHS,
    WindowCode,
    build,
    dequantize_embeddings,
    load_cache,
    npz_array_location,
    npz_memmap,
    retrieve,
    split,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Pre-norm residual block: GroupNorm, window-modulated dilated 3x3 conv, GELU, 1x1 conv."""

    def __init__(self, width: int, code_dim: int, dilation: int) -> None:
        """Modulation starts at zero, so the block starts as identity."""
        super().__init__()
        self.norm = nn.GroupNorm(8, width, affine=False)
        self.conv = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv2d(width, width, 1)
        self.mod = nn.Linear(code_dim, 3 * width)  # shift, scale, gate
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """H + gate * mix(GELU(conv((1 + scale) * norm(h) + shift)))."""
        shift, scale, gate = (x[:, :, None, None] for x in self.mod(c).chunk(3, dim=-1))
        return h + gate * self.mix(
            F.gelu(self.conv(self.norm(h) * (1 + scale) + shift))
        )


class SpatialBranch(nn.Module):
    """Residual correction from a neighbourhood of annual embeddings and per-pixel retrievals."""

    def __init__(
        self,
        dim: int,
        width: int,
        dilations: list[int],
        code_dim: int,
        use_location: bool,
    ) -> None:
        """Zero-initialized output: the branch starts as no correction."""
        super().__init__()
        self.code = WindowCode(code_dim, use_location=use_location)
        self.inp = nn.Conv2d(2 * dim, width, 1)
        self.blocks = nn.ModuleList(ConvBlock(width, code_dim, d) for d in dilations)
        self.out = nn.Conv2d(width, dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        z: Tensor,
        base: Tensor,
        sm: Tensor,
        st: Tensor,
        ln: Tensor,
        latlon: Tensor | None,
    ) -> Tensor:
        """``z``, ``base``: [B, H, W, D] (annual, per-pixel retrieval); window fields: [B]."""
        c = self.code(sm, st, ln, latlon)
        h = self.inp(torch.cat([z, base], dim=-1).permute(0, 3, 1, 2))
        for block in self.blocks:
            h = block(h, c)
        return self.out(h).permute(0, 2, 3, 1)


def per_pixel(
    trunk: nn.Module, z: Tensor, sm: Tensor, st: Tensor, ln: Tensor
) -> Tensor:
    """The frozen trunk on every pixel of [B, H, W, D] maps (window fields per map)."""
    B, H, W, D = z.shape
    rep = lambda x: x[:, None].expand(B, H * W).reshape(-1)  # noqa: E731
    with (
        torch.no_grad(),
        torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.is_cuda),
    ):
        out = retrieve(trunk, z.reshape(-1, D), rep(sm), rep(st), rep(ln))
    return out.float().reshape(B, H, W, D)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def index_chunks(dirs: list[str], workers: int = 32) -> tuple[list[str], list[dict]]:
    """Every chunk file and its small per-sample arrays plus the embeddings' locations."""
    from concurrent.futures import ThreadPoolExecutor

    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "shard*.npz")))
    if not files:
        raise FileNotFoundError(f"no shard*.npz under {dirs}")

    def one(path: str) -> dict:
        with np.load(path) as d:
            out = {
                "index": d["index"],
                "win": d["win"].astype(np.int64),
                "valid": d["present"].sum(-1) > 0,
                "start_month": d["start_month"].astype(np.int64),
            }
        out["loc_z_T"] = npz_array_location(path, "z_T")
        out["loc_z_w"] = npz_array_location(path, "z_w")
        return out

    with ThreadPoolExecutor(workers) as pool:
        return files, list(pool.map(one, files))


def window_order(ch: dict, i: int, gen: np.random.Generator) -> np.ndarray:
    """Sample ``i``'s window slots, valid ones first, random order."""
    order = gen.permutation(ch["win"].shape[1])
    return order[np.argsort(~ch["valid"][i][order], kind="stable")]


class _Crops(torch.utils.data.IterableDataset):
    """Endless batches of random crops, as numpy arrays (built in worker processes)."""

    def __init__(self, files, small, samples, tiles, crop, windows, seed) -> None:
        """Keep what the workers need; memory maps are opened lazily per worker."""
        super().__init__()
        self.files, self.small, self.samples = files, small, samples
        self.tiles, self.crop, self.windows, self.seed = tiles, crop, windows, seed

    def __iter__(self):
        """Yield (z [T,C,C,D], z_w [T,K,C,C,D], sm [T], st [T,K], ln [T,K], valid [T,K], index [T])."""
        info = torch.utils.data.get_worker_info()
        gen = np.random.default_rng([self.seed, info.id if info is not None else 0])
        maps: dict = {}
        C = self.crop
        while True:
            zs, ts, sms, sts, lns, vas, ids = [], [], [], [], [], [], []
            for c, i in self.samples[gen.integers(len(self.samples), size=self.tiles)]:
                ch = self.small[c]
                if c not in maps:
                    maps[c] = (
                        npz_memmap(self.files[c], "z_T", ch["loc_z_T"]),
                        npz_memmap(self.files[c], "z_w", ch["loc_z_w"]),
                    )
                zT, zw = maps[c]
                S = zT.shape[1]
                r0, c0 = gen.integers(S - C + 1), gen.integers(S - C + 1)
                w = window_order(ch, i, gen)[: self.windows]
                zs.append(np.asarray(zT[i, r0 : r0 + C, c0 : c0 + C]))
                ts.append(
                    np.stack(
                        [np.asarray(zw[i, k, r0 : r0 + C, c0 : c0 + C]) for k in w]
                    )
                )
                sms.append(ch["start_month"][i])
                sts.append(ch["win"][i, w, 0])
                lns.append(ch["win"][i, w, 1])
                vas.append(ch["valid"][i, w])
                ids.append(ch["index"][i])
            yield tuple(
                np.stack(x) if j != 2 and j != 6 else np.array(x)
                for j, x in enumerate((zs, ts, sms, sts, lns, vas, ids))
            )


def to_device(batch: tuple, device: str) -> tuple:
    """Dequantize embeddings; move everything to ``device``."""
    z, t, sm, st, ln, va, ids = batch
    deq = lambda a: dequantize_embeddings(torch.as_tensor(a).to(device)).float()  # noqa: E731
    return (
        deq(z),
        deq(t),
        *(torch.as_tensor(a).to(device) for a in (sm, st, ln, va, ids)),
    )


def eval_set(
    files: list[str], small: list[dict], held_out: list[int], n_tiles: int, crop: int
) -> tuple:
    """Centre crops of up to ``n_tiles`` held-out dense tiles, every stored window."""
    want = set(held_out)
    zs, ts, sms, sts, lns, vas, ids = [], [], [], [], [], [], []
    for c, ch in enumerate(small):
        for i, idx in enumerate(ch["index"]):
            if int(idx) not in want or len(ids) >= n_tiles:
                continue
            zT, zw = (
                npz_memmap(files[c], "z_T", ch["loc_z_T"]),
                npz_memmap(files[c], "z_w", ch["loc_z_w"]),
            )
            S = zT.shape[1]
            r0 = (S - crop) // 2
            zs.append(np.asarray(zT[i, r0 : r0 + crop, r0 : r0 + crop]))
            ts.append(np.asarray(zw[i, :, r0 : r0 + crop, r0 : r0 + crop]))
            sms.append(ch["start_month"][i])
            sts.append(ch["win"][i, :, 0])
            lns.append(ch["win"][i, :, 1])
            vas.append(ch["valid"][i])
            ids.append(idx)
    if not ids:
        raise ValueError(
            "no held-out samples found in the dense cache: wrong --cache_dir / --dense_dirs pair?"
        )
    return tuple(
        np.stack(x) if j != 2 and j != 6 else np.array(x)
        for j, x in enumerate((zs, ts, sms, sts, lns, vas, ids))
    )


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------


def predict(trunk, branch, z, sm, st, ln, ll) -> tuple[Tensor, Tensor]:
    """Per-pixel and spatial retrieval for [T, C, C, D] crops and [T, K] windows -> [T, K, C, C, D] each."""
    T, C, _, D = z.shape
    K = st.shape[1]
    zz = z[:, None].expand(T, K, C, C, D).reshape(T * K, C, C, D)
    sm_, st_, ln_ = sm[:, None].expand(T, K).reshape(-1), st.reshape(-1), ln.reshape(-1)
    base = per_pixel(trunk, zz, sm_, st_, ln_)
    ll_ = None if ll is None else ll[:, None].expand(T, K, 2).reshape(-1, 2)
    out = base + branch(zz, base, sm_, st_, ln_, ll_)
    # Full-year windows return the annual vector exactly, as for the per-pixel retriever.
    out = torch.where((ln_ == MONTHS)[:, None, None, None], zz, out)
    return base.reshape(T, K, C, C, D), out.reshape(T, K, C, C, D)


def core(x: Tensor, crop: int, size: int) -> Tensor:
    """Central ``size`` x ``size`` pixels of [..., C, C, D] maps."""
    a = (crop - size) // 2
    return x[..., a : a + size, a : a + size, :]


@torch.no_grad()
def evaluate(trunk, branch, data: tuple, args, latlon_table, device: str) -> dict:
    """Annual, per-pixel and spatial retrieval on the eval crops' cores."""
    branch.eval()
    sums = {
        k: [0.0, 0.0, 0.0] for k in ("floor", "pixel", "spatial")
    }  # cos sum, local se, rows
    z_all, t_all, sm_all, st_all, ln_all, va_all, id_all = data
    for s in range(0, len(z_all), 8):
        z, t, sm, st, ln, va, ids = to_device(
            tuple(
                a[s : s + 8]
                for a in (z_all, t_all, sm_all, st_all, ln_all, va_all, id_all)
            ),
            device,
        )
        ll = latlon_table[ids] if latlon_table is not None else None
        base, out = predict(trunk, branch, z, sm, st, ln, ll)
        T, K = st.shape
        zf = z[:, None].expand_as(t)
        cores = {
            k: core(v, args.crop, args.core)
            for k, v in (("floor", zf), ("pixel", base), ("spatial", out))
        }
        tgt = core(t, args.crop, args.core)
        keep = va & (ln < MONTHS)
        tc = tgt - tgt.mean(dim=(2, 3), keepdim=True)
        for k, v in cores.items():
            cos = F.cosine_similarity(v, tgt, dim=-1)[keep]
            vc = v - v.mean(dim=(2, 3), keepdim=True)
            sums[k][0] += float(cos.sum())
            sums[k][1] += float(((vc - tc) ** 2).mean(-1)[keep].sum())
            sums[k][2] += int(cos.numel())
    branch.train()
    out = {}
    floor_cos = sums["floor"][0] / sums["floor"][2]
    floor_local = sums["floor"][1] / sums["floor"][2]
    for k, (cos_sum, local_se, n) in sums.items():
        out[k] = {
            "cos": cos_sum / n,
            "gap_closed": (cos_sum / n - floor_cos) / max(1 - floor_cos, 1e-8),
            "local_mse": local_se / n,
            "local_closed": 1 - (local_se / n) / max(floor_local, 1e-12),
        }
    return out


def log_eval(label: str, m: dict) -> None:
    """One line per variant."""
    for k, v in m.items():
        logger.info(
            f"{label} {k:8s} cos {v['cos']:.4f} gap closed {100 * v['gap_closed']:5.1f}% "
            f"within-crop closed {100 * v['local_closed']:5.1f}%"
        )


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
        "--cache_dir",
        nargs="+",
        required=True,
        help="calendar cache whose val/test samples are held out",
    )
    parser.add_argument("--dense_dirs", nargs="+", required=True)
    parser.add_argument(
        "--trunk_dir",
        required=True,
        help="per-pixel retriever run dir (metrics.json + model.pt)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--code_dim", type=int, default=128)
    parser.add_argument("--latlon", action="store_true")
    parser.add_argument(
        "--latlon_file",
        default="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
        "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_"
        "worldpop_wri_canopy_height_map/1138828/latlon_distribution.npy",
    )
    parser.add_argument("--tiles", type=int, default=4, help="crops per batch")
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument(
        "--core", type=int, default=32, help="scored centre of each crop"
    )
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_tiles", type=int, default=500)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overfit_tiles",
        type=int,
        default=0,
        help="sanity check: train on this many fixed tiles",
    )
    args = parser.parse_args()
    if args.core + 2 * sum(args.dilations) > args.crop:
        raise ValueError("core plus receptive field does not fit in the crop")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, test = split(load_cache(args.cache_dir, quantize=True), 0.1, 0.1, args.seed)
    held_val = [int(i) for i in val["index"].tolist()]
    held_out = set(held_val) | {int(i) for i in test["index"].tolist()}
    del val, test

    meta = json.loads((Path(args.trunk_dir) / "metrics.json").read_text())
    trunk = build(argparse.Namespace(**meta["args"]), dim=128)
    trunk.load_state_dict(
        torch.load(Path(args.trunk_dir) / "model.pt", map_location="cpu")
    )
    trunk = trunk.to(device).eval().requires_grad_(False)
    branch = SpatialBranch(
        128, args.width, args.dilations, args.code_dim, args.latlon
    ).to(device)
    n_params = sum(p.numel() for p in branch.parameters())
    latlon_table = (
        torch.from_numpy(np.load(args.latlon_file)).float().to(device)
        if args.latlon
        else None
    )

    files, small = index_chunks(args.dense_dirs)
    samples = np.array(
        [
            (c, i)
            for c, ch in enumerate(small)
            for i, idx in enumerate(ch["index"])
            if int(idx) not in held_out
        ],
        dtype=np.int64,
    )
    if args.overfit_tiles:
        # Sanity check: train on a few fixed tiles; the branch should push the
        # loss well below the trunk's on them if it can learn at all.
        samples = samples[
            np.random.default_rng(args.seed).choice(
                len(samples), args.overfit_tiles, replace=False
            )
        ]
    eval_data = eval_set(files, small, held_val, args.eval_tiles, args.crop)
    logger.info(
        f"trunk {meta['args'].get('run_name', args.trunk_dir)} frozen; branch {n_params} params, RF {1 + 2 * sum(args.dilations)} px; "
        f"{len(samples)} train tiles, {len(eval_data[0])} eval tiles"
    )
    loader = torch.utils.data.DataLoader(
        _Crops(files, small, samples, args.tiles, args.crop, args.windows, args.seed),
        batch_size=None,
        num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=bool(args.workers),
    )
    it = iter(loader)

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    m = evaluate(trunk, branch, eval_data, args, latlon_table, device)
    log_eval("step 0", m)

    opt = torch.optim.AdamW(
        branch.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )
    best, best_step, best_state, stale, history = (
        -1.0,
        0,
        copy.deepcopy(branch.state_dict()),
        0,
        [],
    )
    t_log, data_wait = time.time(), 0.0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        z, t, sm, st, ln, va, ids = to_device(next(it), device)
        data_wait += time.time() - t0
        ll = latlon_table[ids] if latlon_table is not None else None
        base, out = predict(trunk, branch, z, sm, st, ln, ll)
        keep = (va & (ln < MONTHS))[:, :, None, None].expand(
            *va.shape, args.core, args.core
        )
        tgt_core = core(t, args.crop, args.core)[keep]
        loss = F.mse_loss(core(out, args.crop, args.core)[keep], tgt_core)
        trunk_loss = F.mse_loss(core(base, args.crop, args.core)[keep], tgt_core).item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            dt = time.time() - t_log
            logger.info(
                f"step {step}/{args.steps} loss {loss.item():.5f} (trunk {trunk_loss:.5f}; {200 / dt:.1f} steps/s, data wait {100 * data_wait / dt:.0f}%)"
            )
            t_log, data_wait = time.time(), 0.0
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(trunk, branch, eval_data, args, latlon_table, device)
            score = (m["spatial"]["gap_closed"] + m["spatial"]["local_closed"]) / 2
            history.append({"step": step, **m})
            logger.info(
                f"step {step}: spatial gap {100 * m['spatial']['gap_closed']:.1f}% within-crop {100 * m['spatial']['local_closed']:.1f}% "
                f"(per-pixel {100 * m['pixel']['gap_closed']:.1f}% / {100 * m['pixel']['local_closed']:.1f}%)"
            )
            if score > best:
                best, best_step, best_state, stale = (
                    score,
                    step,
                    copy.deepcopy(branch.state_dict()),
                    0,
                )
            else:
                stale += 1
                if stale >= args.patience:
                    logger.info(f"early stop at step {step}; best step {best_step}")
                    break
    branch.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "branch.pt"))
    m = evaluate(trunk, branch, eval_data, args, latlon_table, device)
    log_eval(f"FINAL (best step {best_step})", m)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "trunk_args": meta["args"],
                "best_step": best_step,
                "final": m,
                "history": history,
            },
            f,
            indent=2,
        )
    logger.info(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
