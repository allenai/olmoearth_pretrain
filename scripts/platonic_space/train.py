"""Train a three-way vec2vec translator on paired pixel triples (build_pairs.py output).

  python train.py --run <name> [--mode paired|self|unpaired] [--latent 128] [--w_align 1] ...

The whole pair set sits on the GPU (~7.5M train pixels x 320 dims in float16). Every
--eval_every steps it reports translation fidelity on the val split (see fidelity()), and
writes <run_dir>/ckpt.pt, which a restart with the same --run resumes from. At the end it
scores the test split into <run_dir>/test_metrics.json.
"""

import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from vec2vec import (
    DIMS,
    Discriminator,
    Vec2Vec,
    lsgan_disc_loss,
    lsgan_gen_loss,
    paired_loss,
    unpaired_terms,
)

ROOT = "/weka/dfive-default/hadriens/platonic_space"


@dataclasses.dataclass
class Config:
    """Run settings; every field is a --flag."""

    run: str
    data_dir: str = f"{ROOT}/pairs2024"
    run_root: str = f"{ROOT}/runs"
    mode: str = "paired"
    models: str = "aef,oe,tess"
    latent: int = 128
    width: int = 512
    depth: int = 2
    backbone_depth: int = 3
    dropout: float = 0.0
    steps: int = 20000
    batch: int = 4096
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 1000
    w_rec: float = 1.0
    w_trans: float = 1.0
    w_align: float = 1.0
    w_cyc: float = 0.0
    w_vsp: float = 0.0
    tau: float = 0.05
    # unpaired (the paper's unsupervised.toml: LSGAN, cc 10, vsp 1, disc lr = lr / 2)
    w_gan: float = 1.0
    w_latent_gan: float = 1.0
    disc_lr: float = 0.0
    eval_every: int = 2000
    seed: int = 0


def load_split(
    data_dir: str, split: str, models: list[str], device: torch.device
) -> dict:
    """One split's products (float16, NaN = missing) and tile ids, on device."""
    z = np.load(f"{data_dir}/{split}.npz")
    out = {m: torch.from_numpy(z[m]).to(device) for m in models}
    out["tile"] = torch.from_numpy(z["tile"]).to(device)
    return out


def nan_stats(
    x: torch.Tensor, chunk: int = 1 << 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean / std over the rows with finite values."""
    s = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    s2, n = s.clone(), 0
    for i in range(0, len(x), chunk):
        c = x[i : i + chunk].double()
        c = c[torch.isfinite(c).all(1)]
        s += c.sum(0)
        s2 += (c * c).sum(0)
        n += len(c)
    mu = s / n
    return mu.float(), (s2 / n - mu * mu).clamp_min(1e-12).sqrt().float()


def batch_inputs(model: Vec2Vec, data: dict, idx: torch.Tensor, models: list[str]):
    """Standardised inputs (zeros where invalid) and validity masks for rows idx."""
    xs, valid = {}, {}
    for m in models:
        x = data[m][idx].float()
        ok = torch.isfinite(x).all(1)
        xs[m] = torch.where(ok[:, None], model.standardize(m, torch.nan_to_num(x)), 0.0)
        valid[m] = ok
    return xs, valid


@torch.no_grad()
def fidelity(
    model: Vec2Vec, data: dict, models: list[str], max_px: int = 131072
) -> dict:
    """Translation fidelity, the paper's metrics plus R^2.

    - r2_{m}_{n}: variance-weighted R^2 of translation m -> n in standardised units (rec when
      m == n), comparable to piperw's pilot2024 cross-prediction table;
    - cos_{m}_{n}: mean cosine between translation and truth in the product's own units;
    - top1 / rank_{m}_{n}: retrieval among one pixel per tile (truth vs all other tiles'
      pixels), by cosine in the product's own units; lat_top1 / lat_rank use the latents.
    """
    model.eval()
    n_all = len(data["tile"])
    g = torch.Generator(device="cpu").manual_seed(0)
    sub = torch.randperm(n_all, generator=g)[: min(max_px, n_all)].to(
        data["tile"].device
    )
    tile = data["tile"][sub]
    order = torch.argsort(tile, stable=True)
    first = torch.ones_like(order, dtype=torch.bool)
    first[1:] = tile[order][1:] != tile[order][:-1]
    one_per_tile = order[first]
    xs, valid = batch_inputs(model, data, sub, models)
    u = {m: torch.cat([model.latent(m, c) for c in xs[m].split(16384)]) for m in models}
    out = {}
    for m in models:
        for n in models:
            ok = valid[m] & valid[n]
            if ok.sum() < 2:
                continue
            y = torch.cat([model.decode(n, c) for c in u[m].split(16384)])
            t = xs[n][ok]
            out[f"r2_{m}_{n}"] = float(
                1 - ((y[ok] - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()
            )
            if m == n:
                continue
            yr, tr = model.unstandardize(n, y), model.unstandardize(n, xs[n])
            out[f"cos_{m}_{n}"] = float(
                F.cosine_similarity(yr[ok], tr[ok], dim=-1).mean()
            )
            q = one_per_tile[ok[one_per_tile]]
            for tag, a, b in (("", yr[q], tr[q]), ("lat_", u[m][q], u[n][q])):
                sim = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
                rank = (sim > sim.diag()[:, None]).sum(1).float() + 1
                out[f"{tag}top1_{m}_{n}"] = float((rank == 1).float().mean())
                out[f"{tag}rank_{m}_{n}"] = float(rank.mean())
            out[f"n_cand_{m}_{n}"] = len(q)
    model.train()
    return out


def lr_at(step: int, cfg: Config, base: float) -> float:
    """Linear warmup, then cosine decay to 0."""
    if step < cfg.warmup:
        return base * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main() -> None:
    """Train, evaluate on val every eval_every steps, then score test."""
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        p.add_argument(
            f"--{f.name}", type=f.type, default=f.default, required=f.name == "run"
        )
    cfg = Config(**vars(p.parse_args()))
    run_dir = f"{cfg.run_root}/{cfg.run}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = cfg.models.split(",")
    train = load_split(cfg.data_dir, "train", models, device)
    val = load_split(cfg.data_dir, "val", models, device)
    model = Vec2Vec(
        {m: DIMS[m] for m in models},
        cfg.latent,
        cfg.width,
        cfg.depth,
        cfg.backbone_depth,
        cfg.dropout,
    ).to(device)
    model.set_stats({m: nan_stats(train[m]) for m in models})
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    discs, dopt = None, None
    if cfg.mode == "unpaired":
        # Disjoint pixels per product: product j sees only tiles with tile % len(models) == j.
        own = {
            m: (train["tile"] % len(models) == j) & torch.isfinite(train[m][:, 0])
            for j, m in enumerate(models)
        }
        own_idx = {m: o.nonzero().squeeze(1) for m, o in own.items()}
        discs = torch.nn.ModuleDict(
            {f"out_{m}": Discriminator(DIMS[m]) for m in models}
            | {f"lat_{m}": Discriminator(cfg.latent) for m in models}
        ).to(device)
        dopt = torch.optim.AdamW(
            discs.parameters(), lr=cfg.disc_lr or cfg.lr / 2, weight_decay=0.0
        )
    step = 0
    ckpt_path = f"{run_dir}/ckpt.pt"
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if discs is not None:
            discs.load_state_dict(st["discs"])
            dopt.load_state_dict(st["dopt"])
        step = st["step"]
        print(f"resumed {ckpt_path} at step {step}", flush=True)
    json.dump(dataclasses.asdict(cfg), open(f"{run_dir}/config.json", "w"), indent=1)
    wandb = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb  # type: ignore[no-redef]

        wandb.init(
            project="platonic_space",
            name=cfg.run,
            id=cfg.run,
            resume="allow",
            config=dataclasses.asdict(cfg),
        )
    w = {
        "rec": cfg.w_rec,
        "trans": cfg.w_trans,
        "align": cfg.w_align,
        "cyc": cfg.w_cyc,
        "vsp": cfg.w_vsp,
    }
    if cfg.mode == "self":
        w = {"rec": cfg.w_rec}
    n_train = len(train["tile"])
    metrics_f = open(f"{run_dir}/metrics.jsonl", "a")
    t0 = time.time()
    model.train()
    while step < cfg.steps:
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, cfg, cfg.lr)
        logs: dict[str, float] = {}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if cfg.mode in ("paired", "self"):
                idx = torch.randint(0, n_train, (cfg.batch,), device=device)
                xs, valid = batch_inputs(model, train, idx, models)
                loss, logs = paired_loss(model, xs, valid, w, cfg.tau)
            else:
                xs = {}
                for m in models:
                    ii = own_idx[m][
                        torch.randint(0, len(own_idx[m]), (cfg.batch,), device=device)
                    ]
                    xs[m] = batch_inputs(model, train, ii, [m])[0][m]
                terms, fakes, u = unpaired_terms(model, xs)
                gan = sum(
                    lsgan_gen_loss(discs[f"out_{n}"], fakes[n]) for n in models
                ) / len(models)
                lat_gan = sum(
                    lsgan_gen_loss(discs[f"lat_{m}"], u[m]) for m in models
                ) / len(models)
                loss = (
                    cfg.w_rec * terms["rec"]
                    + cfg.w_cyc * terms["cyc"]
                    + cfg.w_vsp * terms["vsp"]
                    + cfg.w_gan * gan
                    + cfg.w_latent_gan * lat_gan
                )
                logs = {k: float(v) for k, v in terms.items()} | {
                    "gen": float(gan),
                    "lat_gen": float(lat_gan),
                }
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if discs is not None:
            for gr in dopt.param_groups:
                gr["lr"] = lr_at(step, cfg, cfg.disc_lr or cfg.lr / 2)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                d_loss = 0.0
                for n in models:
                    d_loss = d_loss + lsgan_disc_loss(
                        discs[f"out_{n}"], xs[n], fakes[n]
                    )
                for m in models:
                    others = torch.cat([u[k] for k in models if k != m])
                    d_loss = d_loss + lsgan_disc_loss(discs[f"lat_{m}"], others, u[m])
            dopt.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discs.parameters(), 1.0)
            dopt.step()
            logs["disc"] = float(d_loss)
        step += 1
        if step % 100 == 0:
            logs |= {
                "loss": float(loss),
                "grad_norm": float(gn),
                "lr": opt.param_groups[0]["lr"],
                "sec": time.time() - t0,
            }
            if step % 1000 == 0:
                print(
                    f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in logs.items()),
                    flush=True,
                )
            if wandb:
                wandb.log({f"train/{k}": v for k, v in logs.items()}, step=step)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = fidelity(model, val, models)
            print(
                f"val step {step} "
                + " ".join(
                    f"{k}={v:.4g}" for k, v in ev.items() if not k.startswith("n_")
                ),
                flush=True,
            )
            metrics_f.write(json.dumps({"step": step, "split": "val", **ev}) + "\n")
            metrics_f.flush()
            if wandb:
                wandb.log({f"val/{k}": v for k, v in ev.items()}, step=step)
            st = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "config": dataclasses.asdict(cfg),
            }
            if discs is not None:
                st |= {"discs": discs.state_dict(), "dopt": dopt.state_dict()}
            torch.save(st, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    del train
    test = load_split(cfg.data_dir, "test", models, device)
    ev = fidelity(model, test, models)
    json.dump({"step": step, **ev}, open(f"{run_dir}/test_metrics.json", "w"), indent=1)
    print(
        "test "
        + " ".join(f"{k}={v:.4g}" for k, v in ev.items() if not k.startswith("n_")),
        flush=True,
    )


if __name__ == "__main__":
    main()
