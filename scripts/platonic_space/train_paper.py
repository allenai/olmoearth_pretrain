"""Train the paper's unsupervised vec2vec (paper.py) on disjoint pixels per product.

  python train_paper.py --run paper_oe_aef_s0 --models oe,aef --seed 0

Product j of --models only ever sees train tiles with tile % len(models) == j, so no pixel
is seen by two products. The first product is the anchor that the latent GAN pulls the
others towards. Every --eval_every steps it scores translation fidelity on the (paired) val
split with train.fidelity, keeps the checkpoint with the best mean translation rank (the
reference's top_rank), and after --min_steps stops once --patience evals pass without a
new best, as the reference's early stopper does. <run_dir>/ckpt.pt makes it resumable; at
the end the best checkpoint is scored on test into <run_dir>/test_metrics.json.
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch
from paper import PaperVec2Vec, generator_terms, init_weights
from train import ROOT, fidelity, load_split
from vec2vec import DIMS, Discriminator, lsgan_disc_loss, lsgan_gen_loss


@dataclasses.dataclass
class Config:
    """Run settings (defaults are configs/unsupervised.toml); every field is a --flag."""

    run: str
    data_dir: str = f"{ROOT}/pairs2024"
    run_root: str = f"{ROOT}/runs"
    models: str = "oe,aef"
    d_model: int = 1024
    steps: int = 1_000_000
    batch: int = 256
    lr: float = 2e-5
    disc_lr: float = 1e-5
    disc_eps: float = 6.25e-10
    warmup: int = 2000
    max_grad_norm: float = 1000.0
    w_rec: float = 1.0
    w_vsp: float = 1.0
    w_cc_trans: float = 10.0
    w_cc_vsp: float = 10.0
    w_gen: float = 1.0
    w_latent_gen: float = 1.0
    eval_every: int = 10000
    min_steps: int = 200000
    patience: int = 20
    seed: int = 0


def mean_rank(ev: dict) -> float:
    """Mean translation retrieval rank over all directions (lower is better)."""
    return float(np.mean([v for k, v in ev.items() if k.startswith("rank_")]))


def main() -> None:
    """Train with early stopping on val rank, then score the best checkpoint on test."""
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        p.add_argument(
            f"--{f.name}", type=f.type, default=f.default, required=f.name == "run"
        )
    cfg = Config(**vars(p.parse_args()))
    run_dir = f"{cfg.run_root}/{cfg.run}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = cfg.models.split(",")
    anchor = models[0]
    train = load_split(cfg.data_dir, "train", models, device)
    val = load_split(cfg.data_dir, "val", models, device)
    own_idx = {
        m: (
            (train["tile"] % len(models) == j) & torch.isfinite(train[m][:, 0])
        ).nonzero()[:, 0]
        for j, m in enumerate(models)
    }
    print({m: len(i) for m, i in own_idx.items()}, "pixels per product", flush=True)
    model = PaperVec2Vec({m: DIMS[m] for m in models}, cfg.d_model).to(device)
    discs = torch.nn.ModuleDict(
        {f"out_{m}": Discriminator(DIMS[m], cfg.d_model) for m in models}
        | {f"lat_{m}": Discriminator(cfg.d_model, cfg.d_model) for m in models[1:]}
    ).to(device)
    init_weights(discs)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    dopts = {
        k: torch.optim.Adam(
            d.parameters(), lr=cfg.disc_lr, eps=cfg.disc_eps, betas=(0.5, 0.999)
        )
        for k, d in discs.items()
    }
    step, best, best_step, bad, stopped = 0, float("inf"), 0, 0, False
    ckpt_path, best_path = f"{run_dir}/ckpt.pt", f"{run_dir}/best.pt"
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        discs.load_state_dict(st["discs"])
        for k, o in dopts.items():
            o.load_state_dict(st["dopts"][k])
        step, best, best_step, bad = st["step"], st["best"], st["best_step"], st["bad"]
        stopped = st["stopped"]
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
    metrics_f = open(f"{run_dir}/metrics.jsonl", "a")
    t0, s0 = time.time(), step
    model.train()
    discs.requires_grad_(False)
    while step < cfg.steps and not stopped:
        warm = min(1.0, step / cfg.warmup)
        for gr in opt.param_groups:
            gr["lr"] = cfg.lr * warm
        for o in dopts.values():
            for gr in o.param_groups:
                gr["lr"] = cfg.disc_lr * warm
        xs = {}
        for m in models:
            ii = own_idx[m][
                torch.randint(0, len(own_idx[m]), (cfg.batch,), device=device)
            ]
            xs[m] = model.standardize(m, train[m][ii].float())
        terms, fakes, u = generator_terms(model, xs)
        # Each discriminator takes its step on this batch, then scores the generator.
        adv = {"gen": 0.0, "latent_gen": 0.0}
        logs: dict[str, float] = {}
        games = [(f"out_{n}", xs[n], fakes[n], "gen") for n in models] + [
            (f"lat_{m}", u[anchor], u[m], "latent_gen") for m in models[1:]
        ]
        for k, real, fake, kind in games:
            d = discs[k]
            d.requires_grad_(True)
            d_loss = lsgan_disc_loss(d, real, fake)
            dopts[k].zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), cfg.max_grad_norm)
            dopts[k].step()
            d.requires_grad_(False)
            adv[kind] = adv[kind] + lsgan_gen_loss(d, fake)
            logs[f"disc_{k}"] = float(d_loss.detach())
        loss = (
            cfg.w_rec * terms["rec"]
            + cfg.w_vsp * terms["vsp"]
            + cfg.w_cc_trans * terms["cc_trans"]
            + cfg.w_cc_vsp * terms["cc_vsp"]
            + cfg.w_gen * adv["gen"]
            + cfg.w_latent_gen * adv["latent_gen"]
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        step += 1
        if step % 100 == 0:
            logs |= {k: float(v.detach()) for k, v in terms.items()} | {
                "gen": float(adv["gen"].detach()),
                "latent_gen": float(adv["latent_gen"].detach()),
                "loss": float(loss.detach()),
                "grad_norm": float(gn),
                "lr": opt.param_groups[0]["lr"],
                "steps_per_sec": (step - s0) / (time.time() - t0),
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
            score = mean_rank(ev)
            if score < best:
                best, best_step, bad = score, step, 0
                torch.save({"model": model.state_dict(), "step": step}, best_path)
            else:
                bad += 1
            print(
                f"val step {step} mean_rank={score:.1f} best={best:.1f}@{best_step} "
                + " ".join(
                    f"{k}={v:.4g}" for k, v in ev.items() if not k.startswith("n_")
                ),
                flush=True,
            )
            metrics_f.write(
                json.dumps({"step": step, "split": "val", "mean_rank": score, **ev})
                + "\n"
            )
            metrics_f.flush()
            if wandb:
                wandb.log(
                    {f"val/{k}": v for k, v in ev.items()} | {"val/mean_rank": score},
                    step=step,
                )
            stopped = step >= cfg.min_steps and bad >= cfg.patience
            st = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "discs": discs.state_dict(),
                "dopts": {k: o.state_dict() for k, o in dopts.items()},
                "step": step,
                "best": best,
                "best_step": best_step,
                "bad": bad,
                "stopped": stopped,
                "config": dataclasses.asdict(cfg),
            }
            torch.save(st, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            if stopped:
                print(f"early stop at step {step} (best {best_step})", flush=True)
    del train
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test = load_split(cfg.data_dir, "test", models, device)
    ev = fidelity(model, test, models)
    json.dump(
        {"step": step, "best_step": best_step, "mean_rank": mean_rank(ev), **ev},
        open(f"{run_dir}/test_metrics.json", "w"),
        indent=1,
    )
    print(
        f"test best_step={best_step} mean_rank={mean_rank(ev):.1f} "
        + " ".join(f"{k}={v:.4g}" for k, v in ev.items() if not k.startswith("n_")),
        flush=True,
    )


if __name__ == "__main__":
    main()
