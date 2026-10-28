"""vec2vec for geospatial embedding products (Jha et al. 2025, arXiv 2505.12540), three-way.

Each product m (aef, oe, tess) has an input adapter A_m into a shared backbone T, whose
unit-norm output u_m = T(A_m(x_m)) is the universal latent, and an output adapter B_m back
to its own space. Translation m -> n is B_n(T(A_m(x_m))). Inputs are standardised per dim
with train-set statistics, so every product and every dim weigh the same in the losses.

Objectives:
- paired (our main line; the products are pixel-aligned): reconstruction, translation,
  InfoNCE between the latents of the same pixel, and optionally the paper's cycle
  consistency and vector-space preservation;
- self (control): reconstruction only, same architecture, no other product involved;
- unpaired (the paper's recipe): reconstruction, cycle consistency, VSP and least-squares
  GANs on the outputs and on the latent, on disjoint pixel sets per product.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = {"aef": 64, "oe": 128, "tess": 128}


class ResBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, d: int, mult: int = 2, dropout: float = 0.0):
        """Width d, hidden d * mult."""
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """X + MLP(LayerNorm(x))."""
        return x + self.drop(self.fc2(F.silu(self.fc1(self.ln(x)))))


class ResMLP(nn.Module):
    """Linear in, `depth` pre-norm residual blocks, LayerNorm + Linear out."""

    def __init__(
        self, d_in: int, d_out: int, width: int, depth: int, dropout: float = 0.0
    ):
        """d_in -> width -> d_out."""
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(
            *[ResBlock(width, dropout=dropout) for _ in range(depth)]
        )
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""
        return self.out(self.blocks(self.inp(x)))


class Vec2Vec(nn.Module):
    """Input adapters A_m, shared backbone T, output adapters B_m."""

    def __init__(
        self,
        dims: dict[str, int],
        latent: int = 128,
        width: int = 1024,
        depth: int = 3,
        backbone_depth: int = 4,
        dropout: float = 0.0,
    ):
        """dims: product -> embedding size; latent: size of the universal latent."""
        super().__init__()
        self.dims, self.latent_dim = dict(dims), latent
        self.A = nn.ModuleDict(
            {m: ResMLP(d, width, width, depth, dropout) for m, d in dims.items()}
        )
        self.T = ResMLP(width, latent, width, backbone_depth, dropout)
        self.B = nn.ModuleDict(
            {m: ResMLP(latent, d, width, depth, dropout) for m, d in dims.items()}
        )
        for m, d in dims.items():
            self.register_buffer(f"mean_{m}", torch.zeros(d))
            self.register_buffer(f"std_{m}", torch.ones(d))

    def set_stats(self, stats: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Store the per-product standardisation (mean, std)."""
        for m, (mu, sd) in stats.items():
            getattr(self, f"mean_{m}").copy_(mu)
            getattr(self, f"std_{m}").copy_(sd)

    def standardize(self, m: str, x: torch.Tensor) -> torch.Tensor:
        """Raw product-m embeddings -> standardised units."""
        return (x - getattr(self, f"mean_{m}")) / getattr(self, f"std_{m}")

    def unstandardize(self, m: str, x: torch.Tensor) -> torch.Tensor:
        """Standardised units -> product-m embeddings."""
        return x * getattr(self, f"std_{m}") + getattr(self, f"mean_{m}")

    def latent(self, m: str, xs: torch.Tensor) -> torch.Tensor:
        """Universal latent of standardised inputs: unit norm, [B, latent]."""
        return F.normalize(self.T(self.A[m](xs)), dim=-1)

    def decode(self, n: str, u: torch.Tensor) -> torch.Tensor:
        """Standardised embedding of product n from a latent."""
        return self.B[n](u * math.sqrt(self.latent_dim))

    @torch.no_grad()
    def embed(self, m: str, x: torch.Tensor) -> torch.Tensor:
        """Raw product-m embeddings -> universal latent (what the evals score)."""
        return self.latent(m, self.standardize(m, x))


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over the valid rows (mean over dims), 0 if none are valid."""
    if not mask.any():
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric InfoNCE between unit-norm rows of a and b (row i of each is a positive pair)."""
    logits = a @ b.T / tau
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def vsp(src: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Vector-space preservation: |cos-Gram(source) - cos-Gram(translation)|, mean."""
    a, b = F.normalize(src, dim=-1), F.normalize(trans, dim=-1)
    return (a @ a.T - b @ b.T).abs().mean()


def paired_loss(
    model: Vec2Vec,
    xs: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    w: dict[str, float],
    tau: float = 0.05,
    gram_rows: int = 1024,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Paired objective. xs: standardised inputs (zeros where invalid); valid: [B] bool."""
    ms = list(xs)
    u = {m: model.latent(m, xs[m]) for m in ms}
    logs: dict[str, torch.Tensor] = {}
    total = 0.0
    rec = [masked_mse(model.decode(m, u[m]), xs[m], valid[m]) for m in ms]
    for m, r in zip(ms, rec):
        logs[f"rec_{m}"] = r
    total = total + w.get("rec", 1.0) * sum(rec) / len(ms)
    if w.get("trans", 0.0) > 0 or w.get("cyc", 0.0) > 0 or w.get("vsp", 0.0) > 0:
        trans, cyc, vs = [], [], []
        for m in ms:
            for n in ms:
                if m == n:
                    continue
                both = valid[m] & valid[n]
                y = model.decode(n, u[m])
                t = masked_mse(y, xs[n], both)
                logs[f"trans_{m}_{n}"] = t
                trans.append(t)
                if w.get("cyc", 0.0) > 0:
                    back = model.decode(m, model.latent(n, y))
                    cyc.append(masked_mse(back, xs[m], valid[m]))
                if w.get("vsp", 0.0) > 0:
                    k = valid[m].nonzero().squeeze(1)[:gram_rows]
                    vs.append(vsp(xs[m][k], y[k]))
        total = total + w.get("trans", 0.0) * sum(trans) / len(trans)
        if cyc:
            logs["cyc"] = sum(cyc) / len(cyc)
            total = total + w["cyc"] * logs["cyc"]
        if vs:
            logs["vsp"] = sum(vs) / len(vs)
            total = total + w["vsp"] * logs["vsp"]
    if w.get("align", 0.0) > 0:
        al = []
        for i, m in enumerate(ms):
            for n in ms[i + 1 :]:
                both = valid[m] & valid[n]
                if both.sum() > 1:
                    a = info_nce(u[m][both], u[n][both], tau)
                    logs[f"align_{m}_{n}"] = a
                    al.append(a)
        if al:
            total = total + w["align"] * sum(al) / len(al)
    return total, {k: float(v) for k, v in logs.items()}


class Discriminator(nn.Module):
    """The paper's discriminator: MLP, SiLU, LayerNorm, no residuals."""

    def __init__(self, d_in: int, width: int = 1024, depth: int = 5):
        """d_in -> width (depth layers) -> 1 logit."""
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, width)]
        for _ in range(depth - 2):
            layers += [nn.SiLU(), nn.Linear(width, width), nn.LayerNorm(width)]
        layers += [nn.SiLU(), nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One logit per row."""
        return self.net(x)


def lsgan_disc_loss(
    d: Discriminator, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    """The paper's least-squares GAN: real -> 0, fake -> 1."""
    return 0.5 * ((d(real.detach()) ** 2).mean() + ((d(fake.detach()) - 1) ** 2).mean())


def lsgan_gen_loss(d: Discriminator, fake: torch.Tensor) -> torch.Tensor:
    """Generator side of the least-squares GAN: fakes -> 0 (real)."""
    return 0.5 * (d(fake) ** 2).mean()


def unpaired_terms(
    model: Vec2Vec, xs: dict[str, torch.Tensor], gram_rows: int = 1024
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Generator-side terms for the unpaired recipe; each product's batch holds different pixels.

    Returns (losses, fake outputs keyed by target product, latents keyed by source product).
    """
    ms = list(xs)
    u = {m: model.latent(m, xs[m]) for m in ms}
    rec, cyc, vs, fakes = [], [], [], {n: [] for n in ms}
    for m in ms:
        rec.append(F.mse_loss(model.decode(m, u[m]), xs[m]))
        for n in ms:
            if m == n:
                continue
            y = model.decode(n, u[m])
            fakes[n].append(y)
            cyc.append(F.mse_loss(model.decode(m, model.latent(n, y)), xs[m]))
            vs.append(vsp(xs[m][:gram_rows], y[:gram_rows]))
    losses = {
        "rec": sum(rec) / len(rec),
        "cyc": sum(cyc) / len(cyc),
        "vsp": sum(vs) / len(vs),
    }
    return losses, {n: torch.cat(f) for n, f in fakes.items()}, u
