"""The paper's own unsupervised vec2vec (rjha18/vec2vec, configs/unsupervised.toml), for our products.

A reimplementation of the reference code's recipe, as opposed to the three-way variant in
vec2vec.py. Where the reference differs from the paper's text, this follows the code:
- inputs and outputs are L2-normalised embeddings (normalize_embeddings = true); the
  universal latent is the backbone's raw 1024-d output (d_adapter), not normalised;
- adapters and backbone are the reference MLPWithResidual (SiLU, BatchNorm, dropout 0.1,
  a zero-padded / truncated residual around every layer, Kaiming init): adapters depth 3,
  backbone depth 4, every layer 1024 wide;
- generator loss = reconstruction (1 - cos) + VSP on reconstructions (x1) + cycle consistency
  (1 - cos, x10) + VSP on cycles (x10) + a least-squares GAN on every output space (x1 each)
  + a latent GAN (x1). The cycle's first leg is detached, and the latent GAN only pushes the
  other products' latents towards the first product's (the reference's sup_emb);
- LSGAN as in utils/gan.py: discriminator real -> 0, fake -> 1, each discriminator stepped
  on the current batch before the generator loss reads it (the config's smooth = 0.9 is
  unused by the least-squares GAN);
- Adam (betas 0.5, 0.999), lr 2e-5, discriminators lr 1e-5 (eps 6.25e-10), 2000 linear
  warmup steps then constant, gradient clipping at 1000, batch 256 per product.
The reference trains two spaces; with three, cycles run m -> n -> m for every ordered pair and
each non-anchor product gets its own latent discriminator. Training is fp32 (TF32 matmuls)
where the reference uses fp16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 1024


def init_weights(module: nn.Module) -> None:
    """The reference's 'kaiming' init: Linear N(fan_in, relu) with zero bias, BN ~ N(1, .02)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0.0, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def pad_to(x: torch.Tensor, d: int) -> torch.Tensor:
    """Zero-pad or truncate the last dim to d (the reference's add_residual)."""
    if x.shape[1] < d:
        return torch.cat([x, x.new_zeros(x.shape[0], d - x.shape[1])], 1)
    return x[:, :d]


class ResidualMLP(nn.Module):
    """The reference MLPWithResidual: `depth` layers, a residual around each one."""

    def __init__(
        self, depth: int, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1
    ):
        """First layer Linear + SiLU, middle ones add BatchNorm + dropout, last maps to d_out."""
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            if i == 0:
                h = d_out if depth == 1 else d_hidden
                layers.append(nn.Sequential(nn.Linear(d_in, h), nn.SiLU()))
            elif i < depth - 1:
                layers.append(
                    nn.Sequential(
                        nn.Linear(d_hidden, d_hidden),
                        nn.SiLU(),
                        nn.BatchNorm1d(d_hidden),
                        nn.Dropout(dropout),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Linear(d_hidden, d_hidden),
                        nn.Dropout(dropout),
                        nn.SiLU(),
                        nn.Linear(d_hidden, d_out),
                    )
                )
        self.layers = nn.ModuleList(layers)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Each layer's output plus its (padded / truncated) input."""
        for layer in self.layers:
            y = layer(x)
            x = y + pad_to(x, y.shape[1])
        return x


class PaperVec2Vec(nn.Module):
    """Input adapters A_m, shared backbone T (latent = T(A_m(x))), output adapters B_m.

    Exposes the same standardize / latent / decode / unstandardize interface as
    vec2vec.Vec2Vec, so train.fidelity scores it unchanged.
    """

    def __init__(
        self,
        dims: dict[str, int],
        d_model: int = D_MODEL,
        depth: int = 3,
        transform_depth: int = 4,
    ):
        """dims: product -> embedding size."""
        super().__init__()
        self.dims, self.latent_dim = dict(dims), d_model
        self.A = nn.ModuleDict(
            {m: ResidualMLP(depth, d, d_model, d_model) for m, d in dims.items()}
        )
        self.T = ResidualMLP(transform_depth, d_model, d_model, d_model)
        self.B = nn.ModuleDict(
            {m: ResidualMLP(depth, d_model, d_model, d) for m, d in dims.items()}
        )

    def standardize(self, m: str, x: torch.Tensor) -> torch.Tensor:
        """Raw embeddings -> unit norm (the reference's normalize_embeddings)."""
        return F.normalize(x, dim=-1)

    def unstandardize(self, m: str, x: torch.Tensor) -> torch.Tensor:
        """Outputs already live in the (unit-norm) input space."""
        return x

    def latent(self, m: str, xs: torch.Tensor) -> torch.Tensor:
        """Universal latent of unit-norm inputs: [B, d_model], not normalised."""
        return self.T(self.A[m](xs))

    def decode(self, n: str, u: torch.Tensor) -> torch.Tensor:
        """Unit-norm product-n embedding from a latent."""
        return F.normalize(self.B[n](u), dim=-1)

    @torch.no_grad()
    def embed(self, m: str, x: torch.Tensor) -> torch.Tensor:
        """Raw product-m embeddings -> universal latent."""
        return self.latent(m, self.standardize(m, x))


def cos_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine (the reference's rec / trans losses)."""
    return 1 - F.cosine_similarity(x, y, dim=-1).mean()


def vsp_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The reference vsp_loss_fn for one (input, output) pair: Gram gap plus reflected gap."""
    b = F.normalize(x.detach(), dim=-1)
    a = F.normalize(y, dim=-1)
    g = b @ b.T
    return (g - a @ a.T).abs().mean() + (g - a @ b.T).abs().mean()


def generator_terms(
    model: PaperVec2Vec, xs: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Non-adversarial generator losses on one unpaired batch per product.

    Returns (losses, translations into each product keyed by target, latents keyed by source).
    """
    ms = list(xs)
    u = {m: model.latent(m, xs[m]) for m in ms}
    rec, vs, cyc, cvs = [], [], [], []
    fakes: dict[str, list[torch.Tensor]] = {n: [] for n in ms}
    for m in ms:
        r = model.decode(m, u[m])
        rec.append(cos_loss(xs[m], r))
        vs.append(vsp_loss(xs[m], r))
        for n in ms:
            if n == m:
                continue
            y = model.decode(n, u[m])
            fakes[n].append(y)
            back = model.decode(m, model.latent(n, y.detach()))
            cyc.append(cos_loss(xs[m], back))
            cvs.append(vsp_loss(xs[m], back))
    losses = {
        "rec": sum(rec) / len(rec),
        "vsp": sum(vs) / len(vs),
        "cc_trans": sum(cyc) / len(cyc),
        "cc_vsp": sum(cvs) / len(cvs),
    }
    return losses, {n: torch.cat(f) for n, f in fakes.items()}, u
