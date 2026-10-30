"""Frozen pretrained (lat, lon) -> vector location encoders, used as model input.

Two encoders are supported, both re-implemented here for inference only with their
weights held as (non-trainable) buffers loaded from a plain ``torch.save`` tensor dict:

* ``tte``: TTE, "Tessellating The Earth" (Cher et al., ECCV 2026; github.com/mvrl/TTE,
  MIT; weights ``MVRL/TTE``). A soft spherical-Voronoi lookup over 4096 learned sites,
  attention over 64 shared semantic tokens and a residual MLP; 512-d, L2-normalized.
* ``satclip_l40``: SatCLIP ResNet50-L40 location encoder (Klemmer et al.; github.com/
  microsoft/satclip, MIT). Real spherical harmonics up to degree 39 (1600 features)
  into a 3-layer SIREN; 256-d. Computed in float64 as in the reference.

``LocationConditioning`` wraps one of them for the encoder: the frozen vector goes
through a LayerNorm and a zero-initialized Linear to the token width and is added to
every encoder token. Samples without a location (no latlon, MISSING_VALUE, or per-sample
dropout during training) get a learned "no location" vector instead.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOCATION_ENCODER_DIMS = {"tte": 512, "satclip_l40": 256}

# latlon arrives min-max normalized by the predefined norm config
# (lat in [-90, 90] -> [0, 1], lon in [-180, 180] -> [0, 1]).
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)


def denormalize_latlon(latlon: Tensor) -> Tensor:
    """Map [0, 1]-normalized (lat, lon) back to degrees."""
    lat = latlon[..., 0] * (LAT_RANGE[1] - LAT_RANGE[0]) + LAT_RANGE[0]
    lon = latlon[..., 1] * (LON_RANGE[1] - LON_RANGE[0]) + LON_RANGE[0]
    return torch.stack([lat, lon], dim=-1)


class TTELocationEncoder(nn.Module):
    """Inference-only TTE location encoder (matches ``tte.TTE.encode``)."""

    output_dim = 512

    def __init__(self, weights_path: str) -> None:
        """Load the TTE location-path tensors (keys as in the HF safetensors)."""
        super().__init__()
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        p = "location."
        r = p + "semantic_registers."
        self.register_buffer("directions", F.normalize(sd[p + "_directions"], dim=-1))
        self.register_buffer("tau", torch.exp(sd[p + "_log_tau"]))
        self.register_buffer("site_embeddings", sd[p + "_embeddings"])
        self.register_buffer("loc_to_logits_w", sd[r + "loc_to_logits.weight"])
        self.register_buffer("loc_to_logits_b", sd[r + "loc_to_logits.bias"])
        # Inference temperature: TTE's anneal progress is 0 outside training, so the
        # location attention uses the learned temperature (clamped as in the reference).
        self.register_buffer(
            "register_temperature",
            torch.exp(sd[r + "log_temperature"]).clamp(min=0.01, max=10.0),
        )
        self.register_buffer("registers", sd[r + "registers"])
        self.register_buffer("site_to_out_w", sd[r + "site_to_out.weight"])
        self.register_buffer("site_to_out_b", sd[r + "site_to_out.bias"])
        self.register_buffer("norm_w", sd[r + "norm.weight"])
        self.register_buffer("norm_b", sd[r + "norm.bias"])
        self.mlp_keys = ["0", "2.w1", "2.w2", "3.w1", "3.w2"]
        for k in self.mlp_keys:
            name = k.replace(".", "_")
            self.register_buffer(f"mlp_{name}_w", sd[f"{p}mlp.{k}.weight"])
            self.register_buffer(f"mlp_{name}_b", sd[f"{p}mlp.{k}.bias"])
        self.register_buffer("mlp_out_w", sd[p + "mlp.4.weight"])
        self.gate = 0.5

    def _lin(self, x: Tensor, name: str) -> Tensor:
        return F.linear(
            x, getattr(self, f"mlp_{name}_w"), getattr(self, f"mlp_{name}_b")
        )

    def forward(self, latlon_deg: Tensor) -> Tensor:
        """(N, 2) (lat, lon) in degrees -> (N, 512) unit vectors."""
        lat = torch.deg2rad(latlon_deg[:, 0])
        lon = torch.deg2rad(latlon_deg[:, 1])
        sphere = torch.stack(
            [
                torch.cos(lat) * torch.cos(lon),
                torch.cos(lat) * torch.sin(lon),
                torch.sin(lat),
            ],
            dim=-1,
        )
        logits = sphere @ (self.directions * self.tau.unsqueeze(-1)).t()
        weights = torch.softmax(logits - logits.amax(dim=-1, keepdim=True), dim=-1)
        site = weights @ self.site_embeddings
        att = torch.softmax(
            F.linear(site, self.loc_to_logits_w, self.loc_to_logits_b)
            / self.register_temperature,
            dim=-1,
        )
        site_out = F.linear(site, self.site_to_out_w, self.site_to_out_b)
        x = (1 - self.gate) * site_out + self.gate * (att @ self.registers)
        x = F.layer_norm(x, (x.shape[-1],), self.norm_w, self.norm_b)
        x = F.relu(self._lin(x, "0"))
        for blk in ("2", "3"):
            y = F.relu(self._lin(x, f"{blk}_w1"))
            y = F.relu(self._lin(y, f"{blk}_w2"))
            x = x + y
        x = F.linear(x, self.mlp_out_w)
        return F.normalize(x, dim=-1)


def real_spherical_harmonics_satclip(
    lat_deg: Tensor, lon_deg: Tensor, num_degrees: int
) -> Tensor:
    """SatCLIP's spherical-harmonic features, (N, num_degrees**2), float64.

    SatCLIP (``positional_encoding/spherical_harmonics_ylm.py``, sympy-generated) uses
    theta = lat + 90 deg, phi = lon + 180 deg, and per (l, m) for m in -l..l the real
    harmonic without the Condon-Shortley phase: sqrt(2) K_l^|m| P_l^|m|(cos theta)
    times cos(m phi) for m > 0 or sin(|m| phi) for m < 0; for m = 0 its generated
    expressions are pi * K_l^0 P_l(cos theta) (a constant factor of the generator).
    Computed here with the standard stable recurrences instead of the generated
    polynomials.
    """
    lat_deg = lat_deg.double()
    lon_deg = lon_deg.double()
    theta = torch.deg2rad(lat_deg + 90.0)
    phi = torch.deg2rad(lon_deg + 180.0)
    x = torch.cos(theta)
    s = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
    big_l = num_degrees
    # Associated Legendre P_l^m(x) without the Condon-Shortley phase, m <= l:
    # P_m^m = (2m-1)!! s^m, P_{m+1}^m = (2m+1) x P_m^m, then the upward recurrence in l.
    pmm = torch.ones_like(x)
    diag = [pmm]
    for m in range(1, big_l):
        pmm = pmm * (2 * m - 1) * s
        diag.append(pmm)
    table: dict[tuple[int, int], Tensor] = {}
    for m in range(big_l):
        table[(m, m)] = diag[m]
        if m + 1 < big_l:
            table[(m + 1, m)] = x * (2 * m + 1) * diag[m]
        for ell in range(m + 2, big_l):
            table[(ell, m)] = (
                (2 * ell - 1) * x * table[(ell - 1, m)]
                - (ell + m - 1) * table[(ell - 2, m)]
            ) / (ell - m)
    feats = []
    for ell in range(big_l):
        for m in range(-ell, ell + 1):
            am = abs(m)
            k = math.sqrt(
                (2 * ell + 1)
                / (4 * math.pi)
                * math.exp(math.lgamma(ell - am + 1) - math.lgamma(ell + am + 1))
            )
            base = k * table[(ell, am)]
            if m == 0:
                feats.append(math.pi * base)
            elif m > 0:
                feats.append(math.sqrt(2.0) * base * torch.cos(m * phi))
            else:
                feats.append(math.sqrt(2.0) * base * torch.sin(am * phi))
    return torch.stack(feats, dim=-1)


class SatCLIPLocationEncoder(nn.Module):
    """Inference-only SatCLIP L40 location encoder (spherical harmonics + SIREN)."""

    output_dim = 256
    num_degrees = 40

    def __init__(self, weights_path: str) -> None:
        """Load the SIREN weights (keys ``layers.{0,1}.*``, ``last_layer.*``)."""
        super().__init__()
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            self.register_buffer(k.replace(".", "_"), v.double())

    def forward(self, latlon_deg: Tensor) -> Tensor:
        """(N, 2) (lat, lon) in degrees -> (N, 256), float64."""
        y = real_spherical_harmonics_satclip(
            latlon_deg[:, 0], latlon_deg[:, 1], self.num_degrees
        )
        # SIREN: sin(30 (W x + b)) for the first layer, sin(W x + b) after, linear out.
        y = torch.sin(30.0 * F.linear(y, self.layers_0_weight, self.layers_0_bias))
        y = torch.sin(F.linear(y, self.layers_1_weight, self.layers_1_bias))
        return F.linear(y, self.last_layer_weight, self.last_layer_bias)


def build_location_encoder(kind: str, weights_path: str) -> nn.Module:
    """Build a frozen location encoder by name."""
    if kind == "tte":
        return TTELocationEncoder(weights_path)
    if kind == "satclip_l40":
        return SatCLIPLocationEncoder(weights_path)
    raise ValueError(f"unknown location encoder {kind!r}")


class LocationConditioning(nn.Module):
    """Frozen location vector -> additive per-sample token conditioning.

    ``forward`` returns a (B, D) tensor to add to every encoder token: LayerNorm then a
    zero-initialized Linear of the frozen location vector, or a learned zero-initialized
    "no location" vector for samples without one (and, during training, for a random
    ``dropout`` fraction of samples). At init the conditioning is exactly zero, so the
    model starts as the unconditioned recipe.
    """

    def __init__(
        self, kind: str, weights_path: str, embedding_size: int, dropout: float
    ) -> None:
        """Build the frozen encoder and the trainable projection."""
        super().__init__()
        self.kind = kind
        self.dropout = dropout
        self.location_encoder = build_location_encoder(kind, weights_path)
        loc_dim = LOCATION_ENCODER_DIMS[kind]
        self.norm = nn.LayerNorm(loc_dim)
        self.proj = nn.Linear(loc_dim, embedding_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.null_embedding = nn.Parameter(torch.zeros(embedding_size))

    def forward(
        self, latlon: Tensor | None, batch_size: int, device: torch.device
    ) -> Tensor:
        """(B, 2) [0, 1]-normalized latlon (or None) -> (B, D) additive conditioning."""
        null = self.null_embedding.unsqueeze(0).expand(batch_size, -1)
        if latlon is None:
            return null
        latlon = latlon.reshape(batch_size, -1)[:, :2].float()
        # MISSING_VALUE (-99999) and anything outside the normalized range = no location.
        has_loc = ((latlon >= 0) & (latlon <= 1)).all(dim=-1)
        if self.training and self.dropout > 0:
            keep = torch.rand(batch_size, device=latlon.device) >= self.dropout
            has_loc = has_loc & keep
        safe = torch.where(has_loc.unsqueeze(-1), latlon, torch.full_like(latlon, 0.5))
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            vec = self.location_encoder(denormalize_latlon(safe)).float()
        with torch.autocast(device_type=device.type, enabled=False):
            cond = self.proj(self.norm(vec))
        return torch.where(has_loc.unsqueeze(-1), cond, null)
