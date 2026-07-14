"""Attention diagnostics for a trained PerceiverEncoder checkpoint.

Loads a DCP train checkpoint (LatentMIM) locally, runs real H5 samples
through the encoder with hooks that capture post-RoPE q/k in the read and
read-out attentions, and reports:

  1. Read re-targeting: per-latent Jensen-Shannon-ish overlap between
     read-1 and read-2 attention distributions (1.0 = identical rows —
     the second read adds nothing spatially).
  2. Read locality: mass-weighted spatial distance between each latent's
     anchor and the tokens it attends to, per read.
  3. Read-out locality: mass-weighted distance from each dense query to
     the latent anchors it reads from + mass on the nearest anchor.

Run: PYTHONPATH=. python scripts/vnext/perceiver/probe_attention.py <ckpt_dir>
"""

from __future__ import annotations

import sys

import torch

CKPT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else (
        "/weka/dfive-default/olmoearth_pretrain/checkpoints/joer/"
        "perceiver_bottleneck_base_2/step60000"
    )
)
H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)


def build_model():
    """Base-size perceiver LatentMIM matching perceiver_base.py."""
    from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
    from olmoearth_pretrain.nn.perceiver import (
        PerceiverEncoderConfig,
        PerceiverPredictorConfig,
    )
    from olmoearth_pretrain.nn.tokenization import (
        ModalityTokenization,
        TokenizationConfig,
    )

    modalities = [
        "sentinel2_l2a",
        "sentinel1",
        "landsat",
        "worldcover",
        "srtm",
        "openstreetmap_raster",
        "wri_canopy_height_map",
        "cdl",
        "worldcereal",
    ]
    tok = TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(
                band_groups=[
                    [
                        "B02",
                        "B03",
                        "B04",
                        "B08",
                        "B05",
                        "B06",
                        "B07",
                        "B8A",
                        "B11",
                        "B12",
                        "B01",
                        "B09",
                    ]
                ]
            ),
            "landsat": ModalityTokenization(
                band_groups=[
                    ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
                ]
            ),
        }
    )
    common = dict(
        supported_modality_names=modalities,
        max_sequence_length=12,
        tokenization_config=tok,
        position_encoding="rope_3d_mixed",
        rope_mixed_base=10000.0,
        rope_temporal_coordinate_scale=1.0 / 30.0,
    )
    enc = PerceiverEncoderConfig(
        embedding_size=768,
        num_heads=12,
        depth=12,
        mlp_ratio=4.0,
        max_patch_size=8,
        min_patch_size=1,
        drop_path=0.1,
        band_dropout_rate=0.2,
        random_band_dropout=True,
        band_dropout_modalities=["sentinel2_l2a", "landsat"],
        patch_embed_hidden_sizes=[64],
        latent_stride_hw=2,
        latent_stride_t=2,
        num_reads=2,
        readout_depth=2,
        **common,
    )
    dec = PerceiverPredictorConfig(
        encoder_embedding_size=768,
        decoder_embedding_size=768,
        depth=0,
        head_depth=2,
        mlp_ratio=4.0,
        num_heads=12,
        **common,
    )
    return LatentMIMConfig(encoder_config=enc, decoder_config=dec).build()


def load_checkpoint(model) -> None:
    """Load DCP train checkpoint into the unsharded model (olmo-core reader)."""
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    load_model_and_optim_state(f"{CKPT}/model_and_optim", model)


def make_batch(indices):
    """Real samples -> all-visible masked batch (eval regime)."""
    from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDatasetConfig
    from olmoearth_pretrain.datatypes import (
        MaskedOlmoEarthSample,
        MaskValue,
    )

    ds = OlmoEarthDatasetConfig(
        h5py_dir=H5_DIR,
        training_modalities=["sentinel2_l2a", "sentinel1", "landsat"],
    ).build()
    fields: dict[str, list] = {}
    for i in indices:
        _, sample = ds[GetItemArgs(idx=i, patch_size=8, sampled_hw_p=12)]
        for name, val in sample.as_dict().items():
            if val is None:
                continue
            fields.setdefault(name, []).append(torch.as_tensor(val))
    batch: dict = {}
    for name, vals in fields.items():
        stacked = torch.stack(
            [v[: min(x.shape[0] for x in vals)] if False else v for v in vals]
        )
        batch[name] = stacked
    out: dict = {"timestamps": batch["timestamps"].long()}
    for name, val in batch.items():
        if name == "timestamps":
            continue
        out[name] = val.float()
        out[f"{name}_mask"] = torch.full(
            val.shape, MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
    return MaskedOlmoEarthSample(**out)


class QKCatcher:
    """Wraps an Attention.sdpa to capture post-RoPE q/k."""

    def __init__(self, attn):
        self.attn = attn
        self.orig = attn.sdpa
        self.q = None
        self.k = None
        attn.sdpa = self._wrapped

    def _wrapped(self, q, k, v, *args, **kwargs):
        self.q = q.detach()
        self.k = k.detach()
        return self.orig(q, k, v, *args, **kwargs)

    def attention(self) -> torch.Tensor:
        """softmax(q k^T / sqrt(d)) averaged over heads: [B, N, M]."""
        scores = (self.q @ self.k.transpose(-2, -1)) * self.attn.scale
        return scores.softmax(dim=-1).mean(dim=1)


def mass_weighted_dist(attn: torch.Tensor, q_pos, k_pos) -> float:
    """Mean attention-mass-weighted spatial (row,col) distance."""
    d = torch.cdist(q_pos[..., -2:], k_pos[..., -2:])  # [B, N, M]
    return float((attn * d).sum(-1).mean())


def main() -> None:
    """Run the probe and print the report."""
    torch.manual_seed(0)
    model = build_model()
    load_checkpoint(model)
    encoder = model.encoder.cuda().eval()
    print(f"loaded {CKPT}")

    batch = make_batch([17, 4242]).to_device(torch.device("cuda"))

    catchers = {
        "read0": QKCatcher(encoder.read_blocks[0].attn),
        "read1": QKCatcher(encoder.read_blocks[1].attn),
        "readout0": QKCatcher(encoder.readout_blocks[0].attn),
        "readout1": QKCatcher(encoder.readout_blocks[1].attn),
    }
    # capture coords by hooking _build_query_grid outputs
    grids = []
    orig_grid = encoder._build_query_grid

    def grid_spy(*args, **kwargs):
        content, coords = orig_grid(*args, **kwargs)
        grids.append(coords.detach())
        return content, coords

    encoder._build_query_grid = grid_spy

    with torch.no_grad():
        encoder.forward(batch, patch_size=8, fast_pass=True)

    latent_pos, query_pos = grids[0].float(), grids[1].float()
    a0 = catchers["read0"].attention().float()
    a1 = catchers["read1"].attention().float()
    # token k positions: from the read catcher's k we can't get coords;
    # reuse encoder rope positions via a second forward? Instead approximate
    # token coords with the dense query grid, which shares the (h,w,t)
    # layout of the multitemporal tokens (first M_grid tokens are S2).
    m = min(a0.shape[-1], query_pos.shape[1])
    print("\n== read re-targeting (per-latent overlap of attention rows) ==")
    overlap = (torch.minimum(a0, a1).sum(-1)).mean()
    print(
        f"read0 vs read1 distribution overlap: {float(overlap):.3f} "
        "(1.0 = second read attends identically; low = re-targeting)"
    )

    ent0 = -(a0 * (a0 + 1e-9).log()).sum(-1).mean()
    ent1 = -(a1 * (a1 + 1e-9).log()).sum(-1).mean()
    uniform = torch.log(torch.tensor(float(a0.shape[-1])))
    print(
        f"attention entropy: read0 {float(ent0):.2f}, read1 {float(ent1):.2f} "
        f"(uniform = {float(uniform):.2f})"
    )

    print(
        "\n== read locality (mass-weighted |anchor - token| in patch units, "
        "S2 block only) =="
    )
    d0 = mass_weighted_dist(a0[..., :m], latent_pos, query_pos[:, :m])
    d1 = mass_weighted_dist(a1[..., :m], latent_pos, query_pos[:, :m])
    rand = mass_weighted_dist(
        torch.full_like(a0[..., :m], 1.0 / m), latent_pos, query_pos[:, :m]
    )
    print(f"read0 {d0:.2f}, read1 {d1:.2f}, uniform-attention baseline {rand:.2f}")

    print("\n== read-out locality (dense queries -> latent anchors) ==")
    for name in ("readout0", "readout1"):
        a = catchers[name].attention().float()
        d = mass_weighted_dist(a, query_pos, latent_pos)
        nearest = a.max(-1).values.mean()
        rand_d = mass_weighted_dist(
            torch.full_like(a, 1.0 / a.shape[-1]), query_pos, latent_pos
        )
        print(
            f"{name}: mass-weighted dist {d:.2f} (uniform {rand_d:.2f}), "
            f"mean top-anchor mass {float(nearest):.3f}"
        )


if __name__ == "__main__":
    main()
