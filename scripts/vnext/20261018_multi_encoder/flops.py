"""MACs: v1.3 control vs multi-encoder arms (same counter as scripts/tools/20251111_flops.py).

Run from the repo root: ``python scripts/vnext/20261018_multi_encoder/flops.py out.json``.
Results (2026-10-18) are in the folder README.

(a) training forward = LatentMIM.forward (online encoder + pooler + decoder + heads) + the
    projection-only target pass, under the v1.3 masking, at shapes the v1.3 sampler draws
    (token budget 3072 over S1+S2+L8, maps excluded), averaged over masking seeds;
(b) inference = unmasked encoder forward on S1+S2+L8 (location tokens + patch tokens).
"""

import json
import logging
import sys
import types

import torch
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

logging.disable(logging.CRITICAL)
sys.path.insert(0, "scripts/official")
sys.path.insert(0, "scripts/vnext/20261018_multi_encoder")
import base as B  # noqa: E402
from v1_2.base import _tokenization_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.datatypes import MaskValue, OlmoEarthSample  # noqa: E402
from olmoearth_pretrain.train.masking import (  # noqa: E402
    MaskedOlmoEarthSample,
)

torch.set_num_threads(8)
INPUTS = ["sentinel2_l2a", "sentinel1", "landsat"]
MAPS = B.ONLY_DECODE_MODALITIES
ALL = INPUTS + MAPS
common = types.SimpleNamespace(
    training_modalities=ALL, tokenization_config=_tokenization_config()
)

_SDPA_CPU = {
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: lambda q,
    k,
    v,
    *a,
    out_shape=None,
    **kw: sdpa_flop_count(q, k, v)
}


def macs(fn) -> float:
    """MACs of one call of ``fn`` (attention included; the CPU SDPA op needs its own formula)."""
    c = FlopCounterMode(display=False, custom_mapping=_SDPA_CPU)
    with c:
        fn()
    return c.get_total_flops() / 2


def timestamps(b, t):
    """Random (day, month, year) stamps; month indices are 0..11."""
    d = torch.randint(0, 28, (b, t, 1))
    m = torch.randint(0, 12, (b, t, 1))
    y = torch.full((b, t, 1), 2022)
    return torch.cat([d, m, y], -1)


def raw_sample(h, w, t, mods):
    """An unmasked OlmoEarthSample with random data for ``mods`` at ``h x w`` pixels, ``t`` steps."""
    kw = {"timestamps": timestamps(1, t)}
    for name in mods:
        spec = Modality.get(name)
        f = spec.image_tile_size_factor
        shape = (
            (1, h * f, w * f)
            + ((t,) if spec.is_multitemporal else ())
            + (spec.num_bands,)
        )
        kw[name] = torch.randn(*shape)
    return OlmoEarthSample(**kw)


def unmasked(h, w, t):
    """S1+S2+L8 sample with every token visible to the online encoder (inference)."""
    s = raw_sample(h, w, t, INPUTS)
    kw = {"timestamps": s.timestamps}
    for name in INPUTS:
        x = getattr(s, name)
        kw[name] = x
        kw[f"{name}_mask"] = torch.full(
            x.shape[:-1] + (1,), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
    return MaskedOlmoEarthSample(**kw)


masking = B.masking_config(common.tokenization_config).build()

models = {
    "control (shared 12)": B.build_v1_3_model_config(common).build(),
    "towers t4 + f8": B.build_multi_encoder_model_config(
        common, tower_depth=4, fusion_depth=8
    ).build(),
    "towers t4 + f0": B.build_multi_encoder_model_config(
        common, tower_depth=4, fusion_depth=0
    ).build(),
}
for m in models.values():
    m.train()

out = {"train": {}, "infer": {}, "visible_frac": {}}
# (patch grid hw, timesteps, patch size): 3*hw^2*T tokens before masking
TRAIN_SHAPES = [(9, 12, 4), (16, 4, 4), (32, 1, 4), (4, 12, 4)]
SEEDS = [0, 1, 2]
for hw, t, p in TRAIN_SHAPES:
    key = f"hw{hw} T{t} p{p} ({3 * hw * hw * t} tok)"
    samples = []
    for s in SEEDS:
        torch.manual_seed(s)
        import numpy as np

        np.random.seed(s)
        samples.append(
            masking.apply_mask(raw_sample(hw * p, hw * p, t, ALL), patch_size=p)
        )
    vis = []
    for x in samples:
        n_vis = n_tot = 0
        for name in INPUTS:
            mk = getattr(x, f"{name}_mask")[:, ::p, ::p]
            n_vis += int((mk == MaskValue.ONLINE_ENCODER.value).sum())
            n_tot += mk.numel()
        vis.append(n_vis / n_tot)
    out["visible_frac"][key] = sum(vis) / len(vis)
    out["train"][key] = {}
    for name, m in models.items():
        vals = []
        for x in samples:

            def fwd(x=x):
                """Online forward + the projection-only target pass (one training forward)."""
                m(x, p)
                with torch.no_grad():
                    m.target_encoder(
                        x.unmask(), patch_size=p, token_exit_cfg={k: 0 for k in ALL}
                    )

            vals.append(macs(fwd))
        out["train"][key][name] = sum(vals) / len(vals)
        print(
            "train", key, name, f"{out['train'][key][name] / 1e9:.1f} GMAC", flush=True
        )

INFER_SHAPES = [(16, 16, 12, 1), (128, 128, 1, 4), (64, 64, 12, 4), (32, 32, 1, 4)]
for h, w, t, p in INFER_SHAPES:
    key = f"{h}x{w} T{t} p{p} ({3 * (h // p) * (w // p) * t} tok)"
    x = unmasked(h, w, t)
    out["infer"][key] = {}
    for name, m in models.items():
        m.eval()
        out["infer"][key][name] = macs(lambda: m.encoder(x, patch_size=p))
        m.train()
        print(
            "infer", key, name, f"{out['infer'][key][name] / 1e9:.1f} GMAC", flush=True
        )

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DONE")
