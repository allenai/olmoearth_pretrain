"""Translate the eval windows' product embeddings into the universal latent (J.F.'s scorer layout).

  python export.py --run <run>            # z_<src> for each source, plus the fused latent
  python export.py --baselines --run <run>  # raw products, concat, concat+PCA (uses the run's stats)

Inputs, one <dataset>.npz (ids, emb) per source:
  aef   favyen's products/gse (read-only): power-dequantised, rows of all -1 are nodata;
        clamped to +-0.9922 (two glance rows hold an impossible code -129)
  tess  favyen's products/tessera_v2 (read-only): float32 forward pass, used as is
  oe    our eval_inputs/oe13_product (oe_eval_inputs.py): the public v1.3 product
(investigation/FINDINGS.md, A, has the evidence that these match the training inputs.)

Common-window protocol: on the AEF tasks every output holds exactly the windows where all three
products exist, so every method is scored on the same windows (score with --allow_missing).
That is the harness window set except on glance (41% have OE), africa (73%), descals (81%) and
us_trees (26 windows short). PASTIS keeps all windows (OE covers them all).

Outputs under ROOT/eval_embeddings/<name>/step<N>/<dataset>.npz:
  <run>_z<src>   T(A_src(x_src)), unit norm (score with --quantization aef_power)
  <run>_zfused   mean of the three latents, renormalised
  raw_<src>      the product itself on the common windows (quantization: none; tess: tessera)
  concat         the three products standardised with the run's train stats, each block scaled
                 to unit total variance (none)
  concat_pca128  PCA of concat fit on 1M pair-set train pixels, unit norm (aef_power)
PASTIS (dense, ~40M pixels) is done one source at a time to bound memory; its raw_<src> files
are symlinks to the inputs.
"""

import argparse
import os
import time

import numpy as np
import torch
from vec2vec import DIMS, Vec2Vec

ROOT = "/weka/dfive-default/hadriens/platonic_space"
PRODUCTS = "/weka/dfive-default/favyen/strata/eval_embeddings/products"
INPUTS = {
    "aef": f"{PRODUCTS}/gse",
    "oe": f"{ROOT}/eval_inputs/oe13_product",
    "tess": f"{PRODUCTS}/tessera_v2",
}
AEF_TASKS = (
    "ethiopia_crops",
    "africa_crop_mask",
    "canada_crops_fine",
    "canada_crops_coarse",
    "descals",
    "lcmap_lu",
    "glance",
    "us_trees",
)
DATASETS = tuple(f"{t}_year_aligned" for t in AEF_TASKS) + ("pastis_year_aligned",)
AEF_MAX = (127 / 127.5) ** 2


def load_source(m: str, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, emb float32, ok) of one source; ok is per row ([N]) or per pixel (PASTIS)."""
    with np.load(f"{INPUTS[m]}/{dataset}.npz") as z:
        ids, emb = z["ids"].astype(str), z["emb"].astype(np.float32)
    if m == "aef":
        ok = ~(emb == -1.0).all(-1)
        emb = np.clip(emb, -AEF_MAX, AEF_MAX)
    else:
        ok = np.isfinite(emb).all(-1) & (np.abs(emb).sum(-1) > 0)
    emb[~ok] = 0.0
    return ids, emb, ok


def load_all(dataset: str, models: list[str]) -> tuple[np.ndarray, dict, dict]:
    """All sources on the common windows, in the first source's id order."""
    src = {m: load_source(m, dataset) for m in models}
    ids0 = src[models[0]][0]
    keep = np.ones(len(ids0), bool)
    rows = {}
    for m, (ids, emb, ok) in src.items():
        pos = dict(zip(ids, range(len(ids))))
        r = np.array([pos.get(i, -1) for i in ids0])
        rows[m] = r
        keep &= r >= 0
        if emb.ndim == 2:
            keep &= np.where(r >= 0, ok[np.maximum(r, 0)], False)
    ids = ids0[keep]
    embs = {m: src[m][1][rows[m][keep]] for m in models}
    oks = {m: src[m][2][rows[m][keep]] for m in models}
    return ids, embs, oks


@torch.no_grad()
def translate(
    model: Vec2Vec, m: str, x: np.ndarray, device: torch.device, chunk: int = 1 << 17
) -> np.ndarray:
    """Rows (any leading shape) of product m -> unit-norm latents, float32."""
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty((len(flat), model.latent_dim), np.float32)
    for i in range(0, len(flat), chunk):
        xb = torch.from_numpy(flat[i : i + chunk]).to(device)
        out[i : i + chunk] = model.embed(m, xb).float().cpu().numpy()
    return out.reshape(*x.shape[:-1], model.latent_dim)


def write(
    name: str,
    step: int,
    dataset: str,
    ids: np.ndarray,
    emb: np.ndarray,
    has_obs: np.ndarray,
) -> None:
    """One scorer npz (float16 for dense PASTIS), atomically."""
    d = f"{ROOT}/eval_embeddings/{name}/step{step}"
    os.makedirs(d, exist_ok=True)
    emb = emb.astype(np.float16 if emb.ndim == 4 else np.float32)
    np.savez(f"{d}/{dataset}.tmp.npz", ids=ids, emb=emb, has_obs=has_obs)
    os.replace(f"{d}/{dataset}.tmp.npz", f"{d}/{dataset}.npz")
    print(f"  wrote {name}/step{step}/{dataset}.npz {emb.shape}", flush=True)


def load_model(run: str, device: torch.device) -> tuple[Vec2Vec, int]:
    """The run's checkpoint (model + standardisation stats) and its step."""
    st = torch.load(
        f"{ROOT}/runs/{run}/ckpt.pt", map_location="cpu", weights_only=False
    )
    c = st["config"]
    models = c["models"].split(",")
    model = Vec2Vec(
        {m: DIMS[m] for m in models},
        c["latent"],
        c["width"],
        c["depth"],
        c["backbone_depth"],
        c["dropout"],
    )
    model.load_state_dict(st["model"])
    return model.to(device).eval(), st["step"]


def concat_blocks(model: Vec2Vec, embs: dict[str, np.ndarray]) -> np.ndarray:
    """Per-product standardised blocks, each scaled to unit total variance, concatenated."""
    parts = []
    for m, x in embs.items():
        mu = getattr(model, f"mean_{m}").cpu().numpy()
        sd = getattr(model, f"std_{m}").cpu().numpy()
        parts.append((x - mu) / sd / np.sqrt(x.shape[-1]))
    return np.concatenate(parts, -1).astype(np.float32)


def fit_pca(model: Vec2Vec, k: int, n: int = 1 << 20) -> tuple[np.ndarray, np.ndarray]:
    """PCA (mean, components [k, D]) of concat on n pair-set train pixels with all three valid."""
    z = np.load(f"{ROOT}/pairs2024/train.npz")
    ok = np.isfinite(z["tess"][:, 0])
    idx = np.random.default_rng(0).choice(
        np.flatnonzero(ok), size=min(n, int(ok.sum())), replace=False
    )
    x = concat_blocks(model, {m: z[m][idx].astype(np.float32) for m in model.dims})
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:k]


def project(
    x: np.ndarray, pca: tuple[np.ndarray, np.ndarray], chunk: int = 1 << 20
) -> np.ndarray:
    """Rows of concat -> unit-norm PCA coordinates (float32, chunked)."""
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty((len(flat), len(pca[1])), np.float32)
    for i in range(0, len(flat), chunk):
        pc = (flat[i : i + chunk].astype(np.float32) - pca[0]) @ pca[1].T
        out[i : i + chunk] = pc / (np.linalg.norm(pc, axis=-1, keepdims=True) + 1e-12)
    return out.reshape(*x.shape[:-1], len(pca[1]))


def export_dense(model, run, step, ds, device, pca, baselines) -> None:  # noqa: ANN001
    """PASTIS: one source in memory at a time; all windows (OE covers them all)."""
    models = list(model.dims)
    ids_of = {}
    for m in models:
        with np.load(f"{INPUTS[m]}/{ds}.npz") as z:
            ids_of[m] = z["ids"].astype(str)
    ids = ids_of[models[0]]
    for m in models:
        assert set(ids_of[m]) == set(ids), f"{m}: PASTIS window sets differ"
    order = {m: np.argsort(ids_of[m])[np.argsort(np.argsort(ids))] for m in models}
    if baselines:
        for m in models:
            d = f"{ROOT}/eval_embeddings/raw_{m}/step0"
            os.makedirs(d, exist_ok=True)
            if not os.path.lexists(f"{d}/{ds}.npz"):
                os.symlink(f"{INPUTS[m]}/{ds}.npz", f"{d}/{ds}.npz")
        cat = None
        off = 0
        for m in models:
            _, x, _ = load_source(m, ds)
            x = x[order[m]]
            if cat is None:
                cat = np.empty(x.shape[:-1] + (sum(model.dims.values()),), np.float16)
            mu = getattr(model, f"mean_{m}").cpu().numpy()
            sd = getattr(model, f"std_{m}").cpu().numpy()
            for i in range(len(x)):
                cat[i, ..., off : off + x.shape[-1]] = (
                    (x[i] - mu) / sd / np.sqrt(x.shape[-1])
                )
            off += x.shape[-1]
            del x
        ok = np.ones(cat.shape[:-1], bool)
        write("concat", 0, ds, ids, cat, ok)
        write(f"concat_pca{len(pca[1])}", 0, ds, ids, project(cat, pca), ok)
        return
    fused = None
    for m in models:
        _, x, ok = load_source(m, ds)
        z = translate(model, m, x[order[m]], device)
        del x
        write(f"{run}_z{m}", step, ds, ids, z, ok[order[m]])
        fused = z if fused is None else fused + z
        del z
    fused /= np.linalg.norm(fused, axis=-1, keepdims=True) + 1e-12
    write(f"{run}_zfused", step, ds, ids, fused, np.ones(fused.shape[:-1], bool))


def main() -> None:
    """Export one run's latents, or the baselines."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--pca_k", type=int, default=128)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = load_model(a.run, device)
    models = list(model.dims)
    pca = fit_pca(model, a.pca_k) if a.baselines else None
    for ds in a.datasets.split(","):
        t0 = time.time()
        if ds.startswith("pastis"):
            export_dense(
                model,
                a.run,
                step,
                ds,
                device,
                pca if a.baselines else None,
                a.baselines,
            )
            print(f"{ds}: done in {time.time() - t0:.0f}s", flush=True)
            continue
        ids, embs, oks = load_all(ds, models)
        print(
            f"{ds}: {len(ids)} common windows ({time.time() - t0:.0f}s load)",
            flush=True,
        )
        if a.baselines:
            for m in models:
                write(f"raw_{m}", 0, ds, ids, embs[m], oks[m])
            cat = concat_blocks(model, embs)
            write("concat", 0, ds, ids, cat, np.ones(len(ids), bool))
            write(
                f"concat_pca{a.pca_k}",
                0,
                ds,
                ids,
                project(cat, pca),
                np.ones(len(ids), bool),
            )
            continue
        zs = {m: translate(model, m, embs[m], device) for m in models}
        for m in models:
            write(f"{a.run}_z{m}", step, ds, ids, zs[m], oks[m])
        fused = sum(zs.values())
        fused /= np.linalg.norm(fused, axis=-1, keepdims=True) + 1e-12
        write(f"{a.run}_zfused", step, ds, ids, fused, np.ones(len(ids), bool))


if __name__ == "__main__":
    main()
