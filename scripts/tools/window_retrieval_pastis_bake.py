r"""Bake the window-retrieval PASTIS eval datasets (``pastis_tw_*``).

Scores retrieved embeddings with the same LP / mIoU harness as every other
precomputed product (``--model=olmoearth_precomputed``). Two steps:

- ``embed`` (GPU): frozen v1.3, S1 + S2 + Landsat, run the way the store is built
  (Searchlight, patch size 2, one location query per pixel, unmasked) over
  every PASTIS window for one source:

  * ``cal2019``: ``pastis_year_aligned`` (calendar 2019);
  * ``cal2018``: ``pastis_cal2018`` (calendar 2018, staging copy);
  * ``sepaug``:  ``pastis_sepaug`` (Sep 2018 - Aug 2019; pastis_rslearn + Landsat, staging copy);
  * ``maraug``:  ``pastis_year_aligned`` sliced to 30-day steps 2-7 (Mar - Aug 2019),
    its own forward pass on the sliced input.

  Embeddings are stored int8 (the store's power scheme), one ``.npy`` per
  window, under ``--work_dir/<source>/``.

- ``bake`` (CPU): builds six sibling datasets of ``pastis_year_aligned``
  (same windows, splits and labels; the ``olmoearth_emb`` layer holds the
  variant), with config.json copied from ``data/rslearn_dataset_configs/<name>``:

  * ``annual2019``   = z_2019
  * ``fresh_sepaug`` = z_SepAug
  * ``avg_sepaug``   = 4/12 z_2018 + 8/12 z_2019
  * ``ret_sepaug``   = 4/12 h(z_2018, Sep-Dec) + 8/12 h(z_2019, Jan-Aug)
  * ``fresh_maraug`` = z_MarAug
  * ``ret_maraug``   = h(z_2019, Mar-Aug)

  ``h`` is a retriever trained by ``window_retrieval_train.py`` on an S1 + S2 +
  Landsat cache, matching these inputs and the store. Every variant is int8 round-tripped, as the
  store is. In 30-day steps from Jan 1, Sep-Dec is steps 8-11 (Aug 29 - Dec 27)
  and Jan-Aug is steps 0-7 (Jan 1 - Aug 29); pastis_rslearn starts Sep 1, so the
  two crop years differ by three days at each end.

Usage::

    python scripts/tools/window_retrieval_pastis_bake.py embed --source cal2019 --work_dir $W
    python scripts/tools/window_retrieval_pastis_bake.py bake --work_dir $W --retriever_dir $R
"""

import argparse
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from rslearn.dataset import Dataset, Window
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.rslearn_builder import parse_model_config
from olmoearth_pretrain.evals.datasets.rslearn_dataset import (
    RslearnToOlmoEarthDataset,
    _window_key,
)
from olmoearth_pretrain.evals.embedding_transforms import (
    dequantize_embeddings,
    quantize_embeddings,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_retrieval_train import build, retrieve  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "data/rslearn_dataset_configs"
LOADER_YAML = CONFIG_ROOT / "window_retrieval_pastis_imagery/model.yaml"
EVAL_DATASETS = "/weka/dfive-default/olmoearth/eval_datasets"
TEMPLATE_DS = f"{EVAL_DATASETS}/pastis_year_aligned"
SOURCES = {
    "cal2019": (f"{EVAL_DATASETS}/pastis_year_aligned", None),
    "cal2018": (
        "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/pastis_cal2018",
        None,
    ),
    "sepaug": (
        "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/pastis_sepaug",
        None,
    ),
    "maraug": (f"{EVAL_DATASETS}/pastis_year_aligned", (2, 8)),
}
INPUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]
SPLITS = ("train", "val", "test")
NODATA_VALUE = -1.0
LABEL_LAYER = "label"
VARIANTS = (
    "annual2019",
    "fresh_sepaug",
    "avg_sepaug",
    "ret_sepaug",
    "fresh_maraug",
    "ret_maraug",
)


def key_to_file(key: str) -> str:
    """'group/name' -> a flat file name."""
    return key.replace("/", "__") + ".npy"


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


def embed_source(args: argparse.Namespace) -> None:
    """Embed every window of one source and store int8 codes per window."""
    # Searchlight (via the cache module) exists only on the branch embed runs on;
    # bake runs on the precomputed-eval branch, which lacks it.
    from window_retrieval_cache import embed, slice_time

    from olmoearth_pretrain.model_loader import load_model_from_path

    path, steps = SOURCES[args.source]
    out = Path(args.work_dir) / args.source
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    encoder = load_model_from_path(UPath(args.checkpoint)).encoder.to(device).eval()
    config = parse_model_config(str(LOADER_YAML))

    done = failed = 0
    start = time.time()
    for split in SPLITS:
        ds = RslearnToOlmoEarthDataset.from_model_config(
            model_config=config,
            source_path=path,
            split=split,
            input_modalities=INPUT_MODALITIES,
            groups_override=[],
            tags_override={"eval_split": split},
        )
        logger.info(f"{args.source} {split}: {len(ds.dataset)} windows from {path}")
        for idx in range(len(ds.dataset)):
            input_dict, target, metadata = ds.dataset[idx]
            key = _window_key(metadata)
            dst = out / key_to_file(key)
            if dst.exists():
                continue
            try:
                sample, _ = ds._transform_sample(input_dict, target, window_key=key)
                sample = sample._replace(
                    **{
                        k: v[None]
                        for k, v in sample.as_dict(include_nones=False).items()
                    }
                )
                if steps is not None:
                    sample = slice_time(sample, *steps)
                z = embed(encoder, sample, device)  # [H, W, D]
                np.save(dst, quantize_embeddings(z.cpu()).numpy())
                done += 1
            except Exception:
                logger.exception(f"{key} failed")
                failed += 1
            if (done + failed) % 100 == 0:
                logger.info(
                    f"  {done + failed} windows ({(done + failed) / (time.time() - start):.2f}/s), failed {failed}"
                )
    logger.info(f"{args.source}: {done} embedded, {failed} failed -> {out}")
    if failed:
        raise SystemExit(f"{failed} windows failed")


# ---------------------------------------------------------------------------
# bake
# ---------------------------------------------------------------------------


def load_retriever(retriever_dir: str) -> torch.nn.Module:
    """Rebuild a trained retriever from its metrics.json args and model.pt."""
    meta = json.loads((Path(retriever_dir) / "metrics.json").read_text())
    model = build(argparse.Namespace(**meta["args"]), dim=128)
    if meta["args"]["arch"] != "floor":
        model.load_state_dict(
            torch.load(Path(retriever_dir) / "model.pt", map_location="cpu")
        )
    return model.eval()


def apply_h(
    model: torch.nn.Module, z: torch.Tensor, window: tuple[int, int]
) -> torch.Tensor:
    """Retrieve a window (start, length in 30-day steps) for every pixel of a calendar-year [H, W, D] embedding.

    Runs on the retriever's device; returns a CPU tensor.
    """
    device = next(model.parameters(), torch.empty(0)).device
    H, W, D = z.shape
    flat = z.reshape(-1, D).to(device)
    n = flat.shape[0]
    start_month = torch.zeros(
        n, dtype=torch.long, device=device
    )  # the store holds calendar years
    start = torch.full((n,), window[0], dtype=torch.long, device=device)
    length = torch.full((n,), window[1], dtype=torch.long, device=device)
    with torch.no_grad():
        return retrieve(model, flat, start_month, start, length).reshape(H, W, D).cpu()


def variants_for(
    codes: dict[str, np.ndarray], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """All six variants for one window, each [H, W, D] float."""
    z = {k: dequantize_embeddings(torch.from_numpy(v)) for k, v in codes.items()}
    return {
        "annual2019": z["cal2019"],
        "fresh_sepaug": z["sepaug"],
        "avg_sepaug": (4 * z["cal2018"] + 8 * z["cal2019"]) / 12,
        "ret_sepaug": (
            4 * apply_h(model, z["cal2018"], (8, 4))
            + 8 * apply_h(model, z["cal2019"], (0, 8))
        )
        / 12,
        "fresh_maraug": z["maraug"],
        "ret_maraug": apply_h(model, z["cal2019"], (2, 6)),
    }


def copy_window(src: Window, dst_dataset: Dataset) -> Window:
    """Recreate a window in a sibling dataset and copy its label layer."""
    dst = Window(
        storage=dst_dataset.storage,
        group=src.group,
        name=src.name,
        projection=src.projection,
        bounds=src.bounds,
        time_range=src.time_range,
        options=src.options,
    )
    dst.save()
    src_dir, dst_dir = src.get_layer_dir(LABEL_LAYER), dst.get_layer_dir(LABEL_LAYER)
    if dst_dir.exists():
        shutil.rmtree(str(dst_dir))
    shutil.copytree(str(src_dir), str(dst_dir))
    dst.mark_layer_completed(LABEL_LAYER)
    return dst


def bake(args: argparse.Namespace) -> None:
    """Write the six sibling datasets."""
    # Only on branches with precomputed-embedding evals; embed runs on older code.
    from olmoearth_pretrain.evals.embedding_materializer.providers import (
        RslearnWindowProvider,
    )

    modality = Modality.OLMOEARTH_EMB
    work = Path(args.work_dir)
    model = load_retriever(args.retriever_dir).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"retriever on {next(model.parameters(), torch.empty(0)).device}")
    targets = {}
    for v in VARIANTS:
        name = f"pastis_tw_{v}"
        dst_path = UPath(f"{EVAL_DATASETS}/{name}")
        if dst_path.exists() and not args.resume:
            raise SystemExit(f"{dst_path} exists (pass --resume to continue)")
        dst_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            CONFIG_ROOT / name / "config.json", str(dst_path / "config.json")
        )
        targets[v] = (Dataset(dst_path), RslearnWindowProvider(dst_path))

    windows = [
        w
        for w in Dataset(UPath(TEMPLATE_DS)).storage.get_windows()
        if w.is_layer_completed(Modality.GSE.name) and w.is_layer_completed(LABEL_LAYER)
    ]
    logger.info(f"{len(windows)} pastis_year_aligned windows carry gse + label")
    missing: list[str] = []

    def handle(w: Window) -> None:
        key = f"{w.group}/{w.name}"
        files = {s: work / s / key_to_file(key) for s in SOURCES}
        if not all(f.exists() for f in files.values()):
            missing.append(key)
            return
        out = variants_for({s: np.load(f) for s, f in files.items()}, model)
        for v, emb in out.items():
            emb = dequantize_embeddings(quantize_embeddings(emb))  # store parity
            dataset, provider = targets[v]
            dst = copy_window(w, dataset)
            provider.write_embedding(
                dst,
                modality,
                emb.permute(2, 0, 1).numpy().astype(np.float32),
                NODATA_VALUE,
            )

    with ThreadPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(pool.map(handle, windows), 1):
            if n % 200 == 0:
                logger.info(f"  baked {n}/{len(windows)}")
    # A window missing from any source (e.g. the 12 PASTIS windows whose S1 footprint
    # fails, which the loader skips) is left out of every variant, so all six share one
    # window set: the same 2421 the forward-pass PASTIS evals are scored on.
    manifest = {
        "baked": len(windows) - len(missing),
        "excluded_missing_source": sorted(missing),
    }
    for v in VARIANTS:
        (
            UPath(f"{EVAL_DATASETS}/pastis_tw_{v}")
            / "window_retrieval_bake_manifest.json"
        ).write_text(
            json.dumps(
                {**manifest, "variant": v, "retriever_dir": args.retriever_dir},
                indent=2,
            )
        )
    logger.info(
        f"baked {manifest['baked']} windows into {len(VARIANTS)} datasets; excluded {len(missing)}"
    )
    if len(missing) > args.max_missing:
        raise SystemExit(
            f"{len(missing)} windows lack an embedding (> {args.max_missing}), e.g. {missing[:3]}"
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
    parser.add_argument("command", choices=["embed", "bake"])
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--checkpoint", help="converted v1.3 checkpoint dir (embed)")
    parser.add_argument("--retriever_dir")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max_missing", type=int, default=20, help="fail if more windows lack a source"
    )
    args = parser.parse_args()
    if args.command == "embed":
        if not args.checkpoint:
            parser.error("embed needs --checkpoint")
        embed_source(args)
    else:
        bake(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
