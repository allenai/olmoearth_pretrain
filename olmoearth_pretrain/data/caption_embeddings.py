"""Text embeddings of a CaptionStore's captions, and lookup by caption key.

embeddings.npy is float16 [num_windows, 17, dim]: row 0 the window caption, rows 1-16 the
cells in row-major order. Empty captions stay all zero. Vectors are unit length at the
full dim; renormalize after truncating to a smaller Matryoshka dim.

Embed with one process per GPU; chunks are marked done as they land, so a rerun resumes:

    torchrun --nproc_per_node 8 -m olmoearth_pretrain.data.caption_embeddings
        <store_dir> <out_dir> [--model google/embeddinggemma-300m]
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from olmoearth_pretrain.data.captions import GRID, CaptionStore
from olmoearth_pretrain.data.constants import MISSING_VALUE

logger = logging.getLogger(__name__)

ROWS = GRID * GRID + 1
CHUNK = 10_000


class CaptionEmbeddings:
    """Caption embeddings by window id, memory-mapped so dataloader workers share them."""

    def __init__(self, path: str, dim: int | None = None):
        """Open embeddings written by embed_store; ``dim`` truncates (Matryoshka)."""
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.array = np.load(os.path.join(path, "embeddings.npy"), mmap_mode="r")
        self.dim = dim or self.array.shape[-1]

    def lookup(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[..., 2] (window id, cell) keys -> [..., dim] float32 embeddings and a valid mask.

        Keys from patch_caption_keys; MISSING_VALUE keys and empty captions are invalid
        and get zeros.
        """
        flat = keys.reshape(-1, 2).cpu().numpy()
        valid = (flat != MISSING_VALUE).all(axis=1)
        out = np.zeros((len(flat), self.dim), dtype=np.float32)
        if valid.any():
            idx = np.flatnonzero(valid)
            window = flat[idx, 0].astype(np.int64)
            row = flat[idx, 1].astype(np.int64) + 1
            vec = self.array[window, row, : self.dim].astype(np.float32)
            norm = np.linalg.norm(vec, axis=1)
            ok = norm > 0
            out[idx[ok]] = vec[ok] / norm[ok, None]
            valid[idx[~ok]] = False
        shape = keys.shape[:-1]
        return (
            torch.from_numpy(out).reshape(*shape, self.dim),
            torch.from_numpy(valid).reshape(shape),
        )


def embed_store(
    store_dir: str,
    out_dir: str,
    model_name: str,
    prompt_name: str | None,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Embed every caption in the store; this process takes every world_size-th chunk."""
    from sentence_transformers import SentenceTransformer

    store = CaptionStore(store_dir)
    device = (
        f"cuda:{rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={
            "torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32
        },
    )
    dim = model.get_sentence_embedding_dimension()
    path = os.path.join(out_dir, "embeddings.npy")
    done_dir = os.path.join(out_dir, "done")
    shape = (len(store), ROWS, dim)
    if rank == 0:
        os.makedirs(done_dir, exist_ok=True)
        if not os.path.exists(path):
            np.lib.format.open_memmap(
                path + ".tmp.npy", "w+", np.float16, shape
            ).flush()
            os.replace(path + ".tmp.npy", path)
        meta = {
            "store": store_dir,
            "model": model_name,
            "prompt_name": prompt_name,
            "dim": dim,
            "shape": list(shape),
        }
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    if world_size > 1:
        torch.distributed.barrier()
    # Plain writes, not the memory map: on Weka, memory-mapped writes from a restarted
    # job were silently lost even after flush().
    offset = np.load(path, mmap_mode="r").offset
    row_bytes = ROWS * dim * np.dtype(np.float16).itemsize

    chunks = range(rank * CHUNK, len(store), world_size * CHUNK)
    for start in chunks:
        marker = os.path.join(done_dir, f"{start:08d}")
        if os.path.exists(marker):
            continue
        end = min(start + CHUNK, len(store))
        texts, where = [], []
        for i in range(start, end):
            window, cells = store.get(i)
            for row, text in enumerate([window, *cells]):
                if text:
                    texts.append(text)
                    where.append((i - start, row))
        block = np.zeros((end - start, ROWS, dim), dtype=np.float16)
        if texts:
            emb = model.encode(
                texts,
                prompt_name=prompt_name,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            idx = np.array(where)
            block[idx[:, 0], idx[:, 1]] = emb.astype(np.float16)
        write_rows(path, offset + start * row_bytes, block)
        open(marker, "w").close()
        logger.info(f"rank {rank}: windows {start}-{end}, {len(texts)} captions")


def write_rows(path: str, at: int, block: np.ndarray) -> None:
    """Write ``block`` at byte ``at``, sync it, and check it reads back."""
    data = block.tobytes()
    fd = os.open(path, os.O_WRONLY)
    try:
        written = 0
        while written < len(data):
            written += os.pwrite(fd, data[written:], at + written)
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(path, os.O_RDONLY)
    try:
        tail = os.pread(
            fd, min(len(data), 1 << 20), at + len(data) - min(len(data), 1 << 20)
        )
    finally:
        os.close(fd)
    if tail != data[len(data) - len(tail) :]:
        raise OSError(f"rows written at byte {at} of {path} did not read back")


def main() -> None:
    """Embed a CaptionStore from the command line, under torchrun or alone."""
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("store_dir")
    p.add_argument("out_dir")
    p.add_argument("--model", default="google/embeddinggemma-300m")
    p.add_argument("--prompt_name", default="document")
    p.add_argument("--batch_size", type=int, default=512)
    a = p.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        torch.distributed.init_process_group("gloo")
    embed_store(
        a.store_dir,
        a.out_dir,
        a.model,
        a.prompt_name or None,
        a.batch_size,
        rank,
        world_size,
    )
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
