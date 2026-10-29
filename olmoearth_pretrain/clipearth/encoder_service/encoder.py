"""EmbeddingGemma through onnxruntime.

The same q4 ONNX file and tokenizer the ClipEarth lab runs in the browser with transformers.js, so a query encodes to the same vector either way.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"
MODEL_FILE = "model_no_gather_q4"
DIMS = 768


class Encoder:
    """The text model with a cache of texts already encoded."""

    def __init__(self, model_dir: str | Path, threads: int = 0, cache_size: int = 4096):
        """Load the tokenizer and ONNX session from a snapshot of the model."""
        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or os.cpu_count() or 1
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / f"{MODEL_FILE}.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self.lock = threading.Lock()
        self.stats = {
            "requests": 0,
            "texts": 0,
            "cache_hits": 0,
            "runs": 0,
            "run_s": 0.0,
        }

    def _run(self, texts: list[str]) -> np.ndarray:
        """Right-padded batch, as transformers.js pads it."""
        encodings = self.tokenizer.encode_batch(texts)
        length = max(len(e.ids) for e in encodings)
        ids = np.full((len(texts), length), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(texts), length), dtype=np.int64)
        for i, e in enumerate(encodings):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        (vectors,) = self.session.run(
            ["sentence_embedding"], {"input_ids": ids, "attention_mask": mask}
        )
        return vectors.astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        """One row per text; texts already encoded come from the cache."""
        with self.lock:
            self.stats["requests"] += 1
            self.stats["texts"] += len(texts)
            found = {t: self.cache[t] for t in texts if t in self.cache}
            for t in found:
                self.cache.move_to_end(t)
            self.stats["cache_hits"] += sum(t in found for t in texts)
        missing = [t for t in dict.fromkeys(texts) if t not in found]
        if missing:
            t0 = time.perf_counter()
            vectors = self._run(missing)
            with self.lock:
                self.stats["runs"] += 1
                self.stats["run_s"] += time.perf_counter() - t0
                for text, v in zip(missing, vectors):
                    found[text] = v
                    self.cache[text] = v
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return np.stack([found[t] for t in texts])
