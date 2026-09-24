r"""Embed the open-set class names with a sentence encoder (CLIP-style probe targets).

Writes, for the frozen ``class_mapping.json``:

* ``class_text_embeddings.npy``: ``(num_classes, dim)`` float16, L2-normalized, row =
  global class id;
* ``class_text_embeddings.json``: sidecar with the encoder name, the text template and
  the sha256 of the class mapping the rows belong to (verified at train time by
  ``train.open_set_probe.load_text_embeddings``).

The text for a class is ``"{class name}; {dataset name}"`` (the dataset name from the
label-bank registry, so e.g. "Wheat; AgriFieldNet India"); merged presence-only
classes use ``"{concept}; {member class names}"``.

The encoder (``sentence-transformers/all-mpnet-base-v2``, 768-d, matching the v1.3
register dim) is imported lazily so ``sentence-transformers`` is NOT a repo
dependency. Run this once in a Beaker session (GPU optional; 3871 short strings take
seconds either way), writing next to the label bank on weka (the location
``scripts/official/v1_3/open_set_base.CLASS_TEXT_EMBEDDINGS_PATH`` reads from)::

    uv pip install sentence-transformers
    python -m olmoearth_pretrain.open_set_segmentation_data.embed_class_names
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .assemble_classes import DEFAULT_OUTPUT_PATH as DEFAULT_CLASS_MAPPING_PATH
from .manifest import OUTPUT_ROOT, REGISTRY_PATH

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Next to the label bank on weka (see manifest.OUTPUT_ROOT); read by the v1.3 scripts.
DEFAULT_OUTPUT_DIR = Path(str(OUTPUT_ROOT)) / "class_text_embeddings"
TEXT_TEMPLATE = "{class_name}; {dataset_name}"
OUTPUT_STEM = "class_text_embeddings"

# ``encode(texts) -> (N, dim)`` float array.
TextEncoder = Callable[[Sequence[str]], np.ndarray]


def class_texts(class_mapping: dict[str, Any], registry: dict[str, Any]) -> list[str]:
    """One text per global class id, in id order."""
    dataset_names = {
        entry["slug"]: entry.get("name") or entry["slug"]
        for entry in registry["datasets"]
    }
    classes = sorted(class_mapping["open_set"]["classes"], key=lambda c: c["global_id"])
    num_classes = int(class_mapping["open_set"]["num_classes"])
    if [c["global_id"] for c in classes] != list(range(num_classes)):
        raise ValueError("class_mapping global ids are not 0..num_classes-1")

    texts = []
    for c in classes:
        members = c.get("members")
        if members:
            # Merged presence-only concept: name the concept and its member classes.
            member_names = sorted({str(m["name"]) for m in members})
            texts.append(
                TEXT_TEMPLATE.format(
                    class_name=str(c["name"]).replace("_", " "),
                    dataset_name=", ".join(member_names),
                )
            )
        else:
            texts.append(
                TEXT_TEMPLATE.format(
                    class_name=str(c["name"]),
                    dataset_name=dataset_names.get(c["slug"], c["slug"]),
                )
            )
    return texts


def sentence_transformer_encoder(model_name: str) -> TextEncoder:
    """Build an encoder from ``sentence-transformers`` (imported lazily)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def _encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        )

    return _encode


def embed_class_names(
    class_mapping_path: Path,
    registry_path: Path,
    output_dir: Path,
    encoder: TextEncoder,
    model_name: str,
) -> tuple[Path, Path]:
    """Embed every class text and write the ``.npy`` + ``.json`` sidecar.

    Returns the two output paths.
    """
    mapping_bytes = class_mapping_path.read_bytes()
    class_mapping = json.loads(mapping_bytes)
    with registry_path.open() as f:
        registry = json.load(f)

    texts = class_texts(class_mapping, registry)
    logger.info("embedding %d class texts with %s", len(texts), model_name)
    embeddings = np.asarray(encoder(texts), dtype=np.float32)
    if embeddings.shape[0] != len(texts) or embeddings.ndim != 2:
        raise ValueError(
            f"encoder returned shape {embeddings.shape}, expected ({len(texts)}, dim)"
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{OUTPUT_STEM}.npy"
    sidecar_path = output_dir / f"{OUTPUT_STEM}.json"
    np.save(npy_path, embeddings.astype(np.float16))
    sidecar = {
        "model_name": model_name,
        "text_template": TEXT_TEMPLATE,
        "class_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "num_classes": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "dtype": "float16",
        "normalized": True,
    }
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("wrote %s and %s", npy_path, sidecar_path)
    return npy_path, sidecar_path


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--class_mapping",
        type=Path,
        default=DEFAULT_CLASS_MAPPING_PATH,
        help="Frozen class_mapping.json to embed the classes of",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_PATH,
        help="Label-bank registry.json (dataset names)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write {OUTPUT_STEM}.npy / .json into",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="sentence-transformers model name",
    )
    args = parser.parse_args()
    embed_class_names(
        args.class_mapping,
        args.registry,
        args.output,
        sentence_transformer_encoder(args.model),
        args.model,
    )


if __name__ == "__main__":
    main()
