"""Bake precomputed embedding products into rslearn eval datasets.

This package materializes precomputed embedding products (AlphaEarth/GSE,
Tessera) as raster layers in existing rslearn eval datasets, one window at a
time. See ``__main__.py`` for the CLI entry point.
"""

from olmoearth_pretrain.evals.embedding_materializer.fetchers import (
    AEFFetcher,
    EmbeddingFetcher,
    TesseraFetcher,
)
from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    materialize_product,
    write_manifest,
)
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)

__all__ = [
    "AEFFetcher",
    "EmbeddingFetcher",
    "RslearnWindowProvider",
    "TesseraFetcher",
    "materialize_product",
    "write_manifest",
]
