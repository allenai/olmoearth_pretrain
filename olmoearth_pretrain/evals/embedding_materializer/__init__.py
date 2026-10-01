"""Bake precomputed embedding products into rslearn eval datasets.

Each product (AlphaEarth/GSE) is declared as an rslearn layer backed by the
product's data source and materialized with rslearn's own prepare/materialize.
See ``__main__.py`` for the CLI entry point.
"""

from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    PRODUCTS,
    EmbeddingProduct,
    materialize_product,
    write_manifest,
)

__all__ = [
    "PRODUCTS",
    "EmbeddingProduct",
    "materialize_product",
    "write_manifest",
]
