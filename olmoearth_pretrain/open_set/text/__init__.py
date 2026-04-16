"""Text encoders and embedding cache for the open-set decoder.

We define a generic ``TextEncoder`` protocol so the SigLIP encoder can be
swapped for RemoteCLIP, GeoRSCLIP, etc. without touching downstream code.
"""

from olmoearth_pretrain.open_set.text.base import TextEncoder, TextEncoding
from olmoearth_pretrain.open_set.text.embedding_cache import (
    TextEmbeddingCache,
    TextEncoderConfig,
)

__all__ = [
    "TextEmbeddingCache",
    "TextEncoder",
    "TextEncoderConfig",
    "TextEncoding",
]
