"""Disk-backed text embedding cache.

Caches pooled and token-sequence embeddings for every ``ClassEntry`` in a
``ClassRegistry``. The cache file is keyed on the encoder's ``name`` so
swapping encoders produces a fresh cache.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.open_set.catalog.registry import ClassEntry, ClassRegistry
from olmoearth_pretrain.open_set.text.base import TextEncoder, TextEncoding

logger = logging.getLogger(__name__)

# Bumped whenever the on-disk format changes.
CACHE_FORMAT_VERSION = 1


def _safe_filename(model_name: str) -> str:
    """Turn a model id like ``"google/siglip2-so400m-patch14-384"`` into a filename."""
    # SHA1 is used purely to disambiguate cache filenames — not for security.
    digest = hashlib.sha1(  # nosec B324
        model_name.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    safe = model_name.replace("/", "_").replace(":", "_")
    return f"text_emb_{safe}_{digest}_v{CACHE_FORMAT_VERSION}.pt"


@dataclass
class _CachedEntry:
    """Per-class cached tensors. All tensors are CPU-resident."""

    tokens: torch.Tensor  # [L, D]
    pooled: torch.Tensor  # [D]
    attention_mask: torch.Tensor  # [L]


def _key(entry: ClassEntry) -> str:
    """Cache key for a single ClassEntry — collisions across sources are not allowed."""
    return f"{entry.source}::{entry.text}"


class TextEmbeddingCache:
    """Holds (and lazily computes + persists) embeddings for a registry.

    The cache stores per-token embeddings (length L) and a pooled embedding
    per class. Looking up by class returns a ``TextEncoding`` of batch size 1
    on the requested device.

    Currently each class is encoded once with its canonical text. Synonym
    ensembling can be added later by mean-pooling pooled embeddings across
    ``entry.all_prompts()`` — the ``rebuild`` method already supports this.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        cache_path: Path | str | None = None,
    ) -> None:
        """Initialize the cache, optionally bound to a path on disk."""
        self._encoder = encoder
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._entries: dict[str, _CachedEntry] = {}

    @property
    def encoder(self) -> TextEncoder:
        """The wrapped text encoder."""
        return self._encoder

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings the encoder produces."""
        return self._encoder.embedding_dim

    def __contains__(self, entry: ClassEntry) -> bool:
        """Return True if this entry has already been cached."""
        return _key(entry) in self._entries

    def __len__(self) -> int:
        """Number of cached classes."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Building / loading
    # ------------------------------------------------------------------

    def populate(
        self, registry: ClassRegistry, batch_size: int = 32, force: bool = False
    ) -> None:
        """Ensure every entry in ``registry`` has cached embeddings.

        If ``self._cache_path`` exists and ``force`` is False we load from
        disk first and only encode classes that are missing.
        """
        if self._cache_path is not None and self._cache_path.exists() and not force:
            self._load()

        missing = [e for e in registry if _key(e) not in self._entries]
        if not missing:
            return

        logger.info(
            "TextEmbeddingCache: encoding %d new class(es) with %s",
            len(missing),
            self._encoder.name,
        )
        self._encode_and_store(missing, batch_size=batch_size)

        if self._cache_path is not None:
            self._save()

    def rebuild(
        self,
        registry: ClassRegistry,
        batch_size: int = 32,
        ensemble_synonyms: bool = False,
    ) -> None:
        """Drop the cache and re-encode every entry from scratch.

        If ``ensemble_synonyms`` is True, the pooled embedding becomes the
        mean over the canonical text plus all synonyms; the token sequence
        and attention mask still come from the canonical text only (they are
        what the decoder cross-attends to).
        """
        self._entries.clear()
        if not ensemble_synonyms:
            self._encode_and_store(list(registry), batch_size=batch_size)
        else:
            for entry in registry:
                prompts = list(entry.all_prompts())
                encoding = self._encoder.encode(prompts)
                pooled = encoding.pooled.mean(dim=0).cpu()
                self._entries[_key(entry)] = _CachedEntry(
                    tokens=encoding.tokens[0].cpu(),
                    pooled=pooled,
                    attention_mask=encoding.attention_mask[0].cpu(),
                )
        if self._cache_path is not None:
            self._save()

    def _encode_and_store(self, entries: Iterable[ClassEntry], batch_size: int) -> None:
        entries = list(entries)
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            encoding = self._encoder.encode([e.text for e in chunk])
            tokens = encoding.tokens.cpu()
            pooled = encoding.pooled.cpu()
            attn = encoding.attention_mask.cpu()
            for i, entry in enumerate(chunk):
                self._entries[_key(entry)] = _CachedEntry(
                    tokens=tokens[i],
                    pooled=pooled[i],
                    attention_mask=attn[i],
                )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(
        self,
        entry: ClassEntry,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> TextEncoding:
        """Return a batch-of-one ``TextEncoding`` for ``entry``."""
        cached = self._entries.get(_key(entry))
        if cached is None:
            raise KeyError(
                f"No cached embedding for {_key(entry)!r}. Call populate() first."
            )
        device = torch.device(device)
        tokens = cached.tokens.to(device=device, dtype=dtype).unsqueeze(0)
        pooled = cached.pooled.to(device=device, dtype=dtype).unsqueeze(0)
        attn = cached.attention_mask.to(device=device).unsqueeze(0)
        return TextEncoding(tokens=tokens, pooled=pooled, attention_mask=attn)

    def get_many(
        self,
        entries: list[ClassEntry],
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> TextEncoding:
        """Stack cached embeddings for several entries into a single batch.

        All cached token tensors share a length (the encoder pads to a fixed
        ``max_length``) so they can be stacked without further padding.
        """
        if not entries:
            raise ValueError("get_many requires at least one entry")
        device = torch.device(device)
        tokens = torch.stack(
            [self._entries[_key(e)].tokens for e in entries], dim=0
        ).to(device=device, dtype=dtype)
        pooled = torch.stack(
            [self._entries[_key(e)].pooled for e in entries], dim=0
        ).to(device=device, dtype=dtype)
        attn = torch.stack(
            [self._entries[_key(e)].attention_mask for e in entries], dim=0
        ).to(device=device)
        return TextEncoding(tokens=tokens, pooled=pooled, attention_mask=attn)

    def all_pooled(
        self,
        registry: ClassRegistry,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return ``[len(registry), D]`` pooled embeddings in registry order.

        Useful for similarity-based hard-negative sampling later.
        """
        return self.get_many(list(registry), device=device, dtype=dtype).pooled

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self._cache_path is not None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_FORMAT_VERSION,
            "encoder_name": self._encoder.name,
            "embedding_dim": self._encoder.embedding_dim,
            "entries": {
                key: {
                    "tokens": cached.tokens,
                    "pooled": cached.pooled,
                    "attention_mask": cached.attention_mask,
                }
                for key, cached in self._entries.items()
            },
        }
        torch.save(payload, self._cache_path)
        logger.info("Wrote text embedding cache to %s", self._cache_path)

    def _load(self) -> None:
        assert self._cache_path is not None
        payload = torch.load(self._cache_path, map_location="cpu", weights_only=False)
        if payload.get("version") != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Cache version mismatch in {self._cache_path}: "
                f"got {payload.get('version')}, expected {CACHE_FORMAT_VERSION}"
            )
        if payload.get("encoder_name") != self._encoder.name:
            raise ValueError(
                f"Cache at {self._cache_path} was built with encoder "
                f"{payload.get('encoder_name')!r}, but the current encoder is "
                f"{self._encoder.name!r}. Use a different cache path."
            )
        for key, payload_entry in payload["entries"].items():
            self._entries[key] = _CachedEntry(
                tokens=payload_entry["tokens"],
                pooled=payload_entry["pooled"],
                attention_mask=payload_entry["attention_mask"],
            )
        logger.info(
            "Loaded text embedding cache (%d entries) from %s",
            len(self._entries),
            self._cache_path,
        )


def default_cache_path(cache_dir: Path | str, encoder: TextEncoder) -> Path:
    """Return the default cache filename for an encoder under ``cache_dir``."""
    return Path(cache_dir) / _safe_filename(encoder.name)


@dataclass
class TextEncoderConfig(Config):
    """Configuration for the text encoder + on-disk cache.

    The text encoder is loaded lazily inside ``build`` so that this config
    can round-trip through the olmo-core serializer without importing
    Hugging Face ``transformers``.

    Attributes:
        model_name: HF model id (or local path) for the SigLIP encoder.
        cache_dir: Directory in which to store the on-disk text-embedding
            cache. If None, the cache is in-memory only — fine for testing
            but means SigLIP runs every time training starts.
    """

    # Default chosen at the entrypoint to avoid a top-level import of the
    # SigLIP module (which would import transformers eagerly).
    model_name: str = "google/siglip2-so400m-patch14-384"
    cache_dir: str | None = None

    def build(self, registry: ClassRegistry) -> TextEmbeddingCache:
        """Construct the encoder, populate (or load) the cache, return it."""
        # Lazy import — keeps the optional ``transformers`` dependency out of
        # the import path until the user actually launches training.
        from olmoearth_pretrain.open_set.text.siglip_encoder import SigLIPTextEncoder

        encoder = SigLIPTextEncoder(model_name=self.model_name, device="cpu")
        cache_path = (
            default_cache_path(self.cache_dir, encoder)
            if self.cache_dir is not None
            else None
        )
        cache = TextEmbeddingCache(encoder, cache_path=cache_path)
        cache.populate(registry)
        return cache
