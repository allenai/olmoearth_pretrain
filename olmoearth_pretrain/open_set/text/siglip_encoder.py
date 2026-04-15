"""SigLIP / SigLIP 2 text encoder wrapper.

Wraps a Hugging Face ``transformers`` model + processor. The model is loaded
lazily so importing this module does not require ``transformers`` to be
installed; it is only required when an instance is actually constructed.
"""

from __future__ import annotations

from typing import Any

import torch

from olmoearth_pretrain.open_set.text.base import TextEncoding

DEFAULT_MODEL_NAME = "google/siglip2-so400m-patch14-384"


class SigLIPTextEncoder:
    """Thin wrapper over the HF SigLIP text tower.

    The model is held in eval mode and has gradients disabled. ``encode`` is
    safe to call inside a normal training loop — its outputs are detached.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize and load the underlying HF model.

        Args:
            model_name: HF model id or local path.
            device: Where to place the model.
            dtype: Parameter dtype for the model.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "SigLIPTextEncoder requires the `transformers` package. "
                "Install with: pip install transformers"
            ) from e

        self._model_name = model_name
        self._device = torch.device(device)
        self._dtype = dtype

        # SigLIP uses the same tokenizer for text in both image-text training
        # and inference. AutoTokenizer routes to SiglipTokenizer. Revision
        # pinning is the caller's responsibility — they pass the exact
        # ``model_name`` they want (which may itself include a revision).
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
        full_model = AutoModel.from_pretrained(  # nosec B615
            model_name, torch_dtype=dtype
        )
        # We only need the text tower. Models that do not expose a sub-module
        # are used as-is.
        self._model = getattr(full_model, "text_model", full_model)
        self._model = self._model.to(self._device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        # Probe the embedding dim by inspecting the model's output shape on a
        # tiny dummy input. Falls back to config inspection if available.
        cfg = getattr(self._model, "config", None)
        hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
        if hidden is None:
            # Best-effort fallback — run one token through the model.
            with torch.no_grad():
                probe = self._tokenizer(
                    ["a"], padding="max_length", return_tensors="pt"
                ).to(self._device)
                out = self._call_model(probe)
                hidden = out["last_hidden_state"].shape[-1]
        self._embedding_dim = int(hidden)

    @property
    def name(self) -> str:
        """Stable identifier used as part of the on-disk cache key."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the per-token / pooled embeddings."""
        return self._embedding_dim

    def _call_model(self, encoded: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Run the underlying text tower; normalises the output to a dict."""
        out = self._model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
        )
        # ``BaseModelOutput`` exposes ``last_hidden_state`` and (for SigLIP)
        # ``pooler_output``. We normalize to a dict.
        return {
            "last_hidden_state": out.last_hidden_state,
            "pooler_output": getattr(out, "pooler_output", None),
        }

    @torch.no_grad()
    def encode(self, texts: list[str]) -> TextEncoding:
        """Encode a batch of strings.

        Returns:
            A ``TextEncoding`` with detached CPU-or-device tensors matching
            ``self._device``.
        """
        if not texts:
            raise ValueError("encode() requires at least one string")
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        out = self._call_model(encoded)
        tokens = out["last_hidden_state"].detach()  # [N, L, D]
        attn = encoded.get(
            "attention_mask",
            torch.ones(tokens.shape[:2], dtype=torch.long, device=self._device),
        ).detach()
        pooled = out["pooler_output"]
        if pooled is None:
            # Mean-pool over real tokens as a sensible fallback.
            mask = attn.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.detach()
        return TextEncoding(tokens=tokens, pooled=pooled, attention_mask=attn)
