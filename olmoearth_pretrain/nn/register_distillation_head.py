"""Distillation of the Perceiver's register grid into its detached student."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from olmoearth_pretrain.config import Config


@dataclass
class RegisterDistillationHeadConfig(Config):
    """Configuration for :class:`RegisterDistillationHead`.

    Attaching this to ``LatentMIMConfig`` turns distillation on for the encoder's
    detached student (``perceiver_config.projection_dims``); without it the student
    is trained by supervision alone, if at all.

    Args:
        back_projection_hidden: Hidden width of the per-prefix back-projection heads.
            None keeps a single ``Linear(d, register_dim)``; an int makes each head a
            2-layer MLP ``Linear(d, H) -> LayerNorm -> ReLU -> Linear(H, register_dim)``.
        gram_max_tokens: Max register cells entering the Gram terms per microbatch
            (one random subsample shared across prefixes; bounds the O(n^2) matrices).
    """

    back_projection_hidden: int | None = None
    gram_max_tokens: int = 2048

    def validate(self) -> None:
        """Validate the configuration."""
        if self.back_projection_hidden is not None and self.back_projection_hidden <= 0:
            raise ValueError(
                "back_projection_hidden must be positive, got "
                f"{self.back_projection_hidden}"
            )
        if self.gram_max_tokens <= 0:
            raise ValueError(
                f"gram_max_tokens must be positive, got {self.gram_max_tokens}"
            )

    def build(
        self, register_dim: int, projection_dims: list[int]
    ) -> "RegisterDistillationHead":
        """Build the head.

        Args:
            register_dim: Width of the teacher register grid (the Perceiver's
                ``register_dim``, resolved by LatentMIMConfig).
            projection_dims: The student's Matryoshka prefix widths, descending.
        """
        self.validate()
        return RegisterDistillationHead(
            register_dim=register_dim,
            projection_dims=projection_dims,
            back_projection_hidden=self.back_projection_hidden,
            gram_max_tokens=self.gram_max_tokens,
        )


class RegisterDistillationHead(nn.Module):
    """Per-prefix back-projections plus the loss that distils teacher into student.

    Each Matryoshka prefix width ``d`` of the student gets its own back-projection
    ``d -> register_dim`` (keyed by ``str(d)`` in ``back_projections``, since
    ``nn.ModuleDict`` keys must be strings). Calling the head returns the distillation
    loss and its per-prefix metrics, mirroring the supervision head: the module owns
    both the trainable parameters and the objective they serve. Training-only: never
    run in the model's forward and discarded at inference.
    """

    def __init__(
        self,
        register_dim: int,
        projection_dims: list[int],
        back_projection_hidden: int | None = None,
        gram_max_tokens: int = 2048,
    ) -> None:
        """Initialize the head."""
        super().__init__()
        self.projection_dims = list(projection_dims)
        self.gram_max_tokens = gram_max_tokens

        def head(prefix_dim: int) -> nn.Module:
            if back_projection_hidden is None:
                return nn.Linear(prefix_dim, register_dim)
            return nn.Sequential(
                nn.Linear(prefix_dim, back_projection_hidden),
                nn.LayerNorm(back_projection_hidden),
                nn.ReLU(),
                nn.Linear(back_projection_hidden, register_dim),
            )

        self.back_projections = nn.ModuleDict(
            {str(d): head(d) for d in self.projection_dims}
        )

    def forward(
        self, registers: Tensor, projected_registers: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Distill the (detached) teacher register grid into the low-dim student.

        For every prefix ``d`` the first ``d`` dims of the student are distilled onto
        the full teacher through their own back-projection (``1 - cos``) and their own
        relational Gram term (MSE between the prefix's and the teacher's token-token
        cosine-similarity matrices), so every listed prefix is trained to be
        self-sufficient. All terms are summed unweighted.

        Args:
            registers: Teacher register grid ``[B, N, D]``; detached here, so the loss
                never reaches the encoder.
            projected_registers: Student grid ``[B, N, max_d]`` (its input was already
                detached inside the encoder, so gradients flow into the projection and
                these heads only).

        Returns:
            loss: Sum of the cosine and Gram terms across prefixes.
            metrics: Detached per-term, per-prefix values for logging.
        """
        teacher = registers.detach().float()
        student = projected_registers.float()
        metrics: dict[str, Tensor] = {}
        total = torch.zeros([], device=student.device, dtype=student.dtype)
        idx: Tensor | None = None
        flat_teacher = F.normalize(teacher.reshape(-1, teacher.shape[-1]), dim=-1)
        num_tokens = flat_teacher.shape[0]
        if num_tokens > self.gram_max_tokens:
            idx = torch.randperm(num_tokens, device=flat_teacher.device)[
                : self.gram_max_tokens
            ]
            flat_teacher = flat_teacher[idx]
        teacher_gram = flat_teacher @ flat_teacher.T
        for dim_str, back_projection in self.back_projections.items():
            prefix = student[..., : int(dim_str)]
            back = back_projection(prefix)
            cosine = (1.0 - F.cosine_similarity(back, teacher, dim=-1)).mean()
            total = total + cosine
            metrics[f"projection/distill_cosine_d{dim_str}"] = cosine.detach()
            flat_prefix = F.normalize(prefix.reshape(-1, prefix.shape[-1]), dim=-1)
            if idx is not None:
                flat_prefix = flat_prefix[idx]
            gram = F.mse_loss(flat_prefix @ flat_prefix.T, teacher_gram)
            total = total + gram
            metrics[f"projection/distill_gram_d{dim_str}"] = gram.detach()
        return total, metrics
