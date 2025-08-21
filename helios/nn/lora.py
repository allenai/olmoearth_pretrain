"""Task-conditioned LoRA layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLoRALinear(nn.Module):
    r"""Task-conditioned LoRA that adds a per-task low-rank delta to a precomputed base output.

    This layer generates a low-rank update :math:`\Delta W` from a task embedding
    and adds the corresponding output delta to an already-computed base output.

    Behavior:
        - If ``task_emb`` is ``None``, returns ``base_out`` unchanged.
        - Otherwise, computes ``\Delta W = A(task_emb) @ B(task_emb)`` and returns
          ``base_out + (alpha / rank) * (x_drop @ (\Delta W)^T)``.

    Shapes:
        - ``x``: ``(..., in_features)``
        - ``base_out``: ``(..., out_features)`` (same leading dims as ``x``)
        - ``task_emb``: ``(B, task_dim)`` broadcast across tokens/samples
        - return: ``(..., out_features)``

    Notes:
        The final layers of the generator MLPs are zero-initialized so that
        :math:`\Delta W = 0` at initialization, preserving ``base_out`` exactly.

    Args:
        in_features: Size of the last dimension of ``x``.
        out_features: Size of the last dimension of ``base_out`` and the output.
        task_dim: Dimensionality of the task embedding vector.
        rank: LoRA rank ``r``; must be > 0.
        alpha: Scaling factor; effective scale is ``alpha / rank``.
        dropout: Dropout probability applied to ``x`` along the LoRA path.
        gen_hidden: Hidden width for the generator MLPs.
        gen_layers: Number of layers in the generator MLPs. ``0`` means a single
            linear layer (no hidden). ``1`` means one hidden -> output.
        gen_activation: Name of activation for generator hidden layers
            (``"gelu"``, ``"relu"``, ``"silu"``) or ``None`` for identity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        task_dim: int,
        *,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.1,
        gen_hidden: int = 64,
        gen_layers: int = 2,
        gen_activation: str | None = "gelu",
    ) -> None:
        """Initialize the TaskLoRALinear module.

        Args:
            in_features: The dimension of the input features.
            out_features: The dimension of the output features.
            task_dim: The dimension of the task embedding.
            rank: The rank of the LoRA matrix.
            alpha: The scaling factor for the LoRA matrix.
            dropout: The dropout probability for the LoRA matrix.
            gen_hidden: The hidden width for the generator MLPs.
            gen_layers: The number of layers in the generator MLPs.
            gen_activation: The activation function for the generator MLPs.
        """
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.in_features = in_features
        self.out_features = out_features
        self.task_dim = task_dim
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank

        # Resolve activation (fallback to Identity)
        acts = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        self.activation_cls = (
            nn.Identity
            if gen_activation is None
            else acts.get(gen_activation.lower(), nn.Identity)
        )

        # Generators that output vec(A) and vec(B)
        self.gen_a = self._make_mlp(out_features * rank, gen_hidden, gen_layers)
        self.gen_b = self._make_mlp(rank * in_features, gen_hidden, gen_layers)

    def _make_mlp(self, out_size: int, hidden: int, layers: int) -> nn.Sequential:
        """Construct a small MLP to output a flattened matrix.

        Args:
            out_size: Output dimensionality (e.g., ``out_features * rank`` or
                ``rank * in_features``).
            hidden: Hidden width for the MLP (if any).
            layers: Number of layers (>= 0). ``0`` means no hidden.

        Returns:
            A :class:`torch.nn.Sequential` generator.
        """
        modules: list[nn.Module] = []
        in_size = self.task_dim

        # (layers - 1) hidden blocks, then the output layer
        for _ in range(max(0, layers - 1)):
            modules.append(nn.Linear(in_size, hidden, bias=True))
            modules.append(self.activation_cls())
            modules.append(nn.Dropout(self.dropout))
            in_size = hidden

        modules.append(nn.Linear(in_size, out_size, bias=True))
        mlp = nn.Sequential(*modules)
        return mlp

    @staticmethod
    def _broadcast_task_emb(
        task_emb: torch.Tensor, b_eff: int, task_dim: int
    ) -> torch.Tensor:
        """Broadcast or trim ``task_emb`` to match the effective batch size.

        Args:
            task_emb: Task embeddings of shape ``(B, task_dim)``.
            b_eff: Effective batch size after flattening the leading dims of ``x``.
            task_dim: Expected embedding dimension (for validation).

        Returns:
            A tensor of shape ``(b_eff, task_dim)`` suitable for per-sample generation.
        """
        if task_emb.dim() != 2 or task_emb.size(-1) != task_dim:
            raise ValueError(
                f"task_emb must be (B, {task_dim}), got {tuple(task_emb.shape)}"
            )
        reps = (b_eff + task_emb.size(0) - 1) // task_emb.size(0)  # ceil division
        return task_emb.repeat(reps, 1)[:b_eff]

    def forward(
        self,
        x: torch.Tensor,
        base_out: torch.Tensor,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add a task-conditioned LoRA delta to a precomputed base output.

        If ``task_emb`` is ``None``, this function returns ``base_out`` unchanged.

        Args:
            x: Input tensor of shape ``(..., in_features)`` used by the LoRA path.
            base_out: Precomputed base output of shape ``(..., out_features)``.
            task_emb: Task embeddings of shape ``(B, task_dim)`` or ``None``.
                These embeddings are broadcast across tokens/samples in the batch.

        Returns:
            Tensor of shape ``(..., out_features)`` equal to:
            ``base_out`` if ``task_emb is None``, else
            ``base_out + scaling * (x_drop @ (Delta W)^T)``.
        """
        if task_emb is None:
            return base_out

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dim of x to be {self.in_features}, got {x.shape[-1]}"
            )
        if (
            base_out.shape[:-1] != x.shape[:-1]
            or base_out.shape[-1] != self.out_features
        ):
            raise ValueError(
                "base_out must match x's leading dimensions and have last dim "
                f"{self.out_features}; got x {tuple(x.shape)}, base_out {tuple(base_out.shape)}"
            )

        # Flatten leading dims to an effective batch
        x_shape = x.shape
        x2d = x.reshape(-1, self.in_features)  # (B_eff, in)
        b_eff = x2d.size(0)

        # Broadcast task embedding across the effective batch
        task_eff = self._broadcast_task_emb(
            task_emb, b_eff, self.task_dim
        )  # (B_eff, task_dim)

        # Generate LoRA factors per sample
        rnk = self.rank
        vec_a = self.gen_a(task_eff)  # (B_eff, out*rnk)
        vec_b = self.gen_b(task_eff)  # (B_eff, rnk*in)
        a = vec_a.view(b_eff, self.out_features, rnk)  # (B_eff, out, rnk)
        b = vec_b.view(b_eff, rnk, self.in_features)  # (B_eff, rnk, in)

        # LoRA delta path: y_delta = x_drop @ ΔW^T per sample
        # Don't materialize the full delta matrix since it's too large
        x2d_drop = F.dropout(x2d, p=self.dropout, training=self.training)
        tmp = torch.bmm(x2d_drop.unsqueeze(1), b.transpose(1, 2)).squeeze(
            1
        )  # (B_eff, rnk)
        y_delta = torch.bmm(tmp.unsqueeze(1), a.transpose(1, 2)).squeeze(
            1
        )  # (B_eff, out)
        print(
            "DELTA NORM",
            y_delta.norm(),
            "BASE NORM",
            base_out.norm(),
        )

        y = base_out.reshape(-1, self.out_features) + self.scaling * y_delta
        return y.view(*x_shape[:-1], self.out_features)
