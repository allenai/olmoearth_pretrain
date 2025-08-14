"""Task-conditioned LoRA layer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLoRALinear(nn.Module):
    r"""Drop-in Linear with task-conditioned LoRA.

    This layer augments a standard linear projection with a per-task, low-rank
    weight update :math:`\\Delta W` that is generated from a task embedding.

    At initialization the LoRA generators' final layers are zero-initialized,
    ensuring :math:`\\Delta W = 0` and preserving the behavior of a plain
    :class:`torch.nn.Linear`. During training, the generators learn to map
    task embeddings to useful low-rank updates.

    Behavior:
        - If ``task_emb`` is ``None``, behaves exactly like :class:`torch.nn.Linear`.
        - Else, computes ``ΔW = A(task_emb) @ B(task_emb)`` and returns
          ``x @ W^T + (alpha/r) * x @ (ΔW)^T + b``.

    Shapes:
        - Input: ``x`` has shape ``(..., in_features)`` (e.g., ``(B, N, C_in)``).
        - Output: same leading dims as input with last dim ``out_features``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        task_dim: int,
        bias: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.1,
        lora_gen_hidden: int = 64,
        lora_gen_layers: int = 2,
        lora_gen_activation: str | None = "gelu",
    ) -> None:
        """Initialize the TaskLoRALinear module.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            task_dim: Dimensionality of the task embedding vector.
            bias: If ``True``, adds a learnable bias to the output.
            lora_rank: LoRA rank ``r``; must be > 0.
            lora_alpha: Scaling factor; effective scale is ``alpha / r``.
            lora_dropout: Dropout applied to input ``x`` before LoRA path.
            lora_gen_hidden: Hidden width used in generator MLP(s).
            lora_gen_layers: Number of hidden layers in generator MLP(s). ``0`` means
                a single linear layer (no hidden).
            lora_gen_activation: Activation for generator hidden layers
                (``"gelu"``, ``"relu"``, ``"silu"``) or ``None`` for identity.
        """
        super().__init__()

        # Base linear
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

        # LoRA config
        self.lora_rank = lora_rank
        self.gen_hidden = lora_gen_hidden
        self.task_dim = task_dim
        self.lora_alpha = lora_alpha
        self.lora_gen_layers = lora_gen_layers
        self.lora_gen_activation = lora_gen_activation
        self.lora_dropout = lora_dropout
        self.lora_scaling = self.lora_alpha / self.lora_rank

        # Resolve activation class (fallback to Identity if None or unrecognized)
        if lora_gen_activation is None:
            self.activation_cls = nn.Identity
        else:
            name = lora_gen_activation.lower()
            acts = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
            self.activation_cls = acts.get(name, nn.Identity)

        # Generators produce vec(A) and vec(B) per token
        self.lora_gen_a = self.make_mlp(self.out_features * self.lora_rank)
        self.lora_gen_b = self.make_mlp(self.lora_rank * self.in_features)

    def reset_parameters(self) -> None:
        """Initialize base linear parameters.

        Uses the same initialization scheme as :class:`torch.nn.Linear`:
        Kaiming-uniform for ``weight`` and a uniform bound for ``bias`` based on
        fan-in.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def make_mlp(self, out_size: int) -> nn.Sequential:
        r"""Construct a small MLP that outputs a flattened matrix.

        The final linear layer is zero-initialized so that at initialization
        the generated LoRA factors produce :math:`\\Delta W = 0`.

        Args:
            out_size: Output dimensionality, typically ``out_features * r`` or
                ``r * in_features`` to represent vectorized ``A`` or ``B``.

        Returns:
            A :class:`torch.nn.Sequential` implementing the generator.
        """
        layers: list[nn.Module] = []
        in_size = self.task_dim
        for _ in range(max(0, self.lora_gen_layers - 1)):
            layers.extend(
                [
                    nn.Linear(in_size, self.gen_hidden, bias=True),
                    nn.Dropout(self.lora_dropout),
                    self.activation_cls(),
                    nn.Dropout(self.lora_dropout),
                ]
            )
            in_size = self.gen_hidden
        layers.append(nn.Linear(in_size, out_size, bias=True))
        mlp = nn.Sequential(*layers)

        # Zero-init the last Linear so ΔW = 0 at init
        last = mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        # Mark so global model init can skip re-initializing this layer.
        last._skip_reinit = True
        return mlp

    def forward(
        self, x: torch.Tensor, task_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the linear projection with optional task-conditioned LoRA.

        If ``task_emb`` is ``None``, this is equivalent to
        ``torch.nn.functional.linear(x, weight, bias)``.

        Args:
            x: Input tensor of shape ``(..., in_features)`` (e.g., ``(B, N, C_in)``).
            task_emb: Task embedding of shape ``(B, task_dim)``. These embeddings are broadcasted
                across all patches (tokens) in the batch.

        Returns:
            Output tensor with shape ``(..., out_features)``.
        """
        # No task embedding -> plain linear
        if task_emb is None:
            return F.linear(x, self.weight, self.bias)

        # Flatten leading dims to an effective batch, keep last feature dim
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")
        x_shape = x.shape  # effective batch = B * N
        x2d = x.reshape(-1, self.in_features)  # (B_eff, C_in)
        b_eff = x2d.shape[0]

        # Broadcast task embedding across the effective batch
        task_emb = task_emb.repeat(b_eff // task_emb.shape[0], 1)  # (B_eff, task_dim)

        # Generate LoRA factors per sample
        r = self.lora_rank
        vec_a = self.lora_gen_a(task_emb)  # (B_eff, out*r)
        vec_b = self.lora_gen_b(task_emb)  # (B_eff, r*in)
        a = vec_a.view(b_eff, self.out_features, r)  # (B_eff, out, r)
        b = vec_b.view(b_eff, r, self.in_features)  # (B_eff, r, in)
        delta = torch.bmm(a, b)  # (B_eff, out, in)

        # Base + LoRA paths
        x2d_drop = F.dropout(x2d, p=self.lora_dropout)  # (B_eff, in)
        y_base = F.linear(x2d, self.weight, self.bias)  # (B_eff, out)

        # Compute y_delta = x2d_drop @ delta^T per sample
        y_delta = torch.einsum("bi,boi->bo", x2d_drop, delta)  # (B_eff, out)
        y = y_base + self.lora_scaling * y_delta

        return y.view(*x_shape[:-1], self.out_features)
