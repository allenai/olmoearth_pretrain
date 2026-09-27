# ERA5 reconstruction and pooled InfoNCE

Objective B can combine reconstruction with symmetric instance InfoNCE over two
independently masked views of each sample. Both views use the configured SWT
masking policy and reconstruct the same clean ERA5 sequence, with their own loss
masks. Their reconstruction losses are averaged:

```text
B = (reconstruction_a + reconstruction_b) / 2 + contrastive_lambda * InfoNCE
training contribution = recon_weight * B
```

InfoNCE normalizes embeddings and contrasts matching sample indices against the
other samples in the opposite view, in both directions. Both views receive
gradients. Its negative set is local to each rank's microbatch: gradient
accumulation and additional GPUs do not increase it. Each microbatch must contain
at least two samples when InfoNCE is enabled.

## Configuration

These overrides are added to an existing reconstruction launch with
`common.enable_reconstruction=True` and `common.encoder_swt_input=True`.

| `common` field | Default | Meaning |
|---|---|---|
| `recon_contrastive_lambda` | `0.0` | Weight of InfoNCE inside B; zero disables it |
| `recon_contrastive_temperature` | `0.1` | Positive cosine-similarity temperature |
| `recon_contrastive_use_projector` | `True` | Project pooled embeddings before InfoNCE |
| `recon_contrastive_projector_hidden_dim` | `384` | Projector hidden width |
| `recon_contrastive_projector_output_dim` | `128` | Projector output width |
| `recon_num_views` | `None` | Automatic: one view at zero lambda, two otherwise; accepts explicit 1 or 2 |

Positive lambda requires two views. An explicit `recon_num_views=2` with lambda
zero enables a reconstruction-only control with the same number of forward
passes. The model's `reconstruction_objective` exposes the same fields without
the `recon_` prefix.

The projector reuses `olmoearth_pretrain.nn.attention.Mlp`:
`Linear(pooled_dim, hidden_dim) → GELU → Linear(hidden_dim, output_dim)`, with
biases and zero dropout. It is shared by the two views and registered under
`objectives.reconstruction.projector` for optimization and checkpoints. The
encoder's original pooled representation remains the output used by supervised
heads, downstream probes and deployment. With lambda zero, no projector is
created and the default path retains the existing single-view behavior and
state-dict keys. Resume training with the same projector configuration; adding a
projector changes the full training checkpoint's parameters and optimizer state.

## Masking the buffer and the window edges

By default, masks never touch the 83-day SWT buffer, and halo spans must fit
inside the target window. Two views then share the whole buffer, and days near
either edge are rarely masked. Two knobs change this; their defaults reproduce
earlier runs.

| `common` field | Default | Meaning |
|---|---|---|
| `recon_mask_buffer` | `False` | Masks may fall anywhere in `[0, 448)`. Losses still start after the buffer. |
| `recon_span_placement` | `"inside"` | `"pin"` draws starts over `[lo − L + 1, T − 1]` and shifts overhanging spans flush with the edge at full length, giving near-uniform coverage |

Both widen the maskable window, so the span count must be re-tuned to keep the
budget. The halo75 recipe (`num_spans=[4,10]`, `span_days=[30,120]`,
`num_variables=[9,14]`, about 75% band / 67% raw in the target window) becomes
`recon_span_num_spans=[5,11]` with `recon_mask_buffer=True recon_span_placement=pin`.

## Experiment overrides

| Configuration | Additional overrides |
|---|---|
| Existing single-view baseline | None |
| Two-view reconstruction control | `common.recon_num_views=2` |
| InfoNCE with projector | `common.recon_contrastive_lambda=0.1` |
| InfoNCE directly on pooled embeddings | `common.recon_contrastive_lambda=0.1 common.recon_contrastive_use_projector=False` |

The example lambda of 0.1 is a starting point, not a selected optimum. Existing
raw/SWT loss weights, pressure gating, masks, buffer handling and pooling retain
their configured behavior. Objective A still uses its own clean-input forward.

## Metrics

Under `train/reconstruction/`, `loss` is the full B contribution after
`recon_weight`. `recon_loss` reports reconstruction before that outer weight;
existing raw/SWT component and mask metrics are averaged across views.
`num_views` reports the resolved view count.

With InfoNCE enabled, `contrastive_loss` reports unweighted InfoNCE,
`contrastive_weighted_loss` includes lambda but not `recon_weight`,
`contrastive_accuracy` averages matching accuracy across both directions, and
`contrastive_batch_size` reports the actual local number of samples (one positive
and B−1 negatives per anchor). `pooled_std` and `projected_std` report mean
per-dimension batch standard deviations after L2 normalization, averaged across
views with population variance. Similarity and loss computations use FP32 even
under mixed precision. Contrastive metrics are omitted when lambda is zero.
