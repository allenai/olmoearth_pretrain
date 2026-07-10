"""Pixel-representation decoders for the dual-resolution encoder.

Two auxiliary heads that operate on the encoder's ONLINE-unit pixel representations
(exposed under ``output_dict["pixel_branch"]`` by
:class:`~olmoearth_pretrain.nn.dual_res_encoder.DualResEncoder`). Both add to the
existing v1.2 coarse-decoding losses rather than replacing them, and both back-propagate
into the main branch (the two branches cross-attend inside the encoder).

* :class:`PixelReconstructionDecoder` -- SSL. For every masked (``DECODER``) imagery
  pixel whose location has at least one visible pixel -- of ANY pixel modality, band
  set or timestep -- a learned per-``(modality, band set)`` query cross-attends to
  all ONLINE pixel representations at that location and predicts the raw (normalized)
  pixel value with MSE. Cross-modality context is essential:
  ``random_time_with_decode`` assigns whole band sets encode-only or decode-only
  roles per instance, so with single-band-set tokenization a modality never contains
  both ONLINE and DECODER units itself. Locations with no visible pixel are skipped.
* :class:`PixelMapProbe` -- auxiliary supervised. Pools the ONLINE pixel
  representations over timestep + modality at each location and applies a linear head
  per map modality: cross-entropy for one-hot categorical maps, multi-label binary
  cross-entropy for multi-band binary maps (e.g. OpenStreetMap / WorldCereal), and MSE
  for regression maps. The label maps live at 10 m -- the same grid as the pixel branch
  -- so prediction is 1:1 per location. MISSING label pixels are ignored.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.decorators import experimental
from olmoearth_pretrain.nn.dual_res_encoder import (
    CrossAttnBlock,
    PixelModalityState,
    chunked_batch_attn,
    get_pixel_branch_modalities,
    unit_grid_coords,
)
from olmoearth_pretrain.nn.encodings import get_1d_sincos_pos_encoding
from olmoearth_pretrain.nn.tokenization import TokenizationConfig

logger = logging.getLogger(__name__)


def _scatter_by_group(
    gid: Tensor, t_of: Tensor, num_groups: int, t_mult: int
) -> tuple[Tensor, Tensor, int, Tensor]:
    """Sort units into per-group padded rows ordered by timestep.

    Args:
        gid: ``[N]`` group id (0..num_groups-1) per unit.
        t_of: ``[N]`` timestep per unit.
        num_groups: number of groups.
        t_mult: value strictly greater than any timestep (for the composite sort key).

    Returns:
        ``(order, scatter_pos, max_len, valid)`` where ``order`` sorts units by
        (group, time), ``scatter_pos`` is the position in the padded
        ``[num_groups * max_len]`` buffer, and ``valid`` is ``[num_groups, max_len]``.
    """
    device = gid.device
    n = gid.numel()
    if n == 0 or num_groups == 0:
        return (
            gid.new_zeros(0),
            gid.new_zeros(0),
            0,
            torch.zeros(num_groups, 0, dtype=torch.bool, device=device),
        )
    order = torch.argsort(gid * t_mult + t_of)
    sgid = gid[order]
    counts = torch.bincount(sgid, minlength=num_groups)
    max_len = int(counts.max().item())
    group_start = counts.cumsum(0) - counts
    rank = torch.arange(n, device=device) - group_start[sgid]
    scatter_pos = sgid * max_len + rank
    valid = torch.zeros(num_groups * max_len, dtype=torch.bool, device=device)
    valid[scatter_pos] = True
    return order, scatter_pos, max_len, valid.view(num_groups, max_len)


@dataclass
class ReconGroupings:
    """Grouping to reconstruct DECODER pixels from ONLINE pixels at the same location.

    Groups are (instance, patch) cells that contain at least one ONLINE unit (pooling
    across band sets, timesteps AND modalities). Within a group the ONLINE units are
    the attention keys and the DECODER units are the queries, so a masked pixel only
    sees visible pixels at its own location. The ``P**2`` intra-patch offsets are
    handled as a batched dimension by the decoder, not by these groups.
    """

    num_groups: int
    # Keys (ONLINE units), aligned to the caller's key order (== encoder packing,
    # concatenated over modalities).
    key_order: Tensor
    key_scatter: Tensor
    key_valid: Tensor  # [num_groups, max_kt]
    max_kt: int
    # Queries: ``keep`` marks the input DECODER units whose location has >= 1 ONLINE
    # key; the query fields below index the KEPT units only.
    keep: Tensor  # [num_queries_in] bool
    query_order: Tensor
    query_scatter: Tensor
    query_valid: Tensor  # [num_groups, max_qt]
    max_qt: int

    @property
    def num_decode(self) -> int:
        """Number of DECODER units to reconstruct (kept queries)."""
        return int(self.query_order.numel())


def build_recon_groupings(
    key_loc: Tensor,
    key_t: Tensor,
    query_loc: Tensor,
    query_t: Tensor,
    t_mult: int,
    location_ratio: float = 1.0,
) -> ReconGroupings:
    """Build the ONLINE-key / DECODER-query grouping for pixel reconstruction.

    We reconstruct each masked (``DECODER``) pixel from the visible (``ONLINE``)
    pixels at the *same location*. ``random_time_with_decode`` assigns whole band sets
    an encode-only or decode-only role per instance, so a decode pixel's context comes
    from **other band sets / modalities / timesteps** at its patch: the caller passes
    flat per-unit location ids spanning ALL pixel modalities (they share the spatial
    grid), and we group by location id (``instance * num_patches + patch``).

    Args:
        key_loc: ``[K]`` location id of every ONLINE (key) unit, in the caller's
            (packed, modality-concatenated) key order.
        key_t: ``[K]`` timestep of each key unit (within-group ordering only).
        query_loc: ``[Q]`` location id of every DECODER (query) unit.
        query_t: ``[Q]`` timestep of each query unit.
        t_mult: Value strictly greater than any timestep (composite sort key).
        location_ratio: Fraction of eligible locations to reconstruct, sampled
            uniformly at random per call. The recon loss stays an unbiased (if
            noisier) estimate of the full loss while its cost -- which is dominated
            by per-location attention rows and padded buffers -- scales linearly.

    Returns:
        A :class:`ReconGroupings`.
    """
    # Groups = the distinct locations (sorted) that contain >= 1 ONLINE unit; only
    # these can provide reconstruction context. Optionally subsample them.
    group_locations = torch.unique(key_loc)
    if location_ratio < 1.0 and group_locations.numel() > 0:
        n_keep = max(1, int(group_locations.numel() * location_ratio))
        sel = torch.randperm(group_locations.numel(), device=group_locations.device)
        group_locations = group_locations[sel[:n_keep]].sort().values
    num_groups = int(group_locations.numel())

    # With subsampled locations, keys outside the kept groups are dropped too.
    if location_ratio < 1.0 and key_loc.numel() > 0 and num_groups > 0:
        pos_k = torch.searchsorted(group_locations, key_loc).clamp(max=num_groups - 1)
        key_member = group_locations[pos_k] == key_loc
        key_idx = torch.nonzero(key_member, as_tuple=False).squeeze(-1)
        key_loc = key_loc[key_idx]
        key_t = key_t[key_idx]
    else:
        key_idx = None

    # Drop DECODER units whose location has no ONLINE context (nothing to attend to).
    # searchsorted finds each query location's slot in the sorted group list; it is a
    # real group iff the value there matches.
    if query_loc.numel() > 0 and num_groups > 0:
        pos = torch.searchsorted(group_locations, query_loc).clamp(max=num_groups - 1)
        keep = group_locations[pos] == query_loc
        gid_q = torch.searchsorted(group_locations, query_loc[keep])
        t_q = query_t[keep]
    else:
        keep = torch.zeros_like(query_loc, dtype=torch.bool)
        gid_q = query_loc[:0]
        t_q = query_t[:0]

    # Group id (0..num_groups-1) for each ONLINE unit.
    gid_k = (
        torch.searchsorted(group_locations, key_loc) if num_groups > 0 else key_loc[:0]
    )

    # Scatter keys and queries into padded [num_groups, max_*] rows (ordered by time).
    key_order, key_scatter, max_kt, key_valid = _scatter_by_group(
        gid_k, key_t, num_groups, t_mult
    )
    if key_idx is not None:
        # Map back to the caller's (unfiltered) key indexing.
        key_order = key_idx[key_order]
    query_order, query_scatter, max_qt, query_valid = _scatter_by_group(
        gid_q, t_q, num_groups, t_mult
    )

    return ReconGroupings(
        num_groups=num_groups,
        key_order=key_order,
        key_scatter=key_scatter,
        key_valid=key_valid,
        max_kt=max_kt,
        keep=keep,
        query_order=query_order,
        query_scatter=query_scatter,
        query_valid=query_valid,
        max_qt=max_qt,
    )


@experimental("Pixel reconstruction decoder is experimental.")
class PixelReconstructionDecoder(nn.Module):
    """MAE-style per-pixel reconstruction of masked imagery from ONLINE pixels.

    Cross-modality: each masked (``DECODER``) unit's queries attend to the ONLINE
    pixel representations of **every** pixel modality at the same location (all pixel
    modalities share the spatial grid). This matters because
    ``random_time_with_decode`` assigns whole band sets an encode-only or decode-only
    role per instance -- with single-band-set tokenization a modality never contains
    both ONLINE and DECODER units, so the reconstruction context necessarily comes
    from the OTHER modalities (and, for mixed-masked single-band-set instances, other
    timesteps). Queries carry a learned per-``(modality, band set)`` embedding plus a
    timestep encoding so one shared attention stack can serve every target.
    """

    def __init__(
        self,
        supported_modality_names: list[str],
        pixel_embedding_size: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        depth: int = 2,
        location_ratio: float = 1.0,
        tokenization_config: TokenizationConfig | None = None,
    ) -> None:
        """Initialize the reconstruction decoder.

        Args:
            supported_modality_names: Modalities to reconstruct (spatial+multitemporal).
            pixel_embedding_size: Pixel embedding dimension (Dp).
            num_heads: Cross-attention heads.
            mlp_ratio: MLP ratio in the decoder blocks.
            depth: Number of cross-attention blocks.
            location_ratio: Fraction of eligible (instance, patch) locations to
                reconstruct per step, sampled uniformly at random -- an unbiased,
                proportionally cheaper estimate of the full loss (its cost is
                dominated by per-location attention rows).
            tokenization_config: Band-grouping config (shared with the encoder).
        """
        super().__init__()
        self.dp = pixel_embedding_size
        self.location_ratio = location_ratio
        self.tokenization_config = tokenization_config or TokenizationConfig()
        specs = [Modality.get(n) for n in supported_modality_names]
        self.modality_names = get_pixel_branch_modalities(specs)

        # Learned query embedding per (modality, band set): with mixed-modality
        # location groups the query must say WHICH band set it is reconstructing.
        self.mask_tokens = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.zeros(
                        self.tokenization_config.get_num_bandsets(modality),
                        pixel_embedding_size,
                    )
                )
                for modality in self.modality_names
            }
        )
        self.blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    q_dim=pixel_embedding_size,
                    kv_dim=pixel_embedding_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    # Affine-free: these norms run over (millions of) query-pixel
                    # rows, where the gamma/beta grad reduction dominates backward.
                    norm_affine=False,
                )
                for _ in range(depth)
            ]
        )
        # Per (modality, bandset) linear reconstruction head + band-index buffers.
        self.heads = nn.ModuleDict({})
        for modality in self.modality_names:
            bandset_indices = self.tokenization_config.get_bandset_indices(modality)
            self.heads[modality] = nn.ModuleList(
                [nn.Linear(pixel_embedding_size, len(b)) for b in bandset_indices]
            )
            for idx, band in enumerate(bandset_indices):
                self.register_buffer(
                    f"{modality}__{idx}_recon_bands",
                    torch.tensor(band, dtype=torch.long),
                    persistent=False,
                )
        for token in self.mask_tokens.values():
            nn.init.normal_(token, std=0.02)

    def forward(
        self,
        pixel_branch: dict[str, PixelModalityState],
        sample: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> Tensor:
        """Compute the masked-pixel reconstruction MSE (over all pixel modalities)."""
        total = self._keep_alive()
        states = {m: st for m, st in pixel_branch.items() if m in self.modality_names}
        if not states:
            return total
        return total + self._reconstruct(states, sample)

    def _keep_alive(self) -> Tensor:
        """Zero term touching every decoder param so grads always flow (FSDP)."""
        term = sum(0.0 * token.sum() for token in self.mask_tokens.values())
        for p in self.blocks.parameters():
            term = term + 0.0 * p.sum()
        for heads in self.heads.values():
            for head in heads:
                for p in head.parameters():
                    term = term + 0.0 * p.sum()
        return term

    def _reconstruct(
        self, states: dict[str, PixelModalityState], sample: MaskedOlmoEarthSample
    ) -> Tensor:
        """Reconstruct every modality's masked pixels from ALL ONLINE pixel reps.

        Keys are the concatenation of every modality's packed ONLINE pixel reps
        (matching the encoder's per-modality ``flat_idx`` order); queries are every
        modality's DECODER units. Both are grouped by (instance, patch) so a masked
        pixel sees exactly the visible pixels -- of any modality, band set or
        timestep -- at its own location and offset.
        """
        first = next(iter(states.values()))
        p1, p2 = first.patch_shape
        p2n = first.pixels_per_unit
        num_patches = first.grid[1] * first.grid[2]
        t_max = max(st.grid[3] for st in states.values())
        device = first.pixels.device

        # --- Keys: every modality's ONLINE units, in concatenation order. ---
        key_reps = torch.cat([st.pixels for st in states.values()], dim=0)
        key_loc, key_t = [], []
        for st in states.values():
            coords = unit_grid_coords(st.flat_idx, st.grid)
            key_loc.append(coords.instance * num_patches + coords.patch)
            key_t.append(coords.t)

        # --- Queries: every modality's DECODER units (per-unit masks recovered by
        #     sampling the top-left pixel of each patch, as in the encoder). ---
        q_loc, q_t, q_embed, q_meta = [], [], [], []
        for modality, st in states.items():
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            full_mask = getattr(sample, mask_name)  # [B, H, W, T, band_sets]
            decode = full_mask[:, 0::p1, 0::p2, ...] == MaskValue.DECODER.value
            decode_flat = torch.nonzero(decode.reshape(-1), as_tuple=False).squeeze(-1)
            coords = unit_grid_coords(decode_flat, st.grid)
            q_loc.append(coords.instance * num_patches + coords.patch)
            q_t.append(coords.t)
            temporal = get_1d_sincos_pos_encoding(
                torch.arange(st.grid[3], device=device, dtype=torch.float32), self.dp
            )  # [T, Dp]
            q_embed.append(
                self.mask_tokens[modality][coords.bandset] + temporal[coords.t]
            )
            q_meta.append((modality, st, decode_flat, coords.bandset))

        rg = build_recon_groupings(
            torch.cat(key_loc),
            torch.cat(key_t),
            torch.cat(q_loc),
            torch.cat(q_t),
            t_mult=t_max,
            location_ratio=self.location_ratio,
        )
        if rg.num_decode == 0 or rg.num_groups == 0:
            return key_reps.new_zeros(())  # nothing to reconstruct
        ng = rg.num_groups

        # --- Keys: scatter the ONLINE pixel reps into padded per-location rows
        #     [ng, max_kt, P**2, Dp], then fold the P**2 offsets into the batch so each
        #     physical pixel attends only to keys at its own offset. ---
        keys = key_reps.new_zeros(ng * rg.max_kt, p2n, self.dp)
        keys[rg.key_scatter] = key_reps[rg.key_order]
        keys = rearrange(
            keys.view(ng, rg.max_kt, p2n, self.dp), "ng kt p d -> (ng p) kt d"
        )
        key_mask = rg.key_valid.repeat_interleave(p2n, dim=0)  # [ng*P**2, max_kt]

        # --- Queries: the kept units' (modality, band set) embeddings + timestep
        #     encodings, scattered into padded per-location rows the same way. ---
        q_flat = torch.cat(q_embed, dim=0)[rg.keep]  # [num_dec, Dp]
        q_flat = q_flat[:, None, :].expand(-1, p2n, -1)  # [num_dec, P**2, Dp]
        queries = key_reps.new_zeros(ng * rg.max_qt, p2n, self.dp)
        # Match the packed-rep dtype: under autocast the mask tokens / sincos encoding
        # stay float32 while ``key_reps`` may be a lower-precision autocast dtype.
        queries[rg.query_scatter] = q_flat[rg.query_order].to(queries.dtype)
        queries = rearrange(
            queries.view(ng, rg.max_qt, p2n, self.dp), "ng qt p d -> (ng p) qt d"
        )

        # --- Cross-attend each query to the ONLINE keys at its location/offset. ---
        x = queries
        for blk in self.blocks:
            x = x + chunked_batch_attn(blk, x, keys, key_mask)

        # Gather the outputs back into kept-DECODER-unit order [num_dec, P**2, Dp].
        out = rearrange(x, "(ng p) qt d -> ng qt p d", ng=ng, p=p2n)
        out = out.reshape(ng * rg.max_qt, p2n, self.dp)
        preds = out.new_empty(rg.num_decode, p2n, self.dp)
        preds[rg.query_order] = out[rg.query_scatter]

        # --- Per-(modality, band set) heads and MSE, token-averaged overall. ---
        loss = preds.new_zeros(())
        count = 0
        offset = 0
        for (modality, st, decode_flat, bandsets), loc in zip(q_meta, q_loc):
            n_in = int(loc.numel())
            keep_m = rg.keep[offset : offset + n_in]
            # Positions of this modality's kept queries within ``preds``.
            pred_pos = int(rg.keep[:offset].sum()) + torch.cumsum(keep_m.long(), 0) - 1
            loss_m, count_m = self._recon_loss(
                modality,
                st,
                preds,
                pred_pos[keep_m],
                decode_flat[keep_m],
                bandsets[keep_m],
                sample,
            )
            loss = loss + loss_m
            count += count_m
            offset += n_in
        if count == 0:
            return preds.new_zeros(())
        return loss / count

    def _recon_loss(
        self,
        modality: str,
        st: PixelModalityState,
        preds: Tensor,
        pred_pos: Tensor,
        decode_flat: Tensor,
        bandsets: Tensor,
        sample: MaskedOlmoEarthSample,
    ) -> tuple[Tensor, int]:
        """Summed squared error (+ element count) for one modality's kept queries."""
        b, g1, g2, t, bs = st.grid
        p1, p2 = st.patch_shape
        data = getattr(sample, modality)  # [B, H, W, T, C] (normalized inputs)

        # Reshape raw pixels into per-(instance, patch, timestep) units, each with its
        # P**2 pixels: [B*G2*T, P**2, C]. The (p1 p2) offset order matches ``preds``.
        data_units = rearrange(
            data,
            "b (g1 p1) (g2 p2) t c -> (b g1 g2 t) (p1 p2) c",
            g1=g1,
            g2=g2,
            p1=p1,
            p2=p2,
        )
        # Row index into ``data_units`` for each kept DECODER unit.
        coords = unit_grid_coords(decode_flat, st.grid)
        unit_bt = (coords.instance * (g1 * g2) + coords.patch) * t + coords.t
        target_all = data_units[unit_bt]  # [num_dec_m, P**2, C]

        loss = preds.new_zeros(())
        count = 0
        for idx in range(self.tokenization_config.get_num_bandsets(modality)):
            sel = bandsets == idx  # queries belonging to band set idx
            if not bool(sel.any()):
                continue
            bands = getattr(self, f"{modality}__{idx}_recon_bands")
            pred = self.heads[modality][idx](preds[pred_pos[sel]])
            # Cast the (float32) normalized target to the prediction dtype, which may
            # be a lower-precision autocast dtype.
            target = target_all[sel][..., bands].to(pred.dtype)  # [n, P**2, num_bands]
            loss = loss + F.mse_loss(pred, target, reduction="sum")
            count += pred.numel()
        return loss, count


@experimental("Pixel map probe is experimental.")
class PixelMapProbe(nn.Module):
    """Linear probe predicting map labels from pooled (time+modality) pixel reps."""

    def __init__(
        self,
        pixel_embedding_size: int,
        map_targets: dict[str, str],
        num_classes: dict[str, int] | None = None,
    ) -> None:
        """Initialize the map probe.

        Args:
            pixel_embedding_size: Pixel embedding dimension (Dp).
            map_targets: Mapping ``modality -> "ce" | "mse" | "bce"`` for each map
                label.

                * ``"ce"`` -- one-hot categorical target (one channel per class, e.g.
                  ``worldcover_onehot``); the per-location label is the argmax over the
                  class channels. One-hot modalities set ``skip_normalization`` so the
                  labels survive the input transform intact.
                * ``"mse"`` -- continuous regression target (e.g. ``srtm``).
                * ``"bce"`` -- multi-label binary target: a multi-band map where each
                  band is an independent 0/1 presence label (e.g.
                  ``openstreetmap_raster``, ``worldcereal``). Each band gets an
                  independent binary cross-entropy loss; the maps normalize to the
                  ``[0, 1]`` range and are binarized at 0.5.
            num_classes: Optional override of the class count for a ``"ce"`` modality.
                Defaults to the modality's band count (one band per class for a one-hot
                modality).
        """
        super().__init__()
        self.dp = pixel_embedding_size
        self.map_targets = map_targets
        num_classes = num_classes or {}
        self.heads = nn.ModuleDict({})
        for modality, kind in map_targets.items():
            if kind == "ce":
                # CE targets must be one-hot categorical modalities: one band per class
                # (e.g. worldcover_onehot), so the band count is the class count. A raw
                # single-band label map is normalized to floats upstream and cannot be
                # used as a CE target.
                if Modality.get(modality).num_bands <= 1:
                    raise ValueError(
                        f"CE map target {modality!r} must be a one-hot categorical "
                        "modality with >1 band (e.g. worldcover_onehot); got a "
                        "single-band map."
                    )
                out = num_classes.get(modality, Modality.get(modality).num_bands)
            elif kind in ("mse", "bce"):
                # One output per band: continuous regression (mse) or independent
                # per-band binary presence (bce, multi-label).
                out = Modality.get(modality).num_bands
            else:
                raise ValueError(f"Unknown map target kind {kind!r} for {modality}")
            self.heads[modality] = nn.Linear(pixel_embedding_size, out)

    def forward(
        self,
        pixel_branch: dict[str, PixelModalityState],
        sample: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> Tensor:
        """Compute the map-prediction loss (summed over map modalities)."""
        pooled, loc_valid, shape = self._pool_online_by_location(pixel_branch)
        # Zero term touching every head param (weights AND biases) so gradients always
        # flow for maps absent from this microbatch, keeping all ranks in sync (FSDP).
        total = 0.0 * sum(p.sum() for h in self.heads.values() for p in h.parameters())
        if pooled is None:
            return total
        assert loc_valid is not None and shape is not None
        for modality, kind in self.map_targets.items():
            if getattr(sample, modality, None) is None:
                continue
            total = total + self._map_loss(
                modality, kind, pooled, loc_valid, shape, sample
            )
        return total

    def _pool_online_by_location(
        self, pixel_branch: dict[str, PixelModalityState]
    ) -> tuple[Tensor | None, Tensor | None, tuple | None]:
        """Mean-pool ONLINE pixel reps over (timestep, band set, modality) per location.

        Every ONLINE unit contributes its P**2 pixels to the pixel *locations*
        ``(instance, patch, offset)`` they occupy. We scatter-add every rep into a shared
        per-location accumulator (shared across modalities) and divide by the count;
        locations with no ONLINE pixel are marked invalid.

        Returns ``(pooled [B*G2*P**2, Dp], valid [B*G2*P**2], (B, G1, G2, P1, P2))`` or
        ``(None, None, None)`` if no modality has ONLINE pixels.
        """
        pooled_sum: Tensor | None = None
        counts: Tensor | None = None
        shape: tuple | None = None
        for st in pixel_branch.values():
            if st.num_online == 0:
                continue
            b, g1, g2, t, bs = st.grid
            p1, p2 = st.patch_shape
            p2n = st.pixels_per_unit
            reps = st.pixels  # [num_online, P**2, Dp]

            # Location id of each ONLINE unit's pixels: flatten (b, patch, offset) to
            # ``b * (G2 * P**2) + patch * P**2 + offset``. ``base`` is offset 0 of each
            # unit; adding arange(P**2) enumerates the unit's P**2 pixel locations.
            coords = unit_grid_coords(st.flat_idx, st.grid)
            base = (coords.instance * (g1 * g2) + coords.patch) * p2n  # [num_online]
            offset = torch.arange(p2n, device=reps.device)
            loc = (base[:, None] + offset[None, :]).reshape(-1)  # [num_online * P**2]

            n_loc = b * g1 * g2 * p2n
            if pooled_sum is None:
                pooled_sum = reps.new_zeros(n_loc, self.dp)
                counts = reps.new_zeros(n_loc)
                shape = (b, g1, g2, p1, p2)
            assert pooled_sum is not None and counts is not None
            # Accumulate every pixel rep + a count into its location bucket.
            pooled_sum.index_add_(0, loc, reps.reshape(-1, self.dp))
            counts.index_add_(0, loc, reps.new_ones(reps.shape[0] * p2n))
        if pooled_sum is None:
            return None, None, None
        assert counts is not None
        valid = counts > 0
        pooled = pooled_sum / counts.clamp(min=1).unsqueeze(-1)
        return pooled, valid, shape

    def _map_loss(
        self,
        modality: str,
        kind: str,
        pooled: Tensor,
        loc_valid: Tensor,
        shape: tuple,
        sample: MaskedOlmoEarthSample,
    ) -> Tensor:
        b, g1, g2, p1, p2 = shape
        data = getattr(sample, modality)  # [B, H, W, (T), C]
        if data.dim() == 5:  # collapse a (usually singleton) time axis for static maps
            data = data.mean(dim=3) if kind == "mse" else data[:, :, :, 0]
        # [B, H, W, C] -> per-location [B*G2*P**2, C].
        target = rearrange(
            data,
            "b (g1 p1) (g2 p2) c -> (b g1 g2 p1 p2) c",
            g1=g1,
            g2=g2,
            p1=p1,
            p2=p2,
        )
        missing = (target == MISSING_VALUE).any(dim=-1)
        keep = loc_valid & ~missing
        if not bool(keep.any()):
            return pooled.new_zeros(())
        pred = self.heads[modality](pooled[keep])
        tgt = target[keep]
        if kind == "ce":
            # One-hot categorical target (e.g. worldcover_onehot): one channel per
            # class, so the per-location label is the argmax over the class channels.
            return F.cross_entropy(pred, tgt.argmax(dim=-1))
        if kind == "bce":
            # Multi-label binary target (e.g. openstreetmap_raster, worldcereal): each
            # band is an independent 0/1 presence label. The maps normalize to ~[0, 1],
            # so binarize at 0.5 and apply per-band binary cross-entropy. The target is
            # cast to the prediction dtype (which may be a lower-precision autocast dtype).
            return F.binary_cross_entropy_with_logits(pred, (tgt > 0.5).to(pred.dtype))
        return F.mse_loss(pred, tgt.to(pred.dtype))
