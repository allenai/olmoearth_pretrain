"""Pixel-representation decoders for the dual-resolution encoder.

Two auxiliary heads that operate on the encoder's ONLINE-unit pixel representations
(exposed under ``output_dict["pixel_branch"]`` by
:class:`~olmoearth_pretrain.nn.dual_res_encoder.DualResEncoder`). Both add to the
existing v1.2 coarse-decoding losses rather than replacing them, and both back-propagate
into the main branch (the two branches cross-attend inside the encoder).

* :class:`PixelReconstructionDecoder` -- SSL. For every masked (``DECODER``) imagery
  pixel that has at least one visible timestep at the same location, a learned query
  cross-attends to the ONLINE pixel representations at that location (other timesteps)
  and predicts the raw (normalized) pixel value with MSE. Locations with no visible
  timestep are skipped.
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
    across band sets and timesteps). Within a group the ONLINE units are the attention
    keys and the DECODER units are the queries, so a masked pixel only sees visible
    pixels at its own location. The ``P**2`` intra-patch offsets are handled as a
    batched dimension by the decoder, not by these groups.
    """

    num_groups: int
    # Keys (ONLINE units), aligned to nonzero(online_mask) order (== encoder packing).
    key_order: Tensor
    key_scatter: Tensor
    key_valid: Tensor  # [num_groups, max_kt]
    max_kt: int
    # Queries (DECODER units in groups that have >=1 ONLINE unit).
    decode_flat: Tensor  # [num_dec] indices into [B * U]
    decode_t: Tensor  # [num_dec] timestep of each decode unit
    decode_bs: Tensor  # [num_dec] bandset of each decode unit
    query_order: Tensor
    query_scatter: Tensor
    query_valid: Tensor  # [num_groups, max_qt]
    max_qt: int

    @property
    def num_decode(self) -> int:
        """Number of DECODER units to reconstruct."""
        return int(self.decode_flat.numel())


def build_recon_groupings(online_mask: Tensor, decode_mask: Tensor) -> ReconGroupings:
    """Build the ONLINE-key / DECODER-query grouping for pixel reconstruction.

    We reconstruct each masked (``DECODER``) pixel from the visible (``ONLINE``) pixels
    at the *same location*. Because ``random_time_with_decode`` makes whole band sets
    encode- or decode-only, a decode pixel's context comes from *other band sets /
    timesteps* at its patch -- so we group by **(instance, patch)** (pooling across band
    sets and timesteps). Within a group, ONLINE units are the keys, DECODER units the
    queries.

    A "unit" is one ``(instance, patch, timestep, band set)`` cell, referenced by its
    flat index into ``[B * U]`` (``U = G*G*T*band_sets``, row-major ``(g1,g2,t,bs)``) --
    the same canonical order the encoder used to pack its ONLINE pixel reps (see
    :func:`~olmoearth_pretrain.nn.dual_res_encoder.unit_grid_coords`).

    Args:
        online_mask: ``[B, G, G, T, band_sets]`` bool (ONLINE_ENCODER units).
        decode_mask: ``[B, G, G, T, band_sets]`` bool (DECODER units).

    Returns:
        A :class:`ReconGroupings`.
    """
    grid = online_mask.shape
    b, g1, g2, t, bs = grid
    num_patches = g1 * g2

    # Flat indices of the ONLINE (key) and DECODER (query) units. The location key
    # ``instance * num_patches + patch`` drops band set + timestep so every unit at one
    # patch shares a group.
    online_flat = torch.nonzero(online_mask.reshape(-1), as_tuple=False).squeeze(-1)
    decode_flat = torch.nonzero(decode_mask.reshape(-1), as_tuple=False).squeeze(-1)
    on = unit_grid_coords(online_flat, grid)
    de = unit_grid_coords(decode_flat, grid)
    key_loc, t_on = on.instance * num_patches + on.patch, on.t
    dec_loc, t_de, bs_de = de.instance * num_patches + de.patch, de.t, de.bandset

    # Groups = the distinct locations (sorted) that contain >= 1 ONLINE unit; only these
    # can provide reconstruction context.
    group_locations = torch.unique(key_loc)
    num_groups = int(group_locations.numel())

    # Drop DECODER units whose location has no ONLINE context (nothing to attend to).
    # searchsorted finds each decode location's slot in the sorted group list; it is a
    # real group iff the value there matches.
    if decode_flat.numel() > 0 and num_groups > 0:
        pos = torch.searchsorted(group_locations, dec_loc).clamp(max=num_groups - 1)
        keep = group_locations[pos] == dec_loc
        decode_flat = decode_flat[keep]
        t_de = t_de[keep]
        bs_de = bs_de[keep]
        gid_de = torch.searchsorted(group_locations, dec_loc[keep])
    else:
        decode_flat = decode_flat[:0]
        t_de = t_de[:0]
        bs_de = bs_de[:0]
        gid_de = decode_flat[:0]

    # Group id (0..num_groups-1) for each ONLINE unit.
    gid_on = (
        torch.searchsorted(group_locations, key_loc) if num_groups > 0 else key_loc[:0]
    )

    # Scatter keys and queries into padded [num_groups, max_*] rows (ordered by time).
    key_order, key_scatter, max_kt, key_valid = _scatter_by_group(
        gid_on, t_on, num_groups, t
    )
    query_order, query_scatter, max_qt, query_valid = _scatter_by_group(
        gid_de, t_de, num_groups, t
    )

    return ReconGroupings(
        num_groups=num_groups,
        key_order=key_order,
        key_scatter=key_scatter,
        key_valid=key_valid,
        max_kt=max_kt,
        decode_flat=decode_flat,
        decode_t=t_de,
        decode_bs=bs_de,
        query_order=query_order,
        query_scatter=query_scatter,
        query_valid=query_valid,
        max_qt=max_qt,
    )


@experimental("Pixel reconstruction decoder is experimental.")
class PixelReconstructionDecoder(nn.Module):
    """MAE-style per-pixel reconstruction of masked imagery from ONLINE pixels."""

    def __init__(
        self,
        supported_modality_names: list[str],
        pixel_embedding_size: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        depth: int = 2,
        tokenization_config: TokenizationConfig | None = None,
    ) -> None:
        """Initialize the reconstruction decoder.

        Args:
            supported_modality_names: Modalities to reconstruct (spatial+multitemporal).
            pixel_embedding_size: Pixel embedding dimension (Dp).
            num_heads: Cross-attention heads.
            mlp_ratio: MLP ratio in the decoder blocks.
            depth: Number of cross-attention blocks.
            tokenization_config: Band-grouping config (shared with the encoder).
        """
        super().__init__()
        self.dp = pixel_embedding_size
        self.tokenization_config = tokenization_config or TokenizationConfig()
        specs = [Modality.get(n) for n in supported_modality_names]
        self.modality_names = get_pixel_branch_modalities(specs)

        self.mask_token = nn.Parameter(torch.zeros(pixel_embedding_size))
        self.blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    q_dim=pixel_embedding_size,
                    kv_dim=pixel_embedding_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
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
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        pixel_branch: dict[str, PixelModalityState],
        sample: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> Tensor:
        """Compute the masked-pixel reconstruction MSE (summed over modalities)."""
        total = self._keep_alive()
        for modality, st in pixel_branch.items():
            if modality not in self.modality_names:
                continue
            total = total + self._reconstruct_modality(modality, st, sample)
        return total

    def _keep_alive(self) -> Tensor:
        """Zero term touching every decoder param so grads always flow (FSDP)."""
        term = 0.0 * self.mask_token.sum()
        for p in self.blocks.parameters():
            term = term + 0.0 * p.sum()
        for heads in self.heads.values():
            for head in heads:
                for p in head.parameters():
                    term = term + 0.0 * p.sum()
        return term

    def _reconstruct_modality(
        self, modality: str, st: PixelModalityState, sample: MaskedOlmoEarthSample
    ) -> Tensor:
        """Reconstruct one modality's masked pixels from its ONLINE pixel reps."""
        online_reps = st.pixels  # [num_online, P**2, Dp] (encoder output)
        b, g1, g2, t, bs = st.grid
        p1, p2 = st.patch_shape
        p2n = st.pixels_per_unit  # pixels per unit (P**2)

        # Recover the coarse-grid (per-unit) ONLINE/DECODER masks by sampling the
        # top-left pixel of each patch (all pixels in a unit share one mask value). This
        # matches the encoder's ONLINE-unit ordering.
        mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
        full_mask = getattr(sample, mask_name)  # [B, H, W, T, band_sets]
        online = full_mask[:, 0::p1, 0::p2, ...] == MaskValue.ONLINE_ENCODER.value
        decode = full_mask[:, 0::p1, 0::p2, ...] == MaskValue.DECODER.value
        rg = build_recon_groupings(online, decode)
        if rg.num_decode == 0 or rg.num_groups == 0:
            return online_reps.new_zeros(())  # nothing to reconstruct

        ng = rg.num_groups

        # --- Keys: scatter the ONLINE pixel reps into padded per-location rows
        #     [ng, max_kt, P**2, Dp], then fold the P**2 offsets into the batch so each
        #     physical pixel attends only to keys at its own offset. ---
        keys = online_reps.new_zeros(ng * rg.max_kt, p2n, self.dp)
        keys[rg.key_scatter] = online_reps[rg.key_order]
        keys = rearrange(
            keys.view(ng, rg.max_kt, p2n, self.dp), "ng kt p d -> (ng p) kt d"
        )
        key_mask = rg.key_valid.repeat_interleave(p2n, dim=0)  # [ng*P**2, max_kt]

        # --- Queries: one learned mask token per masked pixel, plus its timestep
        #     encoding, scattered into padded per-location rows the same way. ---
        temporal = get_1d_sincos_pos_encoding(
            torch.arange(t, device=online_reps.device, dtype=torch.float32), self.dp
        )  # [T, Dp]
        q_flat = self.mask_token[None, :] + temporal[rg.decode_t]  # [num_dec, Dp]
        q_flat = q_flat[:, None, :].expand(-1, p2n, -1)  # [num_dec, P**2, Dp]
        queries = online_reps.new_zeros(ng * rg.max_qt, p2n, self.dp)
        # Match the packed-rep dtype: under autocast the mask token / sincos encoding
        # stay float32 while ``online_reps`` may be a lower-precision autocast dtype.
        queries[rg.query_scatter] = q_flat[rg.query_order].to(queries.dtype)
        queries = rearrange(
            queries.view(ng, rg.max_qt, p2n, self.dp), "ng qt p d -> (ng p) qt d"
        )

        # --- Cross-attend each query to the ONLINE keys at its location/offset. ---
        x = queries
        for blk in self.blocks:
            x = x + chunked_batch_attn(blk, x, keys, key_mask)

        # Gather the outputs back into DECODER-unit order [num_dec, P**2, Dp].
        out = rearrange(x, "(ng p) qt d -> ng qt p d", ng=ng, p=p2n)
        out = out.reshape(ng * rg.max_qt, p2n, self.dp)
        preds = out.new_empty(rg.num_decode, p2n, self.dp)
        preds[rg.query_order] = out[rg.query_scatter]

        return self._recon_loss(modality, preds, rg, sample, st)

    def _recon_loss(
        self,
        modality: str,
        preds: Tensor,
        rg: ReconGroupings,
        sample: MaskedOlmoEarthSample,
        st: PixelModalityState,
    ) -> Tensor:
        """Token-averaged MSE between predicted and true (normalized) pixels."""
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
        # Row index into ``data_units`` for each DECODER unit (its instance/patch/time).
        coords = unit_grid_coords(rg.decode_flat, st.grid)
        unit_bt = (coords.instance * (g1 * g2) + coords.patch) * t + coords.t
        target_all = data_units[unit_bt]  # [num_dec, P**2, C]

        # Each band set has its own linear head predicting that band set's channels.
        # Sum squared error over all queries, then divide by the element count.
        loss = preds.new_zeros(())
        count = 0
        for idx in range(self.tokenization_config.get_num_bandsets(modality)):
            sel = rg.decode_bs == idx  # queries belonging to band set idx
            if not bool(sel.any()):
                continue
            bands = getattr(self, f"{modality}__{idx}_recon_bands")
            pred = self.heads[modality][idx](preds[sel])  # [n, P**2, num_bands]
            # Cast the (float32) normalized target to the prediction dtype, which may be
            # a lower-precision autocast dtype.
            target = target_all[sel][..., bands].to(pred.dtype)  # [n, P**2, num_bands]
            loss = loss + F.mse_loss(pred, target, reduction="sum")
            count += pred.numel()
        if count == 0:
            return preds.new_zeros(())
        return loss / count


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
