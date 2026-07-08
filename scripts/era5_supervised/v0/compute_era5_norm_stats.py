"""Compute ERA5 normalization stats for the SWT-input encoder — two phases.

The packaged ``computed.json`` currently holds placeholder stats for
``era5l_day_10`` (every band ``mean=0, std=1``), so the pipeline does not
actually per-band-normalize the ERA5 input.  This script computes the stats we
need to fix that and to standardize the SWT band channels, in two phases:

* ``--phase raw`` (bird 1): read the *physical* ERA5 values (identity
  normalizer, bypassing the placeholder ``computed.json``) and compute the
  per-band physical ``mean``/``std`` (14 values) — these are what you write into
  ``computed.json`` so the pipeline actually normalizes the raw input (and the
  reconstruction target).  As a byproduct it also logs the per-(var, level) SWT
  stats on the *unnormalized* signal (98 channels), for reference.

* ``--phase swt`` (bird 2): compute the per-(var, level) SWT ``mean``/``std``
  (98 channels) on the *normalized* ERA5 — these are what the encoder uses to
  standardize its SWT band channels (replacing BatchNorm).  There are two ways
  to supply the normalization so the SWT stats match the input the encoder sees
  at train time:

  * ``--raw-stats <phase-raw JSON>`` (recommended when normalization lives in
    the rslearn ``config.yaml`` ``Normalize`` transforms): read the *physical*
    ERA5 (identity normalizer) and apply the phase-``raw`` ``raw_mean``/
    ``raw_std`` in-script, independent of ``computed.json``.
  * omit ``--raw-stats``: defer to the dataset's pipeline normalizer
    (``Strategy.COMPUTED`` / ``computed.json``).  This only reflects real
    normalization if ``computed.json`` was updated with the phase-``raw`` stats
    first; otherwise the placeholder ``mean=0/std=1`` makes it a no-op and the
    result equals the unnormalized SWT reference.

No-data handling: ERA5-Land fills ocean/coast/missing cells with ``-9999``.
All stats ignore these. Raw per-band stats exclude no-data entries per variable
(``str`` matched exactly at ``-9999.0``, others within ``--nodata-tol``). SWT
stats drop an entire ``(window, variable)`` whenever that variable has ANY
no-data timestep, since the wavelet transform smears a single fill across
neighboring coefficients.

Layout: var-major ``c = v*n_bands + s`` (matching
``swt_bands_to_channels``); scale order ``[d0..d{L-1}, approx]``.

Run from the repo root (so ``import direct_registry`` resolves)::

    # Phase 1 (physical raw stats + unnormalized SWT reference):
    python3 scripts/era5_supervised/v0/compute_era5_norm_stats.py --phase raw \
        --task era5enc_pretrain_ssl --max-samples 4000 \
        --output scripts/era5_supervised/v0/era5_raw_norm_stats.json

    # Phase 2 (SWT stats on the raw-normalized input), normalizing in-script
    # with the phase-'raw' stats (mirrors the rslearn Normalize transform):
    python3 scripts/era5_supervised/v0/compute_era5_norm_stats.py --phase swt \
        --task era5enc_pretrain_ssl --max-samples 4000 \
        --raw-stats scripts/era5_supervised/v0/configs/era5_raw_norm_stats.json \
        --output scripts/era5_supervised/v0/configs/swt_input_stats.json

Smoke test on the tiny 32-window subset (``--task era5enc_pretrain_ssl_test``).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

import torch
from direct_registry import DirectRslearnRegistry
from torch.utils.data import DataLoader

from olmoearth_pretrain.data.constants import ERA5_INPUT_SEQUENCE_LENGTH, Modality
from olmoearth_pretrain.data.multi_task_era5_dataset import (
    Era5TaskSpec,
    _build_task_dataset,
    _collate_samples,
)
from olmoearth_pretrain.nn.transforms.era5_swt import (
    StationaryWaveletTransform1d,
    swt_bands_to_channels,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_era5_norm_stats")


class _IdentityNormalizer:
    """No-op normalizer: return the physical ERA5 values unchanged.

    Used by ``--phase raw`` to bypass the (placeholder) ``computed.json`` and
    read true physical units, so we can compute real per-band stats.
    """

    def normalize(self, modality: Any, data: Any) -> Any:  # noqa: D102
        return data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["raw", "swt"],
        required=True,
        help="'raw': physical per-band stats (+ unnormalized SWT ref). "
        "'swt': SWT stats on the pipeline-normalized input.",
    )
    p.add_argument("--task", default="era5enc_pretrain_ssl")
    p.add_argument("--registry-path", default=None)
    p.add_argument("--max-samples", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    # SWT config: MUST match the encoder's swt_input_* config.
    p.add_argument("--wavelet", default="haar")
    p.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--no-approx", action="store_true")
    p.add_argument("--swt-buffer-days", type=int, default=83)
    # No-data handling: ERA5-Land fills ocean/coast/missing cells with -9999.
    p.add_argument("--nodata-value", type=float, default=-9999.0)
    p.add_argument(
        "--nodata-tol",
        type=float,
        default=0.5,
        help="Match |value - nodata| <= tol as no-data (range match).",
    )
    p.add_argument(
        "--exact-match-vars",
        nargs="*",
        default=["str"],
        help="Variables matched by EXACT equality to nodata (tol=0), since "
        "their real values can legitimately sit near -9999 (e.g. 'str'). Pass "
        "with no names to disable exact matching.",
    )
    p.add_argument(
        "--raw-stats",
        default=None,
        help="(phase swt only) Path to a phase-'raw' JSON. When set, the "
        "physical ERA5 input is normalized with its 'raw_mean'/'raw_std' "
        "(bypassing computed.json) before the SWT is applied, so the SWT stats "
        "match the raw-normalized signal the encoder actually sees. When "
        "omitted, the dataset's pipeline normalizer (computed.json) is used.",
    )
    p.add_argument("--output", required=True, help="Output JSON path.")
    return p.parse_args()


def _collate(samples: list) -> Any:
    """Module-level collate (picklable for DataLoader workers)."""
    return _collate_samples(samples) if samples else None


def build_loader(args: argparse.Namespace, identity_norm: bool) -> DataLoader:
    """Resolve the SSL task and build a DataLoader.

    When ``identity_norm`` is True the dataset's normalizer is replaced with a
    no-op so physical values are read (phase ``raw``); otherwise the real
    pipeline normalizer (``Strategy.COMPUTED``) is used (phase ``swt``).
    """
    registry = DirectRslearnRegistry.load(args.registry_path)
    entry = registry.get(args.task)
    if not entry.ssl:
        raise ValueError(f"Task {args.task!r} is not an SSL entry (ssl=False).")
    spec = Era5TaskSpec(
        name=entry.name,
        weight=1.0,
        modality_layer_name=entry.modality_layer_name,
        weka_path=entry.weka_path,
        model_yaml_path=entry.model_yaml_path,
        groups_override=entry.groups or None,
        tags_override=entry.tags or None,
        norm_stats_from_pretrained=entry.norm_stats_from_pretrained,
        split="train",
        max_samples=args.max_samples,
        ssl=True,
    )
    dataset = _build_task_dataset(
        spec, registry=None, max_sequence_length=ERA5_INPUT_SEQUENCE_LENGTH
    )
    if identity_norm:
        dataset.normalizer = _IdentityNormalizer()  # type: ignore[assignment]
    logger.info(
        "Built SSL dataset %r with %d windows (identity_norm=%s)",
        entry.name,
        len(dataset),
        identity_norm,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=False,
    )


def band_labels(
    band_order: list[str], levels: list[int], include_approx: bool
) -> list[str]:
    """Return ``V * n_bands`` var-major channel labels."""
    scale_labels = [f"d{lv}" for lv in levels]
    if include_approx:
        scale_labels.append("approx")
    return [f"{var}_{s}" for var in band_order for s in scale_labels]


def _finalize(sum_v: torch.Tensor, sumsq_v: torch.Tensor, count: torch.Tensor):
    """Return ``(mean, std)`` lists from float64 accumulators.

    ``count`` is a per-element tensor (per variable or per channel) so that
    each entry is normalized only over its own valid (non-no-data) samples.
    """
    denom = count.clamp(min=1.0)
    mean = sum_v / denom
    std = ((sumsq_v / denom) - mean * mean).clamp(min=0.0).sqrt()
    return mean.tolist(), std.tolist()


def build_tol_vec(
    band_order: list[str],
    tol: float,
    exact_match_vars: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Per-variable no-data match tolerance ``[V]``.

    Range ``tol`` for most variables; ``0.0`` (exact ``== nodata``) for the
    variables listed in ``exact_match_vars``.
    """
    tol_vec = torch.full((len(band_order),), float(tol), dtype=torch.float32)
    unknown = sorted(set(exact_match_vars) - set(band_order))
    if unknown:
        logger.warning("Ignoring unknown --exact-match-vars: %s", unknown)
    for name in exact_match_vars:
        if name in band_order:
            tol_vec[band_order.index(name)] = 0.0
    return tol_vec.to(device)


def nodata_mask(
    era5: torch.Tensor, nodata: float, tol_vec: torch.Tensor
) -> torch.Tensor:
    """Boolean ``[B, T, V]`` no-data mask on the *physical* signal.

    ``tol_vec`` ``[V]`` broadcasts over ``[B, T, V]``; an entry is no-data when
    ``|value - nodata| <= tol_vec[v]``.
    """
    return (era5 - nodata).abs() <= tol_vec


def load_raw_norm(
    path: str, band_order: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-band ``(mean, std)`` from a phase-'raw' stats JSON.

    Validates that the JSON's ``band_order`` matches the current modality
    order so the 14 raw values line up with the ``[B, T, V]`` tensor.
    """
    with open(path) as f:
        payload = json.load(f)
    if "raw_mean" not in payload or "raw_std" not in payload:
        raise ValueError(
            f"{path!r} has no 'raw_mean'/'raw_std'; expected a phase-'raw' JSON."
        )
    file_order = payload.get("band_order")
    if file_order is not None and list(file_order) != list(band_order):
        raise ValueError(
            f"band_order mismatch: raw-stats {file_order} != modality {band_order}"
        )
    mean = torch.tensor(payload["raw_mean"], dtype=torch.float64)
    std = torch.tensor(payload["raw_std"], dtype=torch.float64)
    if mean.numel() != len(band_order) or std.numel() != len(band_order):
        raise ValueError(
            f"raw-stats length {mean.numel()}/{std.numel()} != "
            f"num_variables {len(band_order)}"
        )
    return mean, std


def run_raw_phase(args: argparse.Namespace) -> None:
    """Phase 1: physical per-band stats + unnormalized SWT reference stats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band_order = list(Modality.ERA5L_DAY_10.band_order)
    v = len(band_order)
    include_approx = not args.no_approx
    n_bands = len(args.levels) + (1 if include_approx else 0)
    c = v * n_bands
    swt = StationaryWaveletTransform1d(
        num_channels=v, max_levels=max(args.levels) + 1, wavelet=args.wavelet
    ).to(device)

    loader = build_loader(args, identity_norm=True)
    tol_vec = build_tol_vec(band_order, args.nodata_tol, args.exact_match_vars, device)
    # Channel -> variable map for the var-major SWT layout (c = v*n_bands + s).
    var_of_c = torch.arange(c, device=device) // n_bands

    raw_sum = torch.zeros(v, dtype=torch.float64)
    raw_sumsq = torch.zeros(v, dtype=torch.float64)
    raw_count = torch.zeros(v, dtype=torch.float64)  # per-variable valid count
    swt_sum = torch.zeros(c, dtype=torch.float64)
    swt_sumsq = torch.zeros(c, dtype=torch.float64)
    swt_count = torch.zeros(c, dtype=torch.float64)  # per-channel valid count
    n_windows = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch is None:
                continue
            era5 = batch.era5.to(device)  # physical [B, T, V]
            nd = nodata_mask(era5, args.nodata_value, tol_vec)  # [B, T, V]

            # Raw per-band stats over ALL valid timesteps (no-data excluded).
            rf = era5.reshape(-1, v).double()
            valid_f = (~nd).reshape(-1, v)
            rf = rf * valid_f  # zero out no-data so they don't add to the sum
            raw_sum += rf.sum(dim=0).cpu()
            raw_sumsq += (rf * rf).sum(dim=0).cpu()
            raw_count += valid_f.double().sum(dim=0).cpu()

            # SWT-on-physical stats over the target window (boundary excluded).
            # A single -9999 smears across neighboring SWT coefficients, so we
            # drop an entire (window, variable) whenever that variable has ANY
            # no-data in the window, rather than masking post-SWT per timestep.
            valid_bv = ~nd.any(dim=1)  # [B, V] True if var fully valid
            bands = swt(era5.transpose(1, 2), levels=args.levels, target_start=0)
            x = swt_bands_to_channels(bands, include_approx=include_approx)
            x = x[:, args.swt_buffer_days :, :]  # [B, T', C]
            valid_bc = valid_bv[:, var_of_c]  # [B, C]
            valid_btc = valid_bc.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B,T',C]
            bf = x.reshape(-1, c).double()
            vf = valid_btc.reshape(-1, c)
            bf = bf * vf
            swt_sum += bf.sum(dim=0).cpu()
            swt_sumsq += (bf * bf).sum(dim=0).cpu()
            swt_count += vf.double().sum(dim=0).cpu()
            n_windows += era5.shape[0]
            if (bi + 1) % 20 == 0:
                logger.info("  processed %d batches / %d windows", bi + 1, n_windows)

    if n_windows == 0:
        raise RuntimeError("No samples processed; dataset appears empty.")

    raw_mean, raw_std = _finalize(raw_sum, raw_sumsq, raw_count)
    swt_mean, swt_std = _finalize(swt_sum, swt_sumsq, swt_count)
    labels = band_labels(band_order, args.levels, include_approx)

    logger.info("=== Raw per-band physical stats (%d bands) ===", v)
    for i, name in enumerate(band_order):
        logger.info("  %-8s mean=%+.6g std=%.6g", name, raw_mean[i], raw_std[i])
    logger.info("=== SWT stats on UNNORMALIZED signal (%d channels) ===", c)
    for i, name in enumerate(labels):
        logger.info("  %-16s mean=%+.6g std=%.6g", name, swt_mean[i], swt_std[i])

    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": "raw",
        "task": args.task,
        "n_windows": n_windows,
        "wavelet": args.wavelet,
        "levels": list(args.levels),
        "include_approx": include_approx,
        "n_bands_per_var": n_bands,
        "num_variables": v,
        "num_channels": c,
        "swt_buffer_days": args.swt_buffer_days,
        "band_order": band_order,
        "channel_labels": labels,
        "layout": "var_major: c = v*n_bands + s",
        "nodata_value": args.nodata_value,
        "nodata_tol": args.nodata_tol,
        "exact_match_vars": [b for b in args.exact_match_vars if b in band_order],
        "raw_valid_count": [int(x) for x in raw_count.tolist()],
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "swt_unnorm_mean": swt_mean,
        "swt_unnorm_std": swt_std,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote raw-phase stats -> %s", args.output)


def run_swt_phase(args: argparse.Namespace) -> None:
    """Phase 2: SWT stats on the pipeline-normalized input (Strategy.COMPUTED)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band_order = list(Modality.ERA5L_DAY_10.band_order)
    v = len(band_order)
    include_approx = not args.no_approx
    n_bands = len(args.levels) + (1 if include_approx else 0)
    c = v * n_bands
    swt = StationaryWaveletTransform1d(
        num_channels=v, max_levels=max(args.levels) + 1, wavelet=args.wavelet
    ).to(device)

    # Two normalization sources for the "normalized input":
    #   * --raw-stats set  -> read PHYSICAL values (identity normalizer) and
    #     apply the phase-'raw' mean/std ourselves. This matches the rslearn
    #     `Normalize` transform in the config.yaml files (which normalize the
    #     physical raw signal), independent of computed.json.
    #   * --raw-stats unset -> defer to the dataset's pipeline normalizer
    #     (Strategy.COMPUTED / computed.json).
    apply_raw_norm = args.raw_stats is not None
    if apply_raw_norm:
        raw_mean, raw_std = load_raw_norm(args.raw_stats, band_order)
        raw_mean_t = raw_mean.to(device=device, dtype=torch.float32)
        raw_std_t = raw_std.to(device=device, dtype=torch.float32).clamp(min=1e-12)
        logger.info("Applying raw normalization from %s", args.raw_stats)
        loader = build_loader(args, identity_norm=True)
    else:
        logger.info("Using dataset pipeline normalizer (computed.json)")
        loader = build_loader(args, identity_norm=False)

    tol_vec = build_tol_vec(band_order, args.nodata_tol, args.exact_match_vars, device)
    var_of_c = torch.arange(c, device=device) // n_bands

    swt_sum = torch.zeros(c, dtype=torch.float64)
    swt_sumsq = torch.zeros(c, dtype=torch.float64)
    swt_count = torch.zeros(c, dtype=torch.float64)  # per-channel valid count
    n_windows = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch is None:
                continue
            era5 = batch.era5.to(device)  # [B, T, V] (physical if apply_raw_norm)
            # Detect no-data on the PHYSICAL signal, before normalization turns
            # -9999 into some other value.
            nd = nodata_mask(era5, args.nodata_value, tol_vec)  # [B, T, V]
            valid_bv = ~nd.any(dim=1)  # [B, V] True if var fully valid
            if apply_raw_norm:
                era5 = (era5 - raw_mean_t) / raw_std_t
            bands = swt(era5.transpose(1, 2), levels=args.levels, target_start=0)
            x = swt_bands_to_channels(bands, include_approx=include_approx)
            x = x[:, args.swt_buffer_days :, :]  # [B, T', C]
            valid_bc = valid_bv[:, var_of_c]  # [B, C]
            valid_btc = valid_bc.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B,T',C]
            bf = x.reshape(-1, c).double()
            vf = valid_btc.reshape(-1, c)
            bf = bf * vf
            swt_sum += bf.sum(dim=0).cpu()
            swt_sumsq += (bf * bf).sum(dim=0).cpu()
            swt_count += vf.double().sum(dim=0).cpu()
            n_windows += era5.shape[0]
            if (bi + 1) % 20 == 0:
                logger.info("  processed %d batches / %d windows", bi + 1, n_windows)

    if n_windows == 0:
        raise RuntimeError("No samples processed; dataset appears empty.")

    swt_mean, swt_std = _finalize(swt_sum, swt_sumsq, swt_count)
    labels = band_labels(band_order, args.levels, include_approx)
    n_small = sum(1 for s in swt_std if s < 1e-6)
    if n_small:
        logger.warning(
            "%d channel(s) have std < 1e-6; encoder normalization should clamp.",
            n_small,
        )

    norm_src = f"raw-stats ({args.raw_stats})" if apply_raw_norm else "computed.json"
    logger.info(
        "=== SWT stats on NORMALIZED input via %s (%d channels) ===", norm_src, c
    )
    for i, name in enumerate(labels):
        logger.info("  %-16s mean=%+.6g std=%.6g", name, swt_mean[i], swt_std[i])

    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": "swt",
        "task": args.task,
        "n_windows": n_windows,
        "wavelet": args.wavelet,
        "levels": list(args.levels),
        "include_approx": include_approx,
        "n_bands_per_var": n_bands,
        "num_variables": v,
        "num_channels": c,
        "swt_buffer_days": args.swt_buffer_days,
        "band_order": band_order,
        "channel_labels": labels,
        "layout": "var_major: c = v*n_bands + s",
        "nodata_value": args.nodata_value,
        "nodata_tol": args.nodata_tol,
        "exact_match_vars": [b for b in args.exact_match_vars if b in band_order],
        "swt_valid_count": [int(x) for x in swt_count.tolist()],
        "raw_normalized": apply_raw_norm,
        "raw_stats_path": args.raw_stats,
        "mean": swt_mean,
        "std": swt_std,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote swt-phase stats -> %s", args.output)


def main() -> None:
    """Dispatch to the requested phase."""
    args = parse_args()
    if args.phase == "raw":
        run_raw_phase(args)
    else:
        run_swt_phase(args)


if __name__ == "__main__":
    main()
