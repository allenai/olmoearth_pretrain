"""Sample scoring engine for dataset characterization.

Computes cheap-to-evaluate trait vectors for every sample in an H5 dataset.
Each scorer is a function registered via @register_scorer that takes a sample
dict (modality_name -> np.ndarray) plus metadata and returns a dict of
feature_name -> float. All scorers are pure numpy — no GPU needed.

Usage:
    from olmoearth_pretrain.data.sample_scorer import score_sample, SCORER_REGISTRY

    sample_dict, metadata = load_h5_sample(path)
    features = score_sample(sample_dict, metadata)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import entropy

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality

logger = logging.getLogger(__name__)

ScorerFn = Callable[[dict[str, np.ndarray], dict[str, Any]], dict[str, float]]
SCORER_REGISTRY: dict[str, ScorerFn] = {}

# ---------------------------------------------------------------------------
# Worldcover class mapping: raw values -> 0-indexed class IDs
# ESA WorldCover classes: 10=tree, 20=shrub, 30=grass, 40=crop,
# 50=built-up, 60=bare, 70=snow, 80=water, 90=wetland, 95=mangrove, 100=moss
# ---------------------------------------------------------------------------
WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}
WORLDCOVER_RAW_VALUES = sorted(WORLDCOVER_CLASSES.keys())

# Semantic groupings for modality type diversity
MODALITY_TYPE_MAP: dict[str, str] = {
    "sentinel2_l2a": "optical",
    "landsat": "optical",
    "naip": "optical",
    "naip_10": "optical",
    "ndvi": "optical",
    "sentinel1": "sar",
    "worldcover": "label_map",
    "cdl": "label_map",
    "worldcereal": "label_map",
    "eurocrops": "label_map",
    "srtm": "terrain",
    "wri_canopy_height_map": "terrain",
    "openstreetmap_raster": "infrastructure",
    "worldpop": "demographics",
    "era5_10": "weather",
    "gse": "embedding",
}

ALL_TRAINING_MODALITIES = [
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    "worldcover",
    "srtm",
    "openstreetmap_raster",
    "wri_canopy_height_map",
    "cdl",
    "worldcereal",
    "worldpop",
    "era5_10",
    "gse",
    "eurocrops",
    "naip",
    "naip_10",
]


def register_scorer(name: str) -> Callable[[ScorerFn], ScorerFn]:
    """Register a scorer function by name."""

    def decorator(fn: ScorerFn) -> ScorerFn:
        SCORER_REGISTRY[name] = fn
        return fn

    return decorator


def _valid_pixels(arr: np.ndarray) -> np.ndarray:
    """Return mask of pixels that are not MISSING_VALUE."""
    return arr != MISSING_VALUE


def _safe_mean(arr: np.ndarray) -> float:
    """Mean of array, returning 0.0 if empty."""
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def _safe_std(arr: np.ndarray) -> float:
    """Std of array, returning 0.0 if empty."""
    return float(np.std(arr)) if arr.size > 0 else 0.0


# ============================================================================
# Geographic Scorers
# ============================================================================


@register_scorer("geographic")
def score_geographic(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Location-derived features."""
    lat = meta.get("lat", 0.0)
    lon = meta.get("lon", 0.0)

    features: dict[str, float] = {
        "lat": float(lat),
        "lon": float(lon),
        "abs_lat": abs(float(lat)),
        "is_northern_hemisphere": float(lat >= 0),
        # Rough latitude bands for climate proxy
        "is_tropical": float(abs(lat) <= 23.5),
        "is_subtropical": float(23.5 < abs(lat) <= 35.0),
        "is_temperate": float(35.0 < abs(lat) <= 55.0),
        "is_boreal_polar": float(abs(lat) > 55.0),
    }
    return features


# ============================================================================
# Terrain Scorers (SRTM)
# ============================================================================


@register_scorer("terrain")
def score_terrain(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Elevation and terrain features from SRTM."""
    features: dict[str, float] = {}
    srtm = sample.get("srtm")
    if srtm is None:
        return {
            "has_srtm": 0.0,
            "elevation_mean": 0.0,
            "elevation_std": 0.0,
            "elevation_range": 0.0,
            "elevation_median": 0.0,
            "terrain_ruggedness": 0.0,
            "slope_mean": 0.0,
            "is_flat": 0.0,
            "is_mountainous": 0.0,
        }

    valid = _valid_pixels(srtm)
    elev = srtm[valid].astype(np.float64)

    if elev.size == 0:
        return {
            k: 0.0
            for k in [
                "has_srtm",
                "elevation_mean",
                "elevation_std",
                "elevation_range",
                "elevation_median",
                "terrain_ruggedness",
                "slope_mean",
                "is_flat",
                "is_mountainous",
            ]
        }

    features["has_srtm"] = 1.0
    features["elevation_mean"] = _safe_mean(elev)
    features["elevation_std"] = _safe_std(elev)
    features["elevation_range"] = float(np.ptp(elev))
    features["elevation_median"] = float(np.median(elev))

    # Terrain Ruggedness Index: mean absolute difference from neighbors
    # Use a simple 2D approach on the first band, first timestep
    elev_2d = srtm[..., 0, 0].astype(np.float64)  # [H, W]
    elev_2d_valid = np.where(_valid_pixels(srtm[..., 0, 0]), elev_2d, np.nan)
    if elev_2d.shape[0] > 2 and elev_2d.shape[1] > 2:
        dy = np.diff(elev_2d_valid, axis=0)
        dx = np.diff(elev_2d_valid, axis=1)
        slope_y = np.nanmean(np.abs(dy))
        slope_x = np.nanmean(np.abs(dx))
        features["terrain_ruggedness"] = float(np.nanmean([slope_x, slope_y]))
        features["slope_mean"] = float(
            np.nanmean(np.sqrt(dy[:, :-1] ** 2 + dx[:-1, :] ** 2))
        )
    else:
        features["terrain_ruggedness"] = 0.0
        features["slope_mean"] = 0.0

    features["is_flat"] = float(features["elevation_std"] < 10.0)
    features["is_mountainous"] = float(features["elevation_range"] > 500.0)

    return features


# ============================================================================
# Land Cover Scorers (WorldCover)
# ============================================================================


@register_scorer("land_cover")
def score_land_cover(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Land cover composition and diversity from ESA WorldCover."""
    features: dict[str, float] = {}
    wc = sample.get("worldcover")
    if wc is None:
        features["has_worldcover"] = 0.0
        for cls_name in WORLDCOVER_CLASSES.values():
            features[f"lc_frac_{cls_name}"] = 0.0
        features["lc_entropy"] = 0.0
        features["lc_num_classes"] = 0.0
        features["lc_dominant_class_id"] = 0.0
        features["lc_edge_density"] = 0.0
        return features

    wc_flat = wc[_valid_pixels(wc)].flatten()
    if wc_flat.size == 0:
        features["has_worldcover"] = 0.0
        for cls_name in WORLDCOVER_CLASSES.values():
            features[f"lc_frac_{cls_name}"] = 0.0
        features["lc_entropy"] = 0.0
        features["lc_num_classes"] = 0.0
        features["lc_dominant_class_id"] = 0.0
        features["lc_edge_density"] = 0.0
        return features

    features["has_worldcover"] = 1.0
    total = wc_flat.size

    # Per-class fractions
    fracs = []
    for raw_val, cls_name in WORLDCOVER_CLASSES.items():
        frac = float(np.sum(wc_flat == raw_val)) / total
        features[f"lc_frac_{cls_name}"] = frac
        fracs.append(frac)

    frac_arr = np.array(fracs)
    frac_arr = frac_arr[frac_arr > 0]

    features["lc_entropy"] = float(entropy(frac_arr)) if frac_arr.size > 0 else 0.0
    features["lc_num_classes"] = float(frac_arr.size)

    # Dominant class
    max_idx = int(
        np.argmax([features[f"lc_frac_{n}"] for n in WORLDCOVER_CLASSES.values()])
    )
    features["lc_dominant_class_id"] = float(WORLDCOVER_RAW_VALUES[max_idx])

    # Edge density: fraction of pixel transitions in the 2D grid
    wc_2d = wc[..., 0, 0]  # [H, W]
    if wc_2d.shape[0] > 1 and wc_2d.shape[1] > 1:
        horiz_edges = np.sum(wc_2d[:, 1:] != wc_2d[:, :-1])
        vert_edges = np.sum(wc_2d[1:, :] != wc_2d[:-1, :])
        total_adjacencies = (
            wc_2d.shape[0] * (wc_2d.shape[1] - 1)
            + (wc_2d.shape[0] - 1) * wc_2d.shape[1]
        )
        features["lc_edge_density"] = float(horiz_edges + vert_edges) / max(
            total_adjacencies, 1
        )
    else:
        features["lc_edge_density"] = 0.0

    return features


# ============================================================================
# Temporal Signal Scorers
# ============================================================================


def _temporal_features_for_modality(
    data: np.ndarray, name: str, is_spatial: bool
) -> dict[str, float]:
    """Compute temporal features for a single multitemporal modality.

    Args:
        data: array with shape [H, W, T, C] (spatial) or [T, C] (non-spatial)
        name: modality name prefix
        is_spatial: whether the modality has spatial dims
    """
    features: dict[str, float] = {}

    if is_spatial:
        # data shape: [H, W, T, C]
        T = data.shape[2]
        # Per-timestep mean across spatial dims and bands
        band_means = []
        valid_ts = 0
        for t in range(T):
            frame = data[:, :, t, :]
            valid = _valid_pixels(frame)
            if valid.any():
                band_means.append(float(np.mean(frame[valid])))
                valid_ts += 1
            else:
                band_means.append(np.nan)
    else:
        # data shape: [T, C]
        T = data.shape[0]
        band_means = []
        valid_ts = 0
        for t in range(T):
            frame = data[t, :]
            valid = _valid_pixels(frame)
            if valid.any():
                band_means.append(float(np.mean(frame[valid])))
                valid_ts += 1
            else:
                band_means.append(np.nan)

    features[f"{name}_valid_timesteps"] = float(valid_ts)

    ts = np.array(band_means)
    ts_clean = ts[~np.isnan(ts)]

    if ts_clean.size < 2:
        features[f"{name}_temporal_mean"] = _safe_mean(ts_clean)
        features[f"{name}_temporal_std"] = 0.0
        features[f"{name}_seasonal_amplitude"] = 0.0
        features[f"{name}_temporal_autocorr"] = 0.0
        features[f"{name}_abrupt_change_score"] = 0.0
        features[f"{name}_temporal_trend"] = 0.0
        return features

    features[f"{name}_temporal_mean"] = _safe_mean(ts_clean)
    features[f"{name}_temporal_std"] = _safe_std(ts_clean)
    features[f"{name}_seasonal_amplitude"] = float(np.ptp(ts_clean))

    # Lag-1 autocorrelation
    if ts_clean.size > 2:
        mean_c = ts_clean - ts_clean.mean()
        c0 = np.dot(mean_c, mean_c)
        c1 = np.dot(mean_c[:-1], mean_c[1:])
        features[f"{name}_temporal_autocorr"] = float(c1 / c0) if c0 > 1e-10 else 0.0
    else:
        features[f"{name}_temporal_autocorr"] = 0.0

    # Abrupt change: max absolute difference between consecutive valid timesteps
    diffs = np.abs(np.diff(ts_clean))
    features[f"{name}_abrupt_change_score"] = float(np.max(diffs))

    # Linear trend (slope of simple regression)
    x = np.arange(ts_clean.size, dtype=np.float64)
    if ts_clean.size > 1:
        slope = float(np.polyfit(x, ts_clean, 1)[0])
        features[f"{name}_temporal_trend"] = slope
    else:
        features[f"{name}_temporal_trend"] = 0.0

    return features


@register_scorer("temporal")
def score_temporal(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Temporal signal features across all multitemporal modalities."""
    features: dict[str, float] = {}

    # Timestamps metadata
    ts = sample.get("timestamps")
    if ts is not None:
        features["num_timestamps"] = float(ts.shape[0])
        if ts.shape[-1] >= 2:
            months = ts[:, 1]
            features["month_range"] = float(np.ptp(months))
            features["spans_full_year"] = float(np.ptp(months) >= 11)
    else:
        features["num_timestamps"] = 0.0
        features["month_range"] = 0.0
        features["spans_full_year"] = 0.0

    multitemporal = {
        "sentinel2_l2a": True,
        "sentinel1": True,
        "landsat": True,
        "era5_10": False,
    }

    for mod_name, is_spatial in multitemporal.items():
        data = sample.get(mod_name)
        if data is not None:
            features.update(_temporal_features_for_modality(data, mod_name, is_spatial))

    # Aggregate temporal richness across modalities
    valid_ts_keys = [k for k in features if k.endswith("_valid_timesteps")]
    if valid_ts_keys:
        features["total_valid_timesteps"] = sum(features[k] for k in valid_ts_keys)
        features["mean_valid_timesteps"] = features["total_valid_timesteps"] / len(
            valid_ts_keys
        )

    return features


# ============================================================================
# NDVI Scorers (computed from S2 L2A)
# ============================================================================


@register_scorer("ndvi")
def score_ndvi(sample: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, float]:
    """NDVI-derived vegetation features from Sentinel-2 L2A."""
    features: dict[str, float] = {}
    s2 = sample.get("sentinel2_l2a")
    if s2 is None:
        return {
            "ndvi_mean": 0.0,
            "ndvi_std": 0.0,
            "ndvi_range": 0.0,
            "ndvi_seasonal_amplitude": 0.0,
            "ndvi_greenness": 0.0,
            "has_vegetation": 0.0,
        }

    # S2 L2A band order: B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09
    # Red = B04 (index 2), NIR = B08 (index 3)
    band_order = Modality.SENTINEL2_L2A.band_order
    red_idx = band_order.index("B04")
    nir_idx = band_order.index("B08")

    # data is [H, W, T, C]
    T = s2.shape[2]
    ndvi_per_t = []
    for t in range(T):
        red = s2[:, :, t, red_idx].astype(np.float64)
        nir = s2[:, :, t, nir_idx].astype(np.float64)
        valid = _valid_pixels(red) & _valid_pixels(nir)
        if not valid.any():
            continue
        red_v, nir_v = red[valid], nir[valid]
        denom = nir_v + red_v
        safe_denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
        ndvi = (nir_v - red_v) / safe_denom
        ndvi = np.where(np.abs(denom) < 1e-10, 0.0, ndvi)
        ndvi_per_t.append(float(np.mean(ndvi)))

    if not ndvi_per_t:
        return {
            "ndvi_mean": 0.0,
            "ndvi_std": 0.0,
            "ndvi_range": 0.0,
            "ndvi_seasonal_amplitude": 0.0,
            "ndvi_greenness": 0.0,
            "has_vegetation": 0.0,
        }

    ndvi_arr = np.array(ndvi_per_t)
    features["ndvi_mean"] = _safe_mean(ndvi_arr)
    features["ndvi_std"] = _safe_std(ndvi_arr)
    features["ndvi_range"] = float(np.ptp(ndvi_arr))
    features["ndvi_seasonal_amplitude"] = float(np.max(ndvi_arr) - np.min(ndvi_arr))
    features["ndvi_greenness"] = float(np.max(ndvi_arr))
    features["has_vegetation"] = float(np.max(ndvi_arr) > 0.3)

    return features


# ============================================================================
# Modality Structure Scorers
# ============================================================================


@register_scorer("modality_structure")
def score_modality_structure(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Modality presence, completeness, and type diversity."""
    features: dict[str, float] = {}

    present = set()
    for mod in ALL_TRAINING_MODALITIES:
        is_present = mod in sample and sample[mod] is not None
        if is_present:
            valid = _valid_pixels(sample[mod])
            is_present = valid.any()
        features[f"has_{mod}"] = float(is_present)
        if is_present:
            present.add(mod)

    features["num_modalities"] = float(len(present))
    features["completeness_ratio"] = float(len(present)) / max(
        len(ALL_TRAINING_MODALITIES), 1
    )

    # Missingness bitfield for compact representation
    bitfield = 0
    for i, mod in enumerate(ALL_TRAINING_MODALITIES):
        if mod in present:
            bitfield |= 1 << i
    features["missingness_pattern"] = float(bitfield)

    # Type diversity: how many distinct modality *types* are present
    types_present = set()
    for mod in present:
        mod_type = MODALITY_TYPE_MAP.get(mod)
        if mod_type:
            types_present.add(mod_type)
    features["modality_type_diversity"] = float(len(types_present))

    return features


# ============================================================================
# Spectral / Content Diversity Scorers
# ============================================================================


@register_scorer("spectral_content")
def score_spectral_content(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Spectral and content diversity features from optical/SAR bands.

    Vectorized — avoids per-band/per-timestep python loops.
    """
    features: dict[str, float] = {}

    optical_modalities = {
        "sentinel2_l2a": (100, (0, 10000)),
        "sentinel1": (50, (-30, 10)),
        "landsat": (100, (0, 10000)),
    }

    for mod_name, (n_bins, hist_range) in optical_modalities.items():
        prefix = mod_name
        data = sample.get(mod_name)
        if data is None:
            features[f"{prefix}_entropy"] = 0.0
            features[f"{prefix}_brightness_mean"] = 0.0
            features[f"{prefix}_spectral_variance"] = 0.0
            features[f"{prefix}_spatial_variance"] = 0.0
            features[f"{prefix}_missing_frac"] = 1.0
            features[f"{prefix}_dynamic_range"] = 0.0
            continue

        valid = _valid_pixels(data)
        features[f"{prefix}_missing_frac"] = 1.0 - float(valid.mean())

        if not valid.any():
            for k in [
                "entropy",
                "brightness_mean",
                "spectral_variance",
                "spatial_variance",
                "dynamic_range",
            ]:
                features[f"{prefix}_{k}"] = 0.0
            continue

        all_valid = data[valid].astype(np.float32)

        hist = np.histogram(all_valid, bins=n_bins, range=hist_range)[0]
        features[f"{prefix}_entropy"] = float(entropy(hist + 1e-10))
        features[f"{prefix}_brightness_mean"] = float(np.mean(all_valid))

        p5, p95 = np.percentile(all_valid, [5, 95])
        features[f"{prefix}_dynamic_range"] = float(p95 - p5)

        # Spectral variance: variance across bands per pixel, vectorized
        if len(data.shape) == 4 and data.shape[-1] > 1:
            dat_f = data.astype(np.float32)
            dat_f[~valid] = np.nan
            features[f"{prefix}_spectral_variance"] = float(
                np.nanmean(np.nanvar(dat_f, axis=-1))
            )
        else:
            features[f"{prefix}_spectral_variance"] = 0.0

        # Spatial variance: variance across H,W for each (T,C) slice
        if len(data.shape) == 4 and data.shape[0] > 1 and data.shape[1] > 1:
            H, W, T, C = data.shape
            flat = data.reshape(H * W, T * C).astype(np.float32)
            flat[flat == MISSING_VALUE] = np.nan
            features[f"{prefix}_spatial_variance"] = float(
                np.nanmean(np.nanvar(flat, axis=0))
            )
        else:
            features[f"{prefix}_spatial_variance"] = 0.0

    return features


# ============================================================================
# Spectral Indices (cheap band ratios from S2)
# ============================================================================


@register_scorer("spectral_indices")
def score_spectral_indices(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Band-ratio indices beyond NDVI: water, built-up, moisture."""
    features: dict[str, float] = {}
    s2 = sample.get("sentinel2_l2a")
    if s2 is None:
        return {
            "ndwi_mean": 0.0,
            "ndbi_mean": 0.0,
            "bsi_mean": 0.0,
            "ndwi_std": 0.0,
            "ndbi_std": 0.0,
        }

    band_order = Modality.SENTINEL2_L2A.band_order
    green_idx = band_order.index("B03")
    nir_idx = band_order.index("B08")
    swir1_idx = band_order.index("B11")
    red_idx = band_order.index("B04")

    def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = a + b
        safe = np.where(np.abs(denom) < 1e-10, 1.0, denom)
        ratio = (a - b) / safe
        return np.where(np.abs(denom) < 1e-10, 0.0, ratio)

    # Aggregate across all timesteps at once
    # s2 shape: [H, W, T, C]
    green = s2[..., green_idx].astype(np.float32)
    nir = s2[..., nir_idx].astype(np.float32)
    swir1 = s2[..., swir1_idx].astype(np.float32)
    red = s2[..., red_idx].astype(np.float32)

    valid = _valid_pixels(s2[..., green_idx]) & _valid_pixels(s2[..., nir_idx])

    # NDWI: (Green - NIR) / (Green + NIR) — water bodies
    ndwi = _safe_ratio(green, nir)
    ndwi_v = ndwi[valid]
    features["ndwi_mean"] = float(np.mean(ndwi_v)) if ndwi_v.size > 0 else 0.0
    features["ndwi_std"] = float(np.std(ndwi_v)) if ndwi_v.size > 0 else 0.0

    valid_swir = valid & _valid_pixels(s2[..., swir1_idx])

    # NDBI: (SWIR1 - NIR) / (SWIR1 + NIR) — built-up areas
    ndbi = _safe_ratio(swir1, nir)
    ndbi_v = ndbi[valid_swir]
    features["ndbi_mean"] = float(np.mean(ndbi_v)) if ndbi_v.size > 0 else 0.0
    features["ndbi_std"] = float(np.std(ndbi_v)) if ndbi_v.size > 0 else 0.0

    # BSI: ((SWIR1 + Red) - (NIR + Green)) / ((SWIR1 + Red) + (NIR + Green)) — bare soil
    bsi_num = (swir1 + red) - (nir + green)
    bsi_den = (swir1 + red) + (nir + green)
    safe_den = np.where(np.abs(bsi_den) < 1e-10, 1.0, bsi_den)
    bsi = bsi_num / safe_den
    bsi_v = bsi[valid_swir]
    features["bsi_mean"] = float(np.mean(bsi_v)) if bsi_v.size > 0 else 0.0

    return features


# ============================================================================
# Spatial Autocorrelation
# ============================================================================


@register_scorer("spatial_autocorrelation")
def score_spatial_autocorrelation(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Simplified Moran's I: neighbor pixel correlation.

    High = smooth/homogeneous (water, forest). Low = patchy (urban, mixed ag).
    """
    s2 = sample.get("sentinel2_l2a")
    if s2 is None or s2.shape[0] < 3 or s2.shape[1] < 3:
        return {"spatial_autocorr": 0.0}

    # Use first valid timestep, mean of visible bands
    for t in range(s2.shape[2]):
        frame = s2[:, :, t, :3].astype(np.float32)
        if _valid_pixels(frame).all(axis=-1).mean() > 0.5:
            gray = frame.mean(axis=-1)
            break
    else:
        return {"spatial_autocorr": 0.0}

    mean = gray.mean()
    centered = gray - mean
    var = float(np.mean(centered**2))
    if var < 1e-10:
        return {"spatial_autocorr": 1.0}

    # Correlation with right and bottom neighbors
    corr_h = np.mean(centered[:, :-1] * centered[:, 1:])
    corr_v = np.mean(centered[:-1, :] * centered[1:, :])
    return {"spatial_autocorr": float((corr_h + corr_v) / (2.0 * var))}


# ============================================================================
# Distribution Shape
# ============================================================================


@register_scorer("distribution_shape")
def score_distribution_shape(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Shape of pixel value distribution: skewness, bimodality, percentiles."""
    features: dict[str, float] = {}
    s2 = sample.get("sentinel2_l2a")
    if s2 is None:
        return {
            "s2_skewness": 0.0,
            "s2_kurtosis": 0.0,
            "s2_p10": 0.0,
            "s2_p50": 0.0,
            "s2_p90": 0.0,
            "s2_bimodality": 0.0,
        }

    # Use visible bands (B02, B03, B04) — subsample for speed
    visible = s2[..., :3]
    valid = _valid_pixels(visible)
    vals = visible[valid].astype(np.float32)

    if vals.size < 10:
        return {
            "s2_skewness": 0.0,
            "s2_kurtosis": 0.0,
            "s2_p10": 0.0,
            "s2_p50": 0.0,
            "s2_p90": 0.0,
            "s2_bimodality": 0.0,
        }

    mean = float(np.mean(vals))
    std = float(np.std(vals))

    if std < 1e-10:
        features["s2_skewness"] = 0.0
        features["s2_kurtosis"] = 0.0
    else:
        centered = vals - mean
        features["s2_skewness"] = float(np.mean(centered**3) / std**3)
        features["s2_kurtosis"] = float(np.mean(centered**4) / std**4 - 3.0)

    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    features["s2_p10"] = float(p10)
    features["s2_p50"] = float(p50)
    features["s2_p90"] = float(p90)

    # Bimodality coefficient: (skew^2 + 1) / kurtosis
    # Values > 0.555 suggest bimodal distribution (e.g. land + water)
    kurt = features["s2_kurtosis"] + 3.0
    skew2 = features["s2_skewness"] ** 2
    features["s2_bimodality"] = float((skew2 + 1.0) / max(kurt, 1e-10))

    return features


# ============================================================================
# Temporal Change Spatial Pattern
# ============================================================================


@register_scorer("temporal_change_pattern")
def score_temporal_change_pattern(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """How temporal change is distributed spatially.

    Captures whether the whole scene changes uniformly (seasonal) or
    some pixels change while others are stable (land use change, phenology).
    """
    s2 = sample.get("sentinel2_l2a")
    if s2 is None or s2.shape[2] < 2:
        return {
            "frac_pixels_changed": 0.0,
            "change_spatial_std": 0.0,
            "change_concentration": 0.0,
        }

    # Per-pixel temporal std across visible bands mean, shape [H, W]
    visible = s2[..., :3].astype(np.float32)
    valid_all_t = np.all(_valid_pixels(visible), axis=(2, 3))  # [H, W]

    if valid_all_t.sum() < 10:
        return {
            "frac_pixels_changed": 0.0,
            "change_spatial_std": 0.0,
            "change_concentration": 0.0,
        }

    # Mean across visible bands per timestep -> [H, W, T]
    gray_t = visible.mean(axis=-1)
    # Mask invalid
    gray_t[~np.broadcast_to(valid_all_t[..., np.newaxis], gray_t.shape)] = np.nan

    pixel_temporal_std = np.nanstd(gray_t, axis=-1)  # [H, W]
    valid_std = pixel_temporal_std[valid_all_t]

    if valid_std.size == 0:
        return {
            "frac_pixels_changed": 0.0,
            "change_spatial_std": 0.0,
            "change_concentration": 0.0,
        }

    # What fraction of pixels have "significant" temporal change
    threshold = float(np.median(valid_std))
    if threshold < 1e-10:
        threshold = float(np.mean(valid_std))
    features: dict[str, float] = {}
    features["frac_pixels_changed"] = float(np.mean(valid_std > threshold))

    # How variable is the change across space — high = localized change
    features["change_spatial_std"] = float(np.std(valid_std))

    # Concentration: ratio of p90 to p50 temporal std
    p50, p90 = np.percentile(valid_std, [50, 90])
    features["change_concentration"] = float(p90 / max(p50, 1e-10))

    return features


# ============================================================================
# Season of Peak Activity
# ============================================================================


@register_scorer("peak_season")
def score_peak_season(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """When is peak greenness/brightness — captures phenological timing."""
    features: dict[str, float] = {}
    ts = sample.get("timestamps")
    s2 = sample.get("sentinel2_l2a")

    if ts is None or s2 is None or s2.shape[2] < 2:
        return {"peak_brightness_month": 0.0, "peak_greenness_month": 0.0}

    band_order = Modality.SENTINEL2_L2A.band_order
    red_idx = band_order.index("B04")
    nir_idx = band_order.index("B08")

    months = ts[:, 1] if ts.shape[-1] >= 2 else np.zeros(ts.shape[0])
    T = min(s2.shape[2], len(months))

    brightness_per_t = []
    ndvi_per_t = []
    for t in range(T):
        frame = s2[:, :, t, :3]
        valid = _valid_pixels(frame)
        if valid.any():
            brightness_per_t.append(float(np.mean(frame[valid])))
        else:
            brightness_per_t.append(np.nan)

        red = s2[:, :, t, red_idx].astype(np.float32)
        nir = s2[:, :, t, nir_idx].astype(np.float32)
        v = _valid_pixels(red) & _valid_pixels(nir)
        if v.any():
            denom = nir[v] + red[v]
            safe = np.where(np.abs(denom) < 1e-10, 1.0, denom)
            ndvi = (nir[v] - red[v]) / safe
            ndvi_per_t.append(float(np.mean(ndvi)))
        else:
            ndvi_per_t.append(np.nan)

    b_arr = np.array(brightness_per_t)
    n_arr = np.array(ndvi_per_t)

    valid_b = ~np.isnan(b_arr)
    valid_n = ~np.isnan(n_arr)

    if valid_b.any():
        peak_idx = int(np.nanargmax(b_arr))
        features["peak_brightness_month"] = (
            float(months[peak_idx]) if peak_idx < len(months) else 0.0
        )
    else:
        features["peak_brightness_month"] = 0.0

    if valid_n.any():
        peak_idx = int(np.nanargmax(n_arr))
        features["peak_greenness_month"] = (
            float(months[peak_idx]) if peak_idx < len(months) else 0.0
        )
    else:
        features["peak_greenness_month"] = 0.0

    return features


# ============================================================================
# Spatial Texture Scorers
# ============================================================================


@register_scorer("spatial_texture")
def score_spatial_texture(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Spatial texture features: edge density, homogeneity."""
    features: dict[str, float] = {}

    s2 = sample.get("sentinel2_l2a")
    if s2 is None or s2.shape[0] < 3 or s2.shape[1] < 3:
        return {
            "edge_density": 0.0,
            "spatial_homogeneity": 0.0,
            "patch_uniformity": 0.0,
        }

    # Use first valid timestep, average across visible bands (B02, B03, B04 = idx 0,1,2)
    for t in range(s2.shape[2]):
        frame = s2[:, :, t, :3].astype(np.float64)
        valid = _valid_pixels(frame)
        if valid.all(axis=-1).mean() > 0.5:
            gray = frame.mean(axis=-1)
            break
    else:
        return {
            "edge_density": 0.0,
            "spatial_homogeneity": 0.0,
            "patch_uniformity": 0.0,
        }

    # Simple Sobel-like edge detection
    dy = np.abs(np.diff(gray, axis=0))
    dx = np.abs(np.diff(gray, axis=1))
    edge_magnitude = np.sqrt(dy[:, :-1] ** 2 + dx[:-1, :] ** 2)
    features["edge_density"] = float(np.mean(edge_magnitude))

    # Spatial homogeneity: inverse of normalized spatial variance
    spatial_var = float(np.var(gray))
    spatial_mean = float(np.mean(gray))
    cv = (spatial_var**0.5) / max(abs(spatial_mean), 1e-10)
    features["spatial_homogeneity"] = 1.0 / (1.0 + cv)

    # Patch uniformity: fraction of pixels within 1 std of the mean
    std = float(np.std(gray))
    within_1std = np.abs(gray - spatial_mean) < std
    features["patch_uniformity"] = float(np.mean(within_1std))

    return features


# ============================================================================
# OSM / Infrastructure Scorers
# ============================================================================


@register_scorer("infrastructure")
def score_infrastructure(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Infrastructure features from OpenStreetMap raster."""
    features: dict[str, float] = {}
    osm = sample.get("openstreetmap_raster")
    if osm is None:
        return {
            "has_osm": 0.0,
            "osm_feature_count": 0.0,
            "osm_coverage_fraction": 0.0,
            "osm_diversity": 0.0,
            "has_buildings": 0.0,
            "has_roads": 0.0,
        }

    features["has_osm"] = 1.0
    valid = _valid_pixels(osm)

    if not valid.any():
        return {
            "has_osm": 0.0,
            "osm_feature_count": 0.0,
            "osm_coverage_fraction": 0.0,
            "osm_diversity": 0.0,
            "has_buildings": 0.0,
            "has_roads": 0.0,
        }

    # osm shape: [H, W, 1, C] where C = 29 OSM feature layers
    osm_bands = Modality.OPENSTREETMAP_RASTER.band_order
    n_layers = osm.shape[-1]

    # Per-layer: does it have any non-zero pixels?
    layer_active = []
    layer_coverage = []
    for c in range(n_layers):
        layer = osm[..., c]
        lv = layer[_valid_pixels(layer)]
        if lv.size > 0:
            active = float(np.any(lv > 0))
            coverage = float(np.mean(lv > 0))
        else:
            active = 0.0
            coverage = 0.0
        layer_active.append(active)
        layer_coverage.append(coverage)

    features["osm_feature_count"] = float(sum(layer_active))
    features["osm_coverage_fraction"] = _safe_mean(np.array(layer_coverage))

    # Diversity: entropy across layer coverages
    cov_arr = np.array(layer_coverage)
    cov_arr = cov_arr[cov_arr > 0]
    features["osm_diversity"] = float(entropy(cov_arr)) if cov_arr.size > 0 else 0.0

    # Named features
    building_idx = osm_bands.index("building") if "building" in osm_bands else None
    highway_idx = osm_bands.index("highway") if "highway" in osm_bands else None

    features["has_buildings"] = (
        layer_active[building_idx] if building_idx is not None else 0.0
    )
    features["has_roads"] = (
        layer_active[highway_idx] if highway_idx is not None else 0.0
    )

    return features


# ============================================================================
# Population Scorers
# ============================================================================


@register_scorer("population")
def score_population(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Population density features from WorldPop."""
    features: dict[str, float] = {}
    wp = sample.get("worldpop")
    if wp is None:
        return {
            "has_worldpop": 0.0,
            "population_mean": 0.0,
            "population_max": 0.0,
            "population_std": 0.0,
            "is_populated": 0.0,
            "is_dense_urban": 0.0,
        }

    valid = _valid_pixels(wp)
    pop = wp[valid].astype(np.float64)

    if pop.size == 0:
        return {
            "has_worldpop": 0.0,
            "population_mean": 0.0,
            "population_max": 0.0,
            "population_std": 0.0,
            "is_populated": 0.0,
            "is_dense_urban": 0.0,
        }

    features["has_worldpop"] = 1.0
    features["population_mean"] = _safe_mean(pop)
    features["population_max"] = float(np.max(pop))
    features["population_std"] = _safe_std(pop)
    features["is_populated"] = float(np.any(pop > 0))
    features["is_dense_urban"] = float(features["population_mean"] > 1000)

    return features


# ============================================================================
# Agriculture Scorers
# ============================================================================


@register_scorer("agriculture")
def score_agriculture(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Agriculture features from CDL, WorldCereal, canopy height."""
    features: dict[str, float] = {}

    # CDL (Cropland Data Layer)
    cdl = sample.get("cdl")
    if cdl is not None:
        valid = _valid_pixels(cdl)
        cdl_v = cdl[valid].flatten()
        if cdl_v.size > 0:
            features["has_cdl"] = 1.0
            unique_classes = np.unique(cdl_v)
            features["cdl_num_classes"] = float(len(unique_classes))
            hist = np.histogram(
                cdl_v,
                bins=max(int(unique_classes.max()) + 1, 1),
                range=(0, max(unique_classes.max() + 1, 1)),
            )[0]
            hist = hist[hist > 0]
            features["cdl_entropy"] = float(entropy(hist)) if hist.size > 0 else 0.0
        else:
            features["has_cdl"] = 0.0
            features["cdl_num_classes"] = 0.0
            features["cdl_entropy"] = 0.0
    else:
        features["has_cdl"] = 0.0
        features["cdl_num_classes"] = 0.0
        features["cdl_entropy"] = 0.0

    # WorldCereal crop fractions
    wc = sample.get("worldcereal")
    if wc is not None:
        valid = _valid_pixels(wc)
        wc_v = wc[valid].astype(np.float64)
        if wc_v.size > 0:
            features["has_worldcereal"] = 1.0
            features["worldcereal_crop_fraction"] = float(np.mean(wc_v > 0))
        else:
            features["has_worldcereal"] = 0.0
            features["worldcereal_crop_fraction"] = 0.0
    else:
        features["has_worldcereal"] = 0.0
        features["worldcereal_crop_fraction"] = 0.0

    # Canopy height
    ch = sample.get("wri_canopy_height_map")
    if ch is not None:
        valid = _valid_pixels(ch)
        ch_v = ch[valid].astype(np.float64)
        if ch_v.size > 0:
            features["has_canopy_height"] = 1.0
            features["canopy_height_mean"] = _safe_mean(ch_v)
            features["canopy_height_max"] = float(np.max(ch_v))
            features["canopy_height_std"] = _safe_std(ch_v)
            features["has_forest_canopy"] = float(np.max(ch_v) > 5.0)
        else:
            features["has_canopy_height"] = 0.0
            features["canopy_height_mean"] = 0.0
            features["canopy_height_max"] = 0.0
            features["canopy_height_std"] = 0.0
            features["has_forest_canopy"] = 0.0
    else:
        features["has_canopy_height"] = 0.0
        features["canopy_height_mean"] = 0.0
        features["canopy_height_max"] = 0.0
        features["canopy_height_std"] = 0.0
        features["has_forest_canopy"] = 0.0

    return features


# ============================================================================
# Weather / Climate Scorers (ERA5)
# ============================================================================


@register_scorer("weather")
def score_weather(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Climate/weather features from ERA5 reanalysis."""
    features: dict[str, float] = {}
    era5 = sample.get("era5_10")
    if era5 is None:
        return {
            "has_era5": 0.0,
            "temperature_mean": 0.0,
            "temperature_range": 0.0,
            "precipitation_total": 0.0,
            "wind_speed_mean": 0.0,
            "climate_variability": 0.0,
        }

    valid = _valid_pixels(era5)
    if not valid.any():
        return {
            "has_era5": 0.0,
            "temperature_mean": 0.0,
            "temperature_range": 0.0,
            "precipitation_total": 0.0,
            "wind_speed_mean": 0.0,
            "climate_variability": 0.0,
        }

    features["has_era5"] = 1.0

    # ERA5 bands: 2m-temperature, 2m-dewpoint-temperature, surface-pressure,
    #             10m-u-component-of-wind, 10m-v-component-of-wind, total-precipitation
    # Shape: [T, C]
    era5_f = era5.astype(np.float64)

    # Temperature (band 0)
    temp = era5_f[:, 0]
    temp_valid = temp[_valid_pixels(era5[:, 0])]
    features["temperature_mean"] = _safe_mean(temp_valid)
    features["temperature_range"] = (
        float(np.ptp(temp_valid)) if temp_valid.size > 1 else 0.0
    )

    # Precipitation (band 5)
    if era5_f.shape[1] > 5:
        precip = era5_f[:, 5]
        precip_valid = precip[_valid_pixels(era5[:, 5])]
        features["precipitation_total"] = (
            float(np.sum(precip_valid)) if precip_valid.size > 0 else 0.0
        )
    else:
        features["precipitation_total"] = 0.0

    # Wind speed (bands 3, 4: u and v components)
    if era5_f.shape[1] > 4:
        u = era5_f[:, 3]
        v = era5_f[:, 4]
        u_valid = u[_valid_pixels(era5[:, 3])]
        v_valid = v[_valid_pixels(era5[:, 4])]
        if u_valid.size > 0 and v_valid.size > 0:
            wind_speed = np.sqrt(u_valid**2 + v_valid**2)
            features["wind_speed_mean"] = _safe_mean(wind_speed)
        else:
            features["wind_speed_mean"] = 0.0
    else:
        features["wind_speed_mean"] = 0.0

    # Overall climate variability: mean std across all bands over time
    band_stds = []
    for b in range(era5_f.shape[1]):
        bv = era5_f[:, b]
        bv = bv[_valid_pixels(era5[:, b])]
        if bv.size > 1:
            band_stds.append(float(np.std(bv)))
    features["climate_variability"] = (
        _safe_mean(np.array(band_stds)) if band_stds else 0.0
    )

    return features


# ============================================================================
# Data Quality Scorers
# ============================================================================


@register_scorer("data_quality")
def score_data_quality(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Data quality: missingness, constant regions, cloud/shadow proxies, noise."""
    features: dict[str, float] = {}

    # Overall missing pixel fraction across all modalities
    total_pixels = 0
    missing_pixels = 0
    for mod_name in ALL_TRAINING_MODALITIES:
        data = sample.get(mod_name)
        if data is None:
            continue
        total_pixels += data.size
        missing_pixels += int(np.sum(data == MISSING_VALUE))

    features["overall_missing_frac"] = (
        float(missing_pixels) / max(total_pixels, 1) if total_pixels > 0 else 1.0
    )

    # Per-modality degenerate content checks: constant/near-constant regions,
    # saturated values, and missing fractions. Catches clouds, shadows, nodata
    # fill, sensor dead pixels, etc.
    spatial_modalities = {
        "sentinel2_l2a": {"bright_thresh": 8000, "dark_thresh": 100},
        "sentinel1": {"bright_thresh": 5, "dark_thresh": -25},
        "landsat": {"bright_thresh": 8000, "dark_thresh": 100},
        "worldcover": None,
        "srtm": None,
        "openstreetmap_raster": None,
    }

    for mod_name, thresholds in spatial_modalities.items():
        data = sample.get(mod_name)
        if data is None:
            features[f"{mod_name}_missing_pixel_frac"] = 1.0
            features[f"{mod_name}_constant_frac"] = 0.0
            if thresholds:
                features[f"{mod_name}_bright_frac"] = 0.0
                features[f"{mod_name}_dark_frac"] = 0.0
            continue

        valid = _valid_pixels(data)
        features[f"{mod_name}_missing_pixel_frac"] = 1.0 - float(valid.mean())

        if not valid.any():
            features[f"{mod_name}_constant_frac"] = 0.0
            if thresholds:
                features[f"{mod_name}_bright_frac"] = 0.0
                features[f"{mod_name}_dark_frac"] = 0.0
            continue

        # Constant fraction: what fraction of valid pixels equal the mode
        vals = data[valid].flatten().astype(np.float32)
        if vals.size > 10_000:
            check = vals[np.random.choice(vals.size, 10_000, replace=False)]
        else:
            check = vals
        unique, counts = np.unique(check, return_counts=True)
        features[f"{mod_name}_constant_frac"] = float(counts.max()) / check.size

        # Bright/dark pixel fractions (sensor-specific thresholds)
        if thresholds:
            features[f"{mod_name}_bright_frac"] = float(
                np.mean(check > thresholds["bright_thresh"])
            )
            features[f"{mod_name}_dark_frac"] = float(
                np.mean(check < thresholds["dark_thresh"])
            )

    # S1 noise proxy: variance of SAR backscatter
    s1 = sample.get("sentinel1")
    if s1 is not None:
        valid = _valid_pixels(s1)
        if valid.any():
            s1_valid = s1[valid].astype(np.float32)
            features["s1_variance"] = float(np.var(s1_valid))
        else:
            features["s1_variance"] = 0.0
    else:
        features["s1_variance"] = 0.0

    return features


# ============================================================================
# Cross-modal Scorers
# ============================================================================


@register_scorer("cross_modal")
def score_cross_modal(
    sample: dict[str, np.ndarray], meta: dict[str, Any]
) -> dict[str, float]:
    """Features measuring relationships between modalities."""
    features: dict[str, float] = {}

    # Optical-SAR temporal correlation (S2 vs S1 temporal mean series)
    s2 = sample.get("sentinel2_l2a")
    s1 = sample.get("sentinel1")

    if s2 is not None and s1 is not None and s2.shape[2] > 1 and s1.shape[2] > 1:
        T = min(s2.shape[2], s1.shape[2])
        s2_ts, s1_ts = [], []
        for t in range(T):
            s2_frame = s2[:, :, t, :].astype(np.float64)
            s1_frame = s1[:, :, t, :].astype(np.float64)
            s2_v = s2_frame[_valid_pixels(s2[:, :, t, :])]
            s1_v = s1_frame[_valid_pixels(s1[:, :, t, :])]
            if s2_v.size > 0 and s1_v.size > 0:
                s2_ts.append(float(np.mean(s2_v)))
                s1_ts.append(float(np.mean(s1_v)))

        if len(s2_ts) > 2:
            s2_arr = np.array(s2_ts)
            s1_arr = np.array(s1_ts)
            s2_arr = (s2_arr - s2_arr.mean()) / max(s2_arr.std(), 1e-10)
            s1_arr = (s1_arr - s1_arr.mean()) / max(s1_arr.std(), 1e-10)
            features["optical_sar_temporal_corr"] = float(np.mean(s2_arr * s1_arr))
        else:
            features["optical_sar_temporal_corr"] = 0.0
    else:
        features["optical_sar_temporal_corr"] = 0.0

    # Elevation-vegetation correlation (SRTM vs canopy height)
    srtm = sample.get("srtm")
    ch = sample.get("wri_canopy_height_map")
    if srtm is not None and ch is not None:
        srtm_flat = srtm[_valid_pixels(srtm)].astype(np.float64)
        ch_flat = ch[_valid_pixels(ch)].astype(np.float64)
        min_size = min(srtm_flat.size, ch_flat.size)
        if min_size > 10:
            s = srtm_flat[:min_size]
            c = ch_flat[:min_size]
            s = (s - s.mean()) / max(s.std(), 1e-10)
            c = (c - c.mean()) / max(c.std(), 1e-10)
            features["elevation_canopy_corr"] = float(np.mean(s * c))
        else:
            features["elevation_canopy_corr"] = 0.0
    else:
        features["elevation_canopy_corr"] = 0.0

    return features


# ============================================================================
# Public API
# ============================================================================


def load_h5_sample(filepath: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a sample from an H5 file.

    Returns:
        Tuple of (sample_dict, metadata_dict). sample_dict maps modality names
        to numpy arrays. metadata_dict contains 'filepath' and 'sample_index'.
    """
    import h5py
    import hdf5plugin  # noqa: F401

    sample_dict: dict[str, np.ndarray] = {}
    with open(filepath, "rb") as f:
        with h5py.File(f, "r") as h5file:
            for key in h5file.keys():
                if key == "missing_timesteps_masks":
                    continue
                sample_dict[key] = h5file[key][()]

    meta: dict[str, Any] = {"filepath": filepath}

    # Extract sample index from filename pattern "sample_{index}.h5"
    import os

    basename = os.path.basename(filepath)
    if basename.startswith("sample_") and basename.endswith(".h5"):
        try:
            meta["sample_index"] = int(basename[7:-3])
        except ValueError:
            pass

    return sample_dict, meta


def score_sample(
    sample: dict[str, np.ndarray],
    meta: dict[str, Any],
    scorers: list[str] | None = None,
) -> dict[str, float]:
    """Run all (or selected) scorers on a single sample.

    Args:
        sample: modality_name -> np.ndarray
        meta: metadata dict (must include 'lat', 'lon' for geographic scorer)
        scorers: optional list of scorer names to run. If None, runs all.

    Returns:
        Flat dict of feature_name -> float value.
    """
    features: dict[str, float] = {}
    registry = (
        SCORER_REGISTRY
        if scorers is None
        else {k: v for k, v in SCORER_REGISTRY.items() if k in scorers}
    )

    for scorer_name, scorer_fn in registry.items():
        try:
            result = scorer_fn(sample, meta)
            features.update(result)
        except Exception:
            logger.warning(f"Scorer '{scorer_name}' failed", exc_info=True)

    return features


def score_sample_timed(
    sample: dict[str, np.ndarray],
    meta: dict[str, Any],
    scorers: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Like score_sample but also returns per-scorer wall-clock times in seconds."""
    import time

    features: dict[str, float] = {}
    timings: dict[str, float] = {}
    registry = (
        SCORER_REGISTRY
        if scorers is None
        else {k: v for k, v in SCORER_REGISTRY.items() if k in scorers}
    )

    for scorer_name, scorer_fn in registry.items():
        t0 = time.perf_counter()
        try:
            result = scorer_fn(sample, meta)
            features.update(result)
        except Exception:
            logger.warning(f"Scorer '{scorer_name}' failed", exc_info=True)
        timings[scorer_name] = time.perf_counter() - t0

    return features, timings


def list_all_features() -> list[str]:
    """Return the names of all features that would be produced by all scorers.

    Useful for pre-allocating columns or understanding the output schema.
    Runs all scorers on an empty sample to discover feature keys.
    """
    empty_sample: dict[str, np.ndarray] = {}
    empty_meta: dict[str, Any] = {"lat": 0.0, "lon": 0.0}
    return sorted(score_sample(empty_sample, empty_meta).keys())
