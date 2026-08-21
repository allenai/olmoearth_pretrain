"""Landsat 8/9 OLI-TIRS Collection-2 Level-1 radiometric calibration.

Landsat is stored in the rslearn dataset as raw quantized DN (Collection-2
Level-1). For training we want physical units:

* Reflective bands (B1-B9, including the panchromatic B8) -> top-of-atmosphere
  reflectance with sun-angle correction: ``rho = (M_rho * DN + A_rho) / sin(theta_SE)``.
  ``M_rho`` / ``A_rho`` are the universal C2 rescaling constants (2e-5, -0.1) and
  ``theta_SE`` is the per-scene ``SUN_ELEVATION`` from the scene's MTL.
* Thermal bands (B10/B11) -> at-sensor brightness temperature in Kelvin:
  ``L = M_L * DN + A_L``; ``BT = K2 / ln(K1 / L + 1)``. ``M_L`` / ``A_L`` / ``K1`` /
  ``K2`` are fixed per platform and differ between Landsat-8 and Landsat-9.

The MTL fetch / parse mirrors the working prototype in
``/weka/dfive-default/yawenz/landsat/spectral/extract_spectra.py``.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from olmoearth_pretrain.data.constants import MISSING_VALUE

logger = logging.getLogger(__name__)

# Universal Collection-2 Level-1 reflectance rescaling constants (identical for
# every band and both platforms). Reflectance is corrected by the per-scene sun
# elevation afterwards.
REFLECTANCE_MULT = 2e-5
REFLECTANCE_ADD = -0.1

# Reflective bands get TOA reflectance; thermal bands get brightness temperature.
REFLECTIVE_BANDS = frozenset({"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"})
THERMAL_BANDS = frozenset({"B10", "B11"})

# Per-platform thermal calibration for B10/B11. RADIANCE_ADD is 0.1 for both.
# L9 values verified against spectral/cache/*.json; L8 values are the standard
# USGS Collection-2 published constants.
THERMAL_CALIBRATION: dict[str, dict[str, dict[str, float]]] = {
    "LC08": {
        "B10": {"rad_mult": 3.342e-04, "rad_add": 0.1, "k1": 774.8853, "k2": 1321.0789},
        "B11": {"rad_mult": 3.342e-04, "rad_add": 0.1, "k1": 480.8883, "k2": 1201.1442},
    },
    "LC09": {
        "B10": {
            "rad_mult": 3.8000e-04,
            "rad_add": 0.1,
            "k1": 799.0284,
            "k2": 1329.2405,
        },
        "B11": {
            "rad_mult": 3.4900e-04,
            "rad_add": 0.1,
            "k1": 475.6581,
            "k2": 1198.3494,
        },
    },
}


def parse_mtl(text: str) -> dict:
    """Pull the radiometric-rescaling numbers we need out of an MTL.txt.

    Ported from the spectral-review prototype. Only ``sun_elevation`` is used by
    the conversion (the rest is kept for validation / debugging).
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')

    def num(key: str, default: float | None = None) -> float | None:
        try:
            return float(out[key])
        except (KeyError, ValueError):
            return default

    cal: dict = {
        "sun_elevation": num("SUN_ELEVATION"),
        "sun_azimuth": num("SUN_AZIMUTH"),
        "earth_sun_distance": num("EARTH_SUN_DISTANCE"),
        "date_acquired": out.get("DATE_ACQUIRED"),
        "cloud_cover": num("CLOUD_COVER"),
        "refl_mult": {},
        "refl_add": {},
        "rad_mult": {},
        "rad_add": {},
        "k1": {},
        "k2": {},
    }
    for band in REFLECTIVE_BANDS:
        n = band[1:]
        cal["refl_mult"][band] = num(f"REFLECTANCE_MULT_BAND_{n}")
        cal["refl_add"][band] = num(f"REFLECTANCE_ADD_BAND_{n}")
    for band in THERMAL_BANDS:
        n = band[1:]
        cal["rad_mult"][band] = num(f"RADIANCE_MULT_BAND_{n}")
        cal["rad_add"][band] = num(f"RADIANCE_ADD_BAND_{n}")
        cal["k1"][band] = num(f"K1_CONSTANT_BAND_{n}")
        cal["k2"][band] = num(f"K2_CONSTANT_BAND_{n}")
    return cal


# Cache MTL sun elevation by scene id within a process so repeated scenes (a
# scene often covers several monthly layers / windows) are only fetched once.
_SUN_ELEVATION_CACHE: dict[str, float | None] = {}
_S3_CLIENT = None


def _s3_client() -> Any:
    """Lazily construct a boto3 S3 client (avoids importing boto3 at import)."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3

        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _scene_id_from_blob_path(blob_path: str) -> str:
    """Derive the scene/product id from a Landsat blob path.

    ``blob_path`` looks like ``collection02/.../LC09_..._T1/LC09_..._T1_`` (with a
    trailing underscore before the band suffix), so the scene id is the last path
    component with the trailing underscore removed.
    """
    return blob_path.rstrip("/").rsplit("/", 1)[-1].rstrip("_")


def fetch_sun_elevation(blob_path: str) -> float | None:
    """Fetch the scene ``SUN_ELEVATION`` (degrees) from its MTL.txt on S3.

    Reads ``s3://usgs-landsat`` (requester pays), so AWS_ACCESS_KEY_ID /
    AWS_SECRET_ACCESS_KEY must be in the environment. Returns None on any failure
    so a single unreadable MTL does not abort the whole conversion.
    """
    scene_id = _scene_id_from_blob_path(blob_path)
    if scene_id in _SUN_ELEVATION_CACHE:
        return _SUN_ELEVATION_CACHE[scene_id]

    sun_elevation: float | None = None
    try:
        obj = _s3_client().get_object(
            Bucket="usgs-landsat",
            Key=blob_path + "MTL.txt",
            RequestPayer="requester",
        )
        cal = parse_mtl(obj["Body"].read().decode("utf-8", "replace"))
        sun_elevation = cal["sun_elevation"]
    except Exception as e:  # noqa: BLE001 - one bad MTL shouldn't kill the run
        logger.warning(f"failed to fetch/parse MTL for {scene_id}: {e}")

    _SUN_ELEVATION_CACHE[scene_id] = sun_elevation
    return sun_elevation


def platform_from_scene_id(scene_id: str | None) -> str | None:
    """Return the platform code (``LC08`` / ``LC09``) from a scene/product id."""
    if not scene_id:
        return None
    prefix = scene_id[:4]
    return prefix if prefix in THERMAL_CALIBRATION else None


def _normalize_length(values: list, length: int, fill: Any) -> list:
    """Pad or truncate ``values`` to ``length`` (defensive time-axis alignment)."""
    if len(values) == length:
        return values
    if len(values) > length:
        return values[:length]
    return values + [fill] * (length - len(values))


def convert_landsat_to_physical(
    image: npt.NDArray,
    solar_elevations: list[float | None],
    platforms: list[str | None],
    band_order: list[str],
) -> npt.NDArray:
    """Convert a stacked Landsat DN image to reflectance / brightness temperature.

    Args:
        image: DN array of shape ``[H, W, T, C]`` where ``C == len(band_order)``.
        solar_elevations: per-timestep ``SUN_ELEVATION`` in degrees (or None).
        platforms: per-timestep platform code (``LC08`` / ``LC09``) or None.
        band_order: band name per channel (e.g. ``Modality.LANDSAT.band_order``).

    Returns:
        float32 array of the same shape. Reflective channels become TOA
        reflectance, thermal channels become brightness temperature in Kelvin.
        Nodata (DN == 0) and timesteps lacking calibration become MISSING_VALUE.
    """
    dn = image.astype(np.float32)
    _, _, t, c = dn.shape
    if c != len(band_order):
        raise ValueError(
            f"channel count {c} does not match band_order length {len(band_order)}"
        )

    solar_elevations = _normalize_length(list(solar_elevations), t, None)
    platforms = _normalize_length(list(platforms), t, None)

    # Per-timestep sin(sun_elevation); NaN marks a missing/invalid angle.
    elev = np.array(
        [e if e is not None else np.nan for e in solar_elevations], dtype=np.float32
    )
    sin_elev = np.sin(np.radians(elev))
    refl_missing_t = ~np.isfinite(sin_elev) | (sin_elev <= 0)

    out = np.empty_like(dn)
    for ch, band in enumerate(band_order):
        chan = dn[..., ch]  # [H, W, T]
        nodata = chan == 0

        if band in REFLECTIVE_BANDS:
            safe_sin = np.where(refl_missing_t, 1.0, sin_elev)
            val = (REFLECTANCE_MULT * chan + REFLECTANCE_ADD) / safe_sin[None, None, :]
            missing_t = refl_missing_t
        elif band in THERMAL_BANDS:
            rad_mult = np.full(t, np.nan, dtype=np.float32)
            rad_add = np.full(t, np.nan, dtype=np.float32)
            k1 = np.full(t, np.nan, dtype=np.float32)
            k2 = np.full(t, np.nan, dtype=np.float32)
            for ti, plat in enumerate(platforms):
                cal = THERMAL_CALIBRATION.get(plat or "", {}).get(band)
                if cal is not None:
                    rad_mult[ti] = cal["rad_mult"]
                    rad_add[ti] = cal["rad_add"]
                    k1[ti] = cal["k1"]
                    k2[ti] = cal["k2"]
            missing_t = ~np.isfinite(k1)
            radiance = rad_mult[None, None, :] * chan + rad_add[None, None, :]
            # Guard the log: radiance can be <= 0 for nodata, which is masked below.
            safe_radiance = np.where(radiance > 0, radiance, np.nan)
            val = k2[None, None, :] / np.log(k1[None, None, :] / safe_radiance + 1.0)
        else:
            # Unknown band: leave the raw DN untouched.
            out[..., ch] = chan
            continue

        val = np.where(nodata, MISSING_VALUE, val)
        if missing_t.any():
            val[:, :, missing_t] = MISSING_VALUE
        # Any remaining non-finite (e.g. thermal nodata that slipped through) -> missing.
        val = np.where(np.isfinite(val), val, MISSING_VALUE)
        out[..., ch] = val

    return out.astype(np.float32)
