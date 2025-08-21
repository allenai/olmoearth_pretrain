"""Utils for the data module."""

import math
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def to_cartesian(lat: float, lon: float) -> np.ndarray:
    """Convert latitude and longitude to Cartesian coordinates.

    Args:
        lat: Latitude in degrees as a float.
        lon: Longitude in degrees as a float.

    Returns:
        A numpy array of Cartesian coordinates (x, y, z).
    """

    def validate_lat_lon(lat: float, lon: float) -> None:
        """Validate the latitude and longitude.

        Args:
            lat: Latitude in degrees as a float.
            lon: Longitude in degrees as a float.
        """
        assert -90 <= lat <= 90, (
            f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon <= 180, (
            f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        )

    def convert_to_radians(lat: float, lon: float) -> tuple:
        """Convert the latitude and longitude to radians.

        Args:
            lat: Latitude in degrees as a float.
            lon: Longitude in degrees as a float.

        Returns:
            A tuple of the latitude and longitude in radians.
        """
        return lat * math.pi / 180, lon * math.pi / 180

    def compute_cartesian(lat: float, lon: float) -> tuple:
        """Compute the Cartesian coordinates.

        Args:
            lat: Latitude in degrees as a float.
            lon: Longitude in degrees as a float.

        Returns:
            A tuple of the Cartesian coordinates (x, y, z).
        """
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)

        return x, y, z

    validate_lat_lon(lat, lon)
    lat, lon = convert_to_radians(lat, lon)
    x, y, z = compute_cartesian(lat, lon)

    return np.array([x, y, z])


# According to the EE, we need to convert Sentinel1 data to dB using 10*log10(x)
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD#description
def convert_to_db(data: np.ndarray) -> np.ndarray:
    """Convert the data to decibels.

    Args:
        data: The data to convert to decibels.

    Returns:
        The data in decibels.
    """
    # clip data to 1e-10 to avoid log(0)
    data = np.clip(data, 1e-10, None)
    result = 10 * np.log10(data)
    return result


def update_streaming_stats(
    current_count: int,
    current_mean: float,
    current_var: float,
    modality_band_data: np.ndarray,
) -> tuple[int, float, float]:
    """Update the streaming mean and variance for a batch of data.

    Args:
        current_count: The current count of data points.
        current_mean: The current mean of the data.
        current_var: The current variance of the data.
        modality_band_data: The data for the current modality band.

    Returns:
        Updated count, mean, and variance for the modality band.
    """
    band_data_count = np.prod(modality_band_data.shape)

    # Compute updated mean and variance with the new batch of data
    # Reference: https://www.geeksforgeeks.org/expression-for-mean-and-variance-in-a-running-stream/
    new_count = current_count + band_data_count
    new_mean = (
        current_mean
        + (modality_band_data.mean() - current_mean) * band_data_count / new_count
    )
    new_var = (
        current_var
        + ((modality_band_data - current_mean) * (modality_band_data - new_mean)).sum()
    )

    return new_count, new_mean, new_var


def plot_latlon_distribution(latlons: np.ndarray, title: str, s=0.01) -> plt.Figure:
    """Plot the geographic distribution of the data."""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)

    # Plot the data points
    ax.scatter(
        latlons[:, 1],
        latlons[:, 0],
        transform=ccrs.PlateCarree(),
        alpha=0.5,
        s=s,
    )

    ax.set_global()  # Show the entire globe
    ax.gridlines()
    ax.set_title(title)
    return fig


def plot_modality_data_distribution(modality: str, modality_data: dict) -> plt.Figure:
    """Plot the data distribution."""
    fig, axes = plt.subplots(
        len(modality_data), 1, figsize=(10, 5 * len(modality_data))
    )
    if len(modality_data) == 1:
        axes = [axes]
    for ax, (band, values) in zip(axes, modality_data.items()):
        ax.hist(values, bins=50, alpha=0.75)
        ax.set_title(f"{modality} - {band}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# TODO: make earth radius a constant
# TODO: is this haversine helper even used?
def haversine(a, b, radius=6_371_000.0):
    """Compute the haversine distance between two points.

    Parameters
    ----------
    a, b : sequence of two floats
        Latitude and longitude in degrees.
    radius : float
        Sphere radius in metres.

    Returns
    -------
    float
        Great circle distance between a and b on a sphere.
    """
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sdlat2 = math.sin(dlat / 2.0) ** 2
    sdlon2 = math.sin(dlon / 2.0) ** 2
    alpha = sdlat2 + math.cos(lat1) * math.cos(lat2) * sdlon2
    # guard numeric
    alpha = min(1.0, max(0.0, alpha))
    return 2.0 * radius * math.asin(math.sqrt(alpha))

import numpy as np

def haversine_distance_radians(
    coords1, coords2
) -> np.ndarray:
    """
    Calculate the haversine distance between two arrays of points in meters.

    Parameters
    ----------
    coords1 : array-like, shape (M, 2)
        Array of [lat_deg, lon_deg] for the first set of points.
    coords2 : array-like, shape (N, 2)
        Array of [lat_deg, lon_deg] for the second set of points.

    Returns
    -------
    np.ndarray
        Array of distances in meters. If coords1 is (M,2) and coords2 is (N,2),
        returns (M,N) array of distances.
        If coords2 is (2,) (a single point), returns (M,) array of distances.
    """
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)

    # Earth's radius in meters
    R = 6371000.0

    # If coords2 is a single point, reshape for broadcasting
    if coords2.ndim == 1:
        coords2 = coords2[np.newaxis, :]

    lat1 = np.radians(coords1[:, 0])[:, np.newaxis]
    lon1 = np.radians(coords1[:, 1])[:, np.newaxis]
    lat2 = np.radians(coords2[:, 0])[np.newaxis, :]
    lon2 = np.radians(coords2[:, 1])[np.newaxis, :]

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c

    # If coords2 was a single point, return (M,) shape
    if distances.shape[1] == 1:
        return distances[:, 0]
    return distances


def nearest_haversine(
    latlon_a,
    N=None,
    latlon_b=None,
    return_indices=False,
    chunk_size=None,
    radius=6_371_000.0,
):
    """
    Haversine distances with optional k-NN selection and optional second set.

    Args:
        latlon_a: array-like, shape (M, 2) with [lat_deg, lon_deg].
        N: None for all pairwise distances, or int for N nearest neighbors per row of A.
        latlon_b: optional array-like, shape (K, 2). If None, uses latlon_a.
        return_indices: if True, also return column indices (into B; into A if B is None).
        chunk_size: process rows of A in blocks of this size (reduces peak memory).
        radius: Earth radius in meters.

    Returns:
        If N is None (all pairs):
            dists: (M, K) float array of all pairwise distances.
            idx:   (M, K) int array of column indices (0..K-1), if return_indices=True.
        If N is an int (k-NN per row):
            dists: (M, N_eff) float array of nearest distances (ascending per row).
            idx:   (M, N_eff) int array of corresponding column indices (into B/A).
            Note: when latlon_b is None (within-set), self is excluded.
                  N_eff = min(N, K) for cross-set, or min(N, M-1) for within-set.
    """
    latlon_a = np.asarray(latlon_a, dtype=np.float64)
    if latlon_a.ndim != 2 or latlon_a.shape[1] != 2:
        raise ValueError("latlon_a must have shape (M, 2) [lat_deg, lon_deg].")

    if latlon_b is None:
        latlon_b = latlon_a
        same_set = True
    else:
        latlon_b = np.asarray(latlon_b, dtype=np.float64)
        if latlon_b.ndim != 2 or latlon_b.shape[1] != 2:
            raise ValueError("latlon_b must have shape (K, 2) [lat_deg, lon_deg].")
        same_set = False

    M = latlon_a.shape[0]
    K = latlon_b.shape[0]

    # Handle empties early
    if M == 0 or K == 0:
        if N is None:
            out = np.empty((M, K), dtype=np.float64)
            return (out, np.empty((M, K), dtype=np.int64)) if return_indices else out
        # k-NN case with zero columns
        out = np.empty((M, 0), dtype=np.float64)
        return (out, np.empty((M, 0), dtype=np.int64)) if return_indices else out

    # Convert to radians
    lat_a = np.deg2rad(latlon_a[:, 0])
    lon_a = np.deg2rad(latlon_a[:, 1])
    lat_b = np.deg2rad(latlon_b[:, 0])
    lon_b = np.deg2rad(latlon_b[:, 1])

    cos_lat_b = np.cos(lat_b)

    if chunk_size is None:
        chunk_size = M

    if N is None:
        # All pairwise distances
        dists_out = np.empty((M, K), dtype=np.float64)
        # Precompute a reusable column index row if indices requested
        idx_cols = None
        if return_indices:
            idx_cols = np.arange(K, dtype=np.int64)

        for i0 in range(0, M, chunk_size):
            i1 = min(M, i0 + chunk_size)
            L = i1 - i0

            lat_blk = lat_a[i0:i1][:, None]      # (L, 1)
            lon_blk = lon_a[i0:i1][:, None]      # (L, 1)
            dlat = lat_blk - lat_b[None, :]      # (L, K)
            dlon = lon_blk - lon_b[None, :]      # (L, K)

            sdlat2 = np.sin(0.5 * dlat) ** 2
            sdlon2 = np.sin(0.5 * dlon) ** 2
            a = sdlat2 + (np.cos(lat_a[i0:i1])[:, None] * cos_lat_b[None, :]) * sdlon2
            np.clip(a, 0.0, 1.0, out=a)
            dist = 2.0 * radius * np.arcsin(np.sqrt(a))

            dists_out[i0:i1, :] = dist

        if return_indices:
            # Each row simply maps to columns 0..K-1
            idx_out = np.broadcast_to(idx_cols, (M, K)).copy()
            return dists_out, idx_out
        return dists_out

    # ---- k-NN selection per row ----
    if same_set:
        N_eff = int(min(max(M - 1, 0), N))
    else:
        N_eff = int(min(K, N))

    if N_eff <= 0:
        out = np.empty((M, 0), dtype=np.float64)
        return (out, np.empty((M, 0), dtype=np.int64)) if return_indices else out

    dists_out = np.empty((M, N_eff), dtype=np.float64)
    idx_out = np.empty((M, N_eff), dtype=np.int64) if return_indices else None

    for i0 in range(0, M, chunk_size):
        i1 = min(M, i0 + chunk_size)
        L = i1 - i0

        lat_blk = lat_a[i0:i1][:, None]        # (L, 1)
        lon_blk = lon_a[i0:i1][:, None]        # (L, 1)
        dlat = lat_blk - lat_b[None, :]        # (L, K)
        dlon = lon_blk - lon_b[None, :]        # (L, K)

        sdlat2 = np.sin(0.5 * dlat) ** 2
        sdlon2 = np.sin(0.5 * dlon) ** 2
        a = sdlat2 + (np.cos(lat_a[i0:i1])[:, None] * cos_lat_b[None, :]) * sdlon2
        np.clip(a, 0.0, 1.0, out=a)
        dist = 2.0 * radius * np.arcsin(np.sqrt(a))  # (L, K)

        # Exclude self for within-set
        if same_set:
            rows = np.arange(L)
            cols = np.arange(i0, i1)  # global column indices
            dist[rows, cols] = np.inf

        # Argpartition to get N_eff smallest (unsorted)
        part_idx = np.argpartition(dist, N_eff - 1, axis=1)[:, :N_eff]          # (L, N_eff)
        sel_d = np.take_along_axis(dist, part_idx, axis=1)                      # (L, N_eff)

        # Sort those N_eff per row
        order = np.argsort(sel_d, axis=1)
        sel_d_sorted = np.take_along_axis(sel_d, order, axis=1)                 # (L, N_eff)
        dists_out[i0:i1, :] = sel_d_sorted

        if return_indices:
            part_idx_sorted = np.take_along_axis(part_idx, order, axis=1)       # (L, N_eff)
            idx_out[i0:i1, :] = part_idx_sorted

    return (dists_out, idx_out) if return_indices else dists_out