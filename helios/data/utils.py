"""Utils for the data module."""

import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


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


def plot_latlon_distribution(latlons: np.ndarray, title: str) -> plt.Figure:
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
        s=0.01,
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
