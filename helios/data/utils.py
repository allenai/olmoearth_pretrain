"""Utils for the data module."""

import math

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
        assert (
            -90 <= lat <= 90
        ), f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        assert (
            -180 <= lon <= 180
        ), f"lon out of range ({lon}). Make sure you are in EPSG:4326"

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
    """
    return 10 * np.log10(data)
