"""
Test suite for the nearest_haversine function.

This file uses pytest to verify a variety of behaviours of the
nearest_haversine utility, including correctness of computed
distances, handling of duplicate points, behaviour when N exceeds
the number of points, and consistency between chunked and unchunked
execution.
"""

import math
import numpy as np
import pytest

from helios.data.utils import nearest_haversine, haversine




def brute_force_neighbours(latlon, N, radius=6_371_000.0):
    """Brute-force computation of nearest N neighbours for each point.

    This helper function computes the pairwise haversine distances
    explicitly and returns the smallest N distances for each row,
    excluding self-distances. It is used to cross-check the
    nearest_haversine implementation on small datasets.

    Parameters
    ----------
    latlon : array-like, shape (M, 2)
        Sequence of latitude/longitude pairs in degrees.
    N : int
        Number of neighbours to return.
    radius : float
        Sphere radius in metres.

    Returns
    -------
    ndarray of shape (M, N_eff)
        Distances to the N_eff nearest neighbours (where N_eff = min(N, M-1)).
    ndarray of shape (M, N_eff)
        Indices of those neighbours in the original array.
    """
    arr = np.asarray(latlon, dtype=np.float64)
    M = arr.shape[0]
    N_eff = int(min(max(M - 1, 0), N))
    d = np.full((M, M), np.inf, dtype=np.float64)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            d[i, j] = haversine(arr[i], arr[j], radius=radius)
    # Select smallest N_eff distances per row
    idx_sorted = np.argsort(d, axis=1)
    selected_idx = idx_sorted[:, :N_eff]
    selected_dist = np.take_along_axis(d, selected_idx, axis=1)
    return selected_dist, selected_idx


def test_invalid_shape_raises():
    # latlon must be shape (M,2)
    with pytest.raises(ValueError):
        nearest_haversine([0, 1, 2], N=1)
    with pytest.raises(ValueError):
        nearest_haversine([[0, 1, 2], [3, 4, 5]], N=1)


def test_N_exceeds_M_minus_one():
    # When N >= M, only M-1 neighbours should be returned
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ]
    d = nearest_haversine(pts, N=10)
    # M-1 neighbours
    assert d.shape == (len(pts), len(pts) - 1)
    # Distances should be non-negative
    assert np.all(d >= 0.0)


def test_known_distances_london_paris():
    # Coordinates for London and Paris (degrees)
    london = [51.5074, -0.1278]
    paris = [48.8566, 2.3522]
    new_york = [40.7128, -74.0060]
    pts = [london, paris, new_york]
    # Compute nearest neighbours
    dists, idx = nearest_haversine(pts, N=2, return_indices=True)
    # London should have Paris as nearest neighbour
    london_idx = 0
    nearest_to_london = idx[london_idx, 0]
    assert nearest_to_london == 1  # Paris index
    london_paris_dist = dists[london_idx, 0]
    # Haversine distance between London and Paris is ~343 km
    assert math.isclose(london_paris_dist / 1e3, 343.0, rel_tol=0.1)
    # Paris should have London as nearest neighbour
    paris_idx = 1
    nearest_to_paris = idx[paris_idx, 0]
    assert nearest_to_paris == 0
    # New York should find whichever of London/Paris is nearer (Paris is nearer)
    ny_idx = 2
    assert idx[ny_idx, 0] in (0, 1)


def test_duplicate_points():
    # Duplicate points: first two rows are identical
    pts = [
        [10.0, 10.0],
        [10.0, 10.0],
        [20.0, 10.0],
    ]
    dists, idx = nearest_haversine(pts, N=2, return_indices=True)
    # For both duplicate rows, the nearest neighbour should be each other with distance 0
    assert dists[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert idx[0, 0] == 1
    assert dists[1, 0] == pytest.approx(0.0, abs=1e-9)
    assert idx[1, 0] == 0
    # The second neighbour for each duplicate should be the third point
    assert idx[0, 1] == 2
    assert idx[1, 1] == 2


def test_consistency_with_brute_force():
    # Generate a small random dataset and compare with brute force
    rng = np.random.default_rng(123)
    for m in range(3, 7):
        coords = rng.uniform(low=[-60, -120], high=[60, 120], size=(m, 2))
        for N in range(1, m):
            d_fast, idx_fast = nearest_haversine(coords, N=N, return_indices=True)
            d_brute, idx_brute = brute_force_neighbours(coords, N=N)
            # Distances and indices should match exactly for these small sets
            assert np.allclose(d_fast, d_brute, atol=1e-6)
            # Because there can be ties, sort both index lists before comparison
            assert np.all(np.sort(idx_fast, axis=1) == np.sort(idx_brute, axis=1))


def test_chunked_vs_full():
    # Larger random dataset to exercise chunked processing
    rng = np.random.default_rng(456)
    coords = rng.uniform(low=[-80, -170], high=[80, 170], size=(50, 2))
    N = 5
    # Compute without chunking
    d_full, idx_full = nearest_haversine(coords, N=N, return_indices=True)
    # Compute with chunking (small chunk size)
    d_chunked, idx_chunked = nearest_haversine(coords, N=N, return_indices=True, chunk_size=11)
    # Distances and indices should match (order matters)
    assert np.allclose(d_full, d_chunked, atol=1e-6)
    assert np.array_equal(idx_full, idx_chunked)