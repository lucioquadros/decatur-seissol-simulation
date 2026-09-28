import numpy as np
import pandas as pd
import pytest

from decatur.geometry import (best_fit_plane, fault_corners, local_origin, point_on_fault,
                              strike_dip_from_normal, strike_dip_vectors)

CASES = [(0.0, 45.0), (90.0, 30.0), (101.3, 72.7), (183.2, 72.9), (275.5, 64.9), (350.0, 89.0)]


@pytest.mark.parametrize("strike,dip", CASES)
def test_vectors_orthonormal_and_down_dip_descends(strike, dip):
    s, d, n = strike_dip_vectors(strike, dip)
    m = np.array([s, d, n])
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert d[2] < 0
    assert s[2] == 0


def test_known_vectors():
    s, d, n = strike_dip_vectors(0.0, 90.0)
    np.testing.assert_allclose(s, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(d, [0, 0, -1], atol=1e-12)
    # strike N, dip 30 -> dips toward East
    _, d, _ = strike_dip_vectors(0.0, 30.0)
    np.testing.assert_allclose(d, [np.cos(np.radians(30)), 0, -0.5], atol=1e-12)


def _plane_points(strike, dip, rng, center=(500.0, -200.0, -1800.0)):
    s, d, _ = strike_dip_vectors(strike, dip)
    a = rng.uniform(-400, 400, 300)
    b = rng.uniform(-150, 150, 300)
    return np.asarray(center) + a[:, None] * s + b[:, None] * d


@pytest.mark.parametrize("strike,dip", CASES)
def test_best_fit_plane_recovers_strike_and_dip(strike, dip):
    pts = _plane_points(strike, dip, np.random.default_rng(1))
    normal, along_strike, _, centroid = best_fit_plane(pts)
    got_strike, got_dip, got_dip_dir = strike_dip_from_normal(normal)
    assert got_dip == pytest.approx(dip, abs=1e-6)
    assert (got_strike - strike + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)
    assert got_dip_dir == pytest.approx((strike + 90) % 360, abs=1e-6)
    # in-plane axes rotate slightly with the sample layout
    assert abs(along_strike @ strike_dip_vectors(strike, dip)[0]) == pytest.approx(1, abs=1e-4)


def test_corners_of_vertical_ns_fault():
    c = fault_corners((0, 0, -1000), 0.0, 90.0, 200.0, 100.0)
    np.testing.assert_allclose(c, [[0, 100, -950], [0, -100, -950],
                                   [0, -100, -1050], [0, 100, -1050]], atol=1e-9)


@pytest.mark.parametrize("strike,dip", CASES)
def test_corners_lie_on_the_plane(strike, dip):
    c = fault_corners((10, 20, -1500), strike, dip, 300.0, 120.0)
    _, _, n = strike_dip_vectors(strike, dip)
    np.testing.assert_allclose((c - [10, 20, -1500]) @ n, 0, atol=1e-9)
    assert np.linalg.norm(c[0] - c[1]) == pytest.approx(300.0)
    assert np.linalg.norm(c[1] - c[2]) == pytest.approx(120.0)


def test_origin_and_point_on_fault():
    inv = pd.DataFrame({"x_min": [0, 50], "x_max": [10, 100], "y_min": [-20, 0], "y_max": [0, 40]})
    assert local_origin(inv) == (50.0, 10.0, 0.0)
    assert local_origin(inv, 206.0) == (50.0, 10.0, 206.0)
    row = pd.Series({"centroid_x": 1.0, "centroid_y": 2.0, "centroid_z": -100.0,
                     "strike_deg": 90.0, "dip_deg": 90.0})
    np.testing.assert_allclose(point_on_fault(row, 5.0, 10.0), [6, 2, -110], atol=1e-12)
