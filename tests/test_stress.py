import numpy as np
import pytest

from decatur.geometry import strike_dip_vectors
from decatur.stress import (Patch, StepCohesion, StressGradients, cfs, points_in_patch, resolve,
                            stress_tensor)

GRAD = StressGradients(sv=25e3, shmax=47.7e3, shmin=22e3, pf=9.81e3, shmax_azimuth=90.0)


def test_tensor_principal_values_for_east_west_shmax():
    sig = GRAD.tensor(-1000.0)
    np.testing.assert_allclose(np.diag(sig), [-37.89e6, -12.19e6, -15.19e6])
    assert sig[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_tensor_above_surface_is_zero():
    np.testing.assert_array_equal(GRAD.tensor(50.0), 0.0)


def test_cfs_hand_calculation():
    # vertical fault striking N45E at 1 km, SHmax E-W: plane at 45 deg to both axes
    # sn = -(37.89 + 12.19)/2 = -25.04 MPa, tau = (37.89 - 12.19)/2 = 12.85 MPa
    # CFS = 12.85 - 0.6 * 25.04 - 0.5 = -2.674 MPa
    _, _, n = strike_dip_vectors(45.0, 90.0)
    sig = stress_tensor([[0.0, 0.0, -1000.0]], GRAD)
    sn, tau = resolve(sig, n)
    assert sn[0] == pytest.approx(-25.04e6)
    assert tau[0] == pytest.approx(12.85e6)
    assert cfs(sig, n, 0.6, -0.5e6)[0] == pytest.approx(-2.674e6)


def test_patch_lowers_normal_stress_by_dp_only():
    patch = Patch((0.0, 0.0, -1000.0), 45.0, 90.0, radius=70.0, normal_tol=25.0,
                  dp=2e6, dtau=0.0)
    _, _, n = strike_dip_vectors(45.0, 90.0)
    p = [[0.0, 0.0, -1000.0]]
    sn0, tau0 = resolve(stress_tensor(p, GRAD), n)
    sn1, tau1 = resolve(stress_tensor(p, GRAD, patch), n)
    assert sn1[0] - sn0[0] == pytest.approx(2e6)
    assert tau1[0] == pytest.approx(tau0[0])


def test_patch_disk_shape():
    patch = Patch((0.0, 0.0, -1000.0), 0.0, 90.0, radius=70.0, normal_tol=25.0,
                  dp=1.0, dtau=0.0)
    pts = np.array([[0, 69, -1000], [0, 71, -1000], [0, 0, -1069], [24, 0, -1000],
                    [26, 0, -1000], [0, 40, -1040]], dtype=float)
    np.testing.assert_array_equal(patch.contains(pts), [True, False, True, True, False, True])


def test_step_cohesion():
    c = StepCohesion(-1950.0, -0.5e6, -0.1e6)
    np.testing.assert_array_equal(c([-1949.0, -1950.0, -1951.0]), [-0.5e6, -0.5e6, -0.1e6])


def test_points_in_patch_areas():
    # vertical N-S patch, 70 m radius, 1 m tolerance
    patch = Patch((0.0, 0.0, -1000.0), 0.0, 90.0, radius=70.0, normal_tol=1.0,
                  dp=1.0, dtau=0.0)
    same, cell = points_in_patch(patch, (0.0, 0.0, -1000.0), 0.0, 90.0, 400.0, 400.0)
    assert len(same) * cell == pytest.approx(np.pi * 70.0 ** 2, rel=0.01)

    # vertical E-W fault through the center: a 140 m tall, 2 m wide strip
    cross, cell = points_in_patch(patch, (0.0, 0.0, -1000.0), 90.0, 90.0, 400.0, 400.0, n=2000)
    assert len(cross) * cell == pytest.approx(140.0 * 2.0, rel=0.02)

    far, _ = points_in_patch(patch, (500.0, 0.0, -1000.0), 0.0, 90.0, 400.0, 400.0)
    assert len(far) == 0
