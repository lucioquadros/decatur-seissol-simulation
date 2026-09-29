import re

import numpy as np
import pytest

from conftest import SCENARIOS
from decatur.resolution import (ERROR_LIMITS, WOLLHERR_FITS, cohesive_zone_width,
                                critical_radius, element_timestep, lts_clusters,
                                max_fault_element_size, max_frequency, read_fault_output,
                                rupture_errors, static_cohesive_zone, strength_parameter,
                                threshold_size, updates_per_second)
from decatur.scenario import load_scenario, render


def test_cohesive_zone_width_only_where_ruptured():
    w = cohesive_zone_width(ds=[0.5, 0.0, 0.3], rt=[0.4, 0.2, 0.0], vr=[2000.0, 2000.0, 2000.0])
    assert w[0] == pytest.approx(200.0)
    assert np.isnan(w[1]) and np.isnan(w[2])


def test_static_cohesive_zone_hand_value():
    # 9 pi / 32 * 30e9 * 0.02 / (0.15 * 40e6) = 88.36 m
    assert static_cohesive_zone(30e9, 0.02, 0.15, -40e6) == pytest.approx(88.357, rel=1e-4)


def test_critical_radius_hand_value():
    # 7 pi / 24 * 30e9 * 5e6 * 0.02 / (2e6)^2 = 687.2 m
    assert critical_radius(30e9, 0.02, 20e6, 23e6, 18e6) == pytest.approx(687.22, rel=1e-4)


def test_strength_parameter_hand_value():
    # (23 - 20) / (20 - 18) = 1.5, and S < 0 when tau_0 exceeds tau_s
    np.testing.assert_allclose(strength_parameter([20e6, 24e6], 23e6, 18e6), [1.5, -1 / 6])


@pytest.mark.parametrize("order", sorted(WOLLHERR_FITS))
def test_max_fault_element_size_meets_every_limit(order):
    h = max_fault_element_size(order, 50.0)
    ratios = [rupture_errors(order, h, 50.0)[k] / ERROR_LIMITS[k] for k in ERROR_LIMITS]
    assert max(ratios) == pytest.approx(1.0)
    assert all(v <= ERROR_LIMITS[k] for k, v in rupture_errors(order, 0.9 * h, 50.0).items())


def test_max_fault_element_size_scales_with_cohesive_zone():
    assert max_fault_element_size(5, 20.0) == pytest.approx(2 * max_fault_element_size(5, 10.0))


def test_threshold_size_ramp():
    np.testing.assert_allclose(threshold_size([0.0, 200.0, 1600.0, 3000.0, 9000.0],
                                              30.0, 500.0, 200.0, 3000.0),
                               [30.0, 30.0, 265.0, 500.0, 500.0])


def test_max_frequency():
    assert max_frequency(3000.0, 100.0, 3.0) == pytest.approx(10.0)


def test_element_timestep_uses_order():
    assert element_timestep(10.0, 5000.0, 5, 0.5) == pytest.approx(0.5 * 20.0 / (5000.0 * 9))


def test_lts_clusters_rate_two():
    clusters, dt_min = lts_clusters([1.0, 1.9, 2.0, 4.1, 8.0])
    assert dt_min == 1.0
    np.testing.assert_array_equal(clusters, [0, 0, 1, 2, 3])
    assert updates_per_second(clusters, dt_min) == pytest.approx(1 + 1 + 0.5 + 0.25 + 0.125)


def test_read_fault_output(tmp_path):
    h5py = pytest.importorskip("h5py")
    xyz = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, -3]], dtype=float)
    with h5py.File(tmp_path / "run-fault_vertex.h5", "w") as f:
        f["mesh0/geometry"] = xyz
    with h5py.File(tmp_path / "run-fault_cell.h5", "w") as f:
        f["mesh0/connect"] = np.array([[0, 1, 2], [1, 2, 3]])
        f["mesh0/RT"] = np.array([[0.0, 0.0], [0.1, 0.2]])
    centers, fields = read_fault_output(tmp_path / "run-fault.xdmf", ["RT"])
    np.testing.assert_allclose(centers, [[1, 1, 0], [2, 2, -1]])
    np.testing.assert_allclose(fields["RT"], [0.1, 0.2])


@pytest.mark.parametrize("name", SCENARIOS)
def test_friction_law_is_linear_slip_weakening(name):
    par = render(load_scenario(name))["parameters.par"]
    law = re.search(r"^\s*FL\s*=\s*(\d+)", par, re.M | re.I)
    assert law and int(law.group(1)) == 16, (
        "static_cohesive_zone (Day et al. 2005, Eq. 30a) and the Wollherr et al. (2018) "
        "fits in decatur.resolution assume linear slip-weakening (FL = 16). Revisit "
        "estimate_resolution.py before changing the friction law.")
