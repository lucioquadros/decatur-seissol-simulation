import numpy as np
import pandas as pd
import pytest

from decatur.mesh import MeshOptions, build_mesh, corners_outside, domain_bounds, tet_quality

UNIT_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def test_tet_quality_unit_tet():
    q = tet_quality(UNIT_TET, np.array([[0, 1, 2, 3]]))
    assert q["volume_min"] == pytest.approx(1 / 6)
    # r = 3V / A = 0.5 / (1.5 + sqrt(3)/2)
    assert q["insphere_min"] == pytest.approx(0.5 / (1.5 + np.sqrt(3) / 2))
    assert q["inverted"] == 0


def test_tet_quality_flags_inverted_and_sliver():
    xyz = np.vstack([UNIT_TET, [[0.5, 0.5, 1e-7]]])
    cells = np.array([[0, 1, 2, 3]] * 10 + [[0, 2, 1, 3], [0, 1, 2, 4]])
    q = tet_quality(xyz, cells)
    assert q["inverted"] == 1
    assert q["slivers"] == 1
    assert q["timestep_ratio"] > 1e5


def one_fault():
    return pd.DataFrame([{"name": "F", "centroid_x": 1000.0, "centroid_y": 2000.0,
                          "centroid_z": -1500.0, "x_min": 900.0, "x_max": 1100.0,
                          "y_min": 1800.0, "y_max": 2200.0, "depth_bottom_m": 1700.0,
                          "strike_deg": 10.0, "dip_deg": 70.0,
                          "extent_strike_m": 400.0, "extent_dip_m": 300.0}])


def test_domain_bounds():
    d = domain_bounds(one_fault(), (1000.0, 2000.0), 500.0, 1000.0)
    assert d == (-600.0, -700.0, -2700.0, 600.0, 700.0, 0.0)
    assert not corners_outside(np.array([[0, 0, -1500]]), d)
    assert corners_outside(np.array([[0, 0, 10]]), d)


def test_coarse_mesh_builds(tmp_path):
    pytest.importorskip("gmsh")
    opts = MeshOptions(lc_fault=100.0, lc_domain=400.0, dist_min=100.0, dist_max=600.0,
                       buffer=500.0, depth_buffer=500.0, nuc_center=(0.0, 0.0, -1500.0),
                       lc_nuc=50.0, nuc_radius=60.0, nuc_thickness=200.0, threads=2)
    stats = build_mesh(one_fault(), (1000.0, 2000.0), opts, tmp_path, log=lambda *a: None)
    assert stats["tets"] > 1000
    text = (tmp_path / "mesh.msh").read_text()
    assert text.startswith("$MeshFormat\n2.2")
    for tag in ("101", "103", "105"):
        assert f'2 {tag} "' in text
