import numpy as np

from decatur.vtk_io import VTK_TETRA, VTK_TRIANGLE, compact, read_vtu, write_vtu

TETS = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
XYZ = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)


def test_vtu_round_trip(tmp_path):
    path = tmp_path / "t.vtu"
    write_vtu(path, XYZ, TETS, VTK_TETRA, {"unit_id": np.array([1, 4]),
                                           "value": np.array([0.5, -2.0])})
    a = read_vtu(path)
    np.testing.assert_array_equal(a["Points"], XYZ)
    np.testing.assert_array_equal(a["connectivity"], TETS.ravel())
    np.testing.assert_array_equal(a["offsets"], [4, 8])
    np.testing.assert_array_equal(a["types"], [VTK_TETRA] * 2)
    np.testing.assert_array_equal(a["unit_id"], [1, 4])
    np.testing.assert_array_equal(a["value"], [0.5, -2.0])
    assert path.read_bytes().rstrip().endswith(b"</VTKFile>")


def test_compact_keeps_used_points():
    pts, cells = compact(XYZ, np.array([[4, 1, 3]]))
    np.testing.assert_array_equal(pts, XYZ[[1, 3, 4]])
    np.testing.assert_array_equal(pts[cells], XYZ[[[4, 1, 3]]])


def test_triangles(tmp_path):
    path = tmp_path / "f.vtu"
    write_vtu(path, XYZ[:3], np.array([[0, 1, 2]]), VTK_TRIANGLE)
    assert read_vtu(path)["types"].tolist() == [VTK_TRIANGLE]
