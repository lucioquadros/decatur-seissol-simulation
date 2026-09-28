import numpy as np
import pytest

from decatur.ts_io import read_ts, read_tsurf, write_stl

TWO_SURFACES = """GOCAD TSurf 1
HEADER {
name:Plane A
*border:on
}
GOCAD_ORIGINAL_COORDINATE_SYSTEM
ZPOSITIVE Elevation
END_ORIGINAL_COORDINATE_SYSTEM
TFACE
VRTX 1 0 0 -100
VRTX 2 10 0 -100
PVRTX 3 0 10 -100 7.5
ATOM 4 2
VRTX 5 10 10 -100
TRGL 1 2 3
TRGL 4 5 3
END
GOCAD PLine 1
HEADER {
name:ignored
}
VRTX 1 0 0 0
END
GOCAD TSurf 1
HEADER {
name:Depth B
}
GOCAD_ORIGINAL_COORDINATE_SYSTEM
ZPOSITIVE Depth
END_ORIGINAL_COORDINATE_SYSTEM
TFACE
VRTX 0 0 0 250
VRTX 1 1 0 250
VRTX 2 0 1 250
TRGL 0 1 2
END
"""


@pytest.fixture
def ts_file(tmp_path):
    path = tmp_path / "two.ts"
    path.write_text(TWO_SURFACES)
    return path


def test_reads_all_tsurfs_and_skips_other_objects(ts_file):
    surfaces = read_ts(ts_file)
    assert [s.name for s in surfaces] == ["Plane A", "Depth B"]


def test_vertex_ids_and_atoms_map_to_rows(ts_file):
    a = read_ts(ts_file)[0]
    assert a.vertices.shape == (4, 3)
    np.testing.assert_array_equal(a.triangles, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_allclose(a.vertices[2], [0, 10, -100])


def test_depth_positive_is_flipped_to_z_up(ts_file):
    b = read_ts(ts_file)[1]
    np.testing.assert_allclose(b.vertices[:, 2], -250.0)


def test_read_tsurf_requires_one_surface(ts_file):
    with pytest.raises(ValueError):
        read_tsurf(ts_file)


def test_stl_has_one_solid_per_surface(ts_file, tmp_path):
    out = tmp_path / "out.stl"
    write_stl(out, read_ts(ts_file), offset=(10.0, 0.0, 0.0))
    text = out.read_text()
    assert text.count("endsolid") == 2
    assert text.count("endfacet") == 3
    assert "vertex -10.000000 0.000000 -100.000000" in text
