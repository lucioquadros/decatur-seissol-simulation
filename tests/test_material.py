import numpy as np
import pandas as pd
import pytest
import yaml

from decatur.material import (LayeredModel, Unit, axis, build_model, check_units,
                              check_vertical_range, grid_surface, rasterize, read_unit_grid,
                              read_units, unit_at, write_unit_grid)
from decatur.scenario import load_scenario, render, scenario_path
from decatur.ts_io import TSurf
from conftest import data_dir, lua_functions

X = np.arange(0.0, 1001.0, 50.0)
Y = np.arange(0.0, 801.0, 50.0)


def tilted_square(x0=200.0, x1=600.0, y0=200.0, y1=500.0) -> TSurf:
    """z = -1000 + 0.1 x on a rectangle, two triangles plus an overlapping third."""
    v = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x1, (y0 + y1) / 2, 0]])
    v[:, 2] = -1000.0 + 0.1 * v[:, 0]
    return TSurf("tilted", v, np.array([[0, 1, 2], [0, 2, 3], [0, 4, 2]]))


def units(*tops) -> list[Unit]:
    names = ("A", "B", "C", "D")
    return [Unit(i + 1, names[i], t, 3000.0 + 500 * i, 1700.0 + 300 * i, 2300.0 + 100 * i)
            for i, t in enumerate(("surface",) + tops)]


def test_repo_units_are_valid():
    us = read_units()
    assert [u.name for u in us] == ["Eau Claire", "Mt. Simon", "Argenta", "Precambrian"]
    for u in us:
        assert u.mu > 0 and u.lam > 0 and u.vp > np.sqrt(2) * u.vs
    for u in us[:3]:
        assert u.vp / u.vs == pytest.approx(1.7, abs=0.005)


@pytest.mark.parametrize("bad", [
    [Unit(1, "A", "surface", 3000.0, 2200.0, 2400.0)],
    [Unit(2, "A", "surface", 3000.0, 1700.0, 2400.0)],
    [Unit(1, "A", "-100", 3000.0, 1700.0, 2400.0)],
    units("surface")])
def test_bad_units_are_rejected(bad):
    with pytest.raises(ValueError):
        check_units(bad)


def test_rasterize_is_exact_and_nan_outside():
    s = tilted_square()
    z = rasterize(s.vertices, s.triangles, X, Y)
    xx, yy = np.meshgrid(X, Y)
    inside = (xx >= 200) & (xx <= 600) & (yy >= 200) & (yy <= 500)
    np.testing.assert_array_equal(np.isfinite(z), inside)
    np.testing.assert_allclose(z[inside], -1000.0 + 0.1 * xx[inside], atol=1e-9)


def test_extension_blends_edge_into_mean():
    z = grid_surface(tilted_square(), X, Y, (0.0, 0.0, 0.0), taper=200.0)
    mean = np.mean(-1000.0 + 0.1 * X[(X >= 200) & (X <= 600)])
    row = list(Y).index(350.0)
    w = 0.5 - 0.5 * np.cos(np.pi * 50.0 / 200.0)
    assert z[row, list(X).index(650.0)] == pytest.approx((1 - w) * -940.0 + w * mean)
    assert z[row, list(X).index(800.0)] == pytest.approx(mean)
    assert z[row, list(X).index(1000.0)] == pytest.approx(mean)
    assert z[row, list(X).index(600.0)] == pytest.approx(-940.0)


def test_origin_shifts_the_surface():
    s = tilted_square()
    z = grid_surface(s, X, Y, (100.0, 0.0, 200.0), taper=200.0)
    ref = grid_surface(s, X + 100.0, Y, (0.0, 0.0, 0.0), taper=200.0)
    np.testing.assert_allclose(z, ref - 200.0)


def test_interfaces_are_ordered():
    tops = np.stack([np.full((len(Y), len(X)), -500.0),
                     np.tile(np.linspace(-900.0, -300.0, len(X)), (len(Y), 1))])
    m = LayeredModel(units("-", "-"), X, Y, tops)
    assert np.all(np.diff(m.tops, axis=0) <= 0)
    assert np.all(m.tops[1] <= -500.0)
    z = np.arange(-1000.0, 1.0, 10.0)
    assert np.all(np.diff(m.unit_grid(z), axis=0) <= 0)


def test_unit_lookup_and_interface_belongs_above():
    m = build_model(units("-1000", "-1200.0"), X, Y, (0.0, 0.0, 0.0), 100.0)
    pts = np.array([[500, 400, z] for z in (0.0, -999.0, -1000.0, -1001.0, -1200.0, -1300.0)])
    np.testing.assert_array_equal(m.unit_at(pts), [1, 1, 1, 2, 2, 3])
    z = np.array([-1300.0, -1100.0, -500.0])
    g = m.unit_grid(z)
    np.testing.assert_array_equal(g[:, 3, 4], [3, 2, 1])


def test_vertical_range_check():
    m = build_model(units("-1000", "-1200"), X, Y, (0.0, 0.0, 0.0), 100.0)
    check_vertical_range(m, -1300.0, -900.0, 10.0)
    with pytest.raises(ValueError):
        check_vertical_range(m, -1205.0, -900.0, 10.0)


def test_netcdf_round_trip(tmp_path):
    z = axis(-1300.0, -900.0, 10.0)
    uid = build_model(units("-1000", "-1200"), X, Y, (0.0, 0.0, 0.0), 100.0).unit_grid(z)
    path = tmp_path / "material.nc"
    write_unit_grid(path, X, Y, z, uid)
    x2, y2, z2, uid2 = read_unit_grid(path)
    np.testing.assert_array_equal(x2, X)
    np.testing.assert_array_equal(y2, Y)
    np.testing.assert_array_equal(z2, z)
    np.testing.assert_array_equal(uid2, uid)
    from netCDF4 import Dataset
    with Dataset(path) as nc:
        assert nc["unit_id"].dimensions == ("z", "y", "x")


def test_two_layer_reference_is_a_special_case(tmp_path, lua):
    """Mt. Simon over fractured Precambrian split at a flat elevation."""
    csv = tmp_path / "units.csv"
    pd.DataFrame([
        {"unit_id": 1, "unit": "Mt. Simon", "top": "surface", "vp": 2870, "vs": 1720,
         "rho": 2400, "source": "Bondarenko 2025"},
        {"unit_id": 2, "unit": "Precambrian", "top": "-1950", "vp": 5330, "vs": 3160,
         "rho": 2730, "source": "Bondarenko 2025"}]).to_csv(csv, index=False)
    raw = yaml.safe_load(scenario_path("bob_will").read_text())
    raw["ground_elevation"] = 0.0
    raw["material"].update(z_min=-2000.0, z_max=-1900.0)
    raw["friction"]["cohesion"] = {"Mt. Simon": -0.5e6, "Precambrian": -0.1e6}
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    sc = load_scenario(path, material_csv=csv)
    model = sc.layered_model(None)
    np.testing.assert_array_equal(model.unit_at([[0, 0, -1949.0], [0, 0, -1951.0]]), [1, 2])

    files = render(sc)
    src = lua_functions(files["material.yaml"])[("rho", "mu", "lambda")]
    # mu = rho Vs^2, lambda = rho (Vp^2 - 2 Vs^2)
    assert lua(src, unit_id=1.0) == pytest.approx(
        {"rho": 2400.0, "mu": 7.100e9, "lambda": 5.568e9}, rel=1e-3)
    assert lua(src, unit_id=2.0) == pytest.approx(
        {"rho": 2730.0, "mu": 2.726e10, "lambda": 2.303e10}, rel=1e-3)
    coh = lua_functions(files["fault.yaml"])[("cohesion",)]
    assert [lua(coh, unit_id=u)["cohesion"] for u in (1.0, 2.0)] == [-0.5e6, -0.1e6]


# CCS1 in the Petrel frame (Illinois East SPCS27, m), projected from the report's
# NAD83 lat/lon. Tops are depths below KB (KB = 210.46 m) from the IBDP final report.
CCS1_XY = (104487.7, 356482.0)
CCS1_KB = 210.46


@pytest.mark.skipif(data_dir() is None, reason="DATA_DIR with the horizon .ts files not set")
def test_units_at_ccs1_well_tops():
    sc = load_scenario("bob_will")
    model = sc.layered_model(data_dir())

    def unit(md_kb):
        z = CCS1_KB - md_kb - sc.origin[2]
        return int(model.unit_at([[CCS1_XY[0] - sc.origin[0], CCS1_XY[1] - sc.origin[1], z]])[0])

    assert [unit(d) for d in (1000.0, 1685.0, 1695.0, 2143.0, 2153.0, 2250.0)] == \
        [1, 1, 2, 2, 3, 4]


def test_unit_at_takes_nearest_node_and_top_bottom_off_grid():
    x, y, z = axis(0.0, 100.0, 50.0), axis(0.0, 100.0, 50.0), axis(-100.0, 0.0, 10.0)
    ids = np.broadcast_to(np.arange(len(z))[:, None, None] + 10, (len(z), len(y), len(x)))
    pts = np.array([[1.0, 99.0, -100.0], [74.0, 26.0, -55.1], [60.0, 0.0, -44.0],
                    [0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, -100.1]])
    got = unit_at(pts, (x, y, z, ids), top_id=1, bottom_id=4)
    np.testing.assert_array_equal(got, [10, 14, 16, 20, 1, 4])
