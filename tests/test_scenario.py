import re

import numpy as np
import pytest
import yaml

from decatur.geometry import fault_row, point_on_fault, strike_dip_vectors
from decatur.mesh import domain_bounds
from decatur.scenario import BALL_MARGIN, load_scenario, render, scenario_path, write
from decatur.stress import stress_tensor
from conftest import SCENARIOS, load_easi, lua_functions

STRESS = ("s_xx", "s_xy", "s_yy", "s_zz")


def numeric_locals(lua_source: str) -> dict[str, float]:
    return {m[1]: float(m[2]) for m in
            re.finditer(r"local\s+(\w+)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", lua_source)}


def sample_points(sc, n=400, seed=0):
    """Random points around the faults, plus points well inside the patch disk."""
    rng = np.random.default_rng(seed)
    p = sc.patch
    s, d, nrm = strike_dip_vectors(p.strike, p.dip)
    r = rng.uniform(0, 0.95 * p.radius, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    off = rng.uniform(-0.9, 0.9, n) * p.normal_tol
    inside = (np.asarray(p.center) + (r * np.cos(phi))[:, None] * s
              + (r * np.sin(phi))[:, None] * d + off[:, None] * nrm)
    around = np.column_stack([rng.uniform(-1500, 1500, n), rng.uniform(-1500, 1500, n),
                              rng.uniform(-3000, 100, n)])
    return np.vstack([inside, around])


@pytest.fixture(params=SCENARIOS)
def case(request):
    sc = load_scenario(request.param)
    return request.param, sc, render(sc)


def test_lua_constants_match_scenario(case):
    _, sc, files = case
    g, p = sc.gradients, sc.patch
    got = numeric_locals(lua_functions(files["fault.yaml"])[STRESS])
    assert got == {
        "grad_Sv": g.sv, "grad_SH": g.shmax, "grad_Sh": g.shmin, "grad_Pf": g.pf,
        "az_SH": g.shmax_azimuth, "x_nuc": p.center[0], "y_nuc": p.center[1],
        "z_nuc": p.center[2], "strike_deg": p.strike, "dip_deg": p.dip, "r_nuc": p.radius,
        "n_tol": p.normal_tol, "dP": p.dp, "dTau": p.dtau}


def test_friction_constants_match_scenario(case):
    _, sc, files = case
    key = ("mu_s", "mu_d", "d_c", "s_xz", "s_yz")
    got = load_easi(files["fault.yaml"])["value"][key]["value"]["map"]
    fr = sc.raw["friction"]
    assert {k: float(v) for k, v in got.items()} == {
        "mu_s": float(fr["mu_s"]), "mu_d": float(fr["mu_d"]), "d_c": float(fr["d_c"]),
        "s_xz": 0.0, "s_yz": 0.0}


def test_generated_lua_matches_stress_model(case, lua):
    _, sc, files = case
    src = lua_functions(files["fault.yaml"])
    pts = sample_points(sc)
    assert sc.patch.contains(pts).sum() >= 400
    sig = stress_tensor(pts, sc.gradients, sc.patch)
    for (x, y, z), s in zip(pts, sig):
        out = lua(src[STRESS], x, y, z)
        want = {"s_xx": s[0, 0], "s_xy": s[0, 1], "s_yy": s[1, 1], "s_zz": s[2, 2]}
        for k in STRESS:
            assert out[k] == pytest.approx(want[k], rel=1e-12, abs=1e-3), (k, x, y, z)


def unit_filters(easi_any: dict, sc) -> list:
    """The three filter components (grid, above, below), checked against the grid axes."""
    x, y, z = sc.material_domain()
    grid, above, below = easi_any["value"]["components"]
    boxes = [c["value"]["limits"] for c in (grid, above, below)]
    for box in boxes:
        assert (box["x"], box["y"]) == ([x[0], x[-1]], [y[0], y[-1]])
    assert [b["z"] for b in boxes] == [[z[0], z[-1]], [z[-1], np.inf], [-np.inf, z[0]]]
    asagi = grid["value"]["components"]
    assert asagi["tag"] == "ASAGI"
    assert asagi["value"] | {"components": None} == {
        "file": "material.nc", "parameters": ["unit_id"], "var": "unit_id",
        "interpolation": "nearest", "components": None}
    return [grid, above, below]


def test_cohesion_by_unit(case, lua):
    _, sc, files = case
    src = lua_functions(files["fault.yaml"])[("cohesion",)]
    want = {u.id: float(sc.raw["friction"]["cohesion"][u.name]) for u in sc.units}
    for uid, c in want.items():
        assert lua(src, unit_id=uid + 1e-4)["cohesion"] == c
    _, above, below = unit_filters(load_easi(files["fault.yaml"])["value"][("cohesion",)], sc)
    assert above["value"]["components"]["value"]["map"] == {"cohesion": want[1]}
    assert below["value"]["components"]["value"]["map"] == {"cohesion": want[len(sc.units)]}


def test_unknown_cohesion_unit_is_rejected(tmp_path):
    raw = yaml.safe_load(scenario_path("bob_will").read_text())
    raw["friction"]["cohesion"]["Basement"] = raw["friction"]["cohesion"].pop("Precambrian")
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Precambrian"):
        load_scenario(path)


def test_forced_rupture_is_off(case, lua):
    _, sc, files = case
    src = lua_functions(files["fault.yaml"])[
        ("forced_rupture_time", "Tnuc_n", "Tnuc_s", "Tnuc_d")]
    for x, y, z in sample_points(sc, n=20):
        assert lua(src, x, y, z) == {"forced_rupture_time": 1e10, "Tnuc_n": 0.0,
                                     "Tnuc_s": 0.0, "Tnuc_d": 0.0}
    assert ("dynamicrupture", "t_0") not in parse_par(files["parameters.par"])


def scenario_with(tmp_path, **forced_rupture):
    raw = yaml.safe_load(scenario_path("bob_will").read_text())
    raw["forced_rupture"] = forced_rupture
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_scenario(path)


def test_forced_rupture_on(tmp_path, lua):
    sc = scenario_with(tmp_path, radius=40.0, time=0.1, t_0=0.5)
    files = render(sc)
    assert parse_par(files["parameters.par"])[("dynamicrupture", "t_0")] == [0.5]
    src = lua_functions(files["fault.yaml"])[
        ("forced_rupture_time", "Tnuc_n", "Tnuc_s", "Tnuc_d")]
    x, y, z = sc.patch.center
    s, _, _ = strike_dip_vectors(sc.patch.strike, sc.patch.dip)
    assert lua(src, x, y, z)["forced_rupture_time"] == 0.1
    assert lua(src, *(np.array(sc.patch.center) + 39.0 * s))["forced_rupture_time"] == 0.1
    assert lua(src, *(np.array(sc.patch.center) + 41.0 * s))["forced_rupture_time"] == 1e10


def test_forced_rupture_needs_t0(tmp_path):
    with pytest.raises(ValueError):
        render(scenario_with(tmp_path, radius=40.0, time=0.1))


def test_material_matches_units(case, lua):
    _, sc, files = case
    src = lua_functions(files["material.yaml"])[("rho", "mu", "lambda")]
    for u in sc.units:
        m = lua(src, unit_id=float(u.id))
        assert m == pytest.approx({"rho": u.rho, "mu": u.mu, "lambda": u.lam}, rel=1e-6)
        assert (m["lambda"] + 2 * m["mu"]) / m["rho"] == pytest.approx(u.vp ** 2, rel=1e-6)
    _, above, below = unit_filters(load_easi(files["material.yaml"]), sc)
    top, bottom = sc.units[0], sc.units[-1]
    assert above["value"]["components"]["value"]["map"] == pytest.approx(
        {"rho": top.rho, "mu": top.mu, "lambda": top.lam}, rel=1e-6)
    assert below["value"]["components"]["value"]["map"] == pytest.approx(
        {"rho": bottom.rho, "mu": bottom.mu, "lambda": bottom.lam}, rel=1e-6)


def parse_par(text: str) -> dict:
    out, group = {}, None
    for line in text.splitlines():
        line = line.split("!", 1)[0].strip()
        if line.startswith("&"):
            group = line[1:].lower()
        elif "=" in line:
            key, value = (t.strip() for t in line.split("=", 1))
            tokens = value.strip("'\"").split()
            try:
                tokens = [float(t) for t in tokens]
            except ValueError:
                pass
            out[(group, key.lower())] = tokens
    return out


def test_parameters_match_scenario(case):
    _, sc, files = case
    par = parse_par(files["parameters.par"])
    run, out = sc.raw["run"], sc.raw["outputs"]
    expected = {
        ("equations", "materialfilename"): ["material.yaml"],
        ("equations", "usecellhomogenizedmaterial"): [1],
        ("dynamicrupture", "modelfilename"): ["fault.yaml"],
        ("dynamicrupture", "fl"): [16],
        ("meshnml", "meshfile"): ["mesh.puml.hdf5"],
        ("abortcriteria", "endtime"): [run["end_time"]],
        ("discretization", "cfl"): [run["cfl"]],
        ("discretization", "clusteredlts"): [run["lts"]],
        ("elementwise", "printtimeinterval_sec"): [out["fault_interval"]],
        ("elementwise", "outputmask"): out["fault_mask"],
        ("elementwise", "refinement"): [out["fault_refinement"]],
        ("output", "energyoutputinterval"): [out["energy_interval"]],
        ("output", "wavefieldoutput"): [out["wavefield"]],
        ("output", "ioutputmask"): out["wavefield_mask"],
        ("output", "surfaceoutput"): [out["surface"]],
        ("output", "surfaceoutputinterval"): [out["surface_interval"]],
        ("output", "surfaceoutputrefinement"): [out["surface_refinement"]],
        ("output", "receiveroutput"): [out["receivers"]],
    }
    for key, value in expected.items():
        assert par[key] == [v if isinstance(v, str) else float(v) for v in value], key


def test_origin_is_the_all_faults_frame(case):
    _, sc, files = case
    origin = yaml.safe_load(files["origin.json"])
    assert (origin["origin_x"], origin["origin_y"]) == pytest.approx((104551.285, 357396.610))
    assert origin["origin_z"] == sc.raw["ground_elevation"] == 206.0


def test_patch_is_in_the_ground_frame(case):
    _, sc, _ = case
    p = sc.raw["patch"]
    raw = point_on_fault(fault_row(sc.inventory, p["fault"]), p["strike_offset"], p["dip_offset"])
    np.testing.assert_allclose(sc.patch.center, raw - np.array(sc.origin), atol=0.005)


def test_make_scenario_is_idempotent(case, tmp_path):
    name, _, files = case
    assert write(files, tmp_path)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert write(render(load_scenario(name)), tmp_path) == []
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == first


def test_mesh_options_follow_scenario(case):
    _, sc, _ = case
    m = sc.raw["mesh"]
    ball = m["nucleation_ball"]
    o = sc.mesh_options()
    for k in ("lc_fault", "lc_domain", "dist_min", "dist_max", "buffer", "depth_buffer"):
        assert o[k] == float(m[k]), k
    assert o["nuc_center"] == sc.patch.center
    assert o["nuc_radius"] == sc.patch.radius + float(ball["margin"])
    assert (o["nuc_thickness"], o["lc_nuc"]) == (float(ball["thickness"]), float(ball["lc"]))


def test_ball_follows_patch_radius(tmp_path):
    raw = yaml.safe_load(scenario_path("bob_will").read_text())
    raw["patch"]["radius"] = 100.0
    del raw["mesh"]["nucleation_ball"]["margin"]
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_scenario(path).mesh_options()["nuc_radius"] == 100.0 + BALL_MARGIN

    raw["mesh"]["nucleation_ball"]["radius"] = 90.0
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_scenario(path).mesh_options()


def test_refinement_box_around_the_faults(tmp_path):
    raw = yaml.safe_load(scenario_path("bob_will").read_text())
    assert "box" not in load_scenario("bob_will").mesh_options()
    raw["mesh"]["refinement_box"] = {"margin": 1000.0, "thickness": 800.0, "lc": 60.0}
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    sc = load_scenario(path)
    o = sc.mesh_options()
    assert o["box"] == domain_bounds(sc.faults, sc.origin, 1000.0, 1000.0)
    assert o["box"][5] == 0.0
    assert (o["box_thickness"], o["lc_box"]) == (800.0, 60.0)


def test_unknown_keys_are_rejected(tmp_path):
    raw = yaml.safe_load(scenario_path("bob_will").read_text())
    raw["mesh"]["nucleation_ball"]["margn"] = 30.0
    raw["extra"] = 1
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError) as err:
        load_scenario(path)
    assert "mesh.nucleation_ball.margn" in str(err.value)
    assert "extra" in str(err.value)
