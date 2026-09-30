"""Scenario file loading and building of the SeisSol inputs."""

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import yaml

from .config import INVENTORY_CSV, MATERIAL_CSV, SCENARIOS_DIR, TEMPLATES_DIR
from .geometry import fault_row, local_origin, point_on_fault, select_faults
from .material import LayeredModel, Unit, axis, build_model, read_units, unit_at
from .mesh import domain_bounds
from .stress import Patch, StressGradients, stress_tensor

# keeps the patch edge, where rupture starts, inside the fully refined ball
BALL_MARGIN = 20.0

MeshSettings = dict[str, float | tuple[float, float, float]]


def _leaves(*names):
    return dict.fromkeys(names)


# Every key a scenario file may use.
SCHEMA = {
    "name": None,
    "faults": None,
    "ground_elevation": None,
    "mesh": {**_leaves("lc_fault", "lc_domain", "dist_min", "dist_max", "buffer",
                       "depth_buffer"),
             "nucleation_ball": _leaves("margin", "thickness", "lc"),
             "refinement_box": _leaves("margin", "thickness", "lc")},
    "material": _leaves("dx", "dz", "z_min", "z_max", "taper"),
    "friction": {**_leaves("mu_s", "mu_d", "d_c"), "cohesion": None},
    "stress": _leaves("sv", "shmax", "shmin", "pf", "shmax_azimuth"),
    "patch": _leaves("fault", "strike_offset", "dip_offset", "radius", "normal_tol", "dp",
                     "dtau"),
    "forced_rupture": _leaves("radius", "time", "t_0"),
    "run": _leaves("end_time", "cfl", "lts"),
    "outputs": _leaves("fault_interval", "fault_mask", "fault_refinement", "energy_interval",
                       "wavefield", "wavefield_mask", "surface", "surface_interval",
                       "surface_refinement", "receivers"),
}


def unknown_keys(raw: dict, schema: dict = SCHEMA, prefix: str = "") -> list[str]:
    """Dotted names of keys in raw that the schema doesn't know."""
    found = []
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if key not in schema:
            found.append(name)
        elif isinstance(schema[key], dict) and isinstance(value, dict):
            found += unknown_keys(value, schema[key], f"{name}.")
    return found


@dataclass
class Scenario:
    name: str
    raw: dict
    inventory: pd.DataFrame
    origin: tuple[float, float, float]
    gradients: StressGradients
    units: list[Unit]
    cohesion_by_unit: dict[int, float]
    patch: Patch

    @property
    def mu_s(self) -> float:
        return float(self.raw["friction"]["mu_s"])

    @property
    def faults(self) -> pd.DataFrame:
        return select_faults(self.inventory, self.raw.get("faults"))

    def cohesion(self, unit_ids) -> np.ndarray:
        return np.vectorize(self.cohesion_by_unit.__getitem__, otypes=[float])(unit_ids)

    def material_domain(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Material domain plus one cell (z_min..z_max)."""
        m, g = self.raw["mesh"], self.raw["material"]
        dx, dz = float(g["dx"]), float(g["dz"])
        x0, y0, _, x1, y1, _ = domain_bounds(self.faults, self.origin, float(m["buffer"]),
                                             float(m["depth_buffer"]))
        return (axis(x0 - dx, x1 + dx, dx), axis(y0 - dx, y1 + dx, dx),
                axis(float(g["z_min"]), float(g["z_max"]), dz))

    def layered_model(self, data_dir) -> LayeredModel:
        x, y, _ = self.material_domain()
        return build_model(self.units, x, y, self.origin, float(self.raw["material"]["taper"]),
                           data_dir)

    def mesh_options(self) -> MeshSettings:
        m = self.raw["mesh"]
        ball = m.get("nucleation_ball")
        keys = ("lc_fault", "lc_domain", "dist_min", "dist_max", "buffer", "depth_buffer")
        opts: MeshSettings = {k: float(m[k]) for k in keys}
        if ball:
            opts.update(nuc_center=self.patch.center, lc_nuc=float(ball["lc"]),
                        nuc_radius=self.patch.radius + float(ball.get("margin", BALL_MARGIN)),
                        nuc_thickness=float(ball["thickness"]))
        box = m.get("refinement_box")
        if box:
            margin = float(box["margin"])
            opts.update(box=domain_bounds(self.faults, self.origin, margin, margin),
                        box_thickness=float(box["thickness"]), lc_box=float(box["lc"]))
        return opts


def scenario_path(name_or_path) -> Path:
    p = Path(name_or_path)
    return p if p.suffix in (".yaml", ".yml") else SCENARIOS_DIR / str(name_or_path) / "scenario.yaml"


def _cohesion_by_unit(values: dict, units: list[Unit]) -> dict[int, float]:
    names = {u.name for u in units}
    if set(values) != names:
        raise ValueError(f"friction.cohesion must give every unit {sorted(names)}, "
                         f"got {sorted(values)}")
    return {u.id: float(values[u.name]) for u in units}


def load_scenario(name_or_path, inventory_csv=INVENTORY_CSV,
                  material_csv=MATERIAL_CSV) -> Scenario:
    path = scenario_path(name_or_path)
    raw = yaml.safe_load(path.read_text())
    unknown = unknown_keys(raw)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {', '.join(unknown)}")
    inventory = pd.read_csv(inventory_csv)
    origin = local_origin(select_faults(inventory, raw.get("faults")),
                          float(raw["ground_elevation"]))
    units = read_units(material_csv)

    s, p = raw["stress"], raw["patch"]
    gradients = StressGradients(float(s["sv"]), float(s["shmax"]), float(s["shmin"]),
                                float(s["pf"]), float(s["shmax_azimuth"]))
    cohesion = _cohesion_by_unit(raw["friction"]["cohesion"], units)

    row = fault_row(inventory, p["fault"])
    xyz = point_on_fault(row, float(p["strike_offset"]), float(p["dip_offset"])) - origin
    center = tuple(round(float(v), 2) for v in xyz)
    patch = Patch(center, float(row.strike_deg), float(row.dip_deg), float(p["radius"]),
                  float(p["normal_tol"]), float(p["dp"]), float(p["dtau"]))
    return Scenario(raw["name"], raw, inventory, origin, gradients, units, cohesion, patch)


def _num(v) -> str:
    return repr(float(v))


def _forced_rupture_t0(fr: dict) -> str:
    """The &DynamicRupture t_0 line, empty while forced rupture is off."""
    if float(fr["radius"]) <= 0.0:
        return ""
    t0 = float(fr.get("t_0", 0.0))
    if t0 <= 0.0:
        raise ValueError("forced_rupture.radius > 0 needs forced_rupture.t_0 > 0")
    return f"t_0 = {t0!r}".ljust(32) + "! forced-rupture weakening time, s\n"


def _unit_cases(units: list[Unit], returns, indent: int) -> str:
    """Lua lines returning each unit's values for its rounded unit_id."""
    pad = " " * indent
    lines = []
    for u in units:
        values = ", ".join(f"{k} = {v:.6e}" for k, v in returns(u).items())
        lines.append(f"{pad}if u == {u.id} then return {{ {values} }} end   -- {u.name}")
    return "\n".join(lines)


def _material_values(u: Unit) -> dict[str, float]:
    return {"rho": u.rho, "mu": u.mu, "lambda": u.lam}


def _flow_map(values: dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}: {v:.6e}" for k, v in values.items()) + "}"


def _values(sc: Scenario) -> dict[str, str]:
    r, g, p = sc.raw, sc.gradients, sc.patch
    fr, run, out = r["friction"], r["run"], r["outputs"]
    x, y, z = sc.material_domain()
    top, bottom = sc.units[0], sc.units[-1]
    return {
        "name": sc.name,
        "x0": _num(x[0]), "x1": _num(x[-1]), "y0": _num(y[0]), "y1": _num(y[-1]),
        "z0": _num(z[0]), "z1": _num(z[-1]),
        "top_unit": top.name, "bottom_unit": bottom.name,
        "material_cases": _unit_cases(sc.units, _material_values, 16),
        "material_top": _flow_map(_material_values(top)),
        "material_bottom": _flow_map(_material_values(bottom)),
        "cohesion_cases": _unit_cases(sc.units,
                                      lambda u: {"cohesion": sc.cohesion_by_unit[u.id]}, 18),
        "cohesion_top": _flow_map({"cohesion": sc.cohesion_by_unit[top.id]}),
        "cohesion_bottom": _flow_map({"cohesion": sc.cohesion_by_unit[bottom.id]}),
        "mu_s": _num(fr["mu_s"]), "mu_d": _num(fr["mu_d"]), "d_c": _num(fr["d_c"]),
        "grad_Sv": _num(g.sv), "grad_SH": _num(g.shmax), "grad_Sh": _num(g.shmin),
        "grad_Pf": _num(g.pf), "az_SH": _num(g.shmax_azimuth),
        "patch_fault": r["patch"]["fault"],
        "x_nuc": _num(p.center[0]), "y_nuc": _num(p.center[1]), "z_nuc": _num(p.center[2]),
        "strike_deg": _num(p.strike), "dip_deg": _num(p.dip),
        "r_nuc": _num(p.radius), "n_tol": _num(p.normal_tol),
        "dP": _num(p.dp), "dTau": _num(p.dtau),
        "r_seed": _num(r["forced_rupture"]["radius"]),
        "t_rupture": _num(r["forced_rupture"]["time"]),
        "forced_rupture_t0": _forced_rupture_t0(r["forced_rupture"]),
        "end_time": _num(run["end_time"]), "cfl": _num(run["cfl"]), "lts": str(int(run["lts"])),
        "fault_interval": _num(out["fault_interval"]),
        "fault_mask": " ".join(str(int(b)) for b in out["fault_mask"]),
        "fault_refinement": str(int(out["fault_refinement"])),
        "energy_interval": _num(out["energy_interval"]),
        "wavefield": str(int(out["wavefield"])),
        "wavefield_mask": " ".join(str(int(b)) for b in out["wavefield_mask"]),
        "surface": str(int(out["surface"])),
        "surface_interval": _num(out["surface_interval"]),
        "surface_refinement": str(int(out["surface_refinement"])),
        "receivers": str(int(out["receivers"])),
    }


def render(sc: Scenario, templates_dir=TEMPLATES_DIR) -> dict[str, str]:
    """Build content of every SeisSol input file."""
    values = _values(sc)
    files = {name: Template((Path(templates_dir) / name).read_text()).substitute(values)
             for name in ("fault.yaml", "parameters.par", "material.yaml")}
    files["origin.json"] = origin_json(sc.origin, sc.raw.get("faults"))
    return files


def easi_parameters(sc: Scenario, points, grid) -> dict[str, dict[str, np.ndarray]]:
    """What material.yaml and fault.yaml evaluate to at points (N, 3)."""
    ids = unit_at(points, grid, sc.units[0].id, sc.units[-1].id)

    def by_unit(value) -> np.ndarray:
        table = np.full(max(u.id for u in sc.units) + 1, np.nan)
        for u in sc.units:
            table[u.id] = value(u)
        return table[ids]

    material = {k: by_unit(lambda u, k=k: _material_values(u)[k]) for k in ("rho", "mu", "lambda")}
    sig = stress_tensor(points, sc.gradients, sc.patch)
    fr = sc.raw["friction"]
    n = len(ids)
    fault = {"cohesion": by_unit(lambda u: sc.cohesion_by_unit[u.id]),
             "s_xx": sig[:, 0, 0], "s_xy": sig[:, 0, 1], "s_yy": sig[:, 1, 1], "s_zz": sig[:, 2, 2],
             "s_xz": np.zeros(n), "s_yz": np.zeros(n),
             **{k: np.full(n, float(fr[k])) for k in ("mu_s", "mu_d", "d_c")}}
    return {"material.yaml": material, "fault.yaml": fault}


def origin_json(origin, faults=None) -> str:
    return json.dumps({"origin_x": origin[0], "origin_y": origin[1], "origin_z": origin[2],
                       "faults": faults or "all",
                       "note": "local = raw - origin, origin_z is the ground elevation"},
                      indent=2) + "\n"


def write(files: dict[str, str], outdir) -> list[Path]:
    """Write files, touching only those whose content changed."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    changed = []
    for name, text in files.items():
        path = outdir / name
        if not path.is_file() or path.read_text() != text:
            path.write_text(text)
            changed.append(path)
    return changed
