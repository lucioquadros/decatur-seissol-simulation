"""Scenario file loading and building of the SeisSol inputs."""

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

import pandas as pd
import yaml

from .config import INVENTORY_CSV, SCENARIOS_DIR, TEMPLATES_DIR
from .geometry import fault_row, local_origin, point_on_fault, select_faults
from .stress import Patch, StepCohesion, StressGradients

# keeps the patch edge, where rupture starts, inside the fully refined ball
BALL_MARGIN = 20.0

MeshSettings = dict[str, float | tuple[float, float, float]]


def _leaves(*names):
    return dict.fromkeys(names)


# Every key a scenario file may use.
SCHEMA = {
    "name": None,
    "faults": None,
    "mesh": {**_leaves("lc_fault", "lc_domain", "dist_min", "dist_max", "buffer",
                       "depth_buffer"),
             "nucleation_ball": _leaves("margin", "thickness", "lc")},
    "friction": {**_leaves("mu_s", "mu_d", "d_c"),
                 "cohesion": _leaves("z_switch", "above", "below")},
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
    origin: tuple[float, float]
    gradients: StressGradients
    cohesion: StepCohesion
    patch: Patch

    @property
    def mu_s(self) -> float:
        return float(self.raw["friction"]["mu_s"])

    @property
    def faults(self) -> pd.DataFrame:
        return select_faults(self.inventory, self.raw.get("faults"))

    def mesh_options(self) -> MeshSettings:
        m = self.raw["mesh"]
        ball = m.get("nucleation_ball")
        keys = ("lc_fault", "lc_domain", "dist_min", "dist_max", "buffer", "depth_buffer")
        opts: MeshSettings = {k: float(m[k]) for k in keys}
        if ball:
            opts.update(nuc_center=self.patch.center, lc_nuc=float(ball["lc"]),
                        nuc_radius=self.patch.radius + float(ball.get("margin", BALL_MARGIN)),
                        nuc_thickness=float(ball["thickness"]))
        return opts


def scenario_path(name_or_path) -> Path:
    p = Path(name_or_path)
    return p if p.suffix in (".yaml", ".yml") else SCENARIOS_DIR / str(name_or_path) / "scenario.yaml"


def load_scenario(name_or_path, inventory_csv=INVENTORY_CSV) -> Scenario:
    path = scenario_path(name_or_path)
    raw = yaml.safe_load(path.read_text())
    unknown = unknown_keys(raw)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {', '.join(unknown)}")
    inventory = pd.read_csv(inventory_csv)
    origin = local_origin(select_faults(inventory, raw.get("faults")))

    s, c, p = raw["stress"], raw["friction"]["cohesion"], raw["patch"]
    gradients = StressGradients(float(s["sv"]), float(s["shmax"]), float(s["shmin"]),
                                float(s["pf"]), float(s["shmax_azimuth"]))
    cohesion = StepCohesion(float(c["z_switch"]), float(c["above"]), float(c["below"]))

    row = fault_row(inventory, p["fault"])
    xyz = point_on_fault(row, float(p["strike_offset"]), float(p["dip_offset"]))
    center = (round(xyz[0] - origin[0], 2), round(xyz[1] - origin[1], 2), round(xyz[2], 2))
    patch = Patch(center, float(row.strike_deg), float(row.dip_deg), float(p["radius"]),
                  float(p["normal_tol"]), float(p["dp"]), float(p["dtau"]))
    return Scenario(raw["name"], raw, inventory, origin, gradients, cohesion, patch)


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


def _values(sc: Scenario) -> dict[str, str]:
    r, g, p = sc.raw, sc.gradients, sc.patch
    fr, run, out = r["friction"], r["run"], r["outputs"]
    return {
        "name": sc.name,
        "mu_s": _num(fr["mu_s"]), "mu_d": _num(fr["mu_d"]), "d_c": _num(fr["d_c"]),
        "z_switch": _num(sc.cohesion.z_switch),
        "cohesion_above": _num(sc.cohesion.above), "cohesion_below": _num(sc.cohesion.below),
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
             for name in ("fault.yaml", "parameters.par")}
    files["material.yaml"] = (Path(templates_dir) / "material.yaml").read_text()
    files["origin.json"] = origin_json(sc.origin, sc.raw.get("faults"))
    return files


def origin_json(origin, faults=None) -> str:
    return json.dumps({"origin_x": origin[0], "origin_y": origin[1],
                       "faults": faults or "all",
                       "note": "x_local = x - origin_x, y_local = y - origin_y, z unchanged"},
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
