#!/usr/bin/env python3
"""On-fault element size and highest resolved frequency of a scenario.

  1. static cohesive zone per unit on the scenario's faults
  2. cohesive zone (DS - RT) * Vr measured in a fault output
  3. largest fault element size within the Wollherr et al. (2018) error limits
  4. nucleation patch against the Day (1982) critical radius
  5. highest frequency along the mesh size fields

Assumes linear slip-weakening (FL = 16): the static cohesive zone (Day et al. 2005,
Eq. 30a) and the Wollherr et al. (2018). Not applicable for rate-and-state friction law.
"""

import argparse

import numpy as np
import pandas as pd

from decatur.config import load_paths
from decatur.geometry import fault_row, local_centroid, strike_dip_vectors
from decatur.resolution import (ERROR_LIMITS, GMSH_EDGE_RATIO, cohesive_zone_width,
                                critical_radius, max_fault_element_size, max_frequency,
                                read_fault_output, rupture_errors, static_cohesive_zone,
                                strength_parameter, threshold_size)
from decatur.scenario import load_scenario
from decatur.stress import fault_samples, resolve, stress_tensor

# as reported in SeisSol
PERCENTILES = (5, 10, 50) 
# measured / static cohesive zone at the 5th percentile for reference runs in 
# bob_will and rick faults. Used for approximations of "lc_fault" values before the simulation.
# TO DO: double check if after new runs.
DYNAMIC_CONTRACTION = 0.5
# lowest background S of the Galis et al. (2015) critical patch sizes
GALIS_MIN_S = 0.75


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--fault-output", metavar="XDMF",
                   help="fault output of a run of this scenario, e.g. output/decatur-fault.xdmf")
    p.add_argument("--cohesive-zone", type=float, metavar="M",
                   help="design minimum cohesive zone (default: measured 5th percentile, "
                        f"else {DYNAMIC_CONTRACTION:g} x the static one)")
    p.add_argument("--order", type=int, default=5, help="convergence order (default: %(default)s)")
    p.add_argument("--sizes", type=float, nargs="+", default=[30.0, 20.0, 15.0, 10.0],
                   metavar="M", help="fault element sizes to evaluate (default: %(default)s)")
    p.add_argument("--per-wavelength", type=float, default=4.0, metavar="N",
                   help="elements per shortest S wavelength (default: %(default)s)")
    p.add_argument("--data-dir", help="horizon .ts files (default: DATA_DIR in config/paths.yaml)")
    return p.parse_args()


def percentiles(values) -> str:
    return "  ".join(f"p{q} {np.percentile(values, q):6.1f}" for q in PERCENTILES)


def static_zones(sc, model) -> pd.DataFrame:
    fr = sc.raw["friction"]
    mu = np.array([u.mu for u in sc.units])
    parts = []
    for _, r in sc.faults.iterrows():
        pts = fault_samples(local_centroid(r, sc.origin), r.strike_deg, r.dip_deg,
                            r.extent_strike_m, r.extent_dip_m, 60, 60)
        _, _, n = strike_dip_vectors(r.strike_deg, r.dip_deg)
        sn, _ = resolve(stress_tensor(pts, sc.gradients), n)
        uid = model.unit_at(pts)
        width = static_cohesive_zone(mu[uid - 1], fr["d_c"], fr["mu_s"] - fr["mu_d"], sn)
        parts.append(pd.DataFrame({"unit": uid, "width": width}))
    return pd.concat(parts, ignore_index=True)


def measured_zones(sc, model, xdmf) -> pd.DataFrame:
    """Cohesive zone of every ruptured cell, without the nucleation region (2 patch radii)."""
    centers, f = read_fault_output(xdmf, ("DS", "RT", "Vr"))
    width = cohesive_zone_width(f["DS"], f["RT"], f["Vr"])
    far = np.linalg.norm(centers - np.asarray(sc.patch.center), axis=1) > 2 * sc.patch.radius
    keep = np.isfinite(width) & far
    return pd.DataFrame({"unit": model.unit_at(centers[keep]), "width": width[keep]})


def print_zones(title, df, units) -> None:
    print(f"{title}\n  {'all':14s}{percentiles(df.width)}   ({len(df):,} points)")
    for u in units:
        w = df.width[df.unit == u.id]
        if len(w):
            print(f"  {u.name:14s}{percentiles(w)}")


def tractions(sc, model, points, normal, patch=None):
    """Shear traction, static and dynamic strength (Pa, positive) at points on a plane."""
    fr = sc.raw["friction"]
    sn, tau = resolve(stress_tensor(points, sc.gradients, patch), normal)
    coh = sc.cohesion(model.unit_at(points))
    return tau, fr["mu_s"] * -sn - coh, fr["mu_d"] * -sn - coh


def patch_nucleation(sc, model) -> None:
    p = sc.patch
    center = np.array([p.center])
    unit = sc.units[model.unit_at(center)[0] - 1]
    _, _, n = strike_dip_vectors(p.strike, p.dip)
    tau, tau_s, tau_d = (float(v[0]) for v in tractions(sc, model, center, n, p))
    rc = critical_radius(unit.mu, sc.raw["friction"]["d_c"], tau, tau_s, tau_d)
    print(f"\npatch in {unit.name}: radius {p.radius:g} m, Day critical radius {rc:.0f} m "
          f"(tau_0 {tau / 1e6:.2f}, tau_s {tau_s / 1e6:.2f}, tau_d {tau_d / 1e6:.2f} MPa, "
          f"with dP){'  BELOW CRITICAL' if p.radius < rc else ''}")

    row = fault_row(sc.inventory, sc.raw["patch"]["fault"])
    pts = fault_samples(local_centroid(row, sc.origin), row.strike_deg, row.dip_deg,
                        row.extent_strike_m, row.extent_dip_m, 60, 60)
    s_center = float(strength_parameter(*tractions(sc, model, center, n))[0])
    s_fault = strength_parameter(*tractions(sc, model, pts, n))
    inside = "inside" if s_center > GALIS_MIN_S else "outside"
    print(f"  background S {s_center:.2f} at the patch center, {s_fault.min():.2f} .. "
          f"{s_fault.max():.2f} on {row['name']}: {inside} the S > {GALIS_MIN_S:g} range "
          "of Galis et al. (2015)")
    ball = sc.raw["mesh"].get("nucleation_ball")
    lc = float(ball["lc"]) if ball else float(sc.raw["mesh"]["lc_fault"])
    print(f"  {2 * p.radius / lc:.0f} elements of {lc:g} m across the patch")


def frequencies(sc, per_wavelength: float) -> None:
    o = sc.mesh_options()
    lc = [o[k] for k in ("lc_fault", "lc_domain", "dist_min", "dist_max")]
    slowest = min(sc.units, key=lambda u: u.vs)
    surface = float(sc.faults.depth_top_m.min() + sc.origin[2])
    print(f"\nhighest frequency f = Vs / (n h), h = {GMSH_EDGE_RATIO:g} x Gmsh size "
          f"(mean tet edge), {slowest.name} Vs {slowest.vs:g} m/s:")
    print(f"  {'distance (m)':>14s}{'size (m)':>10s}{'h (m)':>8s}{'n = 2':>9s}"
          f"{f'n = {per_wavelength:g}':>9s}  (Hz, without the refinement box)")
    for d in sorted({0.0, lc[2], 1000.0, surface, 2000.0, lc[3]}):
        size = float(threshold_size(d, *lc))
        f2, fn = (float(max_frequency(slowest.vs, GMSH_EDGE_RATIO * size, k))
                  for k in (2.0, per_wavelength))
        label = f"{d:.0f}" + (" surface" if d == surface else "")
        print(f"  {label:>14s}{size:10.0f}{GMSH_EDGE_RATIO * size:8.0f}{f2:9.1f}{fn:9.1f}")
    size = float(threshold_size(surface, *lc))
    if "box" in o:
        size = min(size, o["lc_box"])
    f = float(max_frequency(sc.units[0].vs, GMSH_EDGE_RATIO * size, per_wavelength))
    print(f"  free surface nearest the faults ({surface:.0f} m, {sc.units[0].name}, "
          f"size {size:.0f} m{' in the box' if 'box' in o else ''}): {f:.1f} Hz, "
          f"sample surface and receivers every {1 / (2 * f):.3f} s or less")


def main():
    args = parse_args()
    sc = load_scenario(args.scenario)
    model = sc.layered_model(args.data_dir or load_paths().get("DATA_DIR"))

    static = static_zones(sc, model)
    print_zones(f"{sc.name}: static cohesive zone (m) on {len(sc.faults)} faults", static,
                sc.units)
    design = args.cohesive_zone
    if args.fault_output:
        measured = measured_zones(sc, model, args.fault_output)
        print_zones("\nmeasured cohesive zone (m), outside 2 patch radii", measured, sc.units)
        design = design or float(np.percentile(measured.width, 5))
    design = design or DYNAMIC_CONTRACTION * float(np.percentile(static.width, 5))

    limits = ", ".join(f"{k.replace('_', ' ')} {v:g}%" for k, v in ERROR_LIMITS.items())
    print(f"\ndesign minimum cohesive zone {design:.1f} m, order {args.order}: "
          f"fault size <= {max_fault_element_size(args.order, design):.1f} m for {limits}")
    print(f"  {'h (m)':>7s}{'zone / h':>15s}{'arrival %':>11s}{'PSR %':>8s}{'slip %':>8s}")
    for h in args.sizes:
        e = rupture_errors(args.order, h, design)
        print(f"  {h:7g}{design / h:15.2f}{e['rupture_arrival']:11.3f}"
              f"{e['peak_slip_rate']:8.2f}{e['final_slip']:8.2f}")

    patch_nucleation(sc, model)
    frequencies(sc, args.per_wavelength)


if __name__ == "__main__":
    main()
