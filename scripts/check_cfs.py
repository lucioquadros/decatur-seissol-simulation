#!/usr/bin/env python3
"""Coulomb failure stress on every fault before a run, from a scenario file.

Background CFS must be < 0 on every fault, and > 0 inside the nucleation patch. 
Every fault that passes through the patch slab is listed and only the target 
may fail there.
"""

import argparse
import sys
from dataclasses import replace

import numpy as np

from decatur.geometry import fault_row, strike_dip_vectors
from decatur.scenario import load_scenario
from decatur.stress import cfs, fault_samples, points_in_patch, stress_tensor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--shmax", type=float, metavar="PA_PER_M",
                   help="override the SHmax gradient, e.g. to scan criticality")
    return p.parse_args()


def main():
    args = parse_args()
    sc = load_scenario(args.scenario)
    grad = sc.gradients
    if args.shmax is not None:
        grad = replace(grad, shmax=args.shmax)
    ox, oy = sc.origin

    print(f"{sc.name}: SHmax {grad.shmax / 1e3:.1f} MPa/km, mu_s {sc.mu_s}, "
          f"cohesion {sc.cohesion.above / 1e6:+.2f} / {sc.cohesion.below / 1e6:+.2f} MPa "
          f"above / below z = {sc.cohesion.z_switch:.0f} m\n")
    print(f"{'fault':24s}{'max CFS':>10s}{'mean':>9s}  (MPa)")
    failing = []
    for _, r in sc.faults.iterrows():
        pts = fault_samples((r.centroid_x - ox, r.centroid_y - oy, r.centroid_z),
                            r.strike_deg, r.dip_deg, r.extent_strike_m, r.extent_dip_m)
        _, _, n = strike_dip_vectors(r.strike_deg, r.dip_deg)
        c = cfs(stress_tensor(pts, grad), n, sc.mu_s, sc.cohesion(pts[:, 2])) / 1e6
        flag = "  FAILS WITHOUT THE PATCH" if c.max() > 0 else ""
        print(f"{r['name']:24s}{c.max():10.2f}{c.mean():9.2f}{flag}")
        if c.max() > 0:
            failing.append(r["name"])

    p = sc.patch
    _, _, n = strike_dip_vectors(p.strike, p.dip)
    center = np.array([p.center])
    coh = sc.cohesion(center[:, 2])
    bg = cfs(stress_tensor(center, grad), n, sc.mu_s, coh)[0] / 1e6
    wp = cfs(stress_tensor(center, grad, p), n, sc.mu_s, coh)[0] / 1e6
    row = fault_row(sc.inventory, sc.raw["patch"]["fault"])
    print(f"\npatch on {row['name']} at ({p.center[0]:.2f}, {p.center[1]:.2f}, {p.center[2]:.2f}):"
          f" CFS {bg:+.2f} MPa background, {wp:+.2f} MPa with dP {p.dp / 1e6:g} MPa,"
          f" dTau {p.dtau / 1e6:g} MPa")

    target = row["name"]
    print(f"\nfaults inside the patch slab (in-plane <= {p.radius:g} m, "
          f"off-plane <= {p.normal_tol:g} m):")
    triggered = []
    for _, r in sc.faults.iterrows():
        pts, cell = points_in_patch(p, (r.centroid_x - ox, r.centroid_y - oy, r.centroid_z),
                                    r.strike_deg, r.dip_deg, r.extent_strike_m, r.extent_dip_m)
        if not len(pts):
            continue
        _, _, nf = strike_dip_vectors(r.strike_deg, r.dip_deg)
        c = cfs(stress_tensor(pts, grad, p), nf, sc.mu_s, sc.cohesion(pts[:, 2])) / 1e6
        label = f"{r['name']} (target)" if r["name"] == target else r["name"]
        print(f"  {label:24s}{len(pts) * cell:9.0f} m2   max CFS with dP {c.max():+.2f} MPa")
        if r["name"] != target and c.max() > 0:
            triggered.append(r["name"])

    ok = not failing and not triggered and wp > 0
    if triggered:
        print(f"\nthe patch also fails other fault(s): {triggered}")
    if failing:
        print(f"\n{len(failing)} fault(s) fail spontaneously: {failing}")
    if wp <= 0:
        print("\nthe patch does not reach failure, raise dP or dTau")
    print("\nOK" if ok else "\nCHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
