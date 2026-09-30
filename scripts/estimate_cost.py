#!/usr/bin/env python3
"""Run cost and wavefield output size of a Gmsh mesh for a scenario.

The time step of every tetrahedron (SeisSol's CFL rule, Vp of its unit) gives the
rate-2 LTS clusters and the element updates per simulated second. The cost is
updates x flops per update / sustained flop rate, times an overhead. The defaults
are explained in docs/resolution.md.
"""

import argparse
from pathlib import Path

import numpy as np

from decatur.config import load_paths, work_dir
from decatur.mesh import TAG_DYNAMIC_RUPTURE, TAG_VOLUME, read_msh, tet_volume_inradius
from decatur.resolution import element_timestep, lts_clusters, updates_per_second
from decatur.scenario import load_scenario


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--mesh", help="default: WORK_DIR/<scenario>/mesh.msh")
    p.add_argument("--order", type=int, default=5, help="convergence order (default: %(default)s)")
    p.add_argument("--end-time", type=float, help="simulated time (default: the scenario's)")
    p.add_argument("--flops-per-update", type=float, default=3.15e5,
                   help="hardware flops per element update (default: %(default)g)")
    p.add_argument("--gflops-per-core", type=float, default=9.5,
                   help="sustained hardware GFLOP/s per core (default: %(default)g)")
    p.add_argument("--overhead", type=float, default=1.3,
                   help="factor for the LTS normalization, I/O and setup (default: %(default)g)")
    p.add_argument("--cores-per-node", type=int, default=192, help="default: %(default)s")
    p.add_argument("--box", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                   help="wavefield output region (default: the scenario's OutputRegionBounds)")
    p.add_argument("--fields", type=int, default=3,
                   help="wavefield quantities written (default: %(default)s, velocities)")
    p.add_argument("--interval", type=float,
                   help="wavefield output interval, s (default: the scenario's)")
    p.add_argument("--data-dir", help="horizon .ts files (default: DATA_DIR in config/paths.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    sc = load_scenario(args.scenario)
    mesh = Path(args.mesh) if args.mesh else work_dir(sc.name) / "mesh.msh"
    model = sc.layered_model(args.data_dir or load_paths().get("DATA_DIR"))
    end_time = args.end_time or float(sc.raw["run"]["end_time"])

    xyz, groups = read_msh(mesh)
    tets = groups[TAG_VOLUME]
    centers = xyz[tets].mean(axis=1)
    _, r_in = tet_volume_inradius(xyz, tets)
    vp = np.array([u.vp for u in sc.units])[model.unit_at(centers) - 1]
    dt = element_timestep(r_in, vp, args.order, float(sc.raw["run"]["cfl"]))
    clusters, dt_min = lts_clusters(dt)
    updates = updates_per_second(clusters, dt_min)
    print(f"{mesh}: {len(tets):,} tets, {len(groups[TAG_DYNAMIC_RUPTURE]):,} fault faces")
    print(f"order {args.order}: dt_min {dt_min:.3e} s, {clusters.max() + 1} LTS clusters "
          f"{np.bincount(clusters).tolist()}")
    print(f"{updates:.3e} element updates per simulated second, "
          f"{updates * dt_min / len(tets):.4f} of global time stepping")

    core_s = (updates * end_time * args.flops_per_update * args.overhead
              / (args.gflops_per_core * 1e9))
    node_h = core_s / args.cores_per_node / 3600
    print(f"{end_time:g} s simulated: {core_s / 3600:,.0f} core-hours = {node_h:.1f} node-hours "
          f"({args.cores_per_node} cores per node, perfect scaling)")

    box = args.box or sc.wavefield_bounds()
    if box or int(sc.raw["outputs"]["wavefield"]):
        x0, x1, y0, y1, z0, z1 = box or (-np.inf, np.inf) * 3
        c = centers
        inside = ((c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)
                  & (c[:, 2] >= z0) & (c[:, 2] <= z1))
        interval = args.interval or float(sc.raw["outputs"]["wavefield_interval"])
        snapshots = int(np.floor(end_time / interval)) + 1
        size = inside.sum() * args.fields * 8 * snapshots
        print(f"wavefield: {inside.sum():,} cells in the box x {args.fields} fields x "
              f"{snapshots} snapshots = {size / 1e9:.1f} GB (double precision)")


if __name__ == "__main__":
    main()
