#!/usr/bin/env python3
"""Gmsh tetrahedral mesh with each fault as a planar quad for SeisSol (via PUMGen).

Faults, sizes and the nucleation ball come from the scenario file. The size flags
override the scenario for resolution tests.
Tags: 101 free surface, 103 dynamic rupture, 105 absorbing, volume 1.
"""

import argparse
import os
import sys
from pathlib import Path

from decatur.config import INVENTORY_CSV, work_dir
from decatur.mesh import ALGORITHMS_3D, MeshOptions, build_mesh
from decatur.scenario import load_scenario, origin_json

SIZE_FLAGS = (("lc-fault", "element size on the faults"),
              ("lc-domain", "element size at the domain boundary"),
              ("dist-min", "distance from the faults where sizes start growing"),
              ("dist-max", "distance where sizes reach --lc-domain"),
              ("buffer", "horizontal buffer around the faults"),
              ("depth-buffer", "extra depth below the deepest fault"),
              ("lc-nuc", "element size inside the nucleation ball"),
              ("nuc-radius", "fully refined ball radius (scenario: patch radius + margin)"),
              ("nuc-thickness", "taper from --lc-nuc back to --lc-domain"))
BALL_FLAGS = ("lc_nuc", "nuc_radius", "nuc_thickness")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--inventory", default=str(INVENTORY_CSV), help="default: %(default)s")
    p.add_argument("--outdir", help="default: WORK_DIR/<scenario> from config/paths.yaml")
    for flag, help_ in SIZE_FLAGS:
        p.add_argument(f"--{flag}", type=float, metavar="M", help=help_)
    p.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)),
                   help="Gmsh threads (default: %(default)s)")
    p.add_argument("--algorithm3d", choices=ALGORITHMS_3D, default="delaunay",
                   help="hxt is parallel and faster for large meshes (default: %(default)s)")
    p.add_argument("--no-mesh", action="store_true", help="write geometry (.brep) only")
    p.add_argument("--verbose", action="store_true", help="full Gmsh output")
    return p.parse_args()


def main():
    args = parse_args()
    sc = load_scenario(args.scenario, args.inventory)
    opts = MeshOptions(threads=args.threads, algorithm3d=args.algorithm3d)
    for k, v in sc.mesh_options().items():
        setattr(opts, k, v)
    for flag, _ in SIZE_FLAGS:
        name = flag.replace("-", "_")
        if getattr(args, name) is not None:
            setattr(opts, name, getattr(args, name))

    if opts.nuc_center is None and any(getattr(args, f) is not None for f in BALL_FLAGS):
        sys.exit("--lc-nuc, --nuc-radius and --nuc-thickness need a nucleation_ball "
                 "in the scenario")
    if opts.dist_min >= opts.dist_max:
        sys.exit("--dist-min must be smaller than --dist-max")
    if opts.nuc_center is not None and opts.lc_nuc > opts.lc_fault:
        print("WARNING: --lc-nuc is larger than --lc-fault", file=sys.stderr)
    if opts.nuc_center is not None and opts.nuc_radius < sc.patch.radius:
        print(f"WARNING: ball radius {opts.nuc_radius:g} m is smaller than the patch radius "
              f"{sc.patch.radius:g} m, so the patch edge is not fully refined", file=sys.stderr)

    out = Path(args.outdir) if args.outdir else work_dir(sc.name)
    out.mkdir(parents=True, exist_ok=True)
    faults = sc.faults
    (out / "origin.json").write_text(origin_json(sc.origin, sc.raw.get("faults")))
    print(f"{sc.name}: {len(faults)} faults, origin ({sc.origin[0]:.3f}, {sc.origin[1]:.3f}), "
          f"sizes {opts.lc_fault:g} m on faults -> {opts.lc_domain:g} m "
          f"over {opts.dist_min:g}-{opts.dist_max:g} m")
    if opts.nuc_center is not None:
        c = opts.nuc_center
        print(f"nucleation ball at ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}): {opts.lc_nuc:g} m "
              f"within {opts.nuc_radius:g} m, taper {opts.nuc_thickness:g} m")

    stats = build_mesh(faults, sc.origin, opts, out, generate=not args.no_mesh,
                       verbose=args.verbose)
    if "tets" in stats:
        print(f"wrote {out / 'mesh.msh'}: {stats['tets']:,} tets, "
              f"{stats['triangles']:,} boundary/fault triangles")
        print(f"next: pumgen -s msh2 {out / 'mesh.msh'} {out / 'mesh.puml.hdf5'}")


if __name__ == "__main__":
    main()
