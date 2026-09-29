#!/usr/bin/env python3
"""Export a scenario's Gmsh mesh and layer model to ParaView (.vtu) in the local frame.

  mesh.vtu        tetrahedra mesh, unit_id, unit names in material_units.csv
  faults.vtu      fault surfaces,  unit_id, fault_id (fault_ids.csv), patch (1 inside)
  interfaces.vtu  the unit top interfaces and extended beyond Petrel's horizons, unit_id
"""

import argparse
from pathlib import Path

import numpy as np

from decatur.config import load_paths, work_dir
from decatur.mesh import TAG_DYNAMIC_RUPTURE, TAG_VOLUME, nearest_fault, read_msh
from decatur.scenario import load_scenario
from decatur.vtk_io import VTK_QUAD, VTK_TETRA, VTK_TRIANGLE, compact, write_vtu


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--mesh", help="default: WORK_DIR/<scenario>/mesh.msh")
    p.add_argument("--outdir", help="default: the mesh's directory")
    p.add_argument("--box", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                   help="keep only the tets centered in this box in mesh.vtu")
    p.add_argument("--data-dir", help="horizon .ts files (default: DATA_DIR in config/paths.yaml)")
    return p.parse_args()


def interface_quads(model):
    """Quads of every unit top on the material grid, and the unit below each."""
    ny, nx = len(model.y), len(model.x)
    xx, yy = np.meshgrid(model.x, model.y)
    ij = np.arange(ny * nx).reshape(ny, nx)
    quad = np.stack([ij[:-1, :-1], ij[:-1, 1:], ij[1:, 1:], ij[1:, :-1]], axis=-1).reshape(-1, 4)
    points, cells, unit = [], [], []
    for k, top in enumerate(model.tops):
        points.append(np.column_stack([xx.ravel(), yy.ravel(), top.ravel()]))
        cells.append(quad + k * ny * nx)
        unit.append(np.full(len(quad), k + 2))
    return np.vstack(points), np.vstack(cells), np.concatenate(unit)


def main():
    args = parse_args()
    sc = load_scenario(args.scenario)
    mesh = Path(args.mesh) if args.mesh else work_dir(sc.name) / "mesh.msh"
    out = Path(args.outdir) if args.outdir else mesh.parent
    out.mkdir(parents=True, exist_ok=True)
    model = sc.layered_model(args.data_dir or load_paths().get("DATA_DIR"))

    xyz, groups = read_msh(mesh)
    tets = groups[TAG_VOLUME]
    centers = xyz[tets].mean(axis=1)
    total = len(tets)
    if args.box:
        lo, hi = np.array(args.box[0::2]), np.array(args.box[1::2])
        inside = np.all((centers >= lo) & (centers <= hi), axis=1)
        tets, centers = tets[inside], centers[inside]
    unit = model.unit_at(centers)
    write_vtu(out / "mesh.vtu", *compact(xyz, tets), VTK_TETRA, {"unit_id": unit})
    counts = ", ".join(f"{u.name} {np.sum(unit == u.id):,}" for u in sc.units)
    print(f"wrote {out / 'mesh.vtu'}: {len(tets):,} of {total:,} tets ({counts})")

    tris = groups[TAG_DYNAMIC_RUPTURE]
    centers = xyz[tris].mean(axis=1)
    faults = sc.faults
    fault_id = nearest_fault(centers, faults, sc.origin)
    pts, cells = compact(xyz, tris)
    write_vtu(out / "faults.vtu", pts, cells, VTK_TRIANGLE,
              {"fault_id": fault_id, "unit_id": model.unit_at(centers),
               "patch": sc.patch.contains(centers).astype(np.int32)})
    faults[["name"]].rename_axis("fault_id").to_csv(out / "fault_ids.csv")
    print(f"wrote {out / 'faults.vtu'}: {len(tris):,} triangles on {len(np.unique(fault_id))} "
          f"faults, fault_id names in {out / 'fault_ids.csv'}")

    pts, cells, below = interface_quads(model)
    write_vtu(out / "interfaces.vtu", pts, cells, VTK_QUAD, {"unit_id": below})
    print(f"wrote {out / 'interfaces.vtu'}: tops of "
          + ", ".join(u.name for u in sc.units[1:]))


if __name__ == "__main__":
    main()
