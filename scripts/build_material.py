#!/usr/bin/env python3
"""Grid the material layered model of a scenario into material.nc.
Units and properties: data/material_units.csv.
"""

import argparse
from pathlib import Path

from decatur.config import load_paths, work_dir
from decatur.material import check_vertical_range, write_unit_grid
from decatur.scenario import load_scenario


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--data-dir", help="horizon .ts files (default: DATA_DIR in config/paths.yaml)")
    p.add_argument("--outdir", help="default: WORK_DIR/<scenario> from config/paths.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    sc = load_scenario(args.scenario)
    data_dir = args.data_dir or load_paths().get("DATA_DIR")
    model = sc.layered_model(data_dir)
    x, y, z = sc.material_domain()
    check_vertical_range(model, z[0], z[-1], z[1] - z[0])

    for u in sc.units:
        props = f"Vp {u.vp:g}  Vs {u.vs:g}  rho {u.rho:g}"
        if u.id == 1:
            print(f"  {u.id} {u.name:14s} top: free surface          {props}")
            continue
        lo, hi = model.top_range(u.id)
        print(f"  {u.id} {u.name:14s} top: z {lo:8.1f} .. {hi:8.1f} m   {props}")

    out = Path(args.outdir) if args.outdir else work_dir(sc.name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "material.nc"
    write_unit_grid(path, x, y, z, model.unit_grid(z))
    print(f"wrote {path}: {len(x)} x {len(y)} x {len(z)} cells, x {x[0]:g}..{x[-1]:g}, "
          f"y {y[0]:g}..{y[-1]:g}, z {z[0]:g}..{z[-1]:g} m, "
          f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
