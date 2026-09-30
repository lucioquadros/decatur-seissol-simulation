#!/usr/bin/env python3
"""Compare evaluate_easi output with the Python model of the scenario.

evaluate_easi (SeisSol Meshing/evaluate_material) evaluates material.yaml and
fault.yaml with the real easi/ASAGI at every tetrahedron barycenter:

  evaluate_easi -m mesh.puml.hdf5 -e material.yaml -o easi_material
  evaluate_easi -m mesh.puml.hdf5 -e fault.yaml -o easi_fault

This script reads easi_material.h5 and easi_fault.h5 from the scenario directory
and compares each parameter with material.nc, data/material_units.csv and
stress.py at the same points.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from decatur.config import work_dir
from decatur.material import read_unit_grid
from decatur.scenario import easi_parameters, load_scenario

OUTPUTS = {"material.yaml": "easi_material.h5", "fault.yaml": "easi_fault.h5"}


def read_evaluated(path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Barycenters (N, 3) and every per-cell parameter of an evaluate_easi .h5 file."""
    import h5py
    with h5py.File(path, "r") as f:
        xyz, cells = np.asarray(f["geometry"]), np.asarray(f["connect"])
        params = {k: np.asarray(f[k]) for k in f if k not in ("geometry", "connect", "group")}
    return xyz[cells].mean(axis=1), params


def mismatches(got, want, rtol: float, atol: float) -> tuple[int, float]:
    bad = ~np.isclose(got, want, rtol=rtol, atol=atol)
    return int(bad.sum()), float(np.max(np.abs(got - want), initial=0.0))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--dir", help="scenario directory (default: WORK_DIR/<scenario>)")
    p.add_argument("--max-fraction", type=float, default=1e-4,
                   help="allowed fraction of mismatched cells per parameter (default: %(default)g)")
    args = p.parse_args()

    sc = load_scenario(args.scenario)
    d = Path(args.dir) if args.dir else work_dir(sc.name)
    grid = read_unit_grid(d / "material.nc")
    present = {y: h for y, h in OUTPUTS.items() if (d / h).is_file()}
    if not present:
        sys.exit(f"no evaluate_easi output ({', '.join(OUTPUTS.values())}) in {d}")
    failed = False
    for yaml_name, h5_name in present.items():
        points, got = read_evaluated(d / h5_name)
        want = easi_parameters(sc, points, grid)[yaml_name]
        print(f"{yaml_name} at {len(points):,} barycenters ({h5_name})")
        for name, expected in want.items():
            if name not in got:
                print(f"  {name:10s} not in the easi output")
                failed = True
                continue
            n_bad, max_abs = mismatches(got[name], expected, rtol=1e-6, atol=1e-3)
            frac = n_bad / len(points)
            flag = "FAIL" if frac > args.max_fraction else "ok"
            failed |= flag == "FAIL"
            print(f"  {name:10s} {flag:4s} {n_bad:,} mismatched ({frac:.1e}), "
                  f"max |diff| {max_abs:.3g}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
