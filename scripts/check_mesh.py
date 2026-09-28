#!/usr/bin/env python3
"""Detect inverted, sliver and tiny tetrahedra in a PUMGen mesh.

One near-degenerate element sets a tiny global time step, a common cause of
SeisSol failing at "Computing LTS weights".
"""

import argparse
import sys

from decatur.mesh import read_puml, tet_quality


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mesh", help="PUMGen mesh, e.g. mesh.puml.hdf5")
    p.add_argument("--max-ratio", type=float, default=1000.0,
                   help="fail above this median/min insphere ratio (default: %(default)g)")
    args = p.parse_args()

    q = tet_quality(*read_puml(args.mesh))
    print(f"elements                     : {q['elements']:,}")
    print(f"signed volume min / max      : {q['volume_min']:.3e} / {q['volume_max']:.3e} m^3")
    print(f"insphere radius min / median : {q['insphere_min']:.3e} / {q['insphere_median']:.3e} m")
    print(f"inverted                     : {q['inverted']:,}")
    print(f"slivers (|V| < 1e-4 median)  : {q['slivers']:,}")
    print(f"tiny insphere (< 1e-3 median): {q['tiny_insphere']:,}")
    print(f"worst element                : {q['worst']}")
    print(f"time-step ratio median/min   : {q['timestep_ratio']:.1f}")
    bad = q["inverted"] or q["slivers"] or q["timestep_ratio"] > args.max_ratio
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
