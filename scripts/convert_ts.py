#!/usr/bin/env python3
"""Convert GOCAD TSurf (.ts) files to one ASCII STL, optionally in the local frame."""

import argparse
import json

from decatur.ts_io import read_ts, write_stl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", metavar="FILE.ts")
    p.add_argument("-o", "--output", required=True, metavar="FILE.stl")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--origin", metavar="origin.json", help="shift x, y into this local frame")
    g.add_argument("--translate", nargs=2, type=float, metavar=("X0", "Y0"),
                   help="subtract (X0, Y0) from every vertex")
    args = p.parse_args()

    offset = (0.0, 0.0, 0.0)
    if args.origin:
        with open(args.origin) as fh:
            o = json.load(fh)
        offset = (o["origin_x"], o["origin_y"], 0.0)
    elif args.translate:
        offset = (*args.translate, 0.0)
    surfaces = [s for f in args.inputs for s in read_ts(f)]
    write_stl(args.output, surfaces, offset)
    print(f"wrote {args.output}: {len(surfaces)} surface(s)")


if __name__ == "__main__":
    main()
