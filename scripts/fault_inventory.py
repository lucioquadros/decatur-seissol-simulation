#!/usr/bin/env python3
"""Strike, dip, extents and area of each fault .ts surface -> fault_inventory.csv."""

import argparse
import glob
import sys
from pathlib import Path

from decatur.config import INVENTORY_CSV, load_paths
from decatur.inventory import build_inventory, format_inventory, plot_3d, plot_map, plot_sections
from decatur.ts_io import read_tsurf


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*", metavar="FILE.ts",
                   help="fault surfaces or glob patterns (default: DATA_DIR/Faults/*.ts)")
    p.add_argument("--csv", default=str(INVENTORY_CSV), help="output CSV (default: %(default)s)")
    p.add_argument("--plot-dir", metavar="DIR", help="also write map, section and 3-D PNGs here")
    p.add_argument("--show", action="store_true", help="open the 3-D view interactively")
    return p.parse_args()


def main():
    args = parse_args()
    patterns = args.inputs
    if not patterns:
        data = load_paths().get("DATA_DIR")
        if data is None:
            sys.exit("no inputs given and DATA_DIR not set in config/paths.yaml")
        patterns = [str(data / "Faults" / "*.ts")]
    files = sorted({f for p in patterns for f in glob.glob(p) if f.endswith(".ts")})
    if not files:
        sys.exit(f"no .ts files match {patterns}")

    surfaces = [read_tsurf(f) for f in files]
    df = build_inventory(surfaces)
    print(format_inventory(df))
    df.to_csv(args.csv, index=False)
    print(f"\nwrote {args.csv} ({len(df)} faults)")

    if args.plot_dir:
        out = Path(args.plot_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_map(surfaces, df, out / "fault_map.png")
        plot_sections(surfaces, df, out / "fault_sections.png")
        plot_3d(surfaces, df, out / "fault_3d.png", show=args.show)
        print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
