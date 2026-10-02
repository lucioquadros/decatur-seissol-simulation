#!/usr/bin/env python3
"""Receiver record section and free-surface PGV maps of a SeisSol run, plus receiver_pgv.csv."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from decatur.geometry import fault_row
from decatur.scenario import load_scenario
from decatur.waves import (FaultTrace, figure_receiver_section, figure_surface_pgv,
                           read_receivers, read_surface, ruptured_faults)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", nargs="?", default=".", help="run output directory")
    p.add_argument("--prefix", default="decatur", help="SeisSol OutputFile prefix")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--title", default="")
    p.add_argument("--scenario", help="scenario name or yaml, draws its faults on the PGV maps")
    p.add_argument("--faults", choices=("nucleation", "ruptured"), default="nucleation",
                   help="faults drawn with --scenario: the patch fault, or every fault at least "
                        "half ruptured in the fault output (default: %(default)s)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--bandpass", type=float, nargs=2, metavar=("F_LOW", "F_HIGH"),
                   help="Butterworth band-pass of the record section, Hz, written as "
                        "receiver_section_bandpass.png (receiver_pgv.csv stays unfiltered)")
    p.add_argument("--order", type=int, default=4,
                   help="Butterworth order for --bandpass (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    rec = None
    if any(Path(args.directory).glob(f"{args.prefix}-receiver-*.dat")):
        rec = read_receivers(args.directory, args.prefix)
        print(f"{len(rec.ids)} receivers, {len(rec.t)} samples, "
              f"t = {rec.t[0]:g} to {rec.t[-1]:g} s")
        written.append(figure_receiver_section(rec, outdir / "receiver_section.png", args.title,
                                               args.dpi))
        if args.bandpass:
            written.append(figure_receiver_section(
                rec, outdir / "receiver_section_bandpass.png", args.title, args.dpi,
                band=tuple(args.bandpass), order=args.order))
        pgv = rec.peak()
        table = pd.DataFrame({"receiver": rec.ids, "x": rec.xyz[:, 0], "y": rec.xyz[:, 1],
                              "z": rec.xyz[:, 2], "offset": rec.offsets(),
                              "pgv_horizontal": pgv["horizontal"],
                              "pgv_vertical": pgv["vertical"]})
        table.to_csv(outdir / "receiver_pgv.csv", index=False, float_format="%.6g")
        written.append(outdir / "receiver_pgv.csv")

    if (Path(args.directory) / f"{args.prefix}-surface.xdmf").is_file():
        faults = []
        if args.scenario:
            sc = load_scenario(args.scenario)
            names = ([sc.raw["patch"]["fault"]] if args.faults == "nucleation" else
                     ruptured_faults(args.directory, sc.faults, sc.origin, args.prefix))
            print(f"faults drawn: {', '.join(names)}")
            faults = [FaultTrace.from_inventory(fault_row(sc.inventory, n), sc.origin)
                      for n in names]
        surf = read_surface(args.directory, args.prefix)
        print(f"{surf.corners.shape[0]:,} surface triangles, {len(surf.t)} snapshots, "
              f"t = {surf.t[0]:g} to {surf.t[-1]:g} s")
        written.append(figure_surface_pgv(surf, outdir / "surface_pgv.png", rec, args.title,
                                          args.dpi, faults))

    if not written:
        raise SystemExit(f"no receiver or surface output with prefix '{args.prefix}' in "
                         f"{args.directory}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
