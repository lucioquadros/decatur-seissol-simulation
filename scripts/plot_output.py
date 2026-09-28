#!/usr/bin/env python3
"""Moment rate, source, energy and performance figures plus derived_quantities.csv.

--vs and --rho are the material around the rupture. They set the Brune radius,
stress drop and the rho*Vs^2 rigidity check.
"""

import argparse

from decatur.plotting import run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", nargs="?", default=".", help="run output directory")
    p.add_argument("--vs", type=float, required=True, metavar="M/S")
    p.add_argument("--rho", type=float, required=True, metavar="KG/M3")
    p.add_argument("--prefix", default="decatur", help="SeisSol OutputFile prefix")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--smooth", type=int, default=0, metavar="N",
                   help="Savitzky-Golay window on M0 before differentiating (odd, > 3)")
    p.add_argument("--corner-band", type=float, default=0.1, metavar="FRAC",
                   help="omega^-2 fit stops where the spectrum drops below FRAC * M0")
    p.add_argument("--title")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    run(args.directory, args.vs, args.rho, prefix=args.prefix, outdir=args.outdir,
        smooth=args.smooth, corner_band=args.corner_band, title=args.title,
        dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()
