#!/usr/bin/env python3
"""Render the SeisSol inputs (parameters, easi models, receivers) and origin.json for a scenario."""

import argparse

from decatur.config import work_dir
from decatur.scenario import load_scenario, render, write


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", help="scenario name (scenarios/<name>) or scenario.yaml path")
    p.add_argument("--outdir", help="default: WORK_DIR/<scenario> from config/paths.yaml")
    p.add_argument("--end-time", type=float,
                   help="total simulated time in seconds (default: the scenario's run.end_time)")
    return p.parse_args()


def main():
    args = parse_args()
    sc = load_scenario(args.scenario)
    if args.end_time is not None:
        sc.raw["run"]["end_time"] = args.end_time
    outdir = args.outdir or work_dir(sc.name)
    changed = write(render(sc), outdir)
    c = sc.patch.center
    print(f"{sc.name}: patch on {sc.raw['patch']['fault']} at ({c[0]}, {c[1]}, {c[2]}), "
          f"origin ({sc.origin[0]:.3f}, {sc.origin[1]:.3f}, {sc.origin[2]:g}), "
          f"end time {sc.raw['run']['end_time']:g} s, {len(sc.receivers())} receivers")
    print(f"{outdir}: " + (", ".join(p.name for p in changed) + " updated" if changed
                           else "up to date"))


if __name__ == "__main__":
    main()
