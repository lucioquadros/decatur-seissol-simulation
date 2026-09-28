# Visualization

## Fault overview

```bash
python scripts/fault_inventory.py --csv /tmp/inv.csv --plot-dir figures/faults
```

This writes a map view, E-W and N-S sections, and a 3-D view, colored by
elevation. `--show` opens the 3-D view interactively.
`scripts/convert_ts.py` writes the `.ts` surfaces to STL, optionally shifted
into the local frame (`--origin origin.json`), for overlays in ParaView.

## Run summary (energy output)

```bash
python scripts/plot_output.py <run>/output --vs <Vs> --rho <rho> --outdir figures
```

It writes `moment_rate.png`, `source.png`, `energy.png`,
`performance.png` (when the flops or clustering CSVs exist) and
`derived_quantities.csv`.

`--vs` and `--rho` are required. Use the material around the rupture: they set
the Brune radius, the stress drop and the ρVs² rigidity check. For the 2-layer
reference material:

| Unit | Vs (m/s) | ρ (kg/m³) |
|---|---|---|
| Mt. Simon | 1720 | 2400 |
| Precambrian (fractured) | 3160 | 2730 |

The reference `derived_quantities.csv` files were made with the old defaults
(2600, 2500), so their radius, stress drop and rigidity rows don't correspond
to the run material. M0, Mw and the timing rows don't depend on Vs or ρ.

### Caveats on the derived quantities

Check these against the SeisSol energy-output docs for the version used
before quoting any number:

- **Radiated energy.** `elastic_energy` and `elastic_kinetic_energy` start at
  zero (perturbation field). Their sum is used as a radiated-energy proxy. It is
  not corrected for energy lost through the absorbing boundaries, so it
  underestimates late-time values. Apparent stress built from it is approximate.
- **Breakdown energy** = `total_frictional_work − static_frictional_work`. In
  the reference runs `static_frictional_work` decreases slightly after its peak, and
  the script reports this instead of hiding it.
- **Sampling.** Moment rate is the derivative of the cumulative moment sampled
  at `EnergyOutputInterval`. The script warns when fewer than 50 samples span
  T05–T95. The references had 24 (Nick) at 0.01 s.
- **Corner frequency.** The primary estimate is the half-amplitude crossing,
  with the plateau fixed at M0. The ω² fit is secondary and depends on the band
  (`--corner-band`).
