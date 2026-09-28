# Visualization

## Fault overview

```bash
python scripts/fault_inventory.py --csv /tmp/inv.csv --plot-dir figures/faults
```

This writes a map view, E-W and N-S sections, and a 3-D view, colored by
elevation. `--show` opens the 3-D view interactively.
`scripts/convert_ts.py` writes the `.ts` surfaces to STL, optionally shifted
into the local frame (`--origin origin.json`, x, y and the ground elevation),
for overlays in ParaView.

## Material model

1. **The unit grid.** Open `WORK_DIR/<scenario>/material.nc` in ParaView (NetCDF
   reader, CF conventions, "Spherical Coordinates" off). It holds `unit_id`
   (1 Eau Claire, 2 Mt. Simon, 3 Argenta, 4 Precambrian) over
   z -2350 .. -1600 m. Above the grid everything is Eau Claire, below it
   Precambrian. Use Threshold on `unit_id` to isolate a unit, or Slice / Clip
   to see the interfaces. Set the z scale (Transform, e.g. 5×) to see the
   relief, which is only about 100 m over 12 km.
2. **Horizon and fault overlays** in the same frame:

   ```bash
   D=$DATA_DIR   # from config/paths.yaml
   python scripts/convert_ts.py $D/Mt_Simon.ts $D/Argenta.ts $D/PreCambrian.ts \
       -o horizons.stl --origin WORK_DIR/<scenario>/origin.json
   python scripts/convert_ts.py $D/Faults/*.ts -o faults.stl --origin WORK_DIR/<scenario>/origin.json
   ```

   The STL horizons cover only the Petrel area, the grid extends them flat.
3. **The Gmsh mesh with the layers**, before pumgen:

   ```bash
   python scripts/build_mesh.py bob_will --lc-fault 100 --lc-domain 1500 --lc-nuc 50   # coarse, ~45k tets
   python scripts/export_vtk.py bob_will
   ```

   writes, next to `mesh.msh`:

   | File | Cells | Cell data |
   |---|---|---|
   | `mesh.vtu` | tetrahedra | `unit_id` at the element center |
   | `faults.vtu` | fault triangles as meshed (planar) | `fault_id` (names in `fault_ids.csv`), `unit_id`, `patch` (1 inside the nucleation patch) |
   | `interfaces.vtu` | the unit tops as gridded, extended over the whole domain | `unit_id` of the unit below |

   In ParaView: Slice `mesh.vtu` (e.g. normal y through the patch) colored by
   `unit_id` with "Surface With Edges", and add `faults.vtu` colored by `unit_id`
   or `patch`. Coarse elements straddle the interfaces, and the Argenta (4–86 m)
   is thinner than most of them. SeisSol averages such elements.
4. **On the mesh, with easi.** SeisSol 1.3.1 has no material output (`iOutputMaskMaterial`
   is deprecated and ignored). SeisSol's `Meshing/evaluate_material` tool
   evaluates the easi files at the element barycenters, with the same easi and
   ASAGI as SeisSol:

   ```bash
   evaluate_easi -m mesh.puml.hdf5 -e material.yaml -o material_on_mesh
   ```

   It writes `material_on_mesh.xdmf` (ρ, μ, λ per element) for ParaView. Run it
   in the scenario directory, next to `material.nc`. It samples barycenters,
   while SeisSol averages over each element, so elements that straddle an
   interface differ. The same command on `fault.yaml` checks the cohesion field.

## Run summary (energy output)

```bash
python scripts/plot_output.py <run>/output --vs <Vs> --rho <rho> --outdir figures
```

It writes `moment_rate.png`, `source.png`, `energy.png`,
`performance.png` (when the flops or clustering CSVs exist) and
`derived_quantities.csv`.

`--vs` and `--rho` are required. Use the material around the rupture: they set
the Brune radius, the stress drop and the ρVs² rigidity check. The units are in
`data/material_units.csv`:

| Unit | Vs (m/s) | ρ (kg/m³) |
|---|---|---|
| Eau Claire | 2800 | 2440 |
| Mt. Simon | 2540 | 2220 |
| Argenta | 3070 | 2450 |
| Precambrian (fractured) | 3160 | 2730 |

For the 2-layer reference runs use Mt. Simon 1720 / 2400 and Precambrian
3160 / 2730.

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
