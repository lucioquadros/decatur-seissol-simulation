# decatur-seissol-simulation

Dynamic-rupture simulations of induced seismicity at the Illinois Basin –
Decatur Project (IBDP) CO₂ site with [SeisSol](https://seissol.org), run on
[SDumont](https://github.com/lncc-sered/manual-sdumont2nd/wiki).

Two overpressure scenarios are modeled on the planar approximation of the 28
interpreted Petrel faults:

| Scenario | Nucleation fault | dP | Reference run (order 4, ~1.3M tets) |
|---|---|---|---|
| `bob_will` | Bob Will | 2 MPa | Mw 4.43 |
| `nick` | Nick | 4.32 MPa | Mw 3.42 |

## Layout

```
config/paths.example.yaml   local DATA_DIR, REPORTS_DIR, WORK_DIR (copy to config/paths.yaml)
data/                       small derived tables (fault_inventory.csv)
docs/                       physics.md (stress, nucleation), visualization.md
references/<scenario>/      as-run inputs from seisclass3 and reference metrics (coarse runs)
scenarios/<scenario>/       scenario.yaml: every knob of one scenario
scripts/                    command-line tools (thin wrappers around src/decatur)
src/decatur/                python package
templates/                  fault.yaml, parameters.par, material.yaml templates
tests/                      pytest
```

Raw IBDP data (`.ts` surfaces, reports), meshes and SeisSol outputs are not
in git. Their locations are set in `config/paths.yaml`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp config/paths.example.yaml config/paths.yaml   # then edit
python -m pytest
```

## Workflow

```bash
python scripts/check_cfs.py bob_will                   # background CFS < 0, patch > 0
python scripts/make_scenario.py bob_will               # -> WORK_DIR/bob_will/{fault.yaml,parameters.par,material.yaml,origin.json}
python scripts/build_mesh.py bob_will                  # -> WORK_DIR/bob_will/mesh.msh, same origin.json
pumgen -s msh2 mesh.msh mesh.puml.hdf5
python scripts/check_mesh.py mesh.puml.hdf5
python scripts/plot_output.py output --vs 3160 --rho 2730
```

## Scripts

| Script | Purpose |
|---|---|
| `fault_inventory.py` | strike, dip, extents and area of each fault `.ts` → `data/fault_inventory.csv`, optional figures |
| `make_scenario.py` | render the SeisSol inputs of a scenario |
| `check_cfs.py` | Coulomb failure stress on every fault and at the patch, and the faults inside the patch slab, same model as `fault.yaml` |
| `build_mesh.py` | Gmsh mesh of a scenario: its faults as planar quads and its nucleation ball, size flags override the scenario |
| `check_mesh.py` | inverted, sliver and tiny-insphere tetrahedra in a PUML mesh |
| `convert_ts.py` | `.ts` → ASCII STL, optionally in the local frame |
| `plot_output.py` | moment rate, energy and performance figures, `derived_quantities.csv` |

## References

Temporary development aid, removed once the project works. Comparisons with
these runs are loose (different SeisSol builds and settings), and no config,
code or test depends on them.

`references/<scenario>/` holds the inputs of the reference runs on
seisclass3 (`parameters.par`, `material.yaml`,
`fault_over_stress_pressure.yaml`, the Gmsh `mesh.geo_unrolled` size fields,
the mesh `.xdmf` header) and the small
outputs (`decatur-energy.csv`, `decatur-clustering.csv`,
`derived_quantities.csv`).

| Scenario | seisclass3 inputs | Outputs |
|---|---|---|
| `bob_will` | `~/simplified_decatur_bob_will/` | `overpressure_2/` |
| `nick` | `~/simplified_decatur_nick/` | `output/` |
