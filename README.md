# decatur-seissol-simulation

Dynamic-rupture simulations of induced seismicity at the Illinois Basin –
Decatur Project (IBDP) CO₂ site with [SeisSol](https://seissol.org), run on
[SDumont](https://github.com/lncc-sered/manual-sdumont2nd/wiki).

Two overpressure scenarios are modelled on the planar approximation of the 28
interpreted Petrel faults:

| Scenario | Nucleation fault | dP | Reference run (order 4, ~1.3M tets) |
|---|---|---|---|
| `bob_will` | Bob Will | 2 MPa | Mw 4.43 |
| `nick` | Nick | 4.32 MPa | Mw 3.42 |

## Layout

```
config/paths.example.yaml   local FAULT and LAYERS data locations (copy to config/paths.yaml)
data/                       small derived data tables (fault_inventory.csv)
references/<scenario>/      as-run inputs from seisclass3 and reference metrics (coarse runs) 
scripts/                    command-line tools
src/decatur/                python package
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

## Scripts

| Script | Purpose |
|---|---|
| `fault_inventory.py` | strike, dip, extents and area of each fault `.ts` → `fault_inventory.csv` |
| `build_mesh.py` | Gmsh mesh with the faults as planar quads, optional nucleation ball |
| `check_stresses.py` | Coulomb failure stress on every fault before a run (out of sync with the as-run YAMLs) |
| `check_mesh.py` | degenerate/sliver tetrahedra in a PUML mesh |
| `convertTs.py`, `Face.py` | `.ts` → STL, from [SeisSol/Meshing](https://github.com/SeisSol/Meshing) |
| `plot_seissol_output.py` | moment rate, energy, source and performance figures, `derived_quantities.csv` |

## References

`references/<scenario>/` holds the inputs of the reference runs on
seisclass3 (`parameters.par`, `material.yaml`,
`fault_over_stress_pressure.yaml`, the mesh `.xdmf` header) and the small
outputs (`decatur-energy.csv`, `decatur-clustering.csv`,
`derived_quantities.csv`).

| Scenario | seisclass3 inputs | Outputs |
|---|---|---|
| `bob_will` | `~/simplified_decatur_bob_will/` | `overpressure_2/` |
| `nick` | `~/simplified_decatur_nick/` | `output/` |
