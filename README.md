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
data/                       data tables (fault_inventory.csv, material_units.csv)
references/<scenario>/      as-run inputs from seisclass3 and reference metrics (coarse runs)
scenarios/<scenario>/       scenario.yaml: every knob of one scenario
scripts/                    command-line tools (thin wrappers around src/decatur)
sdumont/                    SDumont job scripts (tools, mesh, pumgen, evaluate_easi, run simulation)
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
python scripts/estimate_resolution.py bob_will         # cohesive zone, fault size, patch, max frequency
python scripts/make_scenario.py bob_will               # -> WORK_DIR/bob_will/{fault.yaml,parameters.par,material.yaml,receivers.dat,origin.json}
python scripts/build_material.py bob_will              # -> WORK_DIR/bob_will/material.nc (unit grid)
python scripts/build_mesh.py bob_will                  # -> WORK_DIR/bob_will/mesh.msh, same origin.json
python scripts/estimate_cost.py bob_will               # LTS clusters, node-hours, wavefield size
python scripts/export_vtk.py bob_will                  # -> mesh.vtu, faults.vtu, interfaces.vtu for ParaView
pumgen -s msh2 mesh.msh mesh.puml.hdf5
python scripts/check_mesh.py mesh.puml.hdf5
# evaluate_easi source: https://github.com/SeisSol/Meshing/tree/master/evaluate_material
evaluate_easi -m mesh.puml.hdf5 -e material.yaml -o easi_material
evaluate_easi -m mesh.puml.hdf5 -e fault.yaml -o easi_fault
python scripts/check_easi.py bob_will
python scripts/plot_output.py output --vs 3160 --rho 2730
python scripts/plot_waves.py output                    # receiver record section, surface PGV maps, receiver_pgv.csv
```

## SDumont

The folder in the project space (`DECATUR_BASE`, default
`/petrobr/parceirosbr/sismo_co2/$USER`) holds this repository as `decatur/`
(with its own `.venv` and `config/paths.yaml`), the horizon `.ts` files, the
work directory `decatur-work/`, and SeisSol's `Meshing/` for `evaluate_easi`.
Spack and `seissol-env` come from
[seissol-spack-installer](https://github.com/lucioquadros/seissol-spack-installer).
Submit the jobs from the repository root.

| File | Where | Purpose |
|---|---|---|
| `sdumont/env.sh` | sourced | paths, OpenMPI module, Spack |
| `sdumont/tools_fetch.sh` | login node | Spack environment `decatur-tools` (pumgen, cmake) concretized and mirrored, `Meshing` cloned |
| `sdumont/tools_build.sbatch` | job | offline install of `decatur-tools`, `evaluate_easi` built against `seissol-env` |
| `sdumont/mesh.sbatch <scenario> <dir> [flags]` | job | `make_scenario`, `build_material`, `build_mesh` (HXT, 32 threads) |
| `sdumont/pumgen.sbatch <dir>` | job | `pumgen -s msh2`, then `check_mesh` |
| `sdumont/evaluate_easi.sbatch <scenario> <dir>` | job | `evaluate_easi` on `material.yaml` and `fault.yaml`, then `check_easi` |
| `sdumont/proxy.sbatch [threads ...]` | job | SeisSol proxy at 100,000 elements, hardware GFLOPS per core for `estimate_cost.py` |
| `sdumont/run.sbatch <dir>` | job | Run SeisSol simulation in `<dir>`. Output in `<dir>/output`. Nodes, ranks per node and cores per rank as `sbatch` flags (default 1 × 4 × 48). One core per rank left for the communication thread |

## Scripts

| Script | Purpose |
|---|---|
| `fault_inventory.py` | strike, dip, extents and area of each fault `.ts` → `data/fault_inventory.csv`, optional figures |
| `make_scenario.py` | render the SeisSol inputs of a scenario (`--end-time` for short test runs) |
| `build_material.py` | grid the material units (horizons from `DATA_DIR`) into `material.nc` for `material.yaml` and the fault cohesion |
| `check_cfs.py` | Coulomb failure stress on every fault and at the patch, and the faults inside the patch slab, same model as `fault.yaml` |
| `build_mesh.py` | Gmsh mesh of a scenario: its faults as planar quads, its nucleation ball and optional refinement box, size flags override the scenario |
| `check_mesh.py` | inverted, sliver and tiny-insphere tetrahedra in a PUML mesh |
| `check_easi.py` | compare the `evaluate_easi` output of `material.yaml` and `fault.yaml` with the Python model |
| `convert_ts.py` | convert `.ts` → ASCII STL, optionally in the local frame |
| `export_vtk.py` | Export scenario's Gmsh mesh → `.vtu` for ParaView {mesh.vtu, faults.vtu, and interfaces.vtu} |
| `estimate_resolution.py` | static and measured cohesive zone, fault element size for the error limits, patch vs critical radius, highest frequency along the size field |
| `estimate_cost.py` | LTS clusters of a mesh, element updates, core- and node-hours, wavefield output size |
| `plot_output.py` | moment rate, energy and performance figures, `derived_quantities.csv` |
| `plot_waves.py` | receiver record section and peak velocity along the line, free-surface PGV maps, `receiver_pgv.csv` |

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
