# Material model

The rock between the free surface and the bottom of the mesh is split into four
elastic units. Their properties and sources are in `data/material_units.csv`,
their tops are the Petrel horizon surfaces. `scripts/build_material.py` grids the
units into `material.nc`, and `make_scenario.py` renders the `material.yaml` that
reads it.

## Datum: z = 0 is the ground

The Petrel horizons and faults (`.ts`, `ZPOSITIVE Elevation`) are elevations
above mean sea level. Checked at CCS1: its well tops (depth below KB minus the KB
elevation) match the horizons at the well within 0.3 m for the Mt. Simon and
Argenta tops. As depths below KB they would miss by about 160 m.

The ground at the IBDP wells is 204–209 m above sea level (NAVD88):

| Well | Ground (m) | Source |
|---|---|---|
| CCS1 | 205.89 | IBDP final report (2021), CCS1 section |
| VW1 | 203.97 | IBDP final report (2021), VW1 section |
| CCS2 | 206 | `Well_Diagrams/Injection_Well-CCS2-2017.pdf` |
| GM2 | 206 | `Well_Diagrams/Geophysical_Well-GM2-2017_070617.pdf` |
| VW2 | 209 | `Well_Diagrams/Verification_Well-VW2-2017.pdf` |

`ground_elevation: 206.0` in the scenario file is subtracted from every z (faults,
patch, horizons), so the free surface z = 0 is the ground and -z is the depth
below ground used by the stress gradients. It is written to `origin.json` as
`origin_z`. The 5 m spread between the wells is far below any element size.

The Petrel x, y are Illinois East State Plane (NAD27) coordinates in meters.
CCS1's NAD83 latitude and longitude project to about (104488, 356482).

## Units

| id | Unit | Top | Vp (m/s) | Vs (m/s) | ρ (kg/m³) | Source |
|---|---|---|---|---|---|---|
| 1 | Eau Claire | free surface | 4760 | 2800 | 2440 | VW2 core, 3 Eau Claire Shale plugs |
| 2 | Mt. Simon | `Mt_Simon.ts` | 4320 | 2540 | 2220 | VW2 core, 7 Mt. Simon plugs |
| 3 | Argenta | `Argenta.ts` | 5220 | 3070 | 2450 | VW2 core, 5 Pre-Mt Simon plugs |
| 4 | Precambrian | `PreCambrian.ts` | 5330 | 3160 | 2730 | Bondarenko 2025, fractured rhyolite |

The Precambrian continues to the bottom of the mesh. `Z_Model_Base.ts` is only
the base of the Petrel model and is not used. The Eau Claire and Mt. Simon
sub-unit horizons are left for a later milestone.

**No overburden unit.** The reports have no velocity or density values for the
1.5 km above the Eau Claire. Sonic logs were run from about 107 m depth in CCS1
and VW1 (IBDP final report, logging tables), but their data are not in the
reports, and the only core is two Maquoketa–Galena plugs (GM2 report, Table 20).
The Eau Claire values are extended to the free surface instead.

### Sediments: VW2 core at in-situ stress

SLB Carbon Services report TR13-404322, *Geomechanics Characterization of
Selected Core Material, IL-ICCS Verification Well #2* (August 2013),
`Geomechanical_Reports/1211523552_20130800_VW2_Geomechanics_Characterization_EReport_404322.pdf`:

- Table 1 (PDF p. 2): sample list, depths, bulk densities.
- Table 9, *Best Fit In-Situ Stress Velocities* (PDF p. 209): ultrasonic Vp and
  Vs at the effective confining pressure of each depth (about 17–18 MPa),
  measured vertical (V), at 45° and horizontal (H).

Vs is the mean of the V and H values of every plug in the unit, ρ the mean bulk
density. Values are rounded to 10.

These are in-situ values and replace the Bondarenko (2025) lab values for the
sediments, which are much slower (Mt. Simon Vs 1720, Argenta 1510 m/s). The VW1
sonic log (`IBDP Phase 4 Geomechanics Report 4_5_15 Final.pdf`, Fig. 2, read by
eye) agrees with the core: Vs about 3000 m/s in the upper Mt. Simon, about 2000
in the porous lower Mt. Simon, about 2300 in the Eau Claire Shale.

Caveats:

- Few plugs, chosen for testing. The Mt. Simon plugs are mostly from the middle
  and lower Mt. Simon (including the injection zone), so the unit average may
  be low for the tight upper Mt. Simon.
- The Eau Claire plugs are all shale. The upper, carbonate part is probably faster.
- The rock is anisotropic: H exceeds V by up to 20% in the Eau Claire.

### Vp from Vp/Vs = 1.7

The core gives Vp/Vs of 1.53–1.55 in all three sedimentary units, which is
implausibly low for saturated sandstone and shale (Poisson's ratio about 0.13).
The sonic log gives 1.65–2.0 in the same units. Vs, which sets the rigidity,
is taken from the core and Vp = 1.7 Vs (Poisson's ratio 0.24).

### Precambrian: fractured rhyolite

No report has elastic data for the basement: the VW2 core tests stop in the
Argenta. The values are the fractured rhyolite column of the Bondarenko (2025)
lab table (intact: 5520 / 3280 / 2760, about 4% faster). Fractured is used
because the upper basement is weathered, fractured rhyolite: MGSC Core
Workshop 2019 (`Stratigraphy/Core_Workshop_2019_final_sm.pdf`, PDF pp. 18–27),
and the wellbore images in the static model report (2021), which call the
basement "highly fractured". 80–90% of the microseismicity was in the
basement (final report, static model report). The measured Vp/Vs (1.69) is kept.

## Interfaces

Each horizon is gridded on the material grid (50 m) in the local frame:

- Inside its triangulated area: linear interpolation on the TSurf triangles.
- Outside: the nearest edge value blended into the surface's mean elevation
  over `material.taper` (1000 m, cosine ramp). The horizons cover only about
  2.4 × 3.9 km, the mesh about 12 × 12.5 km, so most of the domain has flat
  interfaces at the mean depths.
- Surfaces may not cross: each top is clipped to lie at or below the one above.
  With the current horizons nothing is clipped. The thinnest Argenta is 4 m.

Tops in the local frame (min .. max over the grid, m):

| Unit | Top z |
|---|---|
| Mt. Simon | -1731 .. -1652 |
| Argenta | -2211 .. -2095 |
| Precambrian | -2259 .. -2111 |

At CCS1 the Precambrian top from the 2009 well tops is 33 m above the Petrel
surface: the 2019 seismic reinterpretation gives a thicker Argenta there. The
Petrel surfaces are used.

## Grid and easi files

`material.nc` (NetCDF, COARDS) holds one float variable `unit_id(z, y, x)`:

- x, y: the mesh domain plus one cell, every `material.dx` (50 m);
- z: `material.z_min` .. `material.z_max` (-2350 .. -1600 m), every
  `material.dz` (5 m).

For the full fault set that is 245 × 253 × 151 cells, 37 MB.
`build_material.py` refuses a z range that doesn't contain every interface
with a cell to spare.

`material.yaml` is an `!Any` of three boxes:

1. inside the grid: `!ASAGI` (nearest) gives `unit_id`, a `!LuaMap` turns it into
   ρ, μ, λ from `data/material_units.csv`;
2. above the grid: the top unit (Eau Claire) as constants;
3. below the grid: the bottom unit (Precambrian) as constants.

A point outside the grid horizontally matches nothing and easi stops with an
error, which catches a mesh larger than the grid (e.g. a `--buffer` override in
`build_mesh.py`). Nearest lookup keeps the interfaces sharp. The grid stores
only the unit id, so changing a property only needs `make_scenario.py`, not a
new grid.

SeisSol 1.3.1 averages the material over each element by default
(`UseCellHomogenizedMaterial = 1`, set explicitly in `parameters.par`): ρ
arithmetic, μ harmonic, λ from the averaged ν/E. Elements that straddle an
interface get an effective medium instead of the value at their barycenter
(`docs/physical-models.rst` in the SeisSol source,
`src/Initializer/ParameterDB.cpp`).

## Cohesion by unit

`friction.cohesion` gives the fault cohesion per unit, and `fault.yaml` reads
the same grid:

| Unit | Cohesion (MPa) |
|---|---|
| Eau Claire, Mt. Simon, Argenta | -0.5 |
| Precambrian | -0.1 |

This replaces the switch at a fixed elevation of -1950 m (sea level datum)
with the top of the Precambrian. Both patches keep their cohesion: the Bob Will patch is
in the Precambrian, the Nick patch in the Mt. Simon.

## Effect on the stress check

The stresses scale with depth, so moving z = 0 to the ground raises every
stress by the extra 206 m of depth (about 10% at the reservoir). The ratio of
shear to normal stress is unchanged, so every fault moves away from failure
in proportion (Bob Will background max CFS -0.73 → -0.79 MPa). The fixed dP
counts less at the higher normal stress:

| Scenario | Patch z (m) | CFS background | CFS with dP |
|---|---|---|---|
| bob_will | -2306.04 | -0.83 MPa | +0.37 MPa |
| nick | -2029.19 | -2.35 MPa | +0.24 MPa |

Both still fail inside the patch and nowhere else (`check_cfs.py`). Before the
shift these were +0.43 MPa for both.
