# Initial stress and nucleation

Physics behind the generated `fault.yaml`. The numbers live in
`scenarios/<name>/scenario.yaml`. The model is implemented once in
`src/decatur/stress.py` and rendered to Lua from `templates/fault.yaml`.

## Conventions

- Frame: x East, y North, z up, free surface at z = 0. x and y are relative to
  the local origin in `origin.json` (mid-point of the bounding box of the meshed
  faults, which for all 28 faults is 104551.285, 357396.610).
- Stress is tension-positive: compression and cohesion are negative (SeisSol).
- Stresses are effective: hydrostatic pore pressure is removed from the normal
  components.
- Strike/dip follow the right-hand rule. `down_dip = (cos s cos d, -sin s cos d, -sin d)`,
  normal = strike × down-dip.

## Background stress

Andersonian strike-slip regime with depth gradients (Pa/m):

| | Gradient | Note |
|---|---|---|
| Sv | 25.0e3 | overburden |
| SHmax | 47.7e3 | azimuth N70E, inside the published 40–82 MPa/km range (Bauer 2016, Lahann 2017, Langet 2020) |
| Shmin | 22.0e3 | |
| Pf | 9.81e3 | hydrostatic |

SHmax is the knob for background criticality. It was tuned so the most critical
faults sit just below failure while no fault fails without the patch. Check this
with `scripts/check_cfs.py` whenever the stress or friction changes.

## Friction and cohesion

Linear slip-weakening (FL = 16): mu_s 0.60, mu_d 0.45, d_c 0.02 m.

Cohesion switches at z = -1950 m (the 2-layer material interface): -0.5 MPa in
the sediments (cemented Mt. Simon, immature fault segments) and -0.1 MPa in the
basement (mature, gouge-bearing faults). Both reference runs used this model.
M2 replaces the fixed depth with the top of the Precambrian.

## Nucleation patch

The observed overpressure at Decatur is small: about 1 MPa near the well
(Rathmaier et al. 2024), well below the fracture pressure. mu·dP of about
0.6 MPa can't fail a fault on its own. It can only tip a fault that is already
near failure. The recipe:

1. **Near-critical background** on a well-oriented fault. Bob Will strikes
   about 31° from SHmax, close to optimal (45° − φ/2 ≈ 30° for mu = 0.6).
2. **Overpressure dP** inside a disk on the target fault. It is added to
   the three normal stress components, which lowers the effective normal stress
   on the fault by dP and leaves the shear unchanged. Pressure reaches the
   basement locally along faults and where the Argenta seal is thin (Silva 2024,
   Bondarenko 2022 and 2025, Goertz-Allmann 2017). More than 90% of the
   microseismicity was in the basement.
3. **Shear seed dTau** (optional, 0 in both scenarios): a small increase of
   |s_xy| in the disk. It localizes nucleation to a compact, resolved patch.
   Keep it ≤ mu·dP so that injection stays the driver.

The disk is centered on the fault's best-fit plane: the inventory centroid plus
`strike_offset`·strike + `dip_offset`·down-dip. A point is inside when its
in-plane distance is ≤ `radius` and its off-plane distance is ≤ `normal_tol`.

`normal_tol` is 1 m. The target fault is meshed from the same centroid, strike
and dip, so its faces sit within 1 cm of the patch plane. SeisSol reads
`fault.yaml` only on fault faces, so the slab never affects the rock volume,
but it does load any other fault passing through it. The reference runs used
25 m, which in the Nick run also put dP on about 13,000 m² of Si-Yong, Steve
and Dameon (none of them failed). With 1 m only a 414 m² strip of Si-Yong,
where it crosses Nick's plane, remains. `check_cfs.py` lists every fault inside
the slab, with its area and CFS, and fails if a fault other than the target
reaches failure there. Whether dP should load only the target fault or a
pressurized volume is an open question for M2.

A radius of 70 m (a 140 m disk) is resolved by 30 m fault elements, and it
must also exceed the critical nucleation length. Check both before a
production run (M3).

Once the patch fails, the final magnitude is not bounded by the pressurized
volume: dynamic and static stress transfer can carry the rupture across the
network ("runaway" regime, Galis et al. 2017).

| Scenario | Fault | Offset down-dip | Patch (local, m) | dP |
|---|---|---|---|---|
| bob_will | Bob Will (101.3/72.7) | 296 m | (255.19, -393.25, -2100.04) | 2 MPa |
| nick | Nick (275.5/64.9) | 87 m | (42.58, 500.82, -1823.19) | 4.32 MPa |

## Forced rupture

A backup nucleation, disabled (`forced_rupture.radius: 0`). With FL = 16,
SeisSol weakens friction by slip and, where a forced rupture time T is set,
also by time:

  mu = mu_s − (mu_s − mu_d) · max(min(slip / d_c, 1), min((t − T) / t_0, 1)),

with the time term counting only for t ≥ T. Inside a sphere of `radius` around
the patch center (a disk on the fault), T is `time`. Elsewhere T is 1e10 s,
so it is never reached. If the stress patch stalls, set a radius (e.g. 40 m).
`make_scenario.py` then writes `t_0` into `&DynamicRupture`, and it refuses a
radius > 0 without `t_0` > 0.

## Moving the patch or meshing a subset

- The patch coordinates depend on the local origin, which depends on which
  faults are meshed. The only place to choose them is `faults:` in the
  scenario file, which both `make_scenario.py` and `build_mesh.py` read, so the
  inputs and the mesh always share one frame. To mesh a subset for a test,
  write a small scenario file with that `faults:` list.
- To move the patch, change `dip_offset` / `strike_offset` (or `patch.fault`)
  and rerun `make_scenario.py` and `check_cfs.py`. The nucleation ball in the
  mesh follows the patch: same center, and radius = patch radius +
  `nucleation_ball.margin` (20 m by default). The margin keeps the patch edge,
  where the rupture starts, inside the fully refined zone rather than on the
  taper.
