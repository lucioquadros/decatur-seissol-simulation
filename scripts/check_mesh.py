#!/usr/bin/env python3
"""
check_mesh_quality.py — find degenerate / sliver tetrahedra in a PUML mesh.

A single near-zero-volume or inverted element gives a vanishing stable time
step, which is a common cause of a SeisSol segfault at "Computing LTS weights".

Usage:
    python check_mesh_quality.py mesh.puml.h5
Requires: numpy, h5py   (pip install h5py)
"""
import sys
import numpy as np
import h5py

fn = sys.argv[1] if len(sys.argv) > 1 else "mesh.puml.h5"

with h5py.File(fn, "r") as f:
    print("datasets in file:", list(f.keys()))
    # PUMGen/PUML naming: 'geometry' (Nv x 3 float), 'connect' (Ne x 4 int)
    xyz   = f["geometry"][:]
    cells = f["connect"][:]

v0, v1, v2, v3 = (xyz[cells[:, 0]], xyz[cells[:, 1]],
                  xyz[cells[:, 2]], xyz[cells[:, 3]])

# signed volume (negative => inverted element)
vol  = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
absv = np.abs(vol)

# insphere radius r = 3V / (sum of face areas) -- this drives the time step
def area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
A_tot = area(v0, v1, v2) + area(v0, v1, v3) + area(v0, v2, v3) + area(v1, v2, v3)
r_in  = 3.0 * absv / np.maximum(A_tot, 1e-30)

med_v = np.median(absv)
med_r = np.median(r_in)

print(f"\nelements                       : {len(cells):,}")
print(f"signed volume   min / max      : {vol.min():.3e} / {vol.max():.3e}")
print(f"|volume|        min / median   : {absv.min():.3e} / {med_v:.3e} m^3")
print(f"insphere radius min / median   : {r_in.min():.3e} / {med_r:.3e} m")
print(f"inverted (vol<=0) elements     : {int((vol <= 0).sum()):,}")
print(f"slivers (|vol| < 1e-4*median)  : {int((absv < 1e-4 * med_v).sum()):,}")
print(f"tiny insphere (< 1e-3*median)  : {int((r_in < 1e-3 * med_r).sum()):,}")

# the single worst element governs the global minimum time step
w = r_in.argmin()
print(f"\nworst element idx={w}: insphere={r_in[w]:.3e} m, |vol|={absv[w]:.3e} m^3")
print(f"  -> time-step ratio across mesh ~ {med_r / max(r_in.min(), 1e-30):.1f}x "
      f"(healthy meshes: tens, not thousands)")
