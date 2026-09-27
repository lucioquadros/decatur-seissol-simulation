#!/usr/bin/env python3
"""
check_cfs.py — pre-run Coulomb-failure-stress validation for the Decatur
fault_overstress.yaml stress model.

Replicates the stress assembly in fault_overstress.yaml, samples each fault
plane, resolves the stress onto the fault, and computes
    CFS = |tau| - mu_s*|sigma_n'| - cohesion        ( > 0  => slips at t=0 )

GOAL: the BACKGROUND (no patch) must be < 0 on EVERY fault (sub-critical), and
the overstress patch must push its local spot on the target fault > 0.

Usage:
    python check_cfs.py fault_inventory.csv [grad_SH_MPa_per_km]
    e.g.  python check_cfs.py fault_inventory.csv 42
Requires: numpy, pandas. Keep the parameters below in sync with fault_overstress.yaml.
"""
import sys
import numpy as np
import pandas as pd

inv     = sys.argv[1] if len(sys.argv) > 1 else "fault_inventory.csv"
grad_SH = (float(sys.argv[2]) if len(sys.argv) > 2 else 42.0) * 1e3   # Pa/m

# ---- stress / friction (MUST match fault_overstress.yaml) ----
grad_Sv, grad_Sh, grad_Pf = 25.0e3, 22.0e3, 9.81e3
az_SH, mu_s, cohesion = 70.0, 0.60, 1.0e5          # |cohesion| [Pa]

# ---- nucleation patch (all-faults mesh frame) ----
x_nuc, y_nuc, z_nuc = 651.23, -470.49, -2125.88
patch_strike, patch_dip = 183.2, 72.9              # Hannes
r_nuc, n_tol, dTau = 20.0, 25.0, 30.0e6

df = pd.read_csv(inv)
ox = (df.x_min.min() + df.x_max.max()) / 2.0
oy = (df.y_min.min() + df.y_max.max()) / 2.0

th = np.radians(az_SH)
s2, c2, sc = np.sin(th)**2, np.cos(th)**2, np.sin(th)*np.cos(th)


def stress_tensor(p, with_patch=False):
    x, y, z = p
    depth = max(-z, 0.0)
    Sv, SH, Sh, Pf = grad_Sv*depth, grad_SH*depth, grad_Sh*depth, grad_Pf*depth
    Sve, SHe, She = Sv-Pf, SH-Pf, Sh-Pf
    sxx = -(SHe*s2 + She*c2)
    syy = -(SHe*c2 + She*s2)
    sxy = -(SHe - She)*sc
    szz = -Sve
    if with_patch:
        sr, dr = np.radians(patch_strike), np.radians(patch_dip)
        sS, cS, sD, cD = np.sin(sr), np.cos(sr), np.sin(dr), np.cos(dr)
        sv = np.array([sS, cS, 0.0])
        dv = np.array([cS*cD, -sS*cD, -sD])
        nv = np.array([-cS*sD, sS*sD, -cD])
        v = np.array([x-x_nuc, y-y_nuc, z-z_nuc])
        if (v@sv)**2 + (v@dv)**2 <= r_nuc**2 and abs(v@nv) <= n_tol:
            sxy -= dTau
    return np.array([[sxx, sxy, 0.0], [sxy, syy, 0.0], [0.0, 0.0, szz]])


def fault_basis(strike_deg, dip_deg):
    sr, dr = np.radians(strike_deg), np.radians(dip_deg)
    sv = np.array([np.sin(sr), np.cos(sr), 0.0])
    dv = np.array([np.cos(sr)*np.cos(dr), -np.sin(sr)*np.cos(dr), -np.sin(dr)])
    n = np.cross(sv, dv)
    return n/np.linalg.norm(n), sv, -dv          # normal, strike, up-dip


def cfs(p, n, with_patch=False):
    t = stress_tensor(p, with_patch) @ n
    sn = t @ n                                   # normal traction (neg = compression)
    tau = np.linalg.norm(t - sn*n)
    return tau - (mu_s*(-sn) + cohesion)         # > 0 => fails


print(f"grad_SH = {grad_SH/1e3:.0f} MPa/km | mu_s = {mu_s} | cohesion = {cohesion/1e3:.0f} kPa\n")
print(f"{'fault':22s}{'maxCFS(MPa)':>12s}{'meanCFS':>10s}   status")
print("-"*64)
supercritical = []
for _, r in df.iterrows():
    n, sv, uv = fault_basis(r.strike_deg, r.dip_deg)
    c = np.array([r.centroid_x-ox, r.centroid_y-oy, r.centroid_z])
    vals = []
    for a in np.linspace(-r.extent_strike_m/2, r.extent_strike_m/2, 15):
        for b in np.linspace(-r.extent_dip_m/2, r.extent_dip_m/2, 11):
            vals.append(cfs(c + a*sv + b*uv, n))
    vals = np.array(vals)/1e6
    bad = vals.max() > 0
    print(f"{r['name']:22s}{vals.max():12.2f}{vals.mean():10.2f}   "
          f"{'*** SUPER-CRITICAL (fails on its own)' if bad else 'ok (sub-critical)'}")
    if bad:
        supercritical.append(r['name'])

print("-"*64)
if supercritical:
    print(f"*** {len(supercritical)} fault(s) fail spontaneously -> lower grad_SH "
          f"(or raise mu_s) until all are < 0:\n    {supercritical}")
else:
    print("All faults sub-critical in the background. Good.")

# does the patch actually nucleate Hannes?
h = df[df.name == "Hannes"].iloc[0]
n, _, _ = fault_basis(h.strike_deg, h.dip_deg)
bg = cfs((x_nuc, y_nuc, z_nuc), n, with_patch=False)/1e6
wp = cfs((x_nuc, y_nuc, z_nuc), n, with_patch=True)/1e6
print(f"\nHannes patch centre:  background CFS = {bg:+.2f} MPa | with patch = {wp:+.2f} MPa")
print("  want background < 0 (sub-critical) AND with-patch > 0 (nucleates).")
if wp <= 0:
    print("  -> patch does NOT nucleate at this grad_SH; increase dTau in the yaml.")
