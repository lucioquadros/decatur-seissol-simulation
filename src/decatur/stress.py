"""Initial fault stress model.

Sign convention (SeisSol): tension positive; compression and cohesion are
negative. Stresses are effective (hydrostatic Pf removed). Units: Pa, m.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import strike_dip_vectors


@dataclass(frozen=True)
class StressGradients:
    sv: float           # Pa/m
    shmax: float
    shmin: float
    pf: float
    shmax_azimuth: float

    def tensor(self, z) -> np.ndarray:
        """Effective background stress at elevations z, shape (..., 3, 3)."""
        depth = np.maximum(-np.asarray(z, dtype=float), 0.0)
        th = np.radians(self.shmax_azimuth)
        s2, c2, sc = np.sin(th) ** 2, np.cos(th) ** 2, np.sin(th) * np.cos(th)
        sh_e = (self.shmax - self.pf) * depth
        shmin_e = (self.shmin - self.pf) * depth
        sig = np.zeros(depth.shape + (3, 3))
        sig[..., 0, 0] = -(sh_e * s2 + shmin_e * c2) # σxx
        sig[..., 1, 1] = -(sh_e * c2 + shmin_e * s2) # σyy
        sig[..., 0, 1] = sig[..., 1, 0] = -(sh_e - shmin_e) * sc # σxy
        sig[..., 2, 2] = -(self.sv - self.pf) * depth # σzz
        return sig


@dataclass(frozen=True)
class Patch:
    """Disk in a fault plane where overpressure dp and shear dtau are applied."""
    center: tuple[float, float, float]
    strike: float
    dip: float
    radius: float
    normal_tol: float
    dp: float
    dtau: float

    def contains(self, points) -> np.ndarray:
        """True for each point within the disk (plus normal_tol)."""
        v = np.asarray(points, dtype=float) - np.asarray(self.center)
        s, d, n = strike_dip_vectors(self.strike, self.dip)
        return ((v @ s) ** 2 + (v @ d) ** 2 <= self.radius ** 2) & (np.abs(v @ n) <= self.normal_tol)


def stress_tensor(points, gradients: StressGradients, patch: Patch | None = None) -> np.ndarray:
    """Effective stress at points (N, 3), with the patch perturbation if given."""
    points = np.atleast_2d(points)
    sig = gradients.tensor(points[:, 2])
    if patch is not None:
        inside = patch.contains(points)
        for i in range(3):
            sig[inside, i, i] += patch.dp
        sig[inside, 0, 1] -= patch.dtau
        sig[inside, 1, 0] -= patch.dtau
    return sig


def resolve(sig: np.ndarray, normal) -> tuple[np.ndarray, np.ndarray]:
    """Normal traction (compression < 0) and shear-traction magnitude on a plane."""
    n = np.asarray(normal, dtype=float)
    t = sig @ n
    sn = t @ n # normal traction
    tau = np.linalg.norm(t - sn[..., None] * n, axis=-1) # shear traction magnitude
    return sn, tau


def cfs(sig: np.ndarray, normal, mu_s: float, cohesion) -> np.ndarray:
    """Coulomb failure stress tau - (mu_s * max(-sn, 0) - cohesion), > 0 fails."""
    sn, tau = resolve(sig, normal)
    return tau - (mu_s * np.maximum(-sn, 0.0) - np.asarray(cohesion))


def points_in_patch(patch: Patch, center, strike, dip, extent_strike, extent_dip,
                    n: int = 1000) -> tuple[np.ndarray, float]:
    """Grid cell centers on a planar fault that fall inside the patch and the cell area."""
    s, d, _ = strike_dip_vectors(strike, dip)
    center = np.asarray(center, dtype=float)
    v = np.asarray(patch.center) - center
    nearest = (center + np.clip(v @ s, -extent_strike / 2, extent_strike / 2) * s
               + np.clip(v @ d, -extent_dip / 2, extent_dip / 2) * d)
    cell = extent_strike * extent_dip / n ** 2
    if np.linalg.norm(np.asarray(patch.center) - nearest) > np.hypot(patch.radius, patch.normal_tol):
        return np.empty((0, 3)), cell
    u = (np.arange(n) + 0.5) / n - 0.5
    aa, bb = np.meshgrid(u * extent_strike, u * extent_dip, indexing="ij")
    pts = center + aa.reshape(-1, 1) * s + bb.reshape(-1, 1) * d
    return pts[patch.contains(pts)], cell


def fault_samples(center, strike, dip, extent_strike, extent_dip,
                  n_strike: int = 15, n_dip: int = 11) -> np.ndarray:
    """Grid of points on a planar fault patch shape (n_strike * n_dip, 3)."""
    s, d, _ = strike_dip_vectors(strike, dip)
    a = np.linspace(-extent_strike / 2, extent_strike / 2, n_strike)
    b = np.linspace(-extent_dip / 2, extent_dip / 2, n_dip)
    aa, bb = np.meshgrid(a, b, indexing="ij")
    return np.asarray(center) + aa.reshape(-1, 1) * s + bb.reshape(-1, 1) * d
