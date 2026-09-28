"""Layered elastic model (units top-down between horizons, local frame) and its ASAGI grid."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MATERIAL_CSV
from .ts_io import TSurf, read_tsurf

SURFACE = "surface"


@dataclass(frozen=True)
class Unit:
    id: int
    name: str
    top: str   # a horizon .ts stem in DATA_DIR, or a constant elevation
    vp: float
    vs: float
    rho: float

    @property
    def mu(self) -> float:
        return self.rho * self.vs ** 2

    @property
    def lam(self) -> float:
        return self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)


def read_units(path=MATERIAL_CSV) -> list[Unit]:
    df = pd.read_csv(path)
    units = [Unit(int(r.unit_id), str(r.unit), str(r.top), float(r.vp), float(r.vs),
                  float(r.rho)) for r in df.itertuples()]
    check_units(units)
    return units


def check_units(units: list[Unit]) -> None:
    if [u.id for u in units] != list(range(1, len(units) + 1)):
        raise ValueError("unit ids must be 1, 2, ... in top-down order")
    if units[0].top != SURFACE or any(u.top == SURFACE for u in units[1:]):
        raise ValueError(f"only the first unit may (and must) have top '{SURFACE}'")
    for u in units:
        if not (u.rho > 0 and u.vs > 0 and u.vp > np.sqrt(2.0) * u.vs):
            raise ValueError(f"{u.name}: need rho, vs > 0 and vp > sqrt(2) vs (lambda > 0)")


def constant_top(unit: Unit) -> float | None:
    try:
        return float(unit.top)
    except ValueError:
        return None


def axis(lo: float, hi: float, step: float) -> np.ndarray:
    """Regular coordinates for gridding with a step  [lo, hi]."""
    return np.arange(np.floor(lo / step), np.ceil(hi / step) + 1) * step


def rasterize(vertices: np.ndarray, triangles: np.ndarray, x: np.ndarray,
              y: np.ndarray) -> np.ndarray:
    """Elevation of a triangulated surface at every grid node, NaN where no triangle covers.

    Uses barycentric per triangle, because Petrel triangulations can overlap.
    """
    dx, dy = x[1] - x[0], y[1] - y[0]
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    i0 = np.maximum(np.ceil((lo[:, 0] - x[0]) / dx).astype(int), 0)
    i1 = np.minimum(np.floor((hi[:, 0] - x[0]) / dx).astype(int), len(x) - 1)
    j0 = np.maximum(np.ceil((lo[:, 1] - y[0]) / dy).astype(int), 0)
    j1 = np.minimum(np.floor((hi[:, 1] - y[0]) / dy).astype(int), len(y) - 1)
    ni, nj = np.clip(i1 - i0 + 1, 0, None), np.clip(j1 - j0 + 1, 0, None)
    count = ni * nj
    t = np.repeat(np.arange(len(triangles)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    ii, jj = i0[t] + k % ni[t], j0[t] + k // ni[t]
    px, py = x[ii], y[jj]
    ax, ay, bx, by, cx, cy = a[t, 0], a[t, 1], b[t, 0], b[t, 1], c[t, 0], c[t, 1]
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    ok = np.abs(det) > 1e-12
    det = np.where(ok, det, 1.0)
    l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
    l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
    l3 = 1.0 - l1 - l2
    inside = ok & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
    z = np.full((len(y), len(x)), np.nan)
    z[jj[inside], ii[inside]] = (l1 * a[t, 2] + l2 * b[t, 2] + l3 * c[t, 2])[inside]
    return z


def grid_surface(surface: TSurf, x: np.ndarray, y: np.ndarray, origin,
                 taper: float) -> np.ndarray:
    """Surface elevation (ny, nx) in the local frame, edge extended values blend into the mean elevation."""
    from scipy.ndimage import distance_transform_edt

    z = rasterize(surface.vertices - np.asarray(origin), surface.triangles, x, y)
    covered = np.isfinite(z)
    if not covered.any():
        raise ValueError(f"surface '{surface.name}' does not overlap the grid")
    dist, (iy, ix) = distance_transform_edt(~covered, sampling=(y[1] - y[0], x[1] - x[0]),
                                            return_indices=True)
    w = 0.5 - 0.5 * np.cos(np.pi * np.clip(dist / taper, 0.0, 1.0))
    return np.where(covered, z, (1.0 - w) * z[iy, ix] + w * z[covered].mean())


@dataclass
class LayeredModel:
    units: list[Unit]
    x: np.ndarray
    y: np.ndarray
    tops: np.ndarray   # top of units 2..n

    def __post_init__(self):
        self.tops = np.minimum.accumulate(np.minimum(self.tops, 0.0), axis=0)

    def unit_at(self, points) -> np.ndarray:
        """Unit id at points (N, 3). A point on an interface belongs to the upper unit."""
        from scipy.interpolate import RegularGridInterpolator
        p = np.atleast_2d(np.asarray(points, dtype=float))
        uid = np.ones(len(p), dtype=int)
        for top in self.tops:
            f = RegularGridInterpolator((self.y, self.x), top)
            uid += p[:, 2] < f(p[:, [1, 0]])
        return uid

    def unit_grid(self, z: np.ndarray) -> np.ndarray:
        """Unit id on the grid (nz, ny, nx)."""
        uid = np.ones((len(z), len(self.y), len(self.x)), dtype=np.int32)
        for top in self.tops:
            uid += z[:, None, None] < top[None]
        return uid

    def top_range(self, unit_id: int) -> tuple[float, float]:
        top = self.tops[unit_id - 2]
        return float(top.min()), float(top.max())


def build_model(units: list[Unit], x, y, origin, taper: float, data_dir=None) -> LayeredModel:
    """Grid every unit top (except the free surface) over x, y."""
    tops = []
    for u in units[1:]:
        c = constant_top(u)
        if c is not None:
            tops.append(np.full((len(y), len(x)), c))
            continue
        if data_dir is None:
            raise ValueError(f"unit {u.name} needs DATA_DIR for its top '{u.top}'")
        tops.append(grid_surface(read_tsurf(Path(data_dir) / f"{u.top}.ts"), x, y, origin,
                                 taper))
    return LayeredModel(units, np.asarray(x), np.asarray(y), np.array(tops))


def check_vertical_range(model: LayeredModel, z_min: float, z_max: float, dz: float) -> None:
    """Check that the layered material interfaces are within the range of the grid."""
    hi = model.tops.max()
    lo = model.tops.min()
    if hi >= z_max - dz or lo <= z_min + dz:
        raise ValueError(f"interfaces span z {lo:.0f} to {hi:.0f} m, outside the grid's "
                         f"inner range {z_min + dz:g} to {z_max - dz:g} m. "
                         "Widen material.z_min/z_max")


def write_unit_grid(path, x, y, z, unit_id: np.ndarray) -> None:
    """COARDS NetCDF."""
    from netCDF4 import Dataset
    with Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as nc:
        for name, values in (("x", x), ("y", y), ("z", z)):
            nc.createDimension(name, len(values))
            var = nc.createVariable(name, "f8", (name,))
            var.units = "m"
            var[:] = values
        var = nc.createVariable("unit_id", "f4", ("z", "y", "x"))
        var.long_name = "material unit id (data/material_units.csv)"
        var[:] = unit_id.astype(np.float32)


def read_unit_grid(path):
    from netCDF4 import Dataset
    with Dataset(path) as nc:
        return (np.asarray(nc["x"][:]), np.asarray(nc["y"][:]), np.asarray(nc["z"][:]),
                np.rint(np.asarray(nc["unit_id"][:])).astype(np.int32))
