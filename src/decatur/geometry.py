"""Fault-plane geometry. Frame: x East, y North, z up. Angles in degrees."""

import numpy as np
import pandas as pd


def strike_dip_vectors(strike_deg: float, dip_deg: float):
    """Unit (strike, down-dip, normal) vectors, with normal = strike x down-dip."""
    s, d = np.radians(strike_deg), np.radians(dip_deg)
    strike = np.array([np.sin(s), np.cos(s), 0.0])
    down_dip = np.array([np.cos(s) * np.cos(d), -np.sin(s) * np.cos(d), -np.sin(d)])
    normal = np.cross(strike, down_dip)
    return strike, down_dip, normal


def best_fit_plane(vertices: np.ndarray):
    """PCA plane: (upward normal, strike-ish axis, dip-ish axis, centroid)."""
    centroid = vertices.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((vertices - centroid).T))
    normal, along_dip, along_strike = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    if normal[2] < 0:
        normal = -normal
    return normal, along_strike, along_dip, centroid


def strike_dip_from_normal(normal: np.ndarray):
    """(strike, dip, dip direction) from an upward normal, right-hand rule."""
    nx, ny, nz = normal
    dip = float(np.degrees(np.arccos(np.clip(nz, 0.0, 1.0))))
    dip_dir = 0.0 if np.hypot(nx, ny) < 1e-9 else float(np.degrees(np.arctan2(nx, ny)) % 360)
    return (dip_dir - 90.0) % 360.0, dip, dip_dir


def fault_corners(center, strike_deg, dip_deg, extent_strike, extent_dip) -> np.ndarray:
    """Corners (top-right, top-left, bottom-left, bottom-right) of a planar fault."""
    strike, down_dip, _ = strike_dip_vectors(strike_deg, dip_deg)
    c = np.asarray(center, dtype=float)
    hs, hd = 0.5 * extent_strike * strike, 0.5 * extent_dip * down_dip
    return np.array([c + hs - hd, c - hs - hd, c - hs + hd, c + hs + hd])


def local_origin(inventory: pd.DataFrame, ground_elevation: float = 0.0):
    """(x, y, z) local/reference frame: mid-point of the faults' horizontal
    bounding box (x,y) and the ground elevation so that z = 0 is the free surface."""
    return ((inventory.x_min.min() + inventory.x_max.max()) / 2.0,
            (inventory.y_min.min() + inventory.y_max.max()) / 2.0, float(ground_elevation))


def local_centroid(row: pd.Series, origin) -> np.ndarray:
    """Centroid of a fault in the local frame."""
    return np.array([row.centroid_x, row.centroid_y, row.centroid_z]) - np.asarray(origin)


def fault_row(inventory: pd.DataFrame, name: str) -> pd.Series:
    rows = inventory[inventory["name"] == name]
    if rows.empty:
        raise KeyError(f"fault '{name}' not in inventory, "
                       f"available: {sorted(inventory['name'])}")
    return rows.iloc[0]


def point_on_fault(row: pd.Series, strike_offset: float = 0.0,
                   dip_offset: float = 0.0) -> np.ndarray:
    """Projected point at the centroid offseted along a fault plane (dip_offset > 0 is deeper)."""
    strike, down_dip, _ = strike_dip_vectors(row.strike_deg, row.dip_deg)
    centroid = np.array([row.centroid_x, row.centroid_y, row.centroid_z])
    return centroid + strike_offset * strike + dip_offset * down_dip


def select_faults(inventory: pd.DataFrame, names) -> pd.DataFrame:
    """Subset of the inventory (None or empty selects all)."""
    if not names:
        return inventory.reset_index(drop=True)
    missing = [n for n in names if n not in set(inventory["name"])]
    if missing:
        raise KeyError(f"fault(s) not in inventory: {missing}")
    return inventory[inventory["name"].isin(names)].reset_index(drop=True)
