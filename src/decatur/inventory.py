"""Planar summary of each fault surface, and overview figures."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import best_fit_plane, strike_dip_from_normal
from .ts_io import TSurf


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return np.zeros(1)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def fault_geometry(surface: TSurf) -> dict:
    """One inventory row. Depths are positive down, extents come from the PCA axes."""
    v = surface.vertices
    normal, along_strike, along_dip, centroid = best_fit_plane(v)
    strike, dip, dip_dir = strike_dip_from_normal(normal)
    proj_s = (v - centroid) @ along_strike
    proj_d = (v - centroid) @ along_dip
    area = float(triangle_areas(v, surface.triangles).sum())
    depth_top, depth_bottom = float(-v[:, 2].max()), float(-v[:, 2].min())
    return {
        "name": surface.name,
        "n_vertices": len(v),
        "n_triangles": len(surface.triangles),
        "centroid_x": round(float(centroid[0]), 2),
        "centroid_y": round(float(centroid[1]), 2),
        "centroid_z": round(float(centroid[2]), 2),
        "x_min": round(float(v[:, 0].min()), 2),
        "x_max": round(float(v[:, 0].max()), 2),
        "y_min": round(float(v[:, 1].min()), 2),
        "y_max": round(float(v[:, 1].max()), 2),
        "depth_top_m": round(depth_top, 1),
        "depth_bottom_m": round(depth_bottom, 1),
        "depth_range_m": round(depth_bottom - depth_top, 1),
        "strike_deg": round(strike, 1),
        "dip_deg": round(dip, 1),
        "dip_dir_deg": round(dip_dir, 1),
        "extent_strike_m": round(float(np.ptp(proj_s)), 1),
        "extent_dip_m": round(float(np.ptp(proj_d)), 1),
        "area_m2": round(area, 1),
        "area_km2": round(area / 1e6, 6),
    }


def build_inventory(surfaces: list[TSurf]) -> pd.DataFrame:
    return pd.DataFrame([fault_geometry(s) for s in surfaces])


def format_inventory(df: pd.DataFrame) -> str:
    cols = [("name", "Name", "<24"), ("n_vertices", "Verts", ">6"),
            ("depth_top_m", "Top(m)", ">8"), ("depth_bottom_m", "Bot(m)", ">8"),
            ("strike_deg", "Strike", ">7"), ("dip_deg", "Dip", ">5"),
            ("extent_strike_m", "LenStrike", ">10"), ("extent_dip_m", "LenDip", ">8"),
            ("area_km2", "Area(km2)", ">10")]
    lines = [" ".join(format(label, fmt) for _, label, fmt in cols)]
    for _, row in df.iterrows():
        lines.append(" ".join(format(str(row[c]), fmt) for c, _, fmt in cols))
    return "\n".join(lines)


def _depth_norm(surfaces):
    z = np.concatenate([s.vertices[:, 2] for s in surfaces])
    return Normalize(vmin=z.min(), vmax=z.max())


def _label(ax, df, hcol, vcol):
    for _, r in df.iterrows():
        ax.scatter(r[hcol], r[vcol], s=30, color="red", marker="+", zorder=5)
        ax.annotate(r["name"], (r[hcol], r[vcol]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="red", zorder=6)


def plot_map(surfaces: list[TSurf], df: pd.DataFrame, path) -> None:
    norm, cmap = _depth_norm(surfaces), plt.cm.viridis
    fig, ax = plt.subplots(figsize=(10, 8))
    for s in surfaces:
        tri = s.vertices[s.triangles]
        ax.add_collection(PolyCollection(tri[:, :, :2], facecolors=cmap(norm(tri[:, :, 2].mean(1))),
                                         edgecolors="none", alpha=0.7))
    _label(ax, df, "centroid_x", "centroid_y")
    ax.autoscale_view()
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Elevation (m)")
    ax.set(aspect="equal", xlabel="X (m)", ylabel="Y (m)", title="Decatur faults, map view")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sections(surfaces: list[TSurf], df: pd.DataFrame, path) -> None:
    norm, cmap = _depth_norm(surfaces), plt.cm.viridis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, layout="constrained")
    for ax, h, hcol, label in ((axes[0], 0, "centroid_x", "X (m)"),
                               (axes[1], 1, "centroid_y", "Y (m)")):
        for s in surfaces:
            tri = s.vertices[s.triangles]
            ax.add_collection(PolyCollection(tri[:, :, [h, 2]],
                                             facecolors=cmap(norm(tri[:, :, 2].mean(1))),
                                             edgecolors="none", alpha=0.7))
        _label(ax, df, hcol, "centroid_z")
        ax.autoscale_view()
        ax.set(aspect="equal", xlabel=label)
        ax.grid(True, lw=0.3, alpha=0.4)
    axes[0].set(ylabel="Elevation (m)", title="E-W section (X-Z)")
    axes[1].set(title="N-S section (Y-Z)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="Elevation (m)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3d(surfaces: list[TSurf], df: pd.DataFrame, path, show: bool = False) -> None:
    norm, cmap = _depth_norm(surfaces), plt.cm.viridis
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    for s in surfaces:
        tri = s.vertices[s.triangles]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cmap(norm(tri[:, :, 2].mean(1))),
                                             edgecolors=(1, 1, 1, 0.12), linewidths=0.3))
    for _, r in df.iterrows():
        # a solid bbox is needed: 3-D artists are depth-sorted and zorder is ignored
        ax.text(r.centroid_x, r.centroid_y, r.centroid_z, f"  {r['name']}", fontsize=7,
                color="red", bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                       edgecolor="none", alpha=0.85))
    v = np.concatenate([s.vertices for s in surfaces])
    mid, half = (v.max(0) + v.min(0)) / 2, 0.55 * np.ptp(v, axis=0).max()
    ax.set(xlim=(mid[0] - half, mid[0] + half), ylim=(mid[1] - half, mid[1] + half),
           zlim=(mid[2] - half, mid[2] + half), xlabel="X (m)", ylabel="Y (m)",
           zlabel="Elevation (m)", title="Decatur faults")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.12, shrink=0.55,
                 label="Elevation (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
