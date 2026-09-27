#!/usr/bin/env python3
"""
fault_inventory.py
==================
Reads one or more GOCAD TSurf (.ts) files and produces:

  1. fault_inventory.csv   - per-fault plannar geometry table
  2. fault_map.png         - map-view (X-Y) of all faults actual geometry
  3. fault_depth.png       - depth-profile view (X-Z and Y-Z) of all faults
                             actual geometry
  4. fault_3d.png          - snapshot of the interactive 3D view

Plannar geometry computed per fault
-----------------------------------
  - Centroid (X, Y, Z)
  - Bounding box / depth extent
  - Strike and dip (PCA best-fit plane, right-hand rule, X=East / Y=North)
  - Along-strike and down-dip extents
  - Total triangulated fault surface area
  - Mesh resolution (average triangle edge length)

Usage
-----
  python fault_inventory.py path/to/*.ts [OPTIONS]

Dependencies: numpy, scipy, matplotlib, pandas
"""

import sys
import os
import glob
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# -----------------------------------------------------------------------------
# PARSER
# -----------------------------------------------------------------------------

def parse_ts(filepath: str):
    """
    Parse a GOCAD TSurf (.ts) file.

    Returns
    -------
    name      : str
        Fault name from the HEADER block; falls back to the filename stem.
    vertices  : ndarray, shape (N, 3), dtype float64
        XYZ vertex positions.
    triangles : ndarray, shape (M, 3), dtype int32
        Triangle connectivity into *vertices* (empty array if no TRGL lines).

    """
    name = Path(filepath).stem
    vertices: list[list[float, float, float]] = []
    triangles: list[list[int, int, int]] = []

    with open(filepath, "r") as fh:
        in_header = False
        for raw_line in fh:
            line = raw_line.strip()
            tokens = line.split()
            if not tokens:
                continue

            # -- header block ------------------------------------------
            if tokens[0] == "HEADER":
                in_header = True
                continue
            if in_header:
                if line == "}":
                    in_header = False
                elif line.lower().startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                continue

            # -- geometry ----------------------------------------------
            if tokens[0] == "VRTX" and len(tokens) >= 5:
                idx = int(tokens[1])
                vertices.append([float(tokens[2]),
                                   float(tokens[3]),
                                   float(tokens[4])])

            elif tokens[0] == "TRGL" and len(tokens) >= 4:
                triangles.append((int(tokens[1]),
                                   int(tokens[2]),
                                   int(tokens[3])))

    if not vertices:
        raise ValueError(f"No vertices found in '{filepath}'.")

    return name, np.array(vertices), np.array(triangles)

# -----------------------------------------------------------------------------
# FAULT GEOMETRY 
# -----------------------------------------------------------------------------

def best_fit_plane(vertices: np.ndarray):
    """
    Compute the best-fit plane through a point cloud using PCA.

    Returns
    -------
    normal       : ndarray (3,)  - upward-pointing unit normal
    along_strike : ndarray (3,)  - eigenvector of the largest eigenvalue
    along_dip    : ndarray (3,)  - eigenvector of the middle eigenvalue
    centroid     : ndarray (3,)  - mean position
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    # np.linalg.eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))

    normal       = eigenvectors[:, 0].copy()   # smallest eigenvalue
    along_dip    = eigenvectors[:, 1].copy()   # middle
    along_strike = eigenvectors[:, 2].copy()   # largest

    if normal[2] < 0:
        normal = -normal

    return normal, along_strike, along_dip, centroid

def strike_dip_from_normal(normal: np.ndarray):
    """
    Convert an upward-pointing plane normal to strike and dip.

    Coordinate convention: X = East, Y = North, Z = up.

    Derivation
    ----------
    nx, ny, nz = normal
    With nz >= 0 (upward normal):
      dip          = arccos(nz)                - angle from vertical
      dip_dir      = atan2(nx, ny) % 360       - azimuth of steepest descent,
                                                 measured clockwise from North
      strike       = (dip_dir - 90) % 360      - right-hand rule

    Returns
    -------
    strike_deg   : float  0-360°
    dip_deg      : float  0-90°
    dip_dir_deg  : float  0-360°
    """
    nx, ny, nz = normal

    dip_deg = float(np.degrees(np.arccos(np.clip(nz, 0.0, 1.0))))

    # Horizontal component of the normal - dip direction azimuth
    horiz_mag = np.hypot(nx, ny)
    if horiz_mag < 1e-9:
        # Horizontal fault - strike is undefined; return 0 by convention
        dip_dir_deg = 0.0
    else:
        dip_dir_deg = float(np.degrees(np.arctan2(nx, ny)) % 360)

    strike_deg = (dip_dir_deg - 90.0) % 360.0

    return strike_deg, dip_deg, dip_dir_deg

def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Vectorised triangle area computation.

    Uses the cross-product formula: area = 0.5 * |AB x AC|
    Returns a 1-D array of per-triangle areas (m^2).
    Returns [0.0] for meshes with no triangles.
    """
    if len(triangles) == 0: # if FILE has no TRGL lines
        return np.array([0.0])
    A = vertices[triangles[:, 0]]
    B = vertices[triangles[:, 1]]
    C = vertices[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)

def fault_geometry(name: str,
                   vertices: np.ndarray,
                   triangles: np.ndarray) -> dict:
    """
    Compute all scalar geometry metrics for a single fault surface.

    Depth convention: Z is negative below the surface.
      depth_top_m    - shallowest point, reported as positive metres depth
      depth_bottom_m - deepest point, reported as positive metres depth
    """
    normal, along_strike, along_dip, centroid = best_fit_plane(vertices)
    strike, dip, dip_dir = strike_dip_from_normal(normal)

    centered      = vertices - centroid
    proj_strike   = centered @ along_strike
    proj_dip      = centered @ along_dip
    extent_strike = float(proj_strike.max() - proj_strike.min())
    extent_dip    = float(proj_dip.max()    - proj_dip.min())

    areas      = triangle_areas(vertices, triangles)
    total_area = float(areas.sum())

    z      = vertices[:, 2]
    depths = -z                          # positive values
    depth_top_m    = float(depths.min())
    depth_bottom_m = float(depths.max())

    return {
        "name":            name,
        "n_vertices":      int(len(vertices)),
        "n_triangles":     int(len(triangles)),
        "centroid_x":      round(float(centroid[0]), 2),
        "centroid_y":      round(float(centroid[1]), 2),
        "centroid_z":      round(float(centroid[2]), 2),
        "x_min":           round(float(vertices[:, 0].min()), 2),
        "x_max":           round(float(vertices[:, 0].max()), 2),
        "y_min":           round(float(vertices[:, 1].min()), 2),
        "y_max":           round(float(vertices[:, 1].max()), 2),
        "depth_top_m":     round(depth_top_m,    1),
        "depth_bottom_m":  round(depth_bottom_m, 1),
        "depth_range_m":   round(depth_bottom_m - depth_top_m, 1),
        "strike_deg":      round(strike,  1),
        "dip_deg":         round(dip,     1),
        "dip_dir_deg":     round(dip_dir, 1),
        "extent_strike_m": round(extent_strike, 1),
        "extent_dip_m":    round(extent_dip,    1),
        "area_m2":         round(total_area, 1),
        "area_km2":        round(total_area / 1e6, 6),
    }

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

def _depth_cmap(vertices_list: list[np.ndarray]):
    """Return a normaliser anchored to the global depth range."""
    all_z = np.concatenate([v[:, 2] for v in vertices_list])
    return mcolors.Normalize(vmin=all_z.min(), vmax=all_z.max())

def plot_map_view(faults: list[dict], outpath: str):
    """
    Map view (X-Y plane): each fault coloured by mean depth.

    Each triangle is drawn as a filled polygon whose colour encodes the
    mean Z of its three vertices.  Fault centroids are marked and labelled.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap  = plt.cm.viridis_r.reversed()
    norm  = _depth_cmap([f["vertices"] for f in faults])

    for fd in faults:
        verts = fd["vertices"]
        trgls = fd["triangles"]
        geom  = fd["geom"]
        name  = geom["name"]

        if len(trgls) > 0:
            for tri in trgls:
                xy    = verts[tri, :2]          # shape (3, 2)
                mean_z = verts[tri, 2].mean()
                poly   = plt.Polygon(xy,
                                     closed=True,
                                     facecolor=cmap(norm(mean_z)),
                                     edgecolor="none",
                                     alpha=0.7,
                                     zorder=2)
                ax.add_patch(poly)
            # thin mesh outline for legibility
            for tri in trgls:
                xy = np.vstack([verts[tri, :2], verts[tri[0], :2]])
                ax.plot(xy[:, 0], xy[:, 1],
                        color="white", lw=0.2, alpha=0.3, zorder=3)
        else:
            # Point cloud only
            sc = ax.scatter(verts[:, 0], verts[:, 1], c=verts[:, 2],
                            cmap=cmap, norm=norm, s=6, zorder=2)

        # Centroid marker + label
        cx, cy = geom["centroid_x"], geom["centroid_y"]
        ax.scatter(cx, cy, s=30, color="red", zorder=5, marker="+")
        ax.annotate(name, xy=(cx, cy),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="red", zorder=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("Depth (m)", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_title("Decatur CO2 faults map view", fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")

def plot_depth_profiles(faults: list[dict], outpath: str):
    """
    Two depth-profile panels:
      Left  - X-Z (EW cross-section)
      Right - Y-Z (NS cross-section)

    Points coloured by mean triangle depth; fault names annotated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                             layout="constrained")
    cmap = plt.cm.viridis_r.reversed()
    norm = _depth_cmap([f["vertices"] for f in faults])

    for fd in faults:
        verts = fd["vertices"]
        trgls = fd["triangles"]
        geom  = fd["geom"]
        name  = geom["name"]

        for ax_idx, (horiz_col, xlabel) in enumerate(
                [(0, "X (m)"), (1, "Y (m)")]):
            ax = axes[ax_idx]
            if len(trgls) > 0:
                for tri in trgls:
                    pts    = verts[tri]
                    h      = pts[:, horiz_col]
                    z      = pts[:, 2]
                    mean_z = z.mean()
                    poly   = plt.Polygon(
                        np.column_stack([h, z]),
                        closed=True,
                        facecolor=cmap(norm(mean_z)),
                        edgecolor="none",
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.add_patch(poly)
            else:
                ax.scatter(verts[:, horiz_col], verts[:, 2],
                           c=verts[:, 2], cmap=cmap, norm=norm,
                           s=6, zorder=2)

            # Centroid annotation
            cx_h = (geom["centroid_x"] if horiz_col == 0
                    else geom["centroid_y"])
            cz   = geom["centroid_z"]
            ax.scatter(cx_h, cz, s=30, color="red",
                       marker="+", zorder=5)
            ax.annotate(name, xy=(cx_h, cz),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="red", zorder=6)

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_autoscale_on(True)
            ax.relim()
            ax.autoscale_view()

    axes[0].set_ylabel("Depth (m)",
                        fontsize=10)
    axes[0].set_title("E-W depth profile (X-Z)", fontsize=11)
    axes[1].set_title("N-S depth profile (Y-Z)", fontsize=11)
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, pad=0.02,
                 label="Depth (m)")
    fig.suptitle("Fault depth profiles", fontsize=12)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")

# -----------------------------------------------------------------------------
# INTERACTIVE 3-D PLOT
# -----------------------------------------------------------------------------

def plot_3d_interactive(faults: list[dict], outpath: str, show: bool = True):
    """
    Render all fault surfaces in an interactive 3-D matplotlib window.

    Each fault is drawn as a Poly3DCollection so the full triangulation is
    visible.  Triangle faces are coloured by their mean elevation (Z) using
    the viridis_r colourmap (deeper = darker).  The window is rotatable and
    zoomable with the mouse; a PNG snapshot is saved to *outpath*.

    Parameters
    ----------
    faults  : list of fault dicts produced by the loading loop in main()
    outpath : PNG file path for the saved snapshot
    show    : if True, open the interactive window (plt.show()).
              Set to False when running headlessly / in a pipeline.

    Design notes
    ------------
    - Poly3DCollection accepts a list of (3, 3) arrays — one per triangle —
      where columns are X, Y, Z.  We build this in a single vectorised step
      with verts[trgls]  →  shape (M, 3, 3).
    - Face colours are set per-triangle via `facecolors`, keyed to the global
      depth normaliser so colour is comparable across faults.
    - Edge colour is kept white at very low alpha so the mesh is hinted at
      without overwhelming thin fault surfaces.
    - Axis limits are set explicitly from the global vertex cloud so the
      view does not change when faults are added or removed.
    - Equal-aspect ratio is approximated by scaling all three axes to the
      same data range, which mpl's 3D axes does not do automatically.
    - Fault name labels use a solid white bbox background instead of relying
      on zorder.  In matplotlib's 3D engine, all artists are depth-sorted by
      their projected Z and painted back-to-front (painter's algorithm).
      Because text participates in the same sort, zorder cannot override a
      polygon that projects closer to the viewer — it is simply painted on
      top.  A solid bbox ensures the label is always readable even when a
      triangle face partially overlaps it.
    """
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.viridis_r.reversed()
    norm = _depth_cmap([f["vertices"] for f in faults])

    all_verts = np.concatenate([f["vertices"] for f in faults], axis=0)

    for fd in faults:
        verts = fd["vertices"]
        trgls = fd["triangles"]
        geom  = fd["geom"]

        if len(trgls) > 0:
            # Shape: (M, 3, 3) — M triangles × 3 vertices × (X,Y,Z)
            polys    = verts[trgls]
            mean_z   = polys[:, :, 2].mean(axis=1)          # (M,) depth per face
            face_col = cmap(norm(mean_z))                    # (M, 4) RGBA

            collection = Poly3DCollection(
                polys,
                facecolors=face_col,
                edgecolors=(1, 1, 1, 0.12),   # near-transparent white edges
                linewidths=0.3,
                zorder=2,
            )
            ax.add_collection3d(collection)

        else:
            # Point-cloud fallback for meshes with no triangles
            sc = ax.scatter(
                verts[:, 0], verts[:, 1], verts[:, 2],
                c=verts[:, 2], cmap=cmap, norm=norm,
                s=8, zorder=2,
            )

        # Centroid marker
        cx, cy, cz = (geom["centroid_x"],
                      geom["centroid_y"],
                      geom["centroid_z"])
        ax.scatter(cx, cy, cz, color="red", s=40, marker="+",
                   depthshade=False)
        # A solid white bbox is the only reliable way to keep labels readable
        # in matplotlib 3D — zorder cannot override the painter's algorithm
        # depth sort that determines draw order for 3D artists.
        ax.text(cx, cy, cz, f"  {geom['name']}",
                fontsize=7, color="red",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white",
                          edgecolor="none",
                          alpha=0.85))

    # -- axis limits: equal aspect in all three dimensions --------------------
    x_range = all_verts[:, 0].max() - all_verts[:, 0].min()
    y_range = all_verts[:, 1].max() - all_verts[:, 1].min()
    z_range = all_verts[:, 2].max() - all_verts[:, 2].min()
    max_range = max(x_range, y_range, z_range) * 0.55   # half-range

    x_mid = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2
    y_mid = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2
    z_mid = (all_verts[:, 2].max() + all_verts[:, 2].min()) / 2

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    # -- labels & decoration ---------------------------------------------------
    ax.set_xlabel("X (m)",   fontsize=9, labelpad=8)
    ax.set_ylabel("Y (m)",  fontsize=9, labelpad=8)
    ax.set_zlabel("Depth (m)", fontsize=9, labelpad=8)
    ax.set_title(
        "Decatur Faults - interactive 3-D view\n"
        "Left-drag: rotate   |   Right-drag / scroll: zoom",
        fontsize=10,
    )
    ax.tick_params(labelsize=7)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.55, aspect=15,
                 label="Depth (m)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"  Saved: {outpath}")

    if show:
        plt.show()   # blocks until the window is closed

    plt.close(fig)

# -----------------------------------------------------------------------------
# TERMINAL OUTPUT
# -----------------------------------------------------------------------------

def print_inventory(df: pd.DataFrame):
    """Print the inventory table to stdout in a readable columnar format."""
    display_cols = [
        ("name",            "Name",          "<30"),
        ("n_vertices",      "Verts",          ">6"),
        ("n_triangles",     "Tris",           ">6"),
        ("depth_top_m",     "Top(m)",         ">8"),
        ("depth_bottom_m",  "Bot(m)",         ">8"),
        ("depth_range_m",   "dDep(m)",        ">8"),
        ("strike_deg",      "Strike°",        ">8"),
        ("dip_deg",         "Dip°",           ">6"),
        ("extent_strike_m", "ExtStrike(m)",  ">13"),
        ("extent_dip_m",    "ExtDip(m)",     ">10"),
        ("area_km2",        "Area(km²)",     ">10"),
    ]
    header = " ".join(
        format(label, fmt) for _, label, fmt in display_cols
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("FAULT INVENTORY")
    print(sep)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        line = " ".join(
            format(str(row[col]), fmt) for col, _, fmt in display_cols
        )
        print(line)
    print(sep + "\n")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Fault inventory tool for GOCAD TSurf (.ts) files.

            Computes geometry (strike, dip, extent, area, depth) for each fault.
            Writes a CSV inventory table and PNG plots.
        """),
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE.ts",
        help="One or more GOCAD TSurf files (glob patterns accepted).",
    )
    p.add_argument(
        "--outdir",
        default=".",
        metavar="DIR",
        help="Directory for output files (default: current directory).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating all PNG figures.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Save the 3-D figure but do not open the interactive window. "
        ),
    )
    return p

def resolve_inputs(raw: list[str]) -> list[str]:
    """Expand any glob patterns (i.e., issue on Windows not expanding fnames).
    Also removes duplicate entries (e.g, if the same file is passed twice).
    Returns sorted unique .ts file paths."""
    paths = []
    for fname in raw:
        expanded = glob.glob(fname)
        if not expanded:
            print(f"WARNING: No files matched '{fname}'. Skipped.", file=sys.stderr)
            continue
        for fpath in expanded:
            if Path(fpath).suffix != '.ts':
                print(f"WARNING: '{fpath}' is not a .ts file. Skipped.", file=sys.stderr)
                continue
            paths.append(fpath)
    # Remove any duplicate entries
    return sorted(set(paths))

def main():
    args   = build_arg_parser().parse_args()
    files  = resolve_inputs(args.inputs)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    # -- parse ----------------------------------------------------------------
    faults = []
    print(f"\nLoading {len(files)} file(s)...")
    for fpath in files:
        try:
            name, verts, trgls = parse_ts(fpath)
            geom = fault_geometry(name, verts, trgls)
            faults.append({"name": name, "file": fpath,
                            "vertices": verts, "triangles": trgls,
                            "geom": geom})
            print(f"  OK  {name:30s}  "
                  f"{len(verts):4d} verts  "
                  f"{len(trgls):4d} tris  "
                  f"strike={geom['strike_deg']:5.1f}°  "
                  f"dip={geom['dip_deg']:4.1f}°")
        except Exception as exc:
            print(f"  ERROR parsing '{fpath}': {exc}", file=sys.stderr)

    if not faults:
        print("No faults loaded successfully.", file=sys.stderr)
        sys.exit(1)

    # -- tables ---------------------------------------------------------------
    inv_df = pd.DataFrame([f["geom"] for f in faults])

    print_inventory(inv_df)

    inv_path = os.path.join(outdir, "fault_inventory.csv")
    inv_df.to_csv(inv_path, index=False)
    print(f"  Saved: {inv_path}")

    # -- plots ----------------------------------------------------------------
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_map_view(
            faults,
            os.path.join(outdir, "fault_map.png"),
        )
        plot_depth_profiles(
            faults,
            os.path.join(outdir, "fault_depth.png"),
        )
        plot_3d_interactive(
            faults,
            os.path.join(outdir, "fault_3d.png"),
            show=not args.no_show,
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
