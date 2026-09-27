#!/usr/bin/env python3
"""
build_mesh.py
=============
Builds an idealized SeisSol-ready tetrahedral mesh from a fault inventory
CSV, approximating each fault as a planar quadrilateral patch derived from
its centroid, strike, dip, and along-strike / down-dip extents.

Required CSV columns
--------------------
  name            fault identifier (used for display only)
  centroid_x      fault centroid Easting  (m, any consistent projection)
  centroid_y      fault centroid Northing (m)
  centroid_z      fault centroid elevation (m, negative below surface)
  x_min, x_max    fault bounding box Easting  (m) — used for domain sizing
  y_min, y_max    fault bounding box Northing (m) — used for domain sizing
  depth_bottom_m  deepest point of the fault (m, positive down)
  strike_deg      strike azimuth (°, clockwise from North)
  dip_deg         dip angle (°, 0 = horizontal, 90 = vertical)
  extent_strike_m along-strike length of the fault patch (m)
  extent_dip_m    down-dip height of the fault patch (m)

The fault_inventory.py script in this repository produces a CSV with exactly
these columns from GOCAD TSurf (.ts) files.

Workflow
--------
1. Validate required columns and load the fault inventory CSV.
2. Re-centre all coordinates to a local origin at the fault cluster centre
   (avoids floating-point precision issues in Gmsh at large UTM coordinates).
   The origin is written to origin.json so downstream tools (e.g. the
   injection-well coordinates in fault.yaml) can use the same frame.
3. For each fault, compute four corner points of the best-fit plane using
   the centroid, strike, dip, and along-strike / down-dip extents.
   Corners poking above the free surface (Z > 0) trigger a warning.
4. Build the simulation domain as a rectangular box large enough to avoid
   boundary reflections (default: 5 km buffer around the fault cluster).
5. Embed all fault planes into the domain volume using BooleanFragments
   (OpenCASCADE kernel).
6. Classify every surface as free surface, absorbing boundary, or dynamic
   rupture fault by checking whether it lies on the domain box boundary.
   Fragment pieces that fall OUTSIDE the domain volume (no adjacent 3-D
   volume) are reported and left untagged so they never reach SeisSol.
7. Assign SeisSol Physical group tags:
     101  free surface (top face, Z = 0)
     103  dynamic rupture (fault planes)
     105  absorbing boundaries (sides and bottom)
       1  rock volume
8. Apply a Distance/Threshold mesh-size field (distance measured to the
   fault SURFACES, not just their edges) to refine elements near the
   fault surfaces and coarsen them toward the domain boundary.
9. Generate and optimise the 3-D tetrahedral mesh.
10. Write the mesh to mesh.msh (Gmsh format 2.2, readable by PUMGen).

Usage
-----
  python build_mesh.py inventory.csv
  python build_mesh.py inventory.csv --outdir results/
  python build_mesh.py inventory.csv --select FaultA FaultB FaultC
  python build_mesh.py inventory.csv --lc-fault 100 --lc-domain 2000
  python build_mesh.py inventory.csv --lc-fault 25 --dist-min 100

Convert the result to SeisSol format with PUMGen (note: PUMGen takes an
output PREFIX, no extension — it appends .puml.h5 itself):

  pumgen -s msh2 mesh.msh mesh        # produces mesh.puml.h5 + mesh.xdmf

Dependencies: numpy, pandas, gmsh (pip install gmsh)
"""

import sys
import os
import json
import math
import argparse
import textwrap

import numpy as np
import pandas as pd
import gmsh


# -----------------------------------------------------------------------------
# 1.  CONFIGURATION DEFAULTS
# -----------------------------------------------------------------------------

# Mesh element size (metres)
LC_FAULT        = 50.0      # near fault surfaces
LC_DOMAIN       = 1000.0    # at the domain boundary
LC_NUC          = 10.0      # inside the nucleation patch (local refinement)

# Nucleation-patch refinement defaults (metres)
NUC_RADIUS      = 50.0      # radius of the fully-refined ball around the centre
NUC_THICKNESS   = 200.0     # taper over which size grows from LC_NUC back out

# Attractor field distances (metres) — now CLI-configurable (--dist-min/--dist-max)
DIST_MIN        = 200.0     # element size starts increasing beyond this distance
DIST_MAX        = 3000.0    # element size reaches LC_DOMAIN at this distance

# Domain extent (metres) added around the fault cluster bounding box
DOMAIN_BUFFER_XY = 5000.0   # horizontal buffer on all four sides
DOMAIN_BUFFER_Z  = 5000.0   # extra depth below the deepest fault

# Surface classification tolerance (metres)
BOUNDARY_TOL    = 1.0

# SeisSol Physical group tags
TAG_FREE_SURFACE    = 101
TAG_DYNAMIC_RUPTURE = 103
TAG_ABSORBING       = 105
TAG_VOLUME          = 1


# -----------------------------------------------------------------------------
# 2.  GEOMETRY HELPERS
# -----------------------------------------------------------------------------

def local_origin(df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute the re-centring origin as the midpoint of the fault cluster's
    horizontal bounding box.  All X/Y coordinates in the Gmsh model are
    expressed relative to this point to avoid numerical precision issues
    that arise when working at large UTM coordinate values.
    Z is kept as-is (negative below surface).
    """
    ox = (df.x_min.min() + df.x_max.max()) / 2.0
    oy = (df.y_min.min() + df.y_max.max()) / 2.0
    return ox, oy


def fault_corners(cx: float, cy: float, cz: float,
                  strike_deg: float, dip_deg: float,
                  extent_strike: float, extent_dip: float
                  ) -> list[np.ndarray]:
    """
    Compute the four corner points of a planar fault patch.

    Parameters
    ----------
    cx, cy, cz       : fault centroid in local coordinates (metres)
    strike_deg       : strike azimuth (°, clockwise from North/Y-axis)
    dip_deg          : dip angle (°, 0 = horizontal, 90 = vertical)
    extent_strike    : full along-strike length of the patch (m)
    extent_dip       : full down-dip height of the patch (m)

    Returns
    -------
    [P_top_right, P_top_left, P_bottom_left, P_bottom_right] as (3,) arrays.

    Derivation
    ----------
    With X = Easting, Y = Northing, Z = up:
        strike_vec  = (sin(strike), cos(strike), 0)
        down_dip_vec = (cos(strike)*cos(dip), -sin(strike)*cos(dip), -sin(dip))
        up_dip_vec  = -down_dip_vec

    The four corners are the centroid displaced by ±half_strike along
    strike_vec and ±half_dip along up_dip_vec.
    """
    sr = math.radians(strike_deg)
    dr = math.radians(dip_deg)

    strike_vec   = np.array([ math.sin(sr),                      # X
                               math.cos(sr),                      # Y
                               0.0])                              # Z

    down_dip_vec = np.array([ math.cos(sr) * math.cos(dr),       # X
                              -math.sin(sr) * math.cos(dr),       # Y
                              -math.sin(dr)])                      # Z

    up_dip_vec   = -down_dip_vec

    cen = np.array([cx, cy, cz])
    hs  = extent_strike / 2.0
    hd  = extent_dip    / 2.0

    return [cen + hs * strike_vec + hd * up_dip_vec,    # top-right
            cen - hs * strike_vec + hd * up_dip_vec,    # top-left
            cen - hs * strike_vec - hd * up_dip_vec,    # bottom-left
            cen + hs * strike_vec - hd * up_dip_vec]    # bottom-right


def domain_bounds(df: pd.DataFrame, buf_xy: float, buf_z: float,
                  ox: float, oy: float
                  ) -> tuple[float, float, float, float, float, float]:
    """
    Compute domain box corners in local coordinates.

    Returns
    -------
    (x0, y0, z0, x1, y1, z1) where (x0,y0,z0) is the bottom-left-back corner
    and (x1,y1,z1) is the top-right-front corner.
    """
    x0 = df.x_min.min() - ox - buf_xy
    x1 = df.x_max.max() - ox + buf_xy
    y0 = df.y_min.min() - oy - buf_xy
    y1 = df.y_max.max() - oy + buf_xy
    z0 = -(df.depth_bottom_m.max() + buf_z)    # deepest point + buffer
    z1 = 0.0                                    # free surface at Z = 0
    return x0, y0, z0, x1, y1, z1


def check_corners_in_domain(name: str, corners: list[np.ndarray],
                            domain: tuple, tol: float) -> bool:
    """
    Warn if any corner of a fault patch falls outside the domain box
    (most commonly: a shallow fault whose up-dip edge pokes above Z = 0).

    Pieces outside the box survive BooleanFragments as orphan surfaces;
    they are filtered out during classification, but the user should know
    that part of the fault patch has been silently clipped.

    Returns True if all corners are inside the domain.
    """
    x0, y0, z0, x1, y1, z1 = domain
    ok = True
    for c in corners:
        if not (x0 - tol <= c[0] <= x1 + tol and
                y0 - tol <= c[1] <= y1 + tol and
                z0 - tol <= c[2] <= z1 + tol):
            ok = False
            break
    if not ok:
        zmax = max(c[2] for c in corners)
        extra = (f" (top corner Z = {zmax:+.1f} m pokes above the free "
                 f"surface)" if zmax > z1 + tol else "")
        print(f"  WARNING: fault '{name}' extends outside the domain box"
              f"{extra}. The outside portion will be clipped and excluded "
              f"from the dynamic-rupture surface.")
    return ok


def strike_dip_vectors(strike_deg: float, dip_deg: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the (strike_vec, down_dip_vec) unit vectors used to move points
    within a fault plane. IDENTICAL convention to fault_corners() so that a
    nucleation centre placed here lands on the same plane the fault patch
    was built from.

        strike_vec   = (sin s,            cos s,          0)
        down_dip_vec = (cos s · cos d,   -sin s · cos d, -sin d)   (Z<0 = deeper)
    """
    sr, dr = math.radians(strike_deg), math.radians(dip_deg)
    strike_vec   = np.array([math.sin(sr), math.cos(sr), 0.0])
    down_dip_vec = np.array([math.cos(sr) * math.cos(dr),
                             -math.sin(sr) * math.cos(dr),
                             -math.sin(dr)])
    return strike_vec, down_dip_vec


def resolve_nuc_center(df: pd.DataFrame, args, ox: float, oy: float):
    """
    Resolve the nucleation-patch refinement centre in the LOCAL mesh frame
    (the same frame build_mesh.py meshes in, and the frame fault.yaml's
    x_nuc/y_nuc/z_nuc are expressed in). Returns (xc, yc, zc) or None.

    Two mutually exclusive ways to specify it:

      --nuc-fault NAME [--nuc-strike-offset A] [--nuc-dip-offset B]
          centre = centroid + A·strike_vec + B·down_dip_vec   (B>0 = deeper),
          read from the CSV row for NAME, then re-centred with the SAME origin
          (ox, oy) the mesh uses. This is the "adaptable" path: point it at a
          different fault by changing NAME. Because it re-uses the current
          origin, it stays correct even when --select shifts that origin.

      --nuc-center X Y Z
          explicit PROJECTED coordinates (same frame as the CSV centroids),
          re-centred here. Handy if you already have a projected point.

    Returns None when neither flag is given (⇒ no local refinement; the mesh
    is identical to before this feature was added).
    """
    if args.nuc_fault is not None and args.nuc_center is not None:
        print("WARNING: both --nuc-fault and --nuc-center given; "
              "using --nuc-fault.", file=sys.stderr)

    if args.nuc_fault is not None:
        sub = df[df["name"] == args.nuc_fault]
        if sub.empty:
            print(f"ERROR: --nuc-fault '{args.nuc_fault}' not found in the "
                  f"inventory (after --select). Available: "
                  f"{sorted(df['name'].tolist())}", file=sys.stderr)
            sys.exit(1)
        row = sub.iloc[0]
        strike_vec, down_dip_vec = strike_dip_vectors(row.strike_deg,
                                                       row.dip_deg)
        cen = (np.array([row.centroid_x, row.centroid_y, row.centroid_z])
               + args.nuc_strike_offset * strike_vec
               + args.nuc_dip_offset    * down_dip_vec)
        return float(cen[0] - ox), float(cen[1] - oy), float(cen[2])

    if args.nuc_center is not None:
        xp, yp, zp = args.nuc_center
        return float(xp - ox), float(yp - oy), float(zp)

    return None


# -----------------------------------------------------------------------------
# 3.  GMSH HELPERS
# -----------------------------------------------------------------------------

def add_fault_plane(corners: list[np.ndarray]) -> tuple[int, list[int]]:
    """
    Add a planar quadrilateral fault surface to the current Gmsh model.

    Parameters
    ----------
    corners : [P0, P1, P2, P3]  — four coplanar corner points (3-D arrays)

    Returns
    -------
    (surface_tag, [line_tag, line_tag, line_tag, line_tag])

    Note
    ----
    No per-point mesh size is set here: element sizing is governed
    exclusively by the Distance/Threshold background field (see
    add_mesh_size_field), and Mesh.MeshSizeFromPoints is disabled.
    """
    pts = [gmsh.model.occ.addPoint(float(c[0]), float(c[1]), float(c[2]))
           for c in corners]

    lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % 4])
             for i in range(4)]

    loop = gmsh.model.occ.addCurveLoop(lines)
    surf = gmsh.model.occ.addPlaneSurface([loop])

    return surf, lines


def classify_surfaces(domain: tuple, tol: float
                       ) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Classify every 2-D entity in the current Gmsh model as one of:
        free_surface   — top face (Z ≈ Z1 = 0)
        absorbing      — other domain boundary faces
        fault          — internal surfaces embedded in the rock volume
        orphan         — surfaces NOT adjacent to any 3-D volume
                         (fragment pieces that fell outside the domain box)

    Orphan detection matters: BooleanFragments keeps the pieces of a fault
    plane that lie outside the domain volume as standalone surfaces. If
    those were tagged 103, PUMGen/SeisSol would see "fault" faces with no
    adjacent tetrahedra. They are returned separately and left untagged so
    they are excluded from the Physical groups (and therefore from the
    saved mesh, since Mesh.SaveAll = 0).

    Implementation note: orphans are detected GEOMETRICALLY — a surface
    whose bounding box extends beyond the domain box must be a fragment
    piece outside the volume (every retained piece lies inside the box).
    Topological adjacency (gmsh.model.getAdjacencies) cannot be used here:
    OCC stores embedded internal faces as internal shells, so genuine fault
    surfaces report no parent volume even though they are conformally
    meshed with it.

    The boundary classification uses the entity bounding box relative to
    the known domain extents, with a tolerance of *tol* metres.

    Parameters
    ----------
    domain : (x0, y0, z0, x1, y1, z1) — domain box corners in local coords
    tol    : distance tolerance for "on boundary" test (metres)

    Returns
    -------
    (free_tags, absorbing_tags, fault_tags, orphan_tags)
    """
    x0, y0, z0, x1, y1, z1 = domain
    free, absorbing, fault, orphan = [], [], [], []

    for _, tag in gmsh.model.getEntities(2):
        bx0, by0, bz0, bx1, by1, bz1 = gmsh.model.getBoundingBox(2, tag)

        on_top    = bz1 > z1 - tol and bz0 > z1 - tol
        on_bottom = bz0 < z0 + tol and bz1 < z0 + tol
        on_xmin   = bx0 < x0 + tol and bx1 < x0 + tol
        on_xmax   = bx1 > x1 - tol and bx0 > x1 - tol
        on_ymin   = by0 < y0 + tol and by1 < y0 + tol
        on_ymax   = by1 > y1 - tol and by0 > y1 - tol

        # A fragment piece outside the domain box necessarily has a
        # bounding box extending beyond the domain bounds. This test must
        # run FIRST: a fault piece poking above the free surface would
        # otherwise satisfy the on_top test and be mis-tagged as free
        # surface.
        inside = (bx0 >= x0 - tol and bx1 <= x1 + tol and
                  by0 >= y0 - tol and by1 <= y1 + tol and
                  bz0 >= z0 - tol and bz1 <= z1 + tol)

        if not inside:
            orphan.append(tag)
        elif on_top:
            free.append(tag)
        elif on_bottom or on_xmin or on_xmax or on_ymin or on_ymax:
            absorbing.append(tag)
        else:
            fault.append(tag)

    return free, absorbing, fault, orphan


def add_mesh_size_field(fault_surfs: list[int],
                        lc_fault: float, lc_domain: float,
                        dist_min: float, dist_max: float,
                        nuc_center: tuple | None = None,
                        lc_nuc: float = LC_NUC,
                        nuc_radius: float = NUC_RADIUS,
                        nuc_thickness: float = NUC_THICKNESS) -> None:
    """
    Add a Distance + Threshold field to refine the mesh near fault surfaces
    and coarsen it toward the domain boundary, OPTIONALLY combined with a
    spherical Ball field that refines a small nucleation patch further.

    The Distance field measures distance to the fault SURFACES themselves
    (SurfacesList), not just their boundary curves. Measuring to curves —
    a previous version of this script did that — under-refines the interior
    of large fault patches: the centre of a 400–800 m patch is farther than
    dist_min from any edge, so the element size would grow exactly where
    the cohesive zone needs to be resolved.

    The Threshold field applies a linear interpolation:
        size = lc_fault   for distance ≤ dist_min
        size = lc_domain  for distance ≥ dist_max

    Nucleation refinement (when nuc_center is not None)
    ---------------------------------------------------
    A Gmsh ``Ball`` field centred on nuc_center imposes:
        size = lc_nuc                     inside      radius nuc_radius
        size grows lc_nuc → lc_domain     across      nuc_thickness (taper)
        size = lc_domain                  beyond
    The final background size is the ``Min`` of the fault Threshold field and
    the Ball field, so the fine patch wins locally while the rest of the fault
    keeps lc_fault. This is the same pattern the SeisSol example meshes use to
    resolve a nucleation zone (TPV-style) more finely than the host fault.

    nuc_radius should COVER the physical patch (e.g. r_nuc from fault.yaml)
    with a little margin so the whole patch sits in the flat lc_nuc floor
    rather than on the taper; nuc_thickness then grades gently back out.
    """
    dist_tag = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_tag, "SurfacesList", fault_surfs)
    gmsh.model.mesh.field.setNumber(dist_tag,  "Sampling",     100)

    # Threshold field: map distance to mesh size
    thresh_tag = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_tag, "InField",  dist_tag)
    gmsh.model.mesh.field.setNumber(thresh_tag, "SizeMin",  lc_fault)
    gmsh.model.mesh.field.setNumber(thresh_tag, "SizeMax",  lc_domain)
    gmsh.model.mesh.field.setNumber(thresh_tag, "DistMin",  dist_min)
    gmsh.model.mesh.field.setNumber(thresh_tag, "DistMax",  dist_max)

    active_fields = [thresh_tag]

    # Optional spherical refinement around the nucleation patch ---------------
    if nuc_center is not None:
        xc, yc, zc = nuc_center
        ball_tag = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(ball_tag, "XCenter",   xc)
        gmsh.model.mesh.field.setNumber(ball_tag, "YCenter",   yc)
        gmsh.model.mesh.field.setNumber(ball_tag, "ZCenter",   zc)
        gmsh.model.mesh.field.setNumber(ball_tag, "Radius",    nuc_radius)
        gmsh.model.mesh.field.setNumber(ball_tag, "Thickness", nuc_thickness)
        gmsh.model.mesh.field.setNumber(ball_tag, "VIn",       lc_nuc)
        gmsh.model.mesh.field.setNumber(ball_tag, "VOut",      lc_domain)
        active_fields.append(ball_tag)

    # Effective background size = min of all active fields (finest wins) ------
    min_tag = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", active_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

    # Suppress default point/curve size constraints so the field governs alone
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)


# -----------------------------------------------------------------------------
# 4.  REPORTING
# -----------------------------------------------------------------------------

def print_domain_summary(df: pd.DataFrame, domain: tuple,
                         ox: float, oy: float,
                         n_faults: int) -> None:
    x0, y0, z0, x1, y1, z1 = domain
    print("\n" + "=" * 60)
    print("FAULT MESH — DOMAIN SUMMARY")
    print("=" * 60)
    print(f"  Faults included   : {n_faults}")
    print(f"  Local origin      : X={ox:.1f} m, Y={oy:.1f} m")
    print(f"  Domain X          : {x0:.0f} to {x1:.0f} m  ({x1-x0:.0f} m wide)")
    print(f"  Domain Y          : {y0:.0f} to {y1:.0f} m  ({y1-y0:.0f} m wide)")
    print(f"  Domain Z          : {z0:.0f} to {z1:.0f} m  ({-z0:.0f} m deep)")
    print("=" * 60 + "\n")


def print_classification_summary(free: list, absorbing: list,
                                  fault: list, orphan: list,
                                  vol: list) -> None:
    print("\n" + "-" * 40)
    print("Surface classification after BooleanFragments")
    print("-" * 40)
    print(f"  Free surface    (tag 101)  : {len(free):3d} surface(s)")
    print(f"  Dynamic rupture (tag 103)  : {len(fault):3d} surface(s)")
    print(f"  Absorbing       (tag 105)  : {len(absorbing):3d} surface(s)")
    print(f"  Orphan          (untagged) : {len(orphan):3d} surface(s)")
    print(f"  Volumes         (tag 1)    : {len(vol):3d} volume(s)")
    print("-" * 40 + "\n")

    if not fault:
        print("WARNING: No fault surfaces identified. Check BOUNDARY_TOL "
              "or domain dimensions.")
    if not free:
        print("WARNING: No free surface identified. Check domain Z1 = 0.")
    if orphan:
        print(f"WARNING: {len(orphan)} surface fragment(s) lie outside the "
              "domain volume (fault patches extending beyond the box). They "
              "have been excluded from all Physical groups and will not be "
              "written to the mesh.")


def write_origin_sidecar(outdir: str, ox: float, oy: float) -> str:
    """
    Write the re-centring origin to origin.json so downstream tools
    (e.g. fault.yaml's injection-well coordinates) can transform into
    the same local frame:  x_local = x_projected − ox,  y_local = y − oy.
    """
    path = os.path.join(outdir, "origin.json")
    with open(path, "w") as fh:
        json.dump({"origin_x": ox, "origin_y": oy,
                   "note": "local frame: x_local = x - origin_x, "
                           "y_local = y - origin_y, z unchanged"}, fh,
                  indent=2)
    return path


# -----------------------------------------------------------------------------
# 5.  ARGUMENT PARSER
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Build a SeisSol-ready Gmsh tetrahedral mesh from a fault inventory
            CSV, approximating each fault as a planar quadrilateral patch.
        """),
    )
    p.add_argument(
        "inventory",
        metavar="fault_inventory.csv",
        help="Fault inventory CSV produced by fault_inventory.py.",
    )
    p.add_argument(
        "--outdir", default=".", metavar="DIR",
        help="Output directory (default: current directory).",
    )
    p.add_argument(
        "--select", nargs="+", metavar="NAME",
        help="Use only the named faults (default: all faults).",
    )
    p.add_argument(
        "--lc-fault", type=float, default=LC_FAULT, metavar="M",
        help=f"Element size at fault surfaces in metres (default: {LC_FAULT}).",
    )
    p.add_argument(
        "--lc-domain", type=float, default=LC_DOMAIN, metavar="M",
        help=f"Element size at domain boundary in metres (default: {LC_DOMAIN}).",
    )
    p.add_argument(
        "--dist-min", type=float, default=DIST_MIN, metavar="M",
        help=f"Distance from faults at which element size starts growing "
             f"(default: {DIST_MIN}). Scale this with --lc-fault.",
    )
    p.add_argument(
        "--dist-max", type=float, default=DIST_MAX, metavar="M",
        help=f"Distance from faults at which element size reaches "
             f"--lc-domain (default: {DIST_MAX}).",
    )
    # -- nucleation-patch local refinement ------------------------------------
    p.add_argument(
        "--nuc-fault", metavar="NAME",
        help="Centre a local refinement on this fault's plane (name from the "
             "CSV). Combine with --nuc-strike-offset / --nuc-dip-offset to "
             "move the centre along the plane. Change NAME to move the "
             "refinement to a different fault.",
    )
    p.add_argument(
        "--nuc-strike-offset", type=float, default=0.0, metavar="M",
        help="Along-strike shift of the centre from the fault centroid "
             "(default: 0).",
    )
    p.add_argument(
        "--nuc-dip-offset", type=float, default=0.0, metavar="M",
        help="Down-dip shift of the centre from the centroid; >0 = deeper "
             "(default: 0).",
    )
    p.add_argument(
        "--nuc-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="Explicit PROJECTED coordinates of the refinement centre "
             "(alternative to --nuc-fault; same frame as the CSV centroids).",
    )
    p.add_argument(
        "--lc-nuc", type=float, default=LC_NUC, metavar="M",
        help=f"Element size inside the nucleation patch (default: {LC_NUC}). "
             f"Only used when --nuc-fault or --nuc-center is given.",
    )
    p.add_argument(
        "--nuc-radius", type=float, default=NUC_RADIUS, metavar="M",
        help=f"Radius of the fully-refined ball around the nucleation centre "
             f"(default: {NUC_RADIUS}). Set ≥ the fault.yaml patch radius plus "
             f"a small margin.",
    )
    p.add_argument(
        "--nuc-thickness", type=float, default=NUC_THICKNESS, metavar="M",
        help=f"Taper thickness over which element size grows from --lc-nuc "
             f"back to the surrounding size (default: {NUC_THICKNESS}).",
    )
    p.add_argument(
        "--buffer", type=float, default=DOMAIN_BUFFER_XY, metavar="M",
        help=f"Horizontal domain buffer around fault cluster in metres "
             f"(default: {DOMAIN_BUFFER_XY}).",
    )
    p.add_argument(
        "--depth-buffer", type=float, default=DOMAIN_BUFFER_Z, metavar="M",
        help=f"Extra depth below deepest fault in metres "
             f"(default: {DOMAIN_BUFFER_Z}).",
    )
    p.add_argument(
        "--no-mesh", action="store_true",
        help="Build geometry and assign Physical groups, but skip 3-D meshing. "
             "Writes a .brep file for inspection.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Show full Gmsh output.",
    )
    return p


# -----------------------------------------------------------------------------
# 6.  MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    args   = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.dist_min >= args.dist_max:
        print("ERROR: --dist-min must be smaller than --dist-max.",
              file=sys.stderr)
        sys.exit(1)

    # -- load inventory -------------------------------------------------------
    REQUIRED_COLS = {
        "name",
        "centroid_x", "centroid_y", "centroid_z",
        "x_min", "x_max", "y_min", "y_max",
        "depth_bottom_m",
        "strike_deg", "dip_deg",
        "extent_strike_m", "extent_dip_m",
    }
    df = pd.read_csv(args.inventory)
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        print(f"ERROR: missing required column(s): {sorted(missing_cols)}",
              file=sys.stderr)
        sys.exit(1)
    if args.select:
        missing = [n for n in args.select if n not in df["name"].values]
        if missing:
            print(f"ERROR: fault(s) not found in inventory: {missing}",
                  file=sys.stderr)
            sys.exit(1)
        df = df[df["name"].isin(args.select)].reset_index(drop=True)
    n_faults = len(df)
    print(f"Loaded {n_faults} fault(s) from {args.inventory}")

    ox, oy = local_origin(df)
    domain = domain_bounds(df, args.buffer, args.depth_buffer, ox, oy)
    x0, y0, z0, x1, y1, z1 = domain

    print_domain_summary(df, domain, ox, oy, n_faults)
    origin_path = write_origin_sidecar(args.outdir, ox, oy)
    print(f"Local origin written : {origin_path}")

    # -- initialise Gmsh ------------------------------------------------------
    # Everything that follows runs inside try/finally so that any exception
    # (a bad CSV row, an OCC boolean failure, a meshing error) still leaves
    # the Gmsh API cleanly finalised.
    gmsh.initialize()
    try:
        gmsh.model.add("FaultMesh")
        verbosity = 5 if args.verbose else 2
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)

        # -- domain box -------------------------------------------------------
        # Box(x0, y0, z0, dx, dy, dz)
        vol_tag = gmsh.model.occ.addBox(
            x0, y0, z0,
            x1 - x0, y1 - y0, z1 - z0,
        )
        print(f"Domain box created  (tag {vol_tag})")

        # -- fault planes -----------------------------------------------------
        fault_surf_tags = []
        print("Building fault planes...")
        for _, row in df.iterrows():
            cx_local = row.centroid_x - ox
            cy_local = row.centroid_y - oy
            corners  = fault_corners(cx_local, cy_local, row.centroid_z,
                                      row.strike_deg, row.dip_deg,
                                      row.extent_strike_m, row.extent_dip_m)
            check_corners_in_domain(row["name"], corners, domain, BOUNDARY_TOL)
            surf, _ = add_fault_plane(corners)
            fault_surf_tags.append(surf)
            print(f"  {row['name']:22s}  surface tag={surf:3d}"
                  f"  strike={row.strike_deg:5.1f}°  dip={row.dip_deg:4.1f}°")

        gmsh.model.occ.synchronize()
        print(f"\n{n_faults} fault surface(s) defined.")

        # -- BooleanFragments: embed faults into the domain volume ------------
        # This cuts the domain box along every fault plane, creating a
        # conforming interface between the fault and the surrounding rock
        # volume.
        print("Running BooleanFragments (embedding faults into domain)...")
        obj  = [(3, vol_tag)]
        tool = [(2, t) for t in fault_surf_tags]
        gmsh.model.occ.fragment(obj, tool)
        gmsh.model.occ.synchronize()
        print("BooleanFragments complete.")

        # -- classify surfaces ------------------------------------------------
        (free_surfs, absorbing_surfs,
         fault_surfs, orphan_surfs) = classify_surfaces(domain, BOUNDARY_TOL)
        all_vols = [tag for _, tag in gmsh.model.getEntities(3)]
        print_classification_summary(free_surfs, absorbing_surfs,
                                      fault_surfs, orphan_surfs, all_vols)

        # -- assign Physical groups (SeisSol convention) ----------------------
        # Orphan surfaces are deliberately left out of every group; with
        # Mesh.SaveAll = 0 (Gmsh default) they are not written to mesh.msh.
        if free_surfs:
            gmsh.model.addPhysicalGroup(2, free_surfs, TAG_FREE_SURFACE)
            gmsh.model.setPhysicalName(2, TAG_FREE_SURFACE, "free_surface")

        if fault_surfs:
            gmsh.model.addPhysicalGroup(2, fault_surfs, TAG_DYNAMIC_RUPTURE)
            gmsh.model.setPhysicalName(2, TAG_DYNAMIC_RUPTURE,
                                       "dynamic_rupture")

        if absorbing_surfs:
            gmsh.model.addPhysicalGroup(2, absorbing_surfs, TAG_ABSORBING)
            gmsh.model.setPhysicalName(2, TAG_ABSORBING, "absorbing")

        gmsh.model.addPhysicalGroup(3, all_vols, TAG_VOLUME)
        gmsh.model.setPhysicalName(3, TAG_VOLUME, "domain")

        print("Physical groups assigned:")
        print(f"  Physical Surface({TAG_FREE_SURFACE})  -> free_surface")
        print(f"  Physical Surface({TAG_DYNAMIC_RUPTURE}) -> dynamic_rupture")
        print(f"  Physical Surface({TAG_ABSORBING}) -> absorbing")
        print(f"  Physical Volume({TAG_VOLUME}) -> domain")

        # -- mesh size field --------------------------------------------------
        nuc_center = resolve_nuc_center(df, args, ox, oy)
        if nuc_center is not None:
            xc, yc, zc = nuc_center
            if args.lc_nuc > args.lc_fault:
                print(f"WARNING: --lc-nuc ({args.lc_nuc}) is larger than "
                      f"--lc-fault ({args.lc_fault}); the patch will not be "
                      f"refined below the surrounding fault size.")
            print("\nNucleation refinement centre (LOCAL mesh frame):")
            print(f"  x_nuc = {xc:10.2f}  y_nuc = {yc:10.2f}  "
                  f"z_nuc = {zc:10.2f}   (depth {-zc:.1f} m)")
            print("  -> copy these straight into fault.yaml (x_nuc/y_nuc/z_nuc)")
            print(f"  -> refining to {args.lc_nuc} m inside a {args.nuc_radius} m "
                  f"ball, tapering over {args.nuc_thickness} m")

        if fault_surfs:
            add_mesh_size_field(fault_surfs,
                                 args.lc_fault, args.lc_domain,
                                 args.dist_min, args.dist_max,
                                 nuc_center=nuc_center,
                                 lc_nuc=args.lc_nuc,
                                 nuc_radius=args.nuc_radius,
                                 nuc_thickness=args.nuc_thickness)
            print(f"\nMesh size field: {args.lc_fault} m near faults -> "
                  f"{args.lc_domain} m at boundary "
                  f"(transition {args.dist_min}–{args.dist_max} m from "
                  f"fault surfaces)"
                  + (f"; {args.lc_nuc} m at the nucleation patch"
                     if nuc_center is not None else ""))

        # -- generate mesh ----------------------------------------------------
        geo_path = os.path.join(args.outdir, "mesh.geo_unrolled")
        gmsh.write(geo_path)
        print(f"\nGeometry written : {geo_path}")

        if not args.no_mesh:
            print("\nGenerating 3-D tetrahedral mesh...")
            gmsh.model.mesh.generate(3)
            print("Optimising mesh...")
            gmsh.model.mesh.optimize("")           # Laplacian + elem validity

            # Write Gmsh format 2.2 — required by PUMGen
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            msh_path = os.path.join(args.outdir, "mesh.msh")
            gmsh.write(msh_path)
            print(f"Mesh written     : {msh_path}")

            # Report element counts (getElements() with no arguments returns
            # elements of ALL dimensions — points, lines, triangles, tets)
            elem_types, elem_tag_lists, _ = gmsh.model.mesh.getElements()
            type_to_tags = dict(zip(elem_types, elem_tag_lists))
            total  = sum(len(v) for v in type_to_tags.values())
            n_tet  = len(type_to_tags.get(4, []))   # type 4 = 4-node tet
            n_tri  = len(type_to_tags.get(2, []))   # type 2 = 3-node triangle
            print(f"\nMesh statistics:")
            print(f"  All elements (0/1/2/3-D) : {total:,}")
            print(f"  Surface triangles (2-D)  : {n_tri:,}")
            print(f"  Tetrahedra (3-D)         : {n_tet:,}")
            print(f"\nNext step — convert to SeisSol format with PUMGen")
            print(f"(third argument is an output PREFIX, no extension):")
            print(f"  pumgen -s msh2 {msh_path} "
                  f"{os.path.join(args.outdir, 'mesh')}.msh.puml.hdf5")
            print(f"  -> produces "
                  f"{os.path.join(args.outdir, 'mesh')}.puml.h5 "
                  f"(set MeshFile in parameters.par to this)")
        else:
            brep_path = os.path.join(args.outdir, "mesh.brep")
            gmsh.write(brep_path)
            print(f"Geometry (BREP)  : {brep_path}")
            print("(Meshing skipped — run without --no-mesh to generate "
                  "the mesh)")
    finally:
        gmsh.finalize()

    print("\nDone.")


if __name__ == "__main__":
    main()
