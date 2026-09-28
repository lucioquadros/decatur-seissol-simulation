"""Gmsh mesh of planar faults in a box, and PUML mesh quality checks."""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import fault_corners

TAG_FREE_SURFACE = 101
TAG_DYNAMIC_RUPTURE = 103
TAG_ABSORBING = 105
TAG_VOLUME = 1
BOUNDARY_TOL = 1.0
ALGORITHMS_3D = {"delaunay": 1, "hxt": 10}


@dataclass
class MeshOptions:
    lc_fault: float = 50.0
    lc_domain: float = 1000.0
    dist_min: float = 200.0
    dist_max: float = 3000.0
    buffer: float = 5000.0
    depth_buffer: float = 5000.0
    nuc_center: tuple | None = None
    lc_nuc: float = 10.0
    nuc_radius: float = 50.0
    nuc_thickness: float = 200.0
    threads: int = 1
    algorithm3d: str = "delaunay"


def domain_bounds(faults: pd.DataFrame, origin, buffer: float, depth_buffer: float):
    """(x0, y0, z0, x1, y1, z1) in the local frame, top at z = 0."""
    ox, oy = origin
    return (faults.x_min.min() - ox - buffer, faults.y_min.min() - oy - buffer,
            -(faults.depth_bottom_m.max() + depth_buffer),
            faults.x_max.max() - ox + buffer, faults.y_max.max() - oy + buffer, 0.0)


def local_corners(row, origin) -> np.ndarray:
    """Return fault corners in the local frame."""
    center = (row.centroid_x - origin[0], row.centroid_y - origin[1], row.centroid_z)
    return fault_corners(center, row.strike_deg, row.dip_deg,
                         row.extent_strike_m, row.extent_dip_m)


def corners_outside(corners: np.ndarray, domain, tol: float = BOUNDARY_TOL) -> bool:
    lo, hi = np.array(domain[:3]) - tol, np.array(domain[3:]) + tol
    return bool(np.any((corners < lo) | (corners > hi)))


def classify_surfaces(gmsh, domain, tol: float = BOUNDARY_TOL):
    """Split 2-D entities into (free, absorbing, fault, orphan) tags by bounding box."""
    x0, y0, z0, x1, y1, z1 = domain
    free, absorbing, fault, orphan = [], [], [], []
    for _, tag in gmsh.model.getEntities(2):
        bx0, by0, bz0, bx1, by1, bz1 = gmsh.model.getBoundingBox(2, tag)
        inside = (bx0 >= x0 - tol and bx1 <= x1 + tol and by0 >= y0 - tol
                  and by1 <= y1 + tol and bz0 >= z0 - tol and bz1 <= z1 + tol)
        on_side = (bx1 < x0 + tol or bx0 > x1 - tol or by1 < y0 + tol
                   or by0 > y1 - tol or bz1 < z0 + tol)
        if not inside:
            orphan.append(tag)
        elif bz0 > z1 - tol:
            free.append(tag)
        elif on_side:
            absorbing.append(tag)
        else:
            fault.append(tag)
    return free, absorbing, fault, orphan


def _size_field(gmsh, fault_surfs, o: MeshOptions) -> None:
    f = gmsh.model.mesh.field
    dist = f.add("Distance")
    f.setNumbers(dist, "SurfacesList", fault_surfs)
    f.setNumber(dist, "Sampling", 100)
    thr = f.add("Threshold")
    for k, v in (("InField", dist), ("SizeMin", o.lc_fault), ("SizeMax", o.lc_domain),
                 ("DistMin", o.dist_min), ("DistMax", o.dist_max)):
        f.setNumber(thr, k, v)
    fields = [thr]
    if o.nuc_center is not None:
        ball = f.add("Ball")
        for k, v in zip(("XCenter", "YCenter", "ZCenter"), o.nuc_center):
            f.setNumber(ball, k, float(v))
        for k, v in (("Radius", o.nuc_radius), ("Thickness", o.nuc_thickness),
                     ("VIn", o.lc_nuc), ("VOut", o.lc_domain)):
            f.setNumber(ball, k, v)
        fields.append(ball)
    fmin = f.add("Min")
    f.setNumbers(fmin, "FieldsList", fields)
    f.setAsBackgroundMesh(fmin)
    for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{opt}", 0)


def build_mesh(faults: pd.DataFrame, origin, o: MeshOptions, outdir,
               generate: bool = True, verbose: bool = False, log=print) -> dict:
    """Mesh the faults and write mesh.msh (v2.2 for PUMGen), or mesh.brep if not generate."""
    import gmsh

    domain = domain_bounds(faults, origin, o.buffer, o.depth_buffer)
    x0, y0, z0, x1, y1, z1 = domain
    log(f"domain x {x0:.0f}..{x1:.0f}  y {y0:.0f}..{y1:.0f}  z {z0:.0f}..{z1:.0f} m")
    stats: dict[str, tuple | int] = {"domain": domain}
    gmsh.initialize()
    try:
        gmsh.model.add("decatur")
        gmsh.option.setNumber("General.Verbosity", 5 if verbose else 2)
        gmsh.option.setNumber("General.NumThreads", o.threads)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", ALGORITHMS_3D[o.algorithm3d])
        occ = gmsh.model.occ
        box = occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
        planes = []
        for _, row in faults.iterrows():
            corners = local_corners(row, origin)
            if corners_outside(corners, domain):
                log(f"WARNING: fault '{row['name']}' extends outside the domain, "
                    "and will be dropped")
            pts = [occ.addPoint(float(c[0]), float(c[1]), float(c[2])) for c in corners]
            lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
            planes.append(occ.addPlaneSurface([occ.addCurveLoop(lines)]))
        occ.fragment([(3, box)], [(2, t) for t in planes])
        occ.synchronize()

        free, absorbing, fault, orphan = classify_surfaces(gmsh, domain)
        vols = [t for _, t in gmsh.model.getEntities(3)]
        log(f"surfaces: {len(free)} free, {len(fault)} fault, {len(absorbing)} absorbing, "
            f"{len(orphan)} orphan, {len(vols)} volume(s)")
        if not fault or not free:
            raise RuntimeError("no fault or no free-surface entities after fragment")
        for dim, tags, tag, name in ((2, free, TAG_FREE_SURFACE, "free_surface"),
                                     (2, fault, TAG_DYNAMIC_RUPTURE, "dynamic_rupture"),
                                     (2, absorbing, TAG_ABSORBING, "absorbing"),
                                     (3, vols, TAG_VOLUME, "domain")):
            gmsh.model.addPhysicalGroup(dim, tags, tag)
            gmsh.model.setPhysicalName(dim, tag, name)
        _size_field(gmsh, fault, o)
        gmsh.write(os.path.join(outdir, "mesh.geo_unrolled"))

        if not generate:
            gmsh.write(os.path.join(outdir, "mesh.brep"))
            return stats
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(os.path.join(outdir, "mesh.msh"))
        types, tags, _ = gmsh.model.mesh.getElements()
        counts = {t: len(v) for t, v in zip(types, tags)}
        stats.update(tets=counts.get(4, 0), triangles=counts.get(2, 0))
        return stats
    finally:
        gmsh.finalize()


def read_puml(path) -> tuple[np.ndarray, np.ndarray]:
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f["geometry"]), np.asarray(f["connect"])


def tet_quality(xyz: np.ndarray, cells: np.ndarray) -> dict:
    """Tetrahedron quality metrics: signed volume, insphere radius, inverted/sliver/tiny counts."""
    v0, v1, v2, v3 = (xyz[cells[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
    absv = np.abs(vol)

    def area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    faces = area(v0, v1, v2) + area(v0, v1, v3) + area(v0, v2, v3) + area(v1, v2, v3)
    r_in = 3.0 * absv / np.maximum(faces, 1e-30) # 1e-30 avoids divide-by-zero for degenerate tets
    med_v, med_r = np.median(absv), np.median(r_in) 
    worst = int(r_in.argmin())
    return {
        "elements": len(cells),
        "volume_min": float(vol.min()), "volume_max": float(vol.max()),
        "volume_median": float(med_v),
        "insphere_min": float(r_in.min()), "insphere_median": float(med_r),
        "inverted": int((vol <= 0).sum()),
        "slivers": int((absv < 1e-4 * med_v).sum()),
        "tiny_insphere": int((r_in < 1e-3 * med_r).sum()),
        "worst": worst,
        "timestep_ratio": float(med_r / max(r_in.min(), 1e-30)),
    }
