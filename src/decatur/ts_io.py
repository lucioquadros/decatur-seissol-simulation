"""GOCAD TSurf (.ts) reader and STL writer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TSurf:
    name: str
    vertices: np.ndarray   # (N, 3), z up
    triangles: np.ndarray  # (M, 3) row indices into vertices


def read_ts(path) -> list[TSurf]:
    """Read every TSurf object in a file."""
    path = Path(path)
    surfaces = []
    lines = iter(path.read_text().splitlines())
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == "GOCAD":
            if tokens[1].lower() == "tsurf":
                surfaces.append(_read_object(lines, path.stem))
            else:
                _skip_object(lines)
    return surfaces


def read_tsurf(path) -> TSurf:
    """Read a file holding exactly one TSurf."""
    surfaces = read_ts(path)
    if len(surfaces) != 1:
        raise ValueError(f"{path}: expected one TSurf, found {len(surfaces)}")
    return surfaces[0]


def _skip_object(lines) -> None:
    for line in lines:
        if line.strip() == "END":
            return


def _read_object(lines, default_name: str) -> TSurf:
    name = default_name
    z_sign = 1.0
    xyz: list[list[float]] = []
    index: dict[int, int] = {}
    tris: list[list[int]] = []
    in_header = False
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0]
        if in_header:
            if line.strip() == "}":
                in_header = False
            elif line.strip().lower().startswith("name:"):
                name = line.split(":", 1)[1].strip()
            continue
        if key == "HEADER":
            in_header = not line.rstrip().endswith("}")
        elif key == "ZPOSITIVE":
            z_sign = -1.0 if tokens[1].lower() == "depth" else 1.0
        elif key in ("VRTX", "PVRTX"):
            index[int(tokens[1])] = len(xyz)
            xyz.append([float(t) for t in tokens[2:5]])
        elif key in ("ATOM", "PATOM"):
            index[int(tokens[1])] = index[int(tokens[2])]
        elif key == "TRGL":
            tris.append([index[int(t)] for t in tokens[1:4]])
        elif key == "END":
            break
    if not xyz:
        raise ValueError(f"TSurf '{name}' has no vertices")
    vertices = np.array(xyz, dtype=float)
    vertices[:, 2] *= z_sign
    triangles = np.array(tris, dtype=np.int64).reshape(-1, 3)
    return TSurf(name, vertices, triangles)


def write_stl(path, surfaces: list[TSurf], offset=(0.0, 0.0, 0.0)) -> None:
    """Write surfaces as one ASCII STL file, one solid per surface."""
    offset = np.asarray(offset, dtype=float)
    with open(path, "w") as fh:
        for s in surfaces:
            solid = s.name.replace(" ", "_")
            fh.write(f"solid {solid}\n")
            v = s.vertices - offset
            for a, b, c in v[s.triangles]:
                n = np.cross(b - a, c - a)
                n /= max(np.linalg.norm(n), 1e-30)
                fh.write(f"facet normal {n[0]:e} {n[1]:e} {n[2]:e}\n"
                         " outer loop\n")
                for p in (a, b, c):
                    fh.write(f"  vertex {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                fh.write(" endloop\nendfacet\n")
            fh.write(f"endsolid {solid}\n")
