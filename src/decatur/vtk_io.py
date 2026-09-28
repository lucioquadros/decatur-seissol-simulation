"""VTK XML unstructured grids (.vtu, raw appended binary) for ParaView."""

import numpy as np

VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10

_TYPES = {np.dtype("<f8"): "Float64", np.dtype("<f4"): "Float32", np.dtype("<i8"): "Int64",
          np.dtype("<i4"): "Int32", np.dtype("u1"): "UInt8"}


def compact(points: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the points used by cells, renumbered."""
    used, inverse = np.unique(cells, return_inverse=True)
    return points[used], inverse.reshape(cells.shape)


def write_vtu(path, points: np.ndarray, cells: np.ndarray, vtk_type: int,
              cell_data: dict[str, np.ndarray] | None = None) -> None:
    n_cells, per_cell = cells.shape
    blocks = [("Points", np.ascontiguousarray(points, dtype="<f8"), 3),
              ("connectivity", np.ascontiguousarray(cells, dtype="<i8").ravel(), 1),
              ("offsets", np.arange(1, n_cells + 1, dtype="<i8") * per_cell, 1),
              ("types", np.full(n_cells, vtk_type, dtype="u1"), 1)]
    for name, values in (cell_data or {}).items():
        values = np.asarray(values)
        dtype = "<f8" if values.dtype.kind == "f" else "<i4"
        blocks.append((name, np.ascontiguousarray(values, dtype=dtype), 1))

    tags, offset = {}, 0
    for name, values, ncomp in blocks:
        tags[name] = (f'<DataArray type="{_TYPES[values.dtype]}" Name="{name}" '
                      f'NumberOfComponents="{ncomp}" format="appended" offset="{offset}"/>')
        offset += 8 + values.nbytes
    cell_tags = "".join(tags[name] for name, _, _ in blocks[4:])
    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">\n<UnstructuredGrid>\n'
        f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{n_cells}">\n'
        f'<Points>{tags["Points"]}</Points>\n'
        f'<Cells>{tags["connectivity"]}{tags["offsets"]}{tags["types"]}</Cells>\n'
        f'<CellData>{cell_tags}</CellData>\n'
        '</Piece>\n</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')
    with open(path, "wb") as fh:
        fh.write(header.encode())
        for _, values, _ in blocks:
            fh.write(np.uint64(values.nbytes).tobytes())
            fh.write(values.tobytes())
        fh.write(b"\n</AppendedData>\n</VTKFile>\n")


def read_vtu(path) -> dict[str, np.ndarray]:
    """Arrays of a file written by write_vtu, by name."""
    import re
    raw = open(path, "rb").read()
    start = raw.index(b'<AppendedData encoding="raw">') + len(b'<AppendedData encoding="raw">\n_')
    kinds = {v: k for k, v in _TYPES.items()}
    out = {}
    for m in re.finditer(rb'<DataArray type="(\w+)" Name="(\w+)" NumberOfComponents="(\d)" '
                         rb'format="appended" offset="(\d+)"/>', raw):
        dtype, name, ncomp, off = kinds[m[1].decode()], m[2].decode(), int(m[3]), int(m[4])
        nbytes = int(np.frombuffer(raw, "<u8", 1, start + off)[0])
        values = np.frombuffer(raw, dtype, nbytes // dtype.itemsize, start + off + 8)
        out[name] = values.reshape(-1, ncomp) if ncomp > 1 else values
    return out
