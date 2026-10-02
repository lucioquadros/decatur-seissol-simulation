"""Receiver and free-surface output of a SeisSol run: readers, peak ground velocity, figures."""

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.signal import butter, sosfiltfilt

from .geometry import fault_corners, local_centroid, strike_dip_vectors

VELOCITIES = ("v1", "v2", "v3")
COMPONENTS = {"v1": "East", "v2": "North", "v3": "Up"}
RECEIVER_FILE = re.compile(r"-receiver-(\d+)-(\d+)\.dat$")

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_HORIZONTAL, C_VERTICAL = "#2a78d6", "#eb6834"
C_FAULT = "#eb6834"
TEXT_SIZES = {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
              "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 13,
              "figure.titlesize": 17}
PGV_CMAP = LinearSegmentedColormap.from_list("pgv", [
    "#f7f6f3", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


@dataclass
class Receivers:
    ids: np.ndarray                 # SeisSol receiver numbers (receivers.dat line, from 1)
    xyz: np.ndarray                 # (N, 3)
    t: np.ndarray                   # (T,)
    data: dict[str, np.ndarray]     # variable -> (N, T)

    def offsets(self) -> np.ndarray:
        """Horizontal distance from the line's center, positive toward the last receiver."""
        xy = self.xyz[:, :2]
        direction = xy[-1] - xy[0]
        return (xy - xy.mean(axis=0)) @ (direction / np.linalg.norm(direction))

    def peak(self) -> dict[str, np.ndarray]:
        v1, v2, v3 = (self.data[k] for k in VELOCITIES)
        return {"horizontal": np.hypot(v1, v2).max(axis=1), "vertical": np.abs(v3).max(axis=1)}


@dataclass
class Surface:
    t: np.ndarray                   # (T,)
    corners: np.ndarray             # (M, 3, 3) triangle vertices
    data: dict[str, np.ndarray]     # variable -> (T, M)

    def peak(self) -> dict[str, np.ndarray]:
        v1, v2, v3 = (self.data[k] for k in VELOCITIES)
        return {"horizontal": np.hypot(v1, v2).max(axis=0), "vertical": np.abs(v3).max(axis=0)}


@dataclass
class FaultTrace:
    name: str
    corners: np.ndarray             # (4, 3) top-right, top-left, bottom-left, bottom-right
    dip: float
    dip_dir: float

    @classmethod
    def from_inventory(cls, row: pd.Series, origin) -> "FaultTrace":
        """Planar fault of an inventory row, in the local frame."""
        corners = fault_corners(local_centroid(row, origin), row.strike_deg, row.dip_deg,
                                row.extent_strike_m, row.extent_dip_m)
        return cls(row["name"], corners, float(row.dip_deg), float(row.dip_dir_deg))


def ruptured_faults(directory, inventory: pd.DataFrame, origin, prefix: str = "decatur",
                    min_fraction: float = 0.5) -> list[str]:
    """Faults with at least min_fraction of their area ruptured (default 0.5 or 50%) at the
    last fault snapshot, each fault face assigned to the nearest inventory plane."""
    d = Path(directory)
    with h5py.File(d / f"{prefix}-fault_cell.h5") as cell, \
            h5py.File(d / f"{prefix}-fault_vertex.h5") as vertex:
        tri = vertex["mesh0/geometry"][:][cell["mesh0/connect"][:].astype(np.int64)]
        slip, rt = cell["mesh0/ASl"][-1], cell["mesh0/RT"][-1]
    centers = tri.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    distances = []
    for _, row in inventory.iterrows():
        strike, down_dip, normal = strike_dip_vectors(row.strike_deg, row.dip_deg)
        r = centers - local_centroid(row, origin)
        off_strike = np.maximum(np.abs(r @ strike) - row.extent_strike_m / 2, 0.0)
        off_dip = np.maximum(np.abs(r @ down_dip) - row.extent_dip_m / 2, 0.0)
        distances.append(np.sqrt((r @ normal) ** 2 + off_strike ** 2 + off_dip ** 2))
    owner = np.argmin(distances, axis=0)
    ruptured = (rt > 0) & (slip > 1e-3)
    return [row["name"] for i, (_, row) in enumerate(inventory.iterrows())
            if area[(owner == i) & ruptured].sum() >= min_fraction * area[owner == i].sum() > 0]


def read_receiver_file(path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Position and time series of one <prefix>-receiver-<id>-<rank>.dat file."""
    names, xyz, rows = [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("VARIABLES"):
            names = [n.strip().strip('"') for n in line.split("=", 1)[1].split(",")]
        elif line.startswith("#"):
            xyz.append(float(line.split()[2]))
        elif line.strip() and not line.startswith("TITLE"):
            rows.append(line.split())
    table = np.array(rows, dtype=float).reshape(-1, len(names))
    return np.array(xyz), {n: table[:, i] for i, n in enumerate(names)}


def read_receivers(directory, prefix: str = "decatur") -> Receivers:
    files = {}
    for path in sorted(Path(directory).glob(f"{prefix}-receiver-*.dat")):
        m = RECEIVER_FILE.search(path.name)
        rid = int(m[1])
        if rid in files:
            raise ValueError(f"receiver {rid} written twice: {files[rid].name}, {path.name}")
        files[rid] = path
    if not files:
        raise FileNotFoundError(f"no {prefix}-receiver-*.dat in {directory}")
    ids = np.array(sorted(files))
    read = [read_receiver_file(files[i]) for i in ids]
    n = min(len(series["Time"]) for _, series in read)
    names = [k for k in read[0][1] if k != "Time"]
    return Receivers(ids, np.array([xyz for xyz, _ in read]), read[0][1]["Time"][:n],
                     {k: np.array([series[k][:n] for _, series in read]) for k in names})


def read_surface(directory, prefix: str = "decatur") -> Surface:
    d = Path(directory)
    times = re.findall(r'<Time Value="([^"]+)"', (d / f"{prefix}-surface.xdmf").read_text())
    with h5py.File(d / f"{prefix}-surface_cell.h5") as cell, \
            h5py.File(d / f"{prefix}-surface_vertex.h5") as vertex:
        corners = vertex["mesh0/geometry"][:][cell["mesh0/connect"][:].astype(np.int64)]
        n = min(len(times), cell["mesh0/v1"].shape[0])
        data = {k: cell[f"mesh0/{k}"][:n] for k in cell["mesh0"] if cell[f"mesh0/{k}"].ndim == 2}
    return Surface(np.array(times[:n], dtype=float), corners, data)


def bandpass(traces, dt: float, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis (forward and backward, Hz)."""
    nyquist = 0.5 / dt
    if not 0.0 < f_low < f_high < nyquist:
        raise ValueError(f"band-pass needs 0 < f_low < f_high < {nyquist:g} Hz (Nyquist), "
                         f"got {f_low:g}, {f_high:g}")
    sos = butter(order, [f_low, f_high], btype="bandpass", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, np.asarray(traces, dtype=float), axis=-1)


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def _text_sizes(figure):
    @functools.wraps(figure)
    def wrapper(*args, **kwargs):
        with plt.rc_context(TEXT_SIZES):
            return figure(*args, **kwargs)
    return wrapper


@_text_sizes
def figure_receiver_section(rec: Receivers, path, title: str = "", dpi: int = 150,
                            band: tuple[float, float] | None = None, order: int = 4) -> Path:
    """Record section of the three velocity components plus peak velocity along the line.

    band (f_low, f_high, Hz) band-passes the velocities first (Butterworth of the given order,
    zero phase), for the traces and the PGV.
    """
    if band is not None:
        dt = float(np.median(np.diff(rec.t)))
        rec = Receivers(rec.ids, rec.xyz, rec.t,
                        {k: bandpass(rec.data[k], dt, *band, order) for k in VELOCITIES})
        label = f"band-pass {band[0]:g}-{band[1]:g} Hz (Butterworth order {order}, zero phase)"
        title = f"{title}, {label}" if title else label
    offsets_km = rec.offsets() / 1e3
    spacing = np.median(np.abs(np.diff(offsets_km))) if len(offsets_km) > 1 else 1.0
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True, layout="constrained",
                             gridspec_kw={"width_ratios": [3, 3, 3, 1.6]})
    fig.get_layout_engine().set(wspace=0.05)
    for ax, key in zip(axes, VELOCITIES):
        v = rec.data[key]
        vmax = np.abs(v).max()
        # one scale per component, so amplitudes compare across the line
        scale = 0.8 * spacing / vmax if vmax > 0 else 0.0
        for off, trace in zip(offsets_km, v):
            ax.plot(rec.t, off + scale * trace, color=INK, linewidth=0.6)
        ax.set_title(f"{COMPONENTS[key]} ({key}), max {1e3 * vmax:.3g} mm/s", color=INK)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(rec.t[0], rec.t[-1])
        _style(ax)
    axes[0].set_ylabel("Distance along the line from the epicenter (km)")

    ax = axes[3]
    for row, ((label, pgv), color) in enumerate(zip(rec.peak().items(),
                                                    (C_HORIZONTAL, C_VERTICAL))):
        ax.plot(1e3 * pgv, offsets_km, color=color, linewidth=2, label=label)
        ax.annotate(f"max {1e3 * pgv.max():.3g}", (1, 1), xycoords="axes fraction",
                    xytext=(0, -1.4 * 12 * row), textcoords="offset points",
                    ha="right", va="top", color=color, fontsize=12)
    ax.set_title("Peak velocity", color=INK)
    ax.set_xlabel("PGV (mm/s)")
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3g}"))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=2)
    _style(ax)
    if title:
        fig.suptitle(title, color=INK)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def _fault_label(fault: FaultTrace) -> str:
    compass = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[round(fault.dip_dir / 45.0) % 8]
    return f"{fault.name}, dip {fault.dip:.0f}° {compass}"


def _draw_fault(ax, fault: FaultTrace, label: str | None) -> None:
    """Top edge of the fault and a true-scale tick to its projected bottom edge."""
    c = fault.corners[:, :2] / 1e3
    top, tick = c[[1, 0]], np.array([c[[0, 1]].mean(axis=0), c[[2, 3]].mean(axis=0)])
    for xy, width, label in ((top, 3.0, label), (tick, 2.0, None)):
        halo = [patheffects.Stroke(linewidth=width + 2.0, foreground="white"),
                patheffects.Normal()]
        ax.plot(*xy.T, color=C_FAULT, linewidth=width, solid_capstyle="butt",
                path_effects=halo, label=label, zorder=4)


@_text_sizes
def figure_surface_pgv(surf: Surface, path, receivers: Receivers | None = None,
                       title: str = "", dpi: int = 150, faults: list[FaultTrace] = ()) -> Path:
    """Maps of the horizontal and vertical peak ground velocity at the free surface."""
    polygons = surf.corners[:, :, :2] / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    for ax, (label, pgv) in zip(axes, surf.peak().items()):
        mm = 1e3 * pgv
        cells = PolyCollection(polygons, array=mm, cmap=PGV_CMAP, edgecolors="face",
                               linewidths=0.1)
        cells.set_clim(0.0, mm.max() if mm.max() > 0 else 1.0)
        ax.add_collection(cells)
        ax.autoscale_view()
        ax.set_aspect("equal")
        fig.colorbar(cells, ax=ax, shrink=0.8, label=f"{label} PGV (mm/s)")
        if receivers is not None:
            ax.plot(*(receivers.xyz[:, :2].T / 1e3), "o", markersize=3, markerfacecolor="none",
                    markeredgewidth=0.6, color=INK, label="receivers")
        for i, fault in enumerate(faults):
            legend = (_fault_label(fault) if len(faults) == 1 else
                      f"ruptured faults ({len(faults)})" if i == 0 else None)
            _draw_fault(ax, fault, legend)
        if receivers is not None or faults:
            ax.legend(frameon=False, loc="upper right")
        ax.set_title(f"{label.capitalize()}, max {mm.max():.3g} mm/s", color=INK)
        ax.set_xlabel("x, east (km)")
        ax.set_ylabel("y, north (km)")
        ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    if title:
        fig.suptitle(f"{title} (0 to {surf.t[-1]:g} s)", color=INK)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)
