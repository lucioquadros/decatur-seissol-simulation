import h5py
import numpy as np
import pytest

from decatur.waves import (bandpass, figure_receiver_section, figure_surface_pgv,
                           read_receivers, read_surface)

VARIABLES = ("s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_xz", "v1", "v2", "v3")
DT, N_SAMPLES, SPACING, VS = 0.005, 201, 100.0, 2000.0


def pulse(t, t0, width=0.05):
    return np.exp(-((t - t0) / width) ** 2)


def write_receiver(path, rid, xyz, t, series):
    head = [f'TITLE = "Temporal Signal for receiver number {rid:05d}"',
            "VARIABLES = " + ",".join(f'"{n}"' for n in ("Time",) + VARIABLES)]
    head += [f"# x{i + 1}       {v:.12e}" for i, v in enumerate(xyz)]
    rows = np.column_stack([t] + [series[n] for n in VARIABLES])
    path.write_text("\n".join(head) + "\n" + "\n".join(
        " ".join(f"{v:.15e}" for v in row) for row in rows) + "\n")


@pytest.fixture
def line(tmp_path):
    """21 receivers on a line at 30 deg from north, an Up pulse moving out from the center at VS."""
    t = np.arange(N_SAMPLES) * DT
    offsets = (np.arange(21) - 10) * SPACING
    az = np.radians(30.0)
    xyz = np.column_stack([500 + offsets * np.sin(az), -200 + offsets * np.cos(az),
                           np.full(21, -1.0)])
    for i, (off, p) in enumerate(zip(offsets, xyz)):
        amp = 0.01 / (1 + abs(off) / 500)
        series = {n: np.zeros_like(t) for n in VARIABLES}
        series["v3"] = amp * pulse(t, 0.1 + abs(off) / VS)
        series["v1"] = 0.5 * series["v3"]
        rid = i + 1
        write_receiver(tmp_path / f"decatur-receiver-{rid:05d}-{i % 2:05d}.dat", rid, p, t,
                       series)
    return tmp_path, offsets, xyz


def test_receivers_read_in_line_order(line):
    d, offsets, xyz = line
    rec = read_receivers(d)
    np.testing.assert_array_equal(rec.ids, np.arange(1, 22))
    np.testing.assert_allclose(rec.xyz, xyz)
    np.testing.assert_allclose(rec.t, np.arange(N_SAMPLES) * DT)
    assert set(rec.data) == set(VARIABLES)
    np.testing.assert_allclose(rec.offsets(), offsets, atol=1e-9)


def test_receiver_peak_velocity(line):
    d, offsets, _ = line
    pgv = read_receivers(d).peak()
    amp = 0.01 / (1 + np.abs(offsets) / 500)
    np.testing.assert_allclose(pgv["vertical"], amp, rtol=1e-3)
    np.testing.assert_allclose(pgv["horizontal"], 0.5 * amp, rtol=1e-3)


def test_receivers_cut_to_the_shortest(line):
    d, _, _ = line
    path = d / "decatur-receiver-00003-00000.dat"
    path.write_text("\n".join(path.read_text().splitlines()[:-5]) + "\n")
    rec = read_receivers(d)
    assert len(rec.t) == N_SAMPLES - 5
    assert rec.data["v3"].shape == (21, N_SAMPLES - 5)


def test_duplicate_receiver_is_rejected(line):
    d, _, _ = line
    (d / "decatur-receiver-00004-00007.dat").write_text(
        (d / "decatur-receiver-00004-00001.dat").read_text())
    with pytest.raises(ValueError, match="receiver 4"):
        read_receivers(d)


@pytest.fixture
def surface(tmp_path):
    """Two triangles, 3 snapshots written, the h5 datasets one snapshot longer."""
    corners = np.array([[[0, 0, 0], [100, 0, 0], [0, 100, 0]],
                        [[100, 0, 0], [100, 100, 0], [0, 100, 0]]], dtype=float)
    v = {"v1": [[0, 0], [3, 0], [0, 1], [9, 9]], "v2": [[0, 0], [4, 0], [0, 1], [9, 9]],
         "v3": [[0, 0], [-2, 0.5], [1, 0], [9, 9]]}
    with h5py.File(tmp_path / "decatur-surface_vertex.h5", "w") as f:
        f["mesh0/geometry"] = corners.reshape(-1, 3)
    with h5py.File(tmp_path / "decatur-surface_cell.h5", "w") as f:
        f["mesh0/connect"] = np.arange(6, dtype=np.uint64).reshape(2, 3)
        f["mesh0/locationFlag"] = np.full(2, 2, dtype=np.uint32)
        for k, vals in v.items():
            f[f"mesh0/{k}"] = np.array(vals, dtype=float)
    (tmp_path / "decatur-surface.xdmf").write_text(
        "".join(f'<Grid><Time Value="{t}"/></Grid>\n' for t in (0, 0.01, 0.02)))
    return tmp_path, corners


def test_surface_peak_velocity(surface):
    d, corners = surface
    surf = read_surface(d)
    np.testing.assert_allclose(surf.t, [0, 0.01, 0.02])
    np.testing.assert_allclose(surf.corners, corners)
    assert surf.data["v1"].shape == (3, 2)
    pgv = surf.peak()
    np.testing.assert_allclose(pgv["horizontal"], [5.0, np.sqrt(2)])
    np.testing.assert_allclose(pgv["vertical"], [2.0, 0.5])


def test_figures_are_written(line, surface, tmp_path):
    rec = read_receivers(line[0])
    surf = read_surface(surface[0])
    assert figure_receiver_section(rec, tmp_path / "section.png").stat().st_size > 0
    assert figure_surface_pgv(surf, tmp_path / "pgv.png", rec).stat().st_size > 0


def test_bandpass_keeps_the_band_and_removes_the_rest():
    dt = 0.005
    t = np.arange(4000) * dt
    inside, low, high = (np.sin(2 * np.pi * f * t) for f in (5.0, 0.1, 60.0))
    out = bandpass(np.array([inside, low, high]), dt, 1.0, 20.0)
    # the 1 Hz corner rings for about 2 s after the sines switch on at the trace ends
    middle = slice(600, 3400)
    np.testing.assert_allclose(out[0, middle], inside[middle], atol=0.01)
    assert np.abs(out[1:, middle]).max() < 0.02


def test_bandpass_is_zero_phase():
    dt = 0.005
    t = np.arange(800) * dt
    out = bandpass(pulse(t, 2.0, 0.1), dt, 0.2, 10.0)
    assert t[np.argmax(out)] == pytest.approx(2.0, abs=dt)


@pytest.mark.parametrize("band", [(2.0, 1.0), (0.0, 5.0), (1.0, 100.0)])
def test_bandpass_rejects_bad_corners(band):
    with pytest.raises(ValueError, match="band-pass"):
        bandpass(np.zeros(100), 0.005, *band)


def test_filtered_section_is_written(line, tmp_path):
    rec = read_receivers(line[0])
    path = figure_receiver_section(rec, tmp_path / "bp.png", band=(0.5, 5.0), order=4)
    assert path.stat().st_size > 0
