import numpy as np
import pandas as pd
import pytest

from decatur.config import INVENTORY_CSV
from decatur.geometry import strike_dip_vectors
from decatur.inventory import build_inventory, fault_geometry
from decatur.ts_io import TSurf, read_tsurf
from conftest import data_dir


def test_planar_rectangle():
    s, d, _ = strike_dip_vectors(120.0, 60.0)
    c = np.array([100.0, 200.0, -1500.0])
    grid = [c + a * s + b * d for a in np.linspace(-300, 300, 7) for b in np.linspace(-100, 100, 3)]
    tris = [[i * 3 + j, i * 3 + j + 1, (i + 1) * 3 + j] for i in range(6) for j in range(2)]
    row = fault_geometry(TSurf("R", np.array(grid), np.array(tris)))
    assert (row["strike_deg"], row["dip_deg"], row["dip_dir_deg"]) == (120.0, 60.0, 210.0)
    assert (row["extent_strike_m"], row["extent_dip_m"]) == (600.0, 200.0)
    assert row["area_m2"] == pytest.approx(600 * 200 / 2, rel=1e-9)
    assert row["depth_top_m"] < row["depth_bottom_m"]


@pytest.mark.skipif(data_dir() is None, reason="DATA_DIR not configured")
def test_regenerates_committed_inventory():
    files = sorted((data_dir() / "Faults").glob("*.ts"))
    got = build_inventory([read_tsurf(f) for f in files])
    want = pd.read_csv(INVENTORY_CSV)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want, check_dtype=False)
