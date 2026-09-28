import numpy as np
import pandas as pd
import pytest

from decatur.plotting import run

M0_FINAL, VS, RHO = 1e15, 3000.0, 2700.0


@pytest.fixture
def run_dir(tmp_path):
    """Energy CSV with a raised-cosine moment release from 0.2 to 0.6 s."""
    t = np.round(np.arange(0.0, 2.0001, 0.01), 6)
    phase = np.clip((t - 0.2) / 0.4, 0.0, 1.0)
    m0 = M0_FINAL * 0.5 * (1 - np.cos(np.pi * phase))
    rows = [(ti, var, val) for ti, mi in zip(t, m0)
            for var, val in (("seismic_moment", mi), ("potency", mi / (RHO * VS ** 2)))]
    pd.DataFrame(rows, columns=["time", "variable", "measurement"]).to_csv(
        tmp_path / "decatur-energy.csv", index=False)
    return tmp_path


def test_source_quantities(run_dir):
    q = run(str(run_dir), vs=VS, rho=RHO, figures=False).set_index("quantity")["value"]
    assert q["seismic_moment_final"] == pytest.approx(M0_FINAL)
    assert q["moment_magnitude_Mw"] == pytest.approx(2 / 3 * 15 - 6.03)
    assert q["time_of_peak_moment_rate"] == pytest.approx(0.4, abs=0.011)
    # 5% and 95% of a raised cosine: 0.2 + 0.4 * arccos(+-0.9) / pi
    t05 = 0.2 + 0.4 * np.arccos(0.9) / np.pi
    assert q["moment_T05"] == pytest.approx(t05, abs=2e-3)
    assert q["moment_duration"] == pytest.approx(2 * (0.4 - t05), abs=4e-3)
    assert q["effective_rigidity_M0_over_potency"] == pytest.approx(RHO * VS ** 2)
