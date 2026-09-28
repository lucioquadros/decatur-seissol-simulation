"""Moment rate, energy budget and solver-performance figures from SeisSol CSV output.

Reads <prefix>-energy.csv (required) and the flops, clustering and threadPinning
CSVs when present. 
Reference: docs/visualization.md.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scipy.signal import savgol_filter
    HAVE_SCIPY = True
except ImportError:                                    # pragma: no cover
    HAVE_SCIPY = False


# -----------------------------------------------------------------------------
# 1.  CONFIGURATION DEFAULTS
# -----------------------------------------------------------------------------

# Brune (1970) constant in  r = K_BRUNE * Vs / f_c
K_BRUNE      = 0.37

# Fraction of the final moment used to define the rupture duration
DURATION_FRACTIONS = (0.05, 0.90, 0.95)

# Plot styling
FIGSIZE_SOURCE = (16.0, 9.0)
FIGSIZE_ENERGY = (16.0, 5.0)
FIGSIZE_PERF   = (13.0, 9.0)
FIGSIZE_SINGLE = (9.0, 5.5)

C_MOMENT = "#1f4e79"      # cumulative moment
C_RATE   = "#c0392b"      # moment rate
C_FIT    = "#7f8c8d"      # model / reference curves
C_ELAS   = "#2980b9"
C_KIN    = "#e67e22"
C_FRIC   = "#27ae60"
C_BREAK  = "#8e44ad"


# -----------------------------------------------------------------------------
# 2.  LOADERS
# -----------------------------------------------------------------------------

def find_input(directory: str, prefix: str, kind: str) -> str | None:
    """
    Locate '<directory>/<prefix>-<kind>.csv', falling back to a glob on
    '*<kind>.csv' so the script still works if the run prefix differs from
    the default.  Returns None when nothing matches.
    """
    exact = os.path.join(directory, f"{prefix}-{kind}.csv")
    if os.path.isfile(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(directory, f"*{kind}.csv")))
    return matches[0] if matches else None


def load_energy(path: str) -> pd.DataFrame:
    """
    Read the long-format energy CSV and pivot it to wide format
    (index = time, one column per variable).

    Two quirks of the SeisSol output are handled here:

      * `plastic_moment` is written twice per timestep in some builds, so a
        naive pivot raises on duplicate entries.  aggfunc='mean' collapses
        them. For a run with no plasticity both copies are zero anyway.
      * Rows are not guaranteed to be time-sorted, so the index is sorted
        explicitly before any differentiation happens downstream.
    """
    raw = pd.read_csv(path)
    required = {"time", "variable", "measurement"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")

    wide = raw.pivot_table(index="time", columns="variable",
                           values="measurement", aggfunc="mean")
    wide = wide.sort_index()
    wide.columns.name = None

    if "seismic_moment" not in wide.columns:
        raise ValueError(f"{path}: no 'seismic_moment' variable found.")

    return wide


def load_flops(path: str) -> pd.DataFrame | None:
    """
    Read the FLOP/s CSV.  Column names carry the rank number
    (rank_0_accumulated, rank_0_current, ...), so the rank list is discovered
    rather than assumed.
    """
    df = pd.read_csv(path)
    if "time" not in df.columns:
        print(f"  WARNING: {path} has no 'time' column, skipping.",
              file=sys.stderr)
        return None
    return df.sort_values("time").reset_index(drop=True)


def flops_ranks(df: pd.DataFrame) -> list[int]:
    """Return the sorted list of rank numbers present in a flops dataframe."""
    ranks = set()
    for col in df.columns:
        parts = col.split("_")
        if len(parts) == 3 and parts[0] == "rank" and parts[1].isdigit():
            ranks.add(int(parts[1]))
    return sorted(ranks)


def load_clustering(path: str) -> pd.DataFrame | None:
    """
    Read the LTS clustering CSV.  Rows are (profilingId, localId, layerType,
    size, dynamicRuptureSize, rank, localRank). `localId` is the LTS cluster
    index and `layerType` distinguishes Interior from Copy (MPI halo) layers.
    """
    df = pd.read_csv(path)
    needed = {"localId", "layerType", "size", "dynamicRuptureSize"}
    missing = needed - set(df.columns)
    if missing:
        print(f"  WARNING: {path} missing {sorted(missing)}, skipping.",
              file=sys.stderr)
        return None
    return df


def load_thread_pinning(path: str) -> pd.DataFrame | None:
    """Read the thread-pinning CSV.  Used for figure captions, not plotted."""
    df = pd.read_csv(path)
    return df if len(df) else None


def describe_pinning(df: pd.DataFrame | None) -> str:
    """Condense the thread-pinning table into a one-line caption."""
    if df is None or df.empty:
        return "thread pinning: not available"
    n_rank = len(df)
    host = df["hostname"].iloc[0] if "hostname" in df.columns else "?"
    nproc = df["nproc"].iloc[0] if "nproc" in df.columns else "?"
    numa = "?"
    if "workernuma" in df.columns:
        nodes = {n.strip() for n in str(df["workernuma"].iloc[0]).split(",")}
        numa = f"{len(nodes)} NUMA node(s)"
    return (f"host {host} | {n_rank} rank(s) | {nproc} threads/rank | {numa}")


# -----------------------------------------------------------------------------
# 3.  SOURCE-PHYSICS DERIVATIONS
# -----------------------------------------------------------------------------

def moment_magnitude(m0: float) -> float:
    """Hanks & Kanamori (1979) moment magnitude, M0 in N*m."""
    return (2.0 / 3.0) * np.log10(m0) - 6.03


def moment_rate(t: np.ndarray, m0: np.ndarray,
                smooth_window: int = 0) -> np.ndarray:
    """
    Differentiate the cumulative seismic moment to get the moment rate.

    np.gradient is used rather than np.diff: it applies second-order central
    differences in the interior (one-sided at the ends) and returns an array
    the same length as the input, so the result stays on the original time
    axis instead of being offset by half a sample.

    If *smooth_window* > 0 a Savitzky-Golay filter is applied to M0 BEFORE
    differentiating.  Smoothing the cumulative curve is preferable to
    smoothing the derivative: it preserves the total moment (the endpoint
    value) and does not redistribute area under the rate pulse.  The window
    must be odd and larger than the polynomial order (3).
    """
    if smooth_window and smooth_window > 3:
        if not HAVE_SCIPY:
            print("  WARNING: --smooth requested but scipy is unavailable, "
                  "differentiating the raw curve instead.", file=sys.stderr)
        else:
            win = int(smooth_window) | 1                # force odd
            win = min(win, len(m0) - 1 if len(m0) % 2 == 0 else len(m0))
            m0 = savgol_filter(m0, window_length=win, polyorder=3)

    return np.gradient(m0, t)


def duration_metrics(t: np.ndarray, m0: np.ndarray,
                     fractions=DURATION_FRACTIONS) -> dict:
    """
    Times at which the cumulative moment first reaches given fractions of its
    final value, by linear interpolation between samples.

    Returns a dict keyed 'T05', 'T90', ... plus 'duration' = T95 - T05 when
    both are available.
    """
    m_final = m0[-1]
    out = {}
    if m_final <= 0:
        return out
    for frac in fractions:
        target = frac * m_final
        idx = np.searchsorted(m0, target)
        if idx == 0:
            out[f"T{int(frac*100):02d}"] = float(t[0])
        elif idx >= len(m0):
            out[f"T{int(frac*100):02d}"] = float(t[-1])
        else:
            t0, t1 = t[idx - 1], t[idx]
            m_a, m_b = m0[idx - 1], m0[idx]
            w = 0.0 if m_b == m_a else (target - m_a) / (m_b - m_a)
            out[f"T{int(frac*100):02d}"] = float(t0 + w * (t1 - t0))
    if "T05" in out and "T95" in out:
        out["duration"] = out["T95"] - out["T05"]
    return out


def source_spectrum(t: np.ndarray, mdot: np.ndarray,
                    pad_factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Far-field source amplitude spectrum from the moment-rate function.

    The Fourier transform of the moment rate is the source spectrum whose
    zero-frequency asymptote is the total seismic moment. The returned
    amplitude is scaled by dt so that |Omega(0)| == M0_final (a useful
    self-check printed by the summary).

    Zero-padding by *pad_factor* only interpolates the spectrum -- it adds no
    information, but it makes the corner-frequency fit better conditioned by
    supplying more points below f_c.
    """
    dt = float(np.mean(np.diff(t)))
    n = len(mdot)
    nfft = 1 << int(np.ceil(np.log2(max(n * pad_factor, 8))))
    spec = np.fft.rfft(mdot, n=nfft) * dt
    freq = np.fft.rfftfreq(nfft, dt)
    return freq, np.abs(spec)


def corner_frequency(freq: np.ndarray, amp: np.ndarray, m0_final: float,
                     hi_amp_frac: float = 0.1, n_grid: int = 2000) -> dict:
    """
    Estimate the corner frequency of the source spectrum, two ways.

    Why two.  The Brune omega-squared model

        Omega(f) = Omega_0 / (1 + (f / f_c)^2)

    assumes a high-frequency falloff of exactly f^-2.  A smooth simulated
    source-time function decays considerably faster than that, so a
    least-squares fit taken over too wide a band trades a low f_c against an
    inflated Omega_0 and lands on a corner that is a factor of two too low.
    Fitting with a free plateau on this dataset returns Omega_0 = 1.8 * M0,
    which is unphysical: the zero-frequency asymptote of the moment-rate
    spectrum IS the seismic moment, and it is known exactly.

    So:

      f_half  PRIMARY.  The frequency at which the amplitude first crosses
              Omega_0 / 2, with Omega_0 fixed at the final seismic moment.
              For the Brune model this crossing is the definition of f_c, and
              measuring it directly makes no assumption about the falloff
              exponent.  Found by interpolating in log-log space.

      f_fit   SECONDARY.  Grid-search fit of the omega-squared model with the
              plateau pinned to M0, restricted to frequencies where the
              observed amplitude still exceeds hi_amp_frac * Omega_0 (default
              10%, i.e. roughly the corner region rather than the far tail).
              Reported for comparison. It remains sensitive to that band, and
              the summary flags a large discrepancy between the two.

    Returns f_c set to f_half -- the estimate the downstream source
    dimensions use.
    """
    if not np.isfinite(m0_final) or m0_final <= 0:
        return {"f_c": np.nan, "f_half": np.nan, "f_fit": np.nan,
                "omega_0": np.nan, "band_hi": np.nan, "misfit": np.nan}

    omega_0 = m0_final
    pos = freq > 0
    f, a = freq[pos], amp[pos]
    valid = a > 0
    f, a = f[valid], a[valid]

    # -- primary: half-amplitude crossing, interpolated in log-log ------------
    target = omega_0 / 2.0
    below = np.flatnonzero(a < target)
    f_half = np.nan
    if below.size and below[0] > 0:
        i = below[0]
        # Interpolate log10(f) against log10(amp). amp is decreasing here, so
        # the arrays are reversed to keep np.interp's ascending-x contract.
        f_half = 10.0 ** np.interp(np.log10(target),
                                   [np.log10(a[i]), np.log10(a[i - 1])],
                                   [np.log10(f[i]), np.log10(f[i - 1])])

    # -- secondary: band-limited omega-squared fit, plateau pinned to M0 ------
    f_fit, misfit, band_hi = np.nan, np.nan, np.nan
    tail = np.flatnonzero(a < hi_amp_frac * omega_0)
    if tail.size:
        band_hi = float(f[tail[0]])
        band = f <= band_hi
        if band.sum() >= 8:
            log_obs = np.log10(a[band])
            fb = f[band]
            grid = np.logspace(np.log10(max(f[0], 1e-3)),
                               np.log10(band_hi * 4.0), n_grid)
            best = (np.inf, np.nan)
            for fc in grid:
                model = np.log10(omega_0 / (1.0 + (fb / fc) ** 2))
                mis = float(np.mean((log_obs - model) ** 2))
                if mis < best[0]:
                    best = (mis, fc)
            misfit, f_fit = best

    return {"f_c": f_half, "f_half": f_half, "f_fit": f_fit,
            "omega_0": omega_0, "band_hi": band_hi, "misfit": misfit,
            "dc_amplitude": float(amp[0])}


def source_dimensions(m0: float, f_c: float, vs: float) -> dict:
    """
    Convert a corner frequency into a circular-crack source radius and static
    stress drop.

        r        = 0.37 * Vs / f_c                (Brune 1970)
        delta_s  = 7 * M0 / (16 * r^3)            (Eshelby 1957 circular crack)

    Returns NaNs when f_c could not be fitted.
    """
    if not np.isfinite(f_c) or f_c <= 0:
        return {"radius": np.nan, "stress_drop": np.nan, "area": np.nan}
    r = K_BRUNE * vs / f_c
    return {"radius": r,
            "stress_drop": 7.0 * m0 / (16.0 * r ** 3),
            "area": np.pi * r ** 2}


def analyze_source(wide: pd.DataFrame, vs: float, rho: float,
                   smooth_window: int, corner_band: float = 0.1) -> dict:
    """
    Assemble every scalar and series the source figures need, in one place, so
    the plotting functions stay presentational.
    """
    t = wide.index.to_numpy(dtype=float)
    m0 = wide["seismic_moment"].to_numpy(dtype=float)
    mdot = moment_rate(t, m0, smooth_window)

    freq, amp = source_spectrum(t, mdot)
    brune = corner_frequency(freq, amp, float(m0[-1]), corner_band)
    dims = source_dimensions(m0[-1], brune["f_c"], vs)

    res = {
        "t": t, "m0": m0, "mdot": mdot,
        "freq": freq, "amp": amp,
        "brune": brune, "dims": dims,
        "dt_out": float(np.mean(np.diff(t))),
        "m0_final": float(m0[-1]),
        "mw": moment_magnitude(m0[-1]) if m0[-1] > 0 else np.nan,
        "mdot_peak": float(np.max(mdot)),
        "t_peak": float(t[int(np.argmax(mdot))]),
        "mu_theory": rho * vs ** 2,
    }
    res.update(duration_metrics(t, m0))

    # Effective rigidity: M0 = mu * potency, so the ratio is a direct read-out
    # of the shear modulus the solver actually used.  It should be flat.
    if "potency" in wide.columns:
        pot = wide["potency"].to_numpy(dtype=float)
        # Early samples divide two near-zero numbers and produce meaningless
        # spikes that would set the y-limits of the plot.  Only evaluate the
        # ratio once the potency has accumulated 1% of its final value.
        floor = 0.01 * pot[-1] if pot[-1] > 0 else np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_eff = np.where(pot > floor,
                              m0 / np.where(pot > floor, pot, np.nan),
                              np.nan)
        res["potency"] = pot
        res["mu_eff"] = mu_eff
        res["mu_eff_final"] = float(mu_eff[-1]) if pot[-1] > 0 else np.nan
    else:
        res["potency"] = None
        res["mu_eff"] = None
        res["mu_eff_final"] = np.nan

    return res


def analyze_energy(wide: pd.DataFrame, src: dict) -> dict:
    """
    Energy-budget derivations.  Every quantity is guarded on the presence of
    its source column so a run configured with fewer outputs still plots.
    """
    out = {}
    have = set(wide.columns)

    if {"total_frictional_work", "static_frictional_work"} <= have:
        w_tot = wide["total_frictional_work"].to_numpy(dtype=float)
        w_sta = wide["static_frictional_work"].to_numpy(dtype=float)
        e_g = w_tot - w_sta
        out["w_total"] = w_tot
        out["w_static"] = w_sta
        out["e_breakdown"] = e_g
        out["e_breakdown_final"] = float(e_g[-1])
        # Fracture energy per unit area, using the Brune radius as the crack
        # size.  Compare against the analytic linear-slip-weakening value
        #   G_c = 0.5 * (mu_s - mu_d) * |sigma_n'| * d_c
        area = src["dims"]["area"]
        out["g_c"] = (float(e_g[-1] / area)
                      if np.isfinite(area) and area > 0 else np.nan)
        # Flag the non-monotonic static work seen in this dataset instead of
        # silently plotting it.
        out["w_static_decreases"] = bool(w_sta[-1] < w_sta.max() * 0.999)

    if {"elastic_energy", "elastic_kinetic_energy"} <= have:
        e_el = wide["elastic_energy"].to_numpy(dtype=float)
        e_kin = wide["elastic_kinetic_energy"].to_numpy(dtype=float)
        out["e_elastic"] = e_el
        out["e_kinetic"] = e_kin
        out["e_wave"] = e_el + e_kin                   # PROXY, see module docs
        out["e_wave_final"] = float(e_el[-1] + e_kin[-1])
        # Apparent stress sigma_a = mu * E_r / M0.  Approximate here because
        # E_wave is a proxy for E_r (see the caveat block at the top).
        if src["m0_final"] > 0:
            out["apparent_stress"] = (src["mu_theory"] * out["e_wave_final"]
                                      / src["m0_final"])

    mom_cols = [c for c in ("momentumX", "momentumY", "momentumZ")
                if c in have]
    if mom_cols:
        out["momentum"] = {c: wide[c].to_numpy(dtype=float) for c in mom_cols}
        if {"momentumX", "momentumY"} <= have:
            mx = wide["momentumX"].to_numpy(dtype=float)[-1]
            my = wide["momentumY"].to_numpy(dtype=float)[-1]
            # Azimuth clockwise from North, matching the X=East / Y=North
            # convention used throughout build_mesh.py and fault.yaml.
            out["momentum_azimuth"] = float(np.degrees(np.arctan2(mx, my)) % 360)

    return out


# -----------------------------------------------------------------------------
# 4.  FIGURES -- SOURCE PHYSICS
# -----------------------------------------------------------------------------

def _sci_axis(ax, axis: str = "y") -> None:
    """Force scientific notation with a shared exponent on the given axis."""
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


def plot_moment_rate(ax, src: dict, annotate: bool = True) -> None:
    """
    The headline panel: moment rate on the left axis, cumulative moment on a
    twin right axis, so rupture duration can be read off either curve.
    """
    t, mdot, m0 = src["t"], src["mdot"], src["m0"]

    ax.plot(t, mdot, color=C_RATE, lw=1.8, label=r"$\dot{M}_0(t)$")
    ax.fill_between(t, 0, mdot, color=C_RATE, alpha=0.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Moment rate $\dot{M}_0$  (N$\cdot$m/s)", color=C_RATE)
    ax.tick_params(axis="y", labelcolor=C_RATE)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, lw=0.3, alpha=0.4)
    _sci_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(t, m0, color=C_MOMENT, lw=1.6, ls="--",
             label=r"$M_0(t)$ (cumulative)")
    ax2.set_ylabel(r"Cumulative moment $M_0$  (N$\cdot$m)", color=C_MOMENT)
    ax2.tick_params(axis="y", labelcolor=C_MOMENT)
    _sci_axis(ax2)

    if annotate:
        ax.axvline(src["t_peak"], color=C_RATE, lw=0.8, ls=":", alpha=0.7)
        if "T90" in src:
            ax.axvline(src["T90"], color=C_MOMENT, lw=0.8, ls=":", alpha=0.7)
        txt = (f"$M_w$ = {src['mw']:.2f}\n"
               f"$M_0$ = {src['m0_final']:.3g} N$\\cdot$m\n"
               f"peak $\\dot{{M}}_0$ = {src['mdot_peak']:.3g} N$\\cdot$m/s"
               f"  at $t$ = {src['t_peak']:.3f} s")
        if "T90" in src:
            txt += f"\n$T_{{90}}$ = {src['T90']:.3f} s"
        ax.text(0.97, 0.95, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="0.7", alpha=0.9))

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8.5, framealpha=0.9,
              loc="upper right",
              bbox_to_anchor=(0.97, 0.62 if annotate else 0.97))
    ax.set_title("Seismic moment release", fontsize=11)


def plot_spectrum(ax, src: dict) -> None:
    """Source amplitude spectrum with the fitted Brune omega-squared model."""
    freq, amp = src["freq"], src["amp"]
    brune, dims = src["brune"], src["dims"]

    mask = freq > 0
    ax.loglog(freq[mask], amp[mask], color=C_RATE, lw=1.4,
              label="computed spectrum")

    if np.isfinite(brune["omega_0"]):
        ax.axhline(brune["omega_0"], color=C_MOMENT, lw=1.0, ls="-",
                   alpha=0.6, label=r"$\Omega_0 = M_0$")
        ax.axhline(brune["omega_0"] / 2.0, color="0.6", lw=0.8, ls=":")

    if np.isfinite(brune["f_half"]):
        ax.axvline(brune["f_half"], color=C_MOMENT, lw=1.1, ls="--",
                   label=fr"$f_c$ (half-amplitude) = {brune['f_half']:.2f} Hz")

    if np.isfinite(brune["f_fit"]):
        model = brune["omega_0"] / (1.0 + (freq[mask] / brune["f_fit"]) ** 2)
        ax.loglog(freq[mask], model, color=C_FIT, lw=1.5, ls="--",
                  label=fr"$\omega^{{-2}}$ fit, $f_c$ = {brune['f_fit']:.2f} Hz")
    if np.isfinite(brune["band_hi"]):
        # Everything above the fit band is shaded: the simulated source-time
        # function decays faster than omega^-2 there, so the model is not
        # expected to track the data.
        ax.axvspan(brune["band_hi"], freq[-1], color="0.85", alpha=0.5, lw=0,
                   zorder=0)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|\dot{M}_0(f)|$  (N$\cdot$m)")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="lower left")

    if np.isfinite(dims["radius"]):
        ax.text(0.03, 0.06,
                f"$r$ = {dims['radius']:.0f} m\n"
                f"$\\Delta\\sigma$ = {dims['stress_drop']/1e6:.2f} MPa",
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.7", alpha=0.9))
    ax.set_title("Source spectrum and corner frequency", fontsize=11)


def plot_phase(ax, src: dict) -> None:
    """
    Moment rate against cumulative moment, parametric in time.  Nucleation,
    propagation and arrest separate into distinct branches, which a plain
    time series makes harder to see.
    """
    t, m0, mdot = src["t"], src["m0"], src["mdot"]
    sc = ax.scatter(m0, mdot, c=t, cmap="viridis", s=9, zorder=3)
    ax.plot(m0, mdot, color="0.6", lw=0.6, zorder=2)
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Time (s)", fontsize=9)
    ax.set_xlabel(r"$M_0$  (N$\cdot$m)")
    ax.set_ylabel(r"$\dot{M}_0$  (N$\cdot$m/s)")
    ax.grid(True, lw=0.3, alpha=0.4)
    _sci_axis(ax, "both")
    ax.set_title("Rupture phase diagram", fontsize=11)


def plot_rigidity(ax, src: dict) -> None:
    """
    Effective shear modulus mu_eff = M0 / potency against time, with the
    theoretical rho*Vs^2 as a reference line.  A flat curve that misses the
    reference means the solver ran with different material properties than
    assumed here -- a cheap and very effective sanity check.
    """
    if src["mu_eff"] is None:
        ax.text(0.5, 0.5, "no 'potency' variable in output",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return

    t, mu = src["t"], src["mu_eff"] / 1e9
    ax.plot(t, mu, color=C_BREAK, lw=1.6, label=r"$\mu_{eff}=M_0/\mathrm{potency}$")
    ax.axhline(src["mu_theory"] / 1e9, color=C_FIT, lw=1.4, ls="--",
               label=fr"$\rho V_s^2$ = {src['mu_theory']/1e9:.1f} GPa")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\mu$  (GPa)")
    ax.set_xlim(t[0], t[-1])
    finite = mu[np.isfinite(mu)]
    if finite.size:
        lo = min(finite.min(), src["mu_theory"] / 1e9)
        hi = max(finite.max(), src["mu_theory"] / 1e9)
        pad = 0.15 * max(hi - lo, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.legend(fontsize=8.5, loc="best")
    ax.set_title("Effective rigidity check", fontsize=11)


def plot_late_creep(ax, src: dict, eng: dict) -> None:
    """
    Normalized cumulative moment against normalized frictional work.  During
    coseismic slip the two track each other. If the moment keeps creeping
    after the frictional work has plateaued, the tail is either genuine slow
    slip or a numerical artifact -- either way it deserves to be visible.
    """
    t, m0 = src["t"], src["m0"]
    ax.plot(t, m0 / m0[-1], color=C_MOMENT, lw=1.6, label=r"$M_0/M_0^{final}$")
    if "w_total" in eng and eng["w_total"][-1] > 0:
        w = eng["w_total"]
        ax.plot(t, w / w[-1], color=C_FRIC, lw=1.6,
                label=r"$W_{fric}/W_{fric}^{final}$")
    if "T90" in src:
        ax.axvline(src["T90"], color="0.5", lw=0.8, ls=":")
        ax.text(src["T90"], 0.05, f"  $T_{{90}}$ = {src['T90']:.2f} s",
                fontsize=8, color="0.35")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized cumulative value")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1.05)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Coseismic phase vs late-time tail", fontsize=11)


def plot_moment_fraction(ax, src: dict) -> None:
    """
    Cumulative moment as a fraction of the final value, on a log time axis,
    with the duration percentiles marked.  Reads directly as "how much of the
    event had happened by time t".
    """
    t, m0 = src["t"], src["m0"]
    pos = t > 0
    ax.semilogx(t[pos], 100.0 * m0[pos] / m0[-1], color=C_MOMENT, lw=1.8)
    for (key, color), y in zip((("T05", "0.55"), ("T90", C_RATE),
                                 ("T95", "0.35")), (6, 34, 62)):
        if key in src:
            ax.axvline(src[key], color=color, lw=0.9, ls=":")
            ax.text(src[key], y, f" {key}={src[key]:.3f} s", rotation=90,
                    fontsize=7.5, color=color, va="bottom")
    ax.set_xlabel("Time (s, log scale)")
    ax.set_ylabel("Moment released (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.set_title("Moment release percentiles", fontsize=11)


def figure_source(src: dict, eng: dict, outdir: str, dpi: int,
                  title: str) -> str:
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_SOURCE,
                             layout="constrained")
    plot_moment_rate(axes[0, 0], src)
    plot_spectrum(axes[0, 1], src)
    plot_phase(axes[0, 2], src)
    plot_rigidity(axes[1, 0], src)
    plot_late_creep(axes[1, 1], src, eng)
    plot_moment_fraction(axes[1, 2], src)
    fig.suptitle(title, fontsize=13)
    path = os.path.join(outdir, "source.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def figure_moment_rate(src: dict, outdir: str, dpi: int, title: str) -> str:
    """Standalone version of the headline panel, for slides."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, layout="constrained")
    plot_moment_rate(ax, src)
    ax.set_title(title, fontsize=11)
    path = os.path.join(outdir, "moment_rate.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# 5.  FIGURES -- ENERGY BUDGET
# -----------------------------------------------------------------------------

def plot_energy_budget(ax, src: dict, eng: dict) -> None:
    """Elastic, kinetic and frictional terms on one axis."""
    t = src["t"]
    if "e_elastic" in eng:
        ax.plot(t, eng["e_elastic"], color=C_ELAS, lw=1.6,
                label="elastic strain energy")
        ax.plot(t, eng["e_kinetic"], color=C_KIN, lw=1.6,
                label="kinetic energy")
    if "w_total" in eng:
        ax.plot(t, eng["w_total"], color=C_FRIC, lw=1.6,
                label="total frictional work")
        ax.plot(t, eng["w_static"], color=C_FRIC, lw=1.3, ls="--",
                label="static frictional work")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, lw=0.3, alpha=0.4)
    _sci_axis(ax)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Energy budget", fontsize=11)


def plot_breakdown_energy(ax, src: dict, eng: dict) -> None:
    """
    Breakdown (fracture) energy E_G = W_total - W_static, i.e. the work done
    against friction in excess of the residual dynamic level.  For a linear
    slip-weakening law this is the energy consumed inside the cohesive zone.
    """
    if "e_breakdown" not in eng:
        ax.text(0.5, 0.5, "frictional work not in output", ha="center",
                va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return

    t, e_g = src["t"], eng["e_breakdown"]
    ax.plot(t, e_g, color=C_BREAK, lw=1.8, label=r"$E_G = W_{tot}-W_{stat}$")
    ax.fill_between(t, 0, e_g, color=C_BREAK, alpha=0.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Breakdown energy $E_G$  (J)")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, lw=0.3, alpha=0.4)
    _sci_axis(ax)

    lines = [fr"$E_G$ = {eng['e_breakdown_final']:.3g} J"]
    if np.isfinite(eng.get("g_c", np.nan)):
        lines.append(fr"$G_c \approx$ {eng['g_c']:.3g} J/m$^2$"
                     "\n(over Brune crack area)")
    ax.text(0.97, 0.05, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.7", alpha=0.9))
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Breakdown (fracture) energy", fontsize=11)


def plot_momentum(ax, src: dict, eng: dict) -> None:
    """
    Linear momentum components.  The ratio of the horizontal components
    reflects the slip direction and radiation pattern. On a strike-slip fault
    the vertical component should stay comparatively small.
    """
    if "momentum" not in eng:
        ax.text(0.5, 0.5, "no momentum variables in output", ha="center",
                va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return

    t = src["t"]
    styles = {"momentumX": ("#c0392b", "$p_x$ (East)"),
              "momentumY": ("#2980b9", "$p_y$ (North)"),
              "momentumZ": ("#27ae60", "$p_z$ (Up)")}
    for key, arr in eng["momentum"].items():
        color, label = styles.get(key, ("0.4", key))
        ax.plot(t, arr, color=color, lw=1.6, label=label)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Momentum (kg$\cdot$m/s)")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, lw=0.3, alpha=0.4)
    _sci_axis(ax)
    ax.legend(fontsize=8.5, loc="best")
    if "momentum_azimuth" in eng:
        ax.text(0.03, 0.05,
                f"final horizontal azimuth\n"
                f"{eng['momentum_azimuth']:.1f}$^\\circ$ from North",
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.7", alpha=0.9))
    ax.set_title("Linear momentum components", fontsize=11)


def figure_energy(src: dict, eng: dict, outdir: str, dpi: int,
                  title: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_ENERGY,
                             layout="constrained")
    plot_energy_budget(axes[0], src, eng)
    plot_breakdown_energy(axes[1], src, eng)
    plot_momentum(axes[2], src, eng)
    fig.suptitle(title, fontsize=13)
    path = os.path.join(outdir, "energy.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# 6.  FIGURES -- SOLVER PERFORMANCE
# -----------------------------------------------------------------------------

def plot_cluster_sizes(ax, clus: pd.DataFrame) -> None:
    """
    Elements and dynamic-rupture faces per LTS cluster, log y-axis.  This is
    the single most informative figure for explaining where the wall time
    went: cost is dominated by the clusters that are both large and finely
    time-stepped.
    """
    interior = clus[clus["layerType"] == "Interior"]
    grp = interior.groupby("localId")[["size", "dynamicRuptureSize"]].sum()
    grp = grp.sort_index()

    idx = np.arange(len(grp))
    width = 0.4
    ax.bar(idx - width / 2, grp["size"].clip(lower=0.7), width,
           color=C_ELAS, label="elements")
    ax.bar(idx + width / 2, grp["dynamicRuptureSize"].clip(lower=0.7), width,
           color=C_RATE, label="DR faces")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(grp.index.astype(str))
    ax.set_xlabel("LTS cluster id")
    ax.set_ylabel("count (log scale)")
    ax.grid(True, axis="y", which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=8.5)
    ax.set_title(f"LTS clustering: {int(grp['size'].sum()):,} elements, "
                 f"{int(grp['dynamicRuptureSize'].sum()):,} DR faces",
                 fontsize=10.5)


def plot_cluster_dr_fraction(ax, clus: pd.DataFrame) -> None:
    """
    Fraction of each cluster's elements that carry a dynamic-rupture face.
    Shows how the fault is distributed across the time-step hierarchy.
    """
    interior = clus[clus["layerType"] == "Interior"]
    grp = interior.groupby("localId")[["size", "dynamicRuptureSize"]].sum()
    grp = grp.sort_index()
    frac = np.where(grp["size"] > 0,
                    100.0 * grp["dynamicRuptureSize"] / grp["size"].replace(0, np.nan),
                    0.0)

    idx = np.arange(len(grp))
    ax.bar(idx, frac, color=C_BREAK)
    ax.set_xticks(idx)
    ax.set_xticklabels(grp.index.astype(str))
    ax.set_xlabel("LTS cluster id")
    ax.set_ylabel("DR faces / elements (%)")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.set_title("Fault concentration per cluster", fontsize=10.5)


def plot_flops(ax, flops: pd.DataFrame, ranks: list[int],
               window: int = 11) -> None:
    """
    Instantaneous and accumulated performance against wall-clock time.  The
    curve is essentially flat for a well-behaved run, so the informative
    content is the scatter: a rolling mean with a +/- 1 sigma band makes that
    legible where raw samples just look like noise.
    """
    t = flops["time"].to_numpy(dtype=float)
    for rank in ranks:
        cur = f"rank_{rank}_current"
        acc = f"rank_{rank}_accumulated"
        if cur in flops.columns:
            y = flops[cur].to_numpy(dtype=float)
            ax.plot(t, y, color=C_RATE, lw=0.6, alpha=0.35,
                    label=f"rank {rank} instantaneous" if rank == ranks[0] else None)
            roll = pd.Series(y).rolling(window, center=True, min_periods=1)
            mean, std = roll.mean().to_numpy(), roll.std().to_numpy()
            ax.plot(t, mean, color=C_RATE, lw=1.6,
                    label=f"rolling mean ({window})" if rank == ranks[0] else None)
            ax.fill_between(t, mean - std, mean + std, color=C_RATE,
                            alpha=0.15, lw=0)
        if acc in flops.columns:
            ax.plot(t, flops[acc].to_numpy(dtype=float), color=C_MOMENT,
                    lw=1.4, ls="--",
                    label=f"rank {rank} accumulated" if rank == ranks[0] else None)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    total_h = (t[-1] - t[0]) / 3600.0
    ax.set_title(f"Solver throughput  ({total_h:.2f} h of wall time)",
                 fontsize=10.5)


def plot_time_efficiency(ax, src: dict, flops: pd.DataFrame) -> None:
    """
    Simulated seconds per wall-clock hour.

    The energy and flops files share no common column, so this panel ASSUMES
    that the two sample streams are written at the same cadence and pairs
    them index-by-index (truncating to the shorter one).  That assumption is
    stated on the figure because it is not verifiable from the CSVs alone.
    If it holds, the coseismic phase -- small LTS timesteps, many active DR
    faces -- should show up as a visible dip.
    """
    t_sim = src["t"]
    t_wall = flops["time"].to_numpy(dtype=float)
    n = min(len(t_sim), len(t_wall))
    if n < 5:
        ax.set_axis_off()
        return

    ts, tw = t_sim[:n], t_wall[:n]
    d_sim = np.gradient(ts)
    d_wall = np.gradient(tw)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(d_wall > 0, d_sim / d_wall * 3600.0, np.nan)

    ax.plot(ts, eff, color=C_FRIC, lw=1.0, alpha=0.5)
    ax.plot(ts, pd.Series(eff).rolling(9, center=True, min_periods=1).mean(),
            color=C_FRIC, lw=1.8)
    if "T90" in src:
        ax.axvspan(ts[0], min(src["T90"], ts[-1]), color=C_RATE, alpha=0.08,
                   lw=0, label="coseismic phase")
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Simulated s per wall-clock hour")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_title("Throughput vs rupture phase\n"
                 "(assumes energy and flops share a cadence)", fontsize=10)


def figure_performance(src: dict, clus: pd.DataFrame | None,
                       flops: pd.DataFrame | None, pinning_caption: str,
                       outdir: str, dpi: int, title: str) -> str | None:
    if clus is None and flops is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_PERF, layout="constrained")
    if clus is not None:
        plot_cluster_sizes(axes[0, 0], clus)
        plot_cluster_dr_fraction(axes[0, 1], clus)
    else:
        axes[0, 0].set_axis_off()
        axes[0, 1].set_axis_off()

    if flops is not None:
        ranks = flops_ranks(flops)
        plot_flops(axes[1, 0], flops, ranks)
        plot_time_efficiency(axes[1, 1], src, flops)
    else:
        axes[1, 0].set_axis_off()
        axes[1, 1].set_axis_off()

    fig.suptitle(f"{title}\n{pinning_caption}", fontsize=12)
    path = os.path.join(outdir, "performance.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# 7.  SUMMARY REPORT
# -----------------------------------------------------------------------------

def build_summary(src: dict, eng: dict, clus: pd.DataFrame | None,
                  flops: pd.DataFrame | None, vs: float,
                  rho: float) -> pd.DataFrame:
    """Collect every derived scalar into a tidy (quantity, value, unit) table."""
    rows: list[tuple[str, float, str]] = [
        ("seismic_moment_final", src["m0_final"], "N.m"),
        ("moment_magnitude_Mw", src["mw"], "-"),
        ("peak_moment_rate", src["mdot_peak"], "N.m/s"),
        ("time_of_peak_moment_rate", src["t_peak"], "s"),
        ("energy_output_interval", src["dt_out"], "s"),
    ]
    for key in ("T05", "T90", "T95", "duration"):
        if key in src:
            rows.append((f"moment_{key}", src[key], "s"))

    rows += [
        ("corner_frequency_half_amplitude", src["brune"]["f_half"], "Hz"),
        ("corner_frequency_omega2_fit", src["brune"]["f_fit"], "Hz"),
        ("spectral_dc_amplitude", src["brune"].get("dc_amplitude", np.nan),
         "N.m"),
        ("brune_source_radius", src["dims"]["radius"], "m"),
        ("static_stress_drop", src["dims"]["stress_drop"], "Pa"),
        ("assumed_Vs", vs, "m/s"),
        ("assumed_rho", rho, "kg/m3"),
        ("shear_modulus_rho_Vs2", src["mu_theory"], "Pa"),
        ("effective_rigidity_M0_over_potency", src["mu_eff_final"], "Pa"),
    ]
    if src["potency"] is not None:
        rows.append(("potency_final", float(src["potency"][-1]), "m3"))

    for key, unit in (("e_breakdown_final", "J"), ("g_c", "J/m2"),
                      ("e_wave_final", "J"), ("apparent_stress", "Pa")):
        if key in eng and np.isfinite(eng[key]):
            rows.append((key, float(eng[key]), unit))

    if clus is not None:
        interior = clus[clus["layerType"] == "Interior"]
        rows.append(("total_elements", float(interior["size"].sum()), "-"))
        rows.append(("total_dr_faces",
                     float(interior["dynamicRuptureSize"].sum()), "-"))
        rows.append(("n_lts_clusters", float(interior["localId"].nunique()), "-"))

    if flops is not None:
        ranks = flops_ranks(flops)
        t = flops["time"].to_numpy(dtype=float)
        rows.append(("wall_clock_time", float(t[-1]), "s"))
        for rank in ranks:
            col = f"rank_{rank}_current"
            if col in flops.columns:
                rows.append((f"mean_gflops_rank{rank}",
                             float(flops[col].mean()), "GFLOP/s"))

    return pd.DataFrame(rows, columns=["quantity", "value", "unit"])


def print_summary(summary: pd.DataFrame, src: dict, eng: dict) -> None:
    """Console report, including the checks that are easy to overlook."""
    width = 66
    print("\n" + "=" * width)
    print("DERIVED QUANTITIES")
    print("=" * width)
    for _, row in summary.iterrows():
        val = row["value"]
        shown = "n/a" if not np.isfinite(val) else f"{val:>14.6g}"
        print(f"  {row['quantity']:<38s} {shown}  {row['unit']}")
    print("=" * width)

    # -- consistency checks ---------------------------------------------------
    print("\nCHECKS")
    dc = src["brune"].get("dc_amplitude", np.nan)
    if np.isfinite(dc) and src["m0_final"] > 0:
        ratio = dc / src["m0_final"]
        status = "OK" if 0.95 < ratio < 1.05 else "CHECK"
        print(f"  [{status}] spectral DC amplitude / M0 = {ratio:.4f} "
              f"(must be ~1: the zero-frequency")
        print("         asymptote of the moment-rate spectrum is the seismic "
              "moment)")

    f_half, f_fit = src["brune"]["f_half"], src["brune"]["f_fit"]
    if np.isfinite(f_half) and np.isfinite(f_fit):
        spread = max(f_half, f_fit) / min(f_half, f_fit)
        status = "OK" if spread < 1.5 else "CHECK"
        print(f"  [{status}] corner frequency: {f_half:.2f} Hz "
              f"(half-amplitude) vs {f_fit:.2f} Hz (omega^-2 fit)")
        if status == "CHECK":
            print("         -> the source-time function decays faster than "
                  "omega^-2, so the fitted")
            print("            corner depends on the band. The half-amplitude "
                  "value is used downstream.")
            print("            Adjust --corner-band to see the sensitivity.")

    if np.isfinite(src["mu_eff_final"]):
        ratio = src["mu_eff_final"] / src["mu_theory"]
        status = "OK" if 0.9 < ratio < 1.1 else "CHECK"
        print(f"  [{status}] M0/potency = {src['mu_eff_final']/1e9:.2f} GPa "
              f"vs rho*Vs^2 = {src['mu_theory']/1e9:.2f} GPa "
              f"(ratio {ratio:.2f})")
        if status == "CHECK":
            print("         -> the run's material properties differ from the "
                  "--vs/--rho passed here,")
            print("            or 'potency' is defined differently in this "
                  "SeisSol build. Worth resolving")
            print("            before quoting the stress drop or apparent "
                  "stress.")

    if "duration" in src and src["dt_out"] > 0:
        n_in_pulse = src["duration"] / src["dt_out"]
        status = "OK" if n_in_pulse >= 50 else "CHECK"
        print(f"  [{status}] {n_in_pulse:.0f} output samples span the rupture "
              f"(T05-T95)")
        if status == "CHECK":
            print("         -> the moment-rate peak and the high-frequency "
                  "end of the spectrum are")
            print("            under-resolved. Reduce EnergyOutputInterval in "
                  "parameters.par on a rerun.")

    if np.isfinite(src["brune"]["f_c"]):
        f_nyq = 0.5 / src["dt_out"]
        status = "OK" if src["brune"]["f_c"] < 0.2 * f_nyq else "CHECK"
        print(f"  [{status}] f_c = {src['brune']['f_c']:.2f} Hz vs Nyquist "
              f"{f_nyq:.1f} Hz")

    if eng.get("w_static_decreases"):
        print("  [CHECK] static_frictional_work decreases after its peak, "
              "verify the sign")
        print("          convention before interpreting the breakdown energy.")
    print()




# -----------------------------------------------------------------------------
# 8.  DRIVER
# -----------------------------------------------------------------------------

def run(directory: str, vs: float, rho: float, prefix: str = "decatur",
        outdir: str = "figures", smooth: int = 0, corner_band: float = 0.1,
        title: str | None = None, dpi: int = 150, show: bool = False,
        figures: bool = True) -> pd.DataFrame:
    """Analyze one run directory. vs and rho are the rupture-zone material."""
    if not show:
        matplotlib.use("Agg")
    title = title or f"Decatur dynamic rupture ({prefix})"

    print(f"\nLooking for '{prefix}-*.csv' in {os.path.abspath(directory)}")
    paths = {kind: find_input(directory, prefix, kind)
             for kind in ("energy", "flops", "clustering", "threadPinning")}
    for kind, path in paths.items():
        print(f"  {kind:<14s} : {os.path.basename(path) if path else 'not found'}")
    if paths["energy"] is None:
        raise FileNotFoundError(f"no energy CSV in {directory}")

    wide = load_energy(paths["energy"])
    flops = load_flops(paths["flops"]) if paths["flops"] else None
    clus = load_clustering(paths["clustering"]) if paths["clustering"] else None
    pinning = (load_thread_pinning(paths["threadPinning"])
               if paths["threadPinning"] else None)
    caption = describe_pinning(pinning)
    print(f"\n  Energy series : {len(wide)} samples, "
          f"t = {wide.index[0]:.3f} to {wide.index[-1]:.3f} s")
    print(f"  Hardware      : {caption}")

    src = analyze_source(wide, vs, rho, smooth, corner_band)
    eng = analyze_energy(wide, src)
    summary = build_summary(src, eng, clus, flops, vs, rho)
    print_summary(summary, src, eng)
    if not figures:
        return summary

    os.makedirs(outdir, exist_ok=True)
    written = [figure_moment_rate(src, outdir, dpi, title),
               figure_source(src, eng, outdir, dpi, title),
               figure_energy(src, eng, outdir, dpi, title)]
    perf = figure_performance(src, clus, flops, caption, outdir, dpi, title)
    if perf:
        written.append(perf)
    summary_path = os.path.join(outdir, "derived_quantities.csv")
    summary.to_csv(summary_path, index=False)
    written.append(summary_path)
    for path in written:
        print(f"  Saved: {path}")
    if show:
        plt.show()
    return summary
