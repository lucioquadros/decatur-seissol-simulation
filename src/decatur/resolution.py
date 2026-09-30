"""Fault and wavefield resolution: cohesive zone, element size, resolved frequency, LTS cost.
DS = dynamic stress time (ASl > d_c), RT = rupture time (Sr > 0.001 m/s), Vr = rupture velocity."""

import numpy as np

# Wollherr et al. (2018) convergence tests, elastic, as fitted in SeisSol's
# postprocessing/science/Calc_CohesiveZoneError.py:
# log10(error %) = a * log10(h * 162 / cohesive_zone) + b
WOLLHERR_COHESIVE_ZONE = 162.0 # elastic
WOLLHERR_FITS = {
    3: {"rupture_arrival": (1.52739978035, -4.4893475817),
        "peak_slip_rate": (0.836379291573, -0.807389129936),
        "final_slip": (1.0161358622, -2.55381103136)},
    4: {"rupture_arrival": (1.37812937903, -4.37561371471),
        "peak_slip_rate": (1.11589493374, -1.74133399291),
        "final_slip": (1.0066205251, -2.59964604604)},
    5: {"rupture_arrival": (1.25759353097, -4.24036674538),
        "peak_slip_rate": (1.32131510275, -2.45342685486),
        "final_slip": (0.984644208844, -2.66217236352)},
    6: {"rupture_arrival": (1.16913715515, -4.1179187664),
        "peak_slip_rate": (1.38930610536, -2.77826477656),
        "final_slip": (0.97950635571, -2.66901809392)},
}
# Mean tetrahedron edge / Gmsh target size on HXT meshes, the same at every distance
# from the faults (docs/resolution.md). Fault triangles match the target size.
GMSH_EDGE_RATIO = 1.4
# Wollherr's h is the hypotenuse of fault triangles (area h^2 / 4). A Gmsh
# fault triangle has the same sampling density at h = 3^(1/4) a.
WOLLHERR_EDGE_RATIO = 3.0 ** 0.25
# percent, Day et al. (2005)
# TO DO: double check 0.2 rupture arrival error limit, currently this only comes from Wollherr et al. (2018).
# TO DO: check Day et al. (2005) carefully.
# The other two are OK.
ERROR_LIMITS = {"rupture_arrival": 0.2, "peak_slip_rate": 7.0, "final_slip": 1.0}


def cohesive_zone_width(ds, rt, vr) -> np.ndarray:
    """(DS - RT) * Vr from the fault output, NaN where the cell did not rupture and weaken."""
    ds, rt, vr = (np.asarray(a, dtype=float) for a in (ds, rt, vr))
    return np.where((ds > 0) & (rt > 0), (ds - rt) * vr, np.nan)


def static_cohesive_zone(mu, d_c, dmu, sigma_n) -> np.ndarray:
    """static cohesive zone from Eq. 30a from Day et al. (2005) for linear slip-weakening."""
    return 9.0 * np.pi / 32.0 * np.asarray(mu) * d_c / (dmu * np.abs(sigma_n))


def critical_radius(mu, d_c, tau_0, tau_s, tau_d):
    """Day (1982) circular 3-D critical crack radius."""
    return 7.0 * np.pi / 24.0 * mu * (tau_s - tau_d) * d_c / (tau_0 - tau_d) ** 2


def strength_parameter(tau_0, tau_s, tau_d) -> np.ndarray:
    """Relative strength S = (tau_s - tau_0) / (tau_0 - tau_d) (Das and Aki 1977)."""
    return (np.asarray(tau_s) - tau_0) / (np.asarray(tau_0) - tau_d)


def rupture_errors(order: int, h: float, cohesive_zone: float) -> dict[str, float]:
    """Expected average errors (%) of an elastic run of this order and Gmsh fault edge h (Wollherr et al. 2018)."""
    x = np.log10(WOLLHERR_EDGE_RATIO * h * WOLLHERR_COHESIVE_ZONE / cohesive_zone)
    return {k: float(10 ** (a * x + b)) for k, (a, b) in WOLLHERR_FITS[order].items()}


def max_fault_element_edge(order: int, cohesive_zone: float, limits=ERROR_LIMITS) -> float:
    """Largest Gmsh fault edge that keeps every error within its limit."""
    return min(cohesive_zone / (WOLLHERR_EDGE_RATIO * WOLLHERR_COHESIVE_ZONE)
               * 10 ** ((np.log10(limits[k]) - b) / a)
               for k, (a, b) in WOLLHERR_FITS[order].items())


def threshold_size(distance, lc_fault, lc_domain, dist_min, dist_max) -> np.ndarray:
    """Element size of the Gmsh Threshold field at a distance from the faults."""
    t = np.clip((np.asarray(distance, dtype=float) - dist_min) / (dist_max - dist_min), 0.0, 1.0)
    return lc_fault + t * (lc_domain - lc_fault)


def max_frequency(vs, h, per_wavelength: float) -> np.ndarray:
    """Highest frequency with per_wavelength elements of edge h across the S wavelength."""
    return np.asarray(vs) / (per_wavelength * np.asarray(h))


def element_timestep(inradius, vp, order: int, cfl: float) -> np.ndarray:
    """SeisSol's CFL time step per element."""
    return cfl * 2.0 * np.asarray(inradius) / (np.asarray(vp) * (2 * order - 1))


def lts_clusters(dt) -> tuple[np.ndarray, float]:
    """Rate-2 LTS cluster of every element and the smallest time step.

    Ignores SeisSol's cluster merging and the neighbor normalization, which move
    some elements to faster clusters.
    """
    dt = np.asarray(dt, dtype=float)
    dt_min = float(dt.min())
    return np.floor(np.log2(dt / dt_min) + 1e-12).astype(int), dt_min


def updates_per_second(clusters: np.ndarray, dt_min: float) -> float:
    """Element updates per simulated second with rate-2 LTS."""
    return float(np.sum(1.0 / (dt_min * 2.0 ** clusters)))


def read_fault_output(xdmf, fields, step: int = -1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Loads on time step of a SeisSol 1.3 fault output, returning the cell centers and the selected fields."""
    import h5py
    stem = str(xdmf).removesuffix(".xdmf")
    with h5py.File(f"{stem}_vertex.h5", "r") as fv, h5py.File(f"{stem}_cell.h5", "r") as fc:
        xyz = np.asarray(fv["mesh0/geometry"])
        centers = xyz[np.asarray(fc["mesh0/connect"])].mean(axis=1)
        return centers, {k: np.asarray(fc[f"mesh0/{k}"][step]) for k in fields}
