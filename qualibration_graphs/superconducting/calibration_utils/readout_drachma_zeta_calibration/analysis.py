from dataclasses import dataclass

import numpy as np
import xarray as xr

from calibration_utils.readout_drachma_common import STATES, select_parameter_pair


@dataclass
class FitParameters:
    """Stores the fitted zeta values of a single qubit."""

    zeta_ground_hz: float
    zeta_excited_hz: float
    power_ground: float
    power_excited: float
    p_value_ge: float
    zeta_difference_hz: float
    unconstrained_zeta_ground_hz: float
    unconstrained_zeta_excited_hz: float
    constrained: bool
    success: bool


def build_zeta_grid(current_hz: float, step_hz: float, num_points: int) -> np.ndarray:
    """Zeta scan (Hz) for one state of one qubit: num_points points spaced by step_hz around the current
    zeta, which sits on the grid (for an even num_points the extra point is above it). If the lower end
    would go below 0, the whole grid is shifted so that it starts at 0."""
    start = max(current_hz - ((num_points - 1) // 2) * step_hz, 0.0)
    return start + step_hz * np.arange(num_points)


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, dict[str, FitParameters]]:
    """Pick, per qubit, the zeta pair (ground, excited) with the lowest residual power under the limit
    node.parameters.max_zeta_difference_hz.

    The fit fails when no pair satisfies the limit, or when an independent (unconstrained) minimum is on the
    upper edge of the scan (the true minimum is outside the scan) or on the lower edge unless that edge is
    zeta = 0 (the zero-start linear scan). A pair moved onto an edge by the limit is not a failure."""
    limit = node.parameters.max_zeta_difference_hz
    fits: dict[str, FitParameters] = {}
    best = {}
    for q in ds.qubit.values:
        zeta = {s: ds[f"zeta_{s}_hz"].sel(qubit=q).values for s in STATES}
        power = {s: ds["power"].sel(qubit=q, state=s).values for s in STATES}
        err = {s: ds["power_err"].sel(qubit=q, state=s).values for s in STATES}
        free_idx = {s: int(np.argmin(power[s])) for s in STATES}
        n = ds.sizes["point"]
        free_ok = all(free_idx[s] != n - 1 and not (free_idx[s] == 0 and float(zeta[s][0]) != 0.0) for s in STATES)
        pair = select_parameter_pair(
            zeta["ground"],
            power["ground"],
            err["ground"],
            zeta["excited"],
            power["excited"],
            err["excited"],
            limit,
        )
        idx = dict(zip(STATES, pair)) if pair is not None else free_idx
        chosen = {s: float(zeta[s][idx[s]]) for s in STATES}
        # z-test reported at the point whose summed ground+excited power is lowest.
        joint_idx = int(ds["power"].sel(qubit=q).sum("state").argmin("point"))
        p_value = float(ds["p_value_ge"].sel(qubit=q).isel(point=joint_idx))
        fits[str(q)] = FitParameters(
            zeta_ground_hz=chosen["ground"],
            zeta_excited_hz=chosen["excited"],
            power_ground=float(power["ground"][idx["ground"]]),
            power_excited=float(power["excited"][idx["excited"]]),
            p_value_ge=p_value,
            zeta_difference_hz=chosen["excited"] - chosen["ground"],
            unconstrained_zeta_ground_hz=float(zeta["ground"][free_idx["ground"]]),
            unconstrained_zeta_excited_hz=float(zeta["excited"][free_idx["excited"]]),
            constrained=idx != free_idx,
            success=pair is not None and free_ok,
        )
        best[str(q)] = idx

    ds_fit = ds.assign_coords(
        best_point_ground=("qubit", [best[str(q)]["ground"] for q in ds.qubit.values]),
        best_point_excited=("qubit", [best[str(q)]["excited"] for q in ds.qubit.values]),
    )
    return ds_fit, fits


def log_fitted_results(fit_results: dict, alpha: float, log_callable=None):
    """Log one line per qubit with the best zetas and the g-vs-e z-test at the joint-best point."""
    if log_callable is None:
        log_callable = print
    for q, fit in fit_results.items():
        status = (
            "SUCCESS!"
            if fit["success"]
            else "FAIL! (no zeta pair within max_zeta_difference_hz, or minimum at the edge of the scan)"
        )
        verdict = "indistinguishable (depleted)" if fit["p_value_ge"] > alpha else "distinguishable (not depleted)"
        log_callable(
            f"Results for qubit {q}: zeta_ground = {fit['zeta_ground_hz'] * 1e-3:.2f} kHz, "
            f"zeta_excited = {fit['zeta_excited_hz'] * 1e-3:.2f} kHz "
            f"(difference {fit['zeta_difference_hz'] * 1e-3:+.2f} kHz) | "
            f"g-vs-e z-test p = {fit['p_value_ge']:.3g}, {verdict} | --> {status}"
        )
