from dataclasses import dataclass

import numpy as np
import xarray as xr

from calibration_utils.readout_drachma_common import STATES, select_parameter_pair


@dataclass
class FitParameters:
    """Stores the chosen kappa pair of a single qubit."""

    kappa_ground_hz: float
    kappa_excited_hz: float
    power_ground: float
    power_excited: float
    p_value_ge: float
    kappa_difference_hz: float
    unconstrained_kappa_ground_hz: float
    unconstrained_kappa_excited_hz: float
    constrained: bool
    at_edge_ground: bool
    at_edge_excited: bool
    success: bool


def build_kappa_grid(current_hz: float, step_hz: float, num_points: int) -> np.ndarray:
    """Kappa scan (Hz) for one state of one qubit: num_points points spaced by step_hz around the current
    kappa, which sits on the grid (for an even num_points the extra point is above it). If the lower end
    would drop below step_hz, the whole grid is shifted so that it starts at step_hz (kappa must stay > 0)."""
    start = max(current_hz - ((num_points - 1) // 2) * step_hz, step_hz)
    return start + step_hz * np.arange(num_points)


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, dict[str, FitParameters]]:
    """Pick, per qubit, the kappa pair (ground, excited) with the lowest residual power under the limit
    node.parameters.max_kappa_difference_hz.

    A pair on the edge of the scan is still returned (and written to state) but flagged; the fit fails only
    when no pair satisfies the limit."""
    limit = node.parameters.max_kappa_difference_hz
    fits: dict[str, FitParameters] = {}
    best = {}
    for q in ds.qubit.values:
        kappa = {s: ds[f"kappa_{s}_hz"].sel(qubit=q).values for s in STATES}
        power = {s: ds["power"].sel(qubit=q, state=s).values for s in STATES}
        err = {s: ds["power_err"].sel(qubit=q, state=s).values for s in STATES}
        free_idx = {s: int(np.argmin(power[s])) for s in STATES}
        pair = select_parameter_pair(
            kappa["ground"],
            power["ground"],
            err["ground"],
            kappa["excited"],
            power["excited"],
            err["excited"],
            limit,
            geometric=True,
        )
        success = pair is not None
        idx = dict(zip(STATES, pair)) if success else free_idx
        num_points = ds.sizes["point"]
        chosen_kappa = {s: float(kappa[s][idx[s]]) for s in STATES}
        # The z-test compares the two states at the same scan point index, so it is reported at the
        # point whose summed ground+excited power is lowest, not at the chosen pair.
        joint_idx = int(ds["power"].sel(qubit=q).sum("state").argmin("point"))
        p_value = float(ds["p_value_ge"].sel(qubit=q).isel(point=joint_idx))
        fits[str(q)] = FitParameters(
            kappa_ground_hz=chosen_kappa["ground"],
            kappa_excited_hz=chosen_kappa["excited"],
            power_ground=float(power["ground"][idx["ground"]]),
            power_excited=float(power["excited"][idx["excited"]]),
            p_value_ge=p_value,
            kappa_difference_hz=chosen_kappa["excited"] - chosen_kappa["ground"],
            unconstrained_kappa_ground_hz=float(kappa["ground"][free_idx["ground"]]),
            unconstrained_kappa_excited_hz=float(kappa["excited"][free_idx["excited"]]),
            constrained=idx != free_idx,
            at_edge_ground=idx["ground"] in (0, num_points - 1),
            at_edge_excited=idx["excited"] in (0, num_points - 1),
            success=success,
        )
        best[str(q)] = idx

    ds_fit = ds.assign_coords(
        best_point_ground=("qubit", [best[str(q)]["ground"] for q in ds.qubit.values]),
        best_point_excited=("qubit", [best[str(q)]["excited"] for q in ds.qubit.values]),
    )
    return ds_fit, fits


def log_fitted_results(fit_results: dict, alpha: float, log_callable=None):
    """Log one line per qubit with the chosen kappas, their difference and the g-vs-e z-test."""
    if log_callable is None:
        log_callable = print
    for q, fit in fit_results.items():
        status = "SUCCESS!" if fit["success"] else "FAIL! (no kappa pair within max_kappa_difference_hz)"
        verdict = "indistinguishable (depleted)" if fit["p_value_ge"] > alpha else "distinguishable (not depleted)"
        log_callable(
            f"Results for qubit {q}: kappa_ground = {fit['kappa_ground_hz'] * 1e-3:.2f} kHz, "
            f"kappa_excited = {fit['kappa_excited_hz'] * 1e-3:.2f} kHz "
            f"(difference {fit['kappa_difference_hz'] * 1e-3:+.2f} kHz) | "
            f"g-vs-e z-test p = {fit['p_value_ge']:.3g}, {verdict} | --> {status}"
        )
        if fit["constrained"]:
            log_callable(
                f"  {q}: difference limit applied; independent minima were kappa_ground = "
                f"{fit['unconstrained_kappa_ground_hz'] * 1e-3:.2f} kHz, kappa_excited = "
                f"{fit['unconstrained_kappa_excited_hz'] * 1e-3:.2f} kHz."
            )
        for state in STATES:
            if fit[f"at_edge_{state}"]:
                log_callable(
                    f"  WARNING {q}: {state} kappa is at the edge of the scan (still written to state); "
                    "consider rescanning around it."
                )
