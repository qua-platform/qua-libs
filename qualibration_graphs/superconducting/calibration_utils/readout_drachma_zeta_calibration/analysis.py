from dataclasses import dataclass

import numpy as np
import xarray as xr

from calibration_utils.readout_drachma_common import STATES, select_points


@dataclass
class FitParameters:
    """Stores the chosen zeta values of a single qubit."""

    zeta_ground_hz: float
    zeta_excited_hz: float
    power_ground: float
    power_excited: float
    p_value_noop_ground: float
    p_value_noop_excited: float
    zeta_difference_hz: float
    p_flat_ground: float
    p_flat_excited: float
    changed_ground: bool
    changed_excited: bool
    depleted_any_ground: bool
    depleted_any_excited: bool
    success: bool


def build_zeta_grid(current_hz: float, step_hz: float, num_points: int) -> np.ndarray:
    """Zeta scan (Hz) for one state of one qubit: num_points points spaced by step_hz around the current
    zeta, which sits on the grid (for an even num_points the extra point is above it). If the lower end
    would go below 0, the whole grid is shifted so that it starts at 0."""
    start = max(current_hz - ((num_points - 1) // 2) * step_hz, 0.0)
    return start + step_hz * np.arange(num_points)


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, dict[str, FitParameters]]:
    """Pick, per qubit, the zeta of each state from the lockstep scan (see select_points).

    A state whose scan is flat (node.parameters.flatness_alpha) keeps its current zeta; a responsive state
    takes its lowest-cost point, a common one when both respond. The fit fails when a changed state sits on
    the upper edge of the scan (the true minimum is outside the scan) or on the lower edge unless that edge is
    zeta = 0 (the zero-start linear scan)."""
    alpha = node.parameters.alpha
    flatness_alpha = node.parameters.flatness_alpha
    n = ds.sizes["point"]
    fits: dict[str, FitParameters] = {}
    best = {}
    for q in ds.qubit.values:
        zeta = {s: ds[f"zeta_{s}_hz"].sel(qubit=q).values for s in STATES}
        power = {s: ds["power"].sel(qubit=q, state=s).values for s in STATES}
        err = {s: ds["power_err"].sel(qubit=q, state=s).values for s in STATES}
        # Datasets recorded before the current values were stored fall back to the grid centre.
        current = {
            s: (
                float(ds[f"current_zeta_{s}_hz"].sel(qubit=q))
                if f"current_zeta_{s}_hz" in ds.coords
                else float(zeta[s][(n - 1) // 2])
            )
            for s in STATES
        }
        selection = select_points(zeta, power, err, current, flatness_alpha)
        idx, changed = selection["idx"], selection["changed"]
        chosen = {s: float(zeta[s][idx[s]]) if changed[s] else current[s] for s in STATES}
        p_value = {s: float(ds["p_value_noop"].sel(qubit=q, state=s).isel(point=idx[s])) for s in STATES}
        depleted_any = {s: bool((ds["p_value_noop"].sel(qubit=q, state=s) > alpha).any()) for s in STATES}
        edge_ok = all(
            not changed[s] or not (idx[s] == n - 1 or (idx[s] == 0 and float(zeta[s][0]) != 0.0)) for s in STATES
        )
        fits[str(q)] = FitParameters(
            zeta_ground_hz=chosen["ground"],
            zeta_excited_hz=chosen["excited"],
            power_ground=float(power["ground"][idx["ground"]]),
            power_excited=float(power["excited"][idx["excited"]]),
            p_value_noop_ground=p_value["ground"],
            p_value_noop_excited=p_value["excited"],
            zeta_difference_hz=chosen["excited"] - chosen["ground"],
            p_flat_ground=selection["p_flat"]["ground"],
            p_flat_excited=selection["p_flat"]["excited"],
            changed_ground=changed["ground"],
            changed_excited=changed["excited"],
            depleted_any_ground=depleted_any["ground"],
            depleted_any_excited=depleted_any["excited"],
            success=edge_ok,
        )
        best[str(q)] = idx

    ds_fit = ds.assign_coords(
        best_point_ground=("qubit", [best[str(q)]["ground"] for q in ds.qubit.values]),
        best_point_excited=("qubit", [best[str(q)]["excited"] for q in ds.qubit.values]),
    )
    return ds_fit, fits


def log_fitted_results(fit_results: dict, alpha: float, log_callable=None):
    """Log one line per qubit with the chosen zetas and the test-vs-no-operation z-test at each chosen point,
    plus a note per state that kept its current zeta (flat scan) and a warning per state that is never depleted."""
    if log_callable is None:
        log_callable = print
    for q, fit in fit_results.items():
        status = "SUCCESS!" if fit["success"] else "FAIL! (minimum at the edge of the scan)"
        verdicts = {s: "depleted" if fit[f"p_value_noop_{s}"] > alpha else "not depleted" for s in STATES}
        log_callable(
            f"Results for qubit {q}: zeta_ground = {fit['zeta_ground_hz'] * 1e-3:.2f} kHz, "
            f"zeta_excited = {fit['zeta_excited_hz'] * 1e-3:.2f} kHz "
            f"(difference {fit['zeta_difference_hz'] * 1e-3:+.2f} kHz) | "
            f"vs no-operation p: ground = {fit['p_value_noop_ground']:.3g} ({verdicts['ground']}), "
            f"excited = {fit['p_value_noop_excited']:.3g} ({verdicts['excited']}) | --> {status}"
        )
        for state in STATES:
            if not fit[f"changed_{state}"]:
                log_callable(
                    f"  {q}: {state} zeta kept at its current value "
                    f"(flatness p = {fit[f'p_flat_{state}']:.3g}; flat scan or best point is the current one)."
                )
            if not fit[f"depleted_any_{state}"]:
                log_callable(
                    f"  WARNING {q}: {state} is not depleted at any scanned zeta (p <= alpha everywhere); "
                    "another parameter (chi, resonator frequency) is probably mis-set."
                )
