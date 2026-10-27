import logging
from dataclasses import asdict, dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp

from calibration_utils.common_utils.leakage_rb import (
    SHALLOW_LEAKAGE_THRESHOLD,
    LeakageFit,
    correct_populations,
    fit_leakage,
    mean_and_stderr_over_sequences,
    read_confusion_matrix,
)


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    error_per_clifford: float
    error_per_gate: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_fidelity = f"\tSingle qubit gate fidelity: {100 * (1 - fit_results[q]['error_per_gate']):.3f} %\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_fidelity)
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    ds_fit = ds
    if getattr(node.parameters, "measure_leakage", False):
        # g/e/f readout: the survival is P(g); 1 - P(e) would count leaked shots as survived.
        ds_fit["averaged_data"] = ds.pg.mean(dim="nb_of_sequences")
    elif node.parameters.use_state_discrimination:
        ds_fit["averaged_data"] = 1 - ds.state.mean(dim="nb_of_sequences")
    else:
        ds_fit["averaged_data"] = 1 - ds.I.mean(dim="nb_of_sequences")
    # Fit the exponential decay
    fit_data = fit_decay_exp(ds_fit["averaged_data"], "depths")

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Extract the decay rate
    alpha = np.exp(fit.fit_data.sel(fit_vals="decay"))
    # average_gate_per_clifford = 45/24 = 1.875
    average_gate_per_clifford = (1 * 3 + 9 * 2 + 1 * 4 + 2 * 3 + 4 * 2 + 2 * 3) / 24
    # EPC from here: https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html#Step-5:-Fit-the-results
    fit["error_per_clifford"] = (1 - alpha) * (1 - 1 / 2)
    fit["error_per_gate"] = fit["error_per_clifford"] / average_gate_per_clifford
    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.error_per_gate)
    rb_success = (0 < fit.error_per_gate) & (fit.error_per_gate < 1)
    success_criteria = ~nan_success & rb_success
    fit = fit.assign({"success": success_criteria})

    # Save fitting results
    fit_results = {
        q: FitParameters(
            error_per_clifford=float(fit.sel(qubit=q)["error_per_clifford"]),
            error_per_gate=float(fit.sel(qubit=q)["error_per_gate"]),
            success=bool(fit.sel(qubit=q).success),
        )
        for q in fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in fit.qubit.values}

    return fit, fit_results


# --- Leakage RB (measure_leakage=True) -------------------------------------------------------
#
# ``pg``, ``pe`` and ``pf`` hold, per random sequence and depth, the fraction of shots read as g, e
# and f. The leakage model, its fit and the population correction are shared with the two-qubit RB
# (``calibration_utils.common_utils.leakage_rb``).


@dataclass
class LeakageResult:
    """Leakage RB result of one qubit: its leakage ``l1`` and seepage ``l2`` per Clifford.

    ``failure`` is set when the g/e/f readout looks wrong (nothing is then written to the state).
    Non-finite numbers in ``per_clifford`` are ``None``: the result goes to the state as it is, and
    QUAM would write NaN, which is not valid JSON.
    """

    confusion_corrected: bool
    per_clifford: LeakageFit
    success: bool
    failure: str = ""


def fit_leakage_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, LeakageResult]]:
    """Fit the leaked population P(f) against circuit depth for each qubit.

    Each sequence's (P(g), P(e), P(f)) is corrected with the qubit's g/e/f confusion matrix when it
    has one, then P(f) is averaged over sequences and fitted. A qubit fails, so nothing is written
    to the state, if its leaked population at the shallowest depth is above
    ``SHALLOW_LEAKAGE_THRESHOLD``: so few gates cannot leak that much, so the g/e/f readout or
    reset is failing.
    """
    qubits = node.namespace["qubits"]
    depths = ds.depths.values
    shallow = int(np.argmin(depths))
    means, stderrs, fits, results = [], [], [], {}
    for qubit in qubits:
        populations = ds[["pg", "pe", "pf"]].sel(qubit=qubit.name).transpose("depths", "nb_of_sequences")
        measured = np.stack([populations[p].values for p in ("pg", "pe", "pf")], axis=-1)
        confusion = read_confusion_matrix(qubit)
        leaked = correct_populations(measured, [confusion])[..., 2]
        mean, stderr = mean_and_stderr_over_sequences(leaked, shots_per_sequence=node.parameters.num_shots)
        fit = fit_leakage(depths, mean, stderr)
        failure = ""
        if mean[shallow] > SHALLOW_LEAKAGE_THRESHOLD:
            failure = (
                f"Leaked population at depth {depths[shallow]:g} is {mean[shallow]:.3f}, above "
                f"{SHALLOW_LEAKAGE_THRESHOLD:g}: the g/e/f readout or reset is suspect, so nothing is written "
                "to the state. Recalibrate the g/e/f readout (15_iq_blobs_gef) and rerun."
            )
        means.append(mean)
        stderrs.append(stderr)
        fits.append(fit)
        results[qubit.name] = LeakageResult(
            confusion_corrected=confusion is not None,
            per_clifford=LeakageFit(**{k: _none_if_nan(v) for k, v in asdict(fit).items()}),
            success=fit.success and not failure,
            failure=failure,
        )

    per_qubit = lambda values: ("qubit", np.asarray(values))  # noqa: E731
    ds_leakage = xr.Dataset(
        {
            "leaked": (["qubit", "depths"], np.asarray(means)),
            "leaked_stderr": (["qubit", "depths"], np.asarray(stderrs)),
            "leaked_fit": (["qubit", "depths"], np.asarray([fit.curve(depths) for fit in fits])),
            "leakage_l1": per_qubit([fit.l1 for fit in fits]),
            "leakage_l1_stderr": per_qubit([fit.l1_stderr for fit in fits]),
            "leakage_l2": per_qubit([fit.l2 for fit in fits]),
            "leakage_l2_stderr": per_qubit([fit.l2_stderr for fit in fits]),
            "leakage_method": per_qubit([fit.method for fit in fits]),
            "success": per_qubit([results[qubit.name].success for qubit in qubits]),
        },
        coords={"qubit": [qubit.name for qubit in qubits], "depths": depths},
    )
    ds_leakage.leaked.attrs = {"long_name": "leaked population P(f)"}
    ds_leakage.leakage_l1.attrs = {"long_name": "leakage L1 per Clifford"}
    ds_leakage.leakage_l2.attrs = {"long_name": "seepage L2 per Clifford"}
    return ds_leakage, results


def format_leakage_rate(rate: float | None, stderr: float | None = None) -> str:
    """Format a leakage rate as ``value +/- stderr``, or ``n/a``."""
    if rate is None or not np.isfinite(rate):
        return "n/a"
    if stderr is None or not np.isfinite(stderr):
        return f"{rate:.2e}"
    return f"{rate:.2e} +/- {stderr:.1e}"


def log_leakage_results(leakage_results: Dict, log_callable=None):
    """Log L1 (and L2 when the fit has it) per Clifford for each qubit, with fit issues and warnings."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in leakage_results.items():
        fit = result["per_clifford"]
        status = "" if result["success"] else " FAILED"
        lines = [
            f"Leakage for qubit {q}{status}: L1 per Clifford = {format_leakage_rate(fit['l1'], fit['l1_stderr'])}, "
            f"L2 = {format_leakage_rate(fit['l2'], fit['l2_stderr'])} ({fit['method'] or 'no fit'})"
        ]
        if result["failure"]:
            lines.append(f"\t- {result['failure']}")
        lines += [f"\t- {issue}" for issue in fit["fit_issues"]]
        lines += [f"\tWarning: {warning}" for warning in fit["fit_warnings"]]
        if not result["confusion_corrected"]:
            lines.append("\tNot corrected for g/e/f readout misassignment (no gef_confusion_matrix).")
        log_callable("\n".join(lines))


def _none_if_nan(value):
    return None if isinstance(value, float) and not np.isfinite(value) else value
