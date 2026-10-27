import logging
from dataclasses import asdict, dataclass
from datetime import datetime
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
    leakage_per_gate,
    mean_and_stderr_over_sequences,
    read_confusion_matrix,
)


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    error_per_clifford: float
    error_per_gate: float
    success: bool
    error_per_clifford_stderr: float | None = None
    error_per_gate_stderr: float | None = None


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
        epc, epc_stderr = fit_results[q]["error_per_clifford"], fit_results[q].get("error_per_clifford_stderr")
        s_fidelity += f"\tError per Clifford: {epc:.2e}" + (f" +/- {epc_stderr:.1e}\n" if epc_stderr else "\n")
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_fidelity)
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # A leakage run always reads g/e/f states, whatever use_state_discrimination says.
    if not node.parameters.use_state_discrimination and not getattr(node.parameters, "measure_leakage", False):
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
    # Standard errors from the fit covariance: alpha = exp(decay), so stderr(alpha) = alpha * stderr(decay)
    alpha_stderr = alpha * np.sqrt(fit.fit_data.sel(fit_vals="decay_decay").clip(min=0))
    fit["error_per_clifford_stderr"] = alpha_stderr * (1 - 1 / 2)
    fit["error_per_gate_stderr"] = fit["error_per_clifford_stderr"] / average_gate_per_clifford
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
            error_per_clifford_stderr=_none_if_nan(float(fit.sel(qubit=q)["error_per_clifford_stderr"])),
            error_per_gate_stderr=_none_if_nan(float(fit.sel(qubit=q)["error_per_gate_stderr"])),
        )
        for q in fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in fit.qubit.values}

    return fit, fit_results


# --- Leakage RB (measure_leakage=True) -------------------------------------------------------
#
# ``pg``, ``pe`` and ``pf`` hold, per random sequence and depth, the fraction of shots read as g, e
# and f. The leakage model, its fit and the population correction are shared with the two-qubit RB
# (``calibration_utils.common_utils.leakage_rb``). 11a fits the leakage per Clifford and stores it in
# ``qubit.extras["leakage_per_clifford"]``; 11b (a gate interleaved after every Clifford) fits it per
# Clifford + gate step and subtracts that reference to give the leakage of the gate, stored in
# ``qubit.extras["leakage_<gate>"]``. The gate fidelity is written as in a normal run.

# Where 11a stores its leakage result, the reference 11b subtracts.
LEAKAGE_REFERENCE_KEY = "leakage_per_clifford"


@dataclass
class LeakageResult:
    """Leakage RB result of one qubit: its leakage ``l1`` and seepage ``l2`` per sequence step.

    A step is one Clifford (11a), or one Clifford plus the interleaved gate (11b), which also gets
    ``per_gate`` (``l1``, ``l1_stderr``, ``l2``, ``l2_stderr`` of the gate) and the time the 11a
    reference was stored (``reference_updated_at``), or ``per_gate_note`` when no reference was stored. ``failure`` is set
    when the g/e/f readout looks wrong (nothing is then written to the state). Non-finite numbers
    are ``None``: the result goes to the state as it is, and QUAM would write NaN, which is not
    valid JSON.
    """

    confusion_corrected: bool
    per_clifford: LeakageFit
    success: bool
    failure: str = ""
    per_gate: dict | None = None
    reference_updated_at: str = ""
    per_gate_note: str = ""


def fit_leakage_data(
    ds: xr.Dataset, node: QualibrationNode, *, interleaved_gate: str | None = None
) -> Tuple[xr.Dataset, Dict[str, LeakageResult]]:
    """Fit the leaked population P(f) against circuit depth for each qubit.

    Each sequence's (P(g), P(e), P(f)) is corrected with the qubit's g/e/f confusion matrix when it
    has one, then P(f) is averaged over sequences and fitted. A qubit fails, so nothing is written
    to the state, if its leaked population at the shallowest depth is above
    ``SHALLOW_LEAKAGE_THRESHOLD``: so few gates cannot leak that much, so the g/e/f readout or
    reset is failing. With ``interleaved_gate`` (11b) the leakage of that gate is the fitted rate
    minus the 11a reference in ``qubit.extras["leakage_per_clifford"]``.
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
        if interleaved_gate is not None:
            _add_leakage_per_gate(results[qubit.name], fit, qubit, interleaved_gate)

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
            **(
                {
                    f"{key}_per_gate": per_qubit(
                        [_nan_if_none((results[q.name].per_gate or {}).get(key)) for q in qubits]
                    )
                    for key in ("l1", "l1_stderr", "l2", "l2_stderr")
                }
                if interleaved_gate is not None
                else {}
            ),
        },
        coords={"qubit": [qubit.name for qubit in qubits], "depths": depths},
    )
    ds_leakage.leaked.attrs = {"long_name": "leaked population P(f)"}
    step = "Clifford" if interleaved_gate is None else f"Clifford + {interleaved_gate}"
    ds_leakage.leakage_l1.attrs = {"long_name": f"leakage L1 per {step}"}
    ds_leakage.leakage_l2.attrs = {"long_name": f"seepage L2 per {step}"}
    if interleaved_gate is not None:
        ds_leakage.l1_per_gate.attrs = {"long_name": f"leakage L1 per {interleaved_gate}"}
        ds_leakage.l2_per_gate.attrs = {"long_name": f"seepage L2 per {interleaved_gate}"}
    return ds_leakage, results


def _add_leakage_per_gate(result: LeakageResult, fit: LeakageFit, qubit, interleaved_gate: str) -> None:
    """Fill the leakage of the interleaved gate in ``result``: this run minus the 11a reference."""
    reference = (getattr(qubit, "extras", None) or {}).get(LEAKAGE_REFERENCE_KEY) or {}
    if reference.get("l1") is None:
        result.per_gate_note = (
            f"No extras[{LEAKAGE_REFERENCE_KEY!r}] for {qubit.name}: run 11a with measure_leakage=True "
            f"first. Only the leakage per Clifford + {interleaved_gate} step is reported."
        )
        return
    per_gate = leakage_per_gate([fit], [reference])
    result.per_gate = {key: _none_if_nan(float(values[0])) for key, values in per_gate.items()}
    result.reference_updated_at = reference.get("updated_at") or ""


def leakage_reference_entry(leakage_result: dict) -> dict:
    """11a's ``qubit.extras["leakage_per_clifford"]``: L1 and L2 per Clifford and when they were stored
    (``updated_at``, so 11b can tell how old its reference is).

    ``leakage_result`` is this qubit's ``node.results["leakage_results"]`` entry; the fit details
    stay in the run's saved results.
    """
    rates = leakage_result["per_clifford"]
    return {**{key: rates[key] for key in ("l1", "l1_stderr", "l2", "l2_stderr")}, "updated_at": _now()}


def leakage_per_gate_entry(leakage_result: dict) -> dict | None:
    """11b's ``qubit.extras["leakage_<gate>"]``: L1 of the interleaved gate, or None without a reference."""
    per_gate = leakage_result["per_gate"]
    if per_gate is None:
        return None
    return {"l1": per_gate["l1"], "l1_stderr": per_gate["l1_stderr"], "updated_at": _now()}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_leakage_rate(rate: float | None, stderr: float | None = None, *, pm: str = "+/-") -> str:
    """Format a leakage rate as ``(value +/- stderr)e-07``, both on one exponent so a rate consistent with
    zero reads as such, or ``n/a``; plots pass ``pm="±"`` and get ``(value ± stderr) × 10⁻⁷``."""
    if rate is None or not np.isfinite(rate):
        return "n/a"
    if stderr is None or not np.isfinite(stderr):
        return f"{rate:.2e}"
    scale = max(abs(rate), stderr)
    exponent = int(np.floor(np.log10(scale))) if scale > 0 else 0
    mantissa = f"({rate / 10**exponent:.2f} {pm} {stderr / 10**exponent:.2f})"
    if pm == "±":
        return f"{mantissa} × 10{str(exponent).translate(_SUPERSCRIPT)}"
    return f"{mantissa}e{exponent:+03d}"


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_leakage_results(leakage_results: Dict, log_callable=None, *, interleaved_gate: str | None = None):
    """Log L1 (and L2 when the fit has it) per sequence step for each qubit, and per gate for 11b."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    step = "Clifford" if interleaved_gate is None else f"Clifford + {interleaved_gate}"
    for q, result in leakage_results.items():
        fit = result["per_clifford"]
        status = "" if result["success"] else " FAILED"
        lines = [
            f"Leakage for qubit {q}{status}: L1 per {step} = {format_leakage_rate(fit['l1'], fit['l1_stderr'])}, "
            f"L2 = {format_leakage_rate(fit['l2'], fit['l2_stderr'])} ({fit['method'] or 'no fit'})"
        ]
        if interleaved_gate is not None:
            if result["per_gate"] is None:
                lines.append(f"\t{result['per_gate_note']}")
            else:
                per_gate = result["per_gate"]
                lines.append(
                    f"\tL1 per {interleaved_gate} = {format_leakage_rate(per_gate['l1'], per_gate['l1_stderr'])} "
                    f"(minus the 11a reference stored {result['reference_updated_at'] or 'at an unknown time'})"
                )
        if result["failure"]:
            lines.append(f"\t- {result['failure']}")
        lines += [f"\t- {issue}" for issue in fit["fit_issues"]]
        lines += [f"\tWarning: {warning}" for warning in fit["fit_warnings"]]
        if not result["confusion_corrected"]:
            lines.append("\tNot corrected for g/e/f readout misassignment (no gef_confusion_matrix).")
        log_callable("\n".join(lines))


def _none_if_nan(value):
    return None if isinstance(value, float) and not np.isfinite(value) else value


def _nan_if_none(value):
    return np.nan if value is None else value
