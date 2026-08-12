import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode
from qualibration_libs.analysis import peaks_dips
from calibration_utils.measurement_utils import process_streams


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy chirp fit parameters for a single qubit"""

    frequency: float
    relative_freq: float
    fwhm: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None, label: str = ""):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    label : str, optional
        Prefix label to distinguish result source (e.g. "Threshold", "Peak fit").
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    prefix = f"[{label}] " if label else ""
    for q in fit_results.keys():
        s_qubit = f"{prefix}Results for qubit {q}: "
        s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Build the conditional parity expectations from the explicitly named raw streams."""
    qubits = node.namespace["qubits"]
    stream_item_names = [f"{q.name}_parity_diff" for q in qubits]
    ds = process_streams(
        ds,
        stream_item_names,
        parity_measurement=node.parameters.parity_measurement,
        sweep_dims=("detuning",),
    )

    rename_map = {}
    for qubit in qubits:
        stream_name = f"{qubit.name}_parity_diff"
        for prefix in ("E_p1_given_p0_0", "E_p1_given_p0_1"):
            source_name = f"{prefix}_{stream_name}"
            target_name = f"{prefix}_{qubit.name}"
            if source_name in ds.data_vars:
                rename_map[source_name] = target_name

    return ds.rename(rename_map) if rename_map else ds


def find_frequency_by_threshold(
    ds: xr.Dataset, node: QualibrationNode
) -> Dict[str, FitParameters]:
    """Find the qubit frequency by locating the above-threshold region of the signal.

    For each qubit, reads ``{analysis_signal}_{qname}`` directly from the
    processed dataset.  All detuning points where the signal is at or above
    ``signal_threshold`` are collected.  The centre frequency is the
    signal-weighted mean of those detunings, and the reported FWHM is the full
    span of the above-threshold region.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset containing ``{analysis_signal}_{qname}`` variables
        (1-D over ``detuning``) as produced by ``process_raw_dataset``.
    node : QualibrationNode
        Node whose ``parameters.signal_threshold`` and
        ``parameters.analysis_signal`` are used.

    Returns
    -------
    dict[str, FitParameters]
    """
    qubits = node.namespace["qubits"]
    analysis_signal = node.parameters.analysis_signal
    threshold = node.parameters.signal_threshold
    fit_results: Dict[str, FitParameters] = {}

    for q in qubits:
        signal_var = f"{analysis_signal}_{q.name}"
        if signal_var not in ds.data_vars:
            fit_results[q.name] = FitParameters(
                frequency=np.nan,
                relative_freq=np.nan,
                fwhm=np.nan,
                success=False,
            )
            continue

        signal = ds[signal_var].values
        detuning = ds.detuning.values

        above = signal >= threshold
        if not np.any(above):
            fit_results[q.name] = FitParameters(
                frequency=np.nan,
                relative_freq=np.nan,
                fwhm=np.nan,
                success=False,
            )
            continue

        above_detunings = detuning[above]
        above_signal = signal[above]

        center_detuning = float(np.average(above_detunings, weights=above_signal))
        width = float(above_detunings.max() - above_detunings.min())
        abs_frequency = center_detuning + q.xy.RF_frequency

        fit_results[q.name] = FitParameters(
            frequency=float(abs_frequency),
            relative_freq=float(center_detuning),
            fwhm=width,
            success=True,
        )

    return fit_results


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit the qubit Larmor frequency and FWHM for each qubit in the dataset.

    Expects ``ds`` to already contain ``{analysis_signal}_{qname}`` variables
    (1-D over ``detuning``) as produced by ``process_raw_dataset``.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the processed parity-stream signal variables.
    node : QualibrationNode
        The node containing parameters and namespace.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    dict
        Dictionary of FitParameters per qubit.
    """
    qubits = node.namespace["qubits"]
    analysis_signal = node.parameters.analysis_signal
    qubit_names = [q.name for q in qubits]

    # Build the (qubit, detuning) pdiff DataArray from the per-qubit signal variables
    arrays = []
    for qname in qubit_names:
        var = f"{analysis_signal}_{qname}"
        if var not in ds.data_vars:
            raise KeyError(
                f"Expected variable {var!r} not found in dataset. "
                "Did you call process_raw_dataset before fit_raw_data?"
            )
        arrays.append(ds[var].values)

    pdiff = xr.DataArray(
        np.array(arrays),
        dims=["qubit", "detuning"],
        coords={"qubit": qubit_names, "detuning": ds.detuning},
    )

    ds_fit = ds.assign({"pdiff": pdiff})

    # Find the peak with minimal prominence; returns nan if no peak found
    fit_vals = peaks_dips(pdiff, dim="detuning", prominence_factor=5)
    ds_fit = xr.merge([ds_fit, fit_vals])

    # Add full-frequency coordinate (carrier + detuning per qubit)
    rf_freqs = np.array([q.xy.RF_frequency for q in qubits])
    full_freq = ds.detuning.values[np.newaxis, :] + rf_freqs[:, np.newaxis]
    ds_fit = ds_fit.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds_fit.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted qubit frequency
    full_freq = np.array([q.xy.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    rel_freq = fit.position
    fit = fit.assign({"res_freq": ("qubit", res_freq.data)})
    fit = fit.assign({"relative_freq": ("qubit", rel_freq.data)})
    fit.res_freq.attrs = {"long_name": "qubit Larmor frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign({"fwhm": fwhm})
    fit.fwhm.attrs = {"long_name": "qubit fwhm", "units": "Hz"}

    # Assess whether the fit was successful or not
    freq_success = (
        np.abs(res_freq) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    )
    fwhm_success = (
        np.abs(fwhm) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    )
    success_criteria = freq_success & fwhm_success
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            relative_freq=fit.sel(qubit=q).relative_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results


def analyse_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
    *,
    log_callable=None,
) -> tuple[Optional[xr.Dataset], dict, Optional[dict], dict]:
    """Run the chirp analysis flow and return the datasets, fit results, and outcomes."""
    threshold_results = find_frequency_by_threshold(ds, node)
    fit_results = {k: vars(v) for k, v in threshold_results.items()}
    log_fitted_results(fit_results, log_callable=log_callable, label="Threshold")

    ds_fit: Optional[xr.Dataset] = None
    peak_fit_results: Optional[dict] = None
    if node.parameters.fit_peak:
        ds_fit, peak_results = fit_raw_data(ds, node)
        peak_fit_results = {k: vars(v) for k, v in peak_results.items()}
        log_fitted_results(peak_fit_results, log_callable=log_callable, label="Peak fit")

        for q_name, thr in fit_results.items():
            peak = peak_fit_results.get(q_name, {})
            if not (thr.get("success") and peak.get("success")):
                continue
            tolerance = thr["fwhm"] / 2 if thr["fwhm"] > 0 else np.inf
            diff = abs(peak["frequency"] - thr["frequency"])
            if diff > tolerance and log_callable is not None:
                log_callable(
                    f"WARNING {q_name}: peak fit ({1e-9 * peak['frequency']:.4f} GHz) and "
                    f"threshold ({1e-9 * thr['frequency']:.4f} GHz) disagree by "
                    f"{1e-3 * diff:.1f} kHz (tolerance: {1e-3 * tolerance:.1f} kHz)"
                )

    outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in fit_results.items()
    }
    return ds_fit, fit_results, peak_fit_results, outcomes
