import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit


@dataclass
class FitResults:
    """Fit results for a single qubit pair from the CZ phase compensation calibration.

    Attributes:
    -----------
    control_phase_correction : float
        Residual single-qubit phase accumulated by the control qubit during the CZ gate (in 2π units).
        This value is subtracted from ``phase_shift_control`` in the state update.
    target_phase_correction : float
        Residual single-qubit phase accumulated by the target qubit during the CZ gate (in 2π units).
        This value is subtracted from ``phase_shift_target`` in the state update.
    success : bool
        True if the sinusoidal fit converged for both qubits.
    """

    control_phase_correction: float
    target_phase_correction: float
    success: bool


def fix_oscillation_phi_2pi(fit_data):
    """Extract and normalise the oscillation phase to the [0, 1) range (representing 0 to 2π).

    Parameters:
    -----------
    fit_data : xr.DataArray
        Oscillation fit result with a ``fit_vals`` dimension containing ``"phi"``.

    Returns:
    --------
    xr.DataArray
        Phase values normalised to [0, 1).
    """
    phase = fit_data.sel(fit_vals="phi")
    phase = (phase / (2 * np.pi)) % 1
    return phase


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """
    Logs the node-specific fitted results for all qubit pairs.

    Parameters:
    -----------
    fit_results : Dict[str, FitResults]
        Dictionary containing FitResults for each qubit pair.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        s_qubit = f"Results for qubit pair {qp_name}: "
        s_control = f"\tControl qubit phase correction: {fit_results[qp_name]['control_phase_correction']:.6f} a.u."
        s_target = f"\tTarget qubit phase correction: {fit_results[qp_name]['target_phase_correction']:.6f} a.u."

        if fit_results[qp_name]["success"]:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_control + s_target

        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert raw IQ data to volts when state discrimination is not used.

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset from the experiment, containing either ``I_control`` / ``Q_control`` /
        ``I_target`` / ``Q_target`` (raw IQ) or ``state_control`` / ``state_target``
        (state-discrimination) variables.
    node : QualibrationNode
        Calibration node providing ``qubit_pairs`` from its namespace (required by
        ``convert_IQ_to_V``).

    Returns:
    --------
    xr.Dataset
        Dataset with IQ variables converted to volts, or unchanged if state discrimination
        was used.
    """
    if hasattr(ds, "I_control"):
        ds = convert_IQ_to_V(
            ds, qubit_pairs=node.namespace["qubit_pairs"], IQ_list=["I_control", "Q_control", "I_target", "Q_target"]
        )
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Fit the CZ conditional phase data for each qubit pair.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the processed data.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with fit results and dictionary of fit results for each qubit pair.
    """
    ds_fit = ds.groupby("qubit_pair").apply(fit_routine)

    # Extract the relevant fitted parameters
    ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)

    return ds_fit, fit_results


def fit_routine(da):
    """Fit Ramsey frame-rotation oscillations and extract phase corrections for one qubit pair.

    Fits a sinusoidal oscillation over the ``frame`` axis separately for the control qubit
    and the target qubit.  The fitted phase of each oscillation gives the residual single-qubit
    phase error introduced by the CZ gate.

    Parameters:
    -----------
    da : xr.Dataset
        Single-pair dataset with ``state_control`` / ``state_target`` (state-discrimination)
        or ``I_control`` / ``I_target`` (raw IQ) variables, and a ``frame`` dimension.

    Returns:
    --------
    xr.Dataset
        Input dataset extended with ``fitted_control``, ``fitted_target``,
        ``fitted_control_phase``, ``fitted_target_phase``, and ``success`` data variables.
    """
    if hasattr(da, "state_target"):
        data_control = "state_control"
        data_target = "state_target"
    else:
        data_control = "I_control"
        data_target = "I_target"

    try:
        # Fit oscillation for each control state and amplitude
        if data_control == "state_control":
            fit_control = fit_oscillation(da[data_control], "frame")
            fit_target = fit_oscillation(da[data_target], "frame")
        else:
            fit_control = fit_oscillation(da[data_control].sel(control_target="c"), "frame")
            fit_target = fit_oscillation(da[data_target].sel(control_target="t"), "frame")
        # Add fitted oscillation curves to the dataset
        da = da.assign(
            {
                "fitted_control": oscillation(
                    da.frame,
                    fit_control.sel(fit_vals="a"),
                    fit_control.sel(fit_vals="f"),
                    fit_control.sel(fit_vals="phi"),
                    fit_control.sel(fit_vals="offset"),
                )
            }
        )

        da = da.assign(
            {
                "fitted_target": oscillation(
                    da.frame,
                    fit_target.sel(fit_vals="a"),
                    fit_target.sel(fit_vals="f"),
                    fit_target.sel(fit_vals="phi"),
                    fit_target.sel(fit_vals="offset"),
                )
            }
        )

        # Extract phase and calculate phase difference
        phase_control = fix_oscillation_phi_2pi(fit_control)
        phase_target = fix_oscillation_phi_2pi(fit_target)

        da = da.assign(
            {
                "fitted_control_phase": phase_control,
            }
        )
        da = da.assign(
            {
                "fitted_target_phase": phase_target,
            }
        )

        da = da.assign(
            {
                "success": True,
            }
        )
    except Exception as e:
        print(f"Error fitting data: {e}")

        da = da.assign(
            {
                "success": False,
            }
        )

    return da


def _extract_relevant_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Assign xarray metadata attributes and build the FitResults dictionary.

    Parameters:
    -----------
    ds_fit : xr.Dataset
        Dataset produced by applying ``fit_routine`` per qubit pair. Must contain
        ``fitted_control_phase``, ``fitted_target_phase``, and ``success`` data variables.
    node : QualibrationNode
        Calibration node providing ``qubit_pairs`` from its namespace.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with ``long_name`` / ``units`` attrs set on key variables, and a
        dictionary of ``FitResults`` keyed by qubit pair name.
    """
    qubit_pairs = node.namespace["qubit_pairs"]

    # Add metadata attributes to the dataset
    if "fitted_control_phase" in ds_fit.data_vars:
        ds_fit.fitted_control_phase.attrs = {"long_name": "control qubit phase correction", "units": "2π"}
    if "fitted_target_phase" in ds_fit.data_vars:
        ds_fit.fitted_target_phase.attrs = {"long_name": "target qubit phase correction", "units": "2π"}

    # Create FitResults for each qubit pair
    fit_results = {}
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)
        if qp_data.success.values:
            fit_results[qp_name] = FitResults(
                control_phase_correction=qp_data.fitted_control_phase.values,
                target_phase_correction=qp_data.fitted_target_phase.values,
                success=True,
            )
        else:
            fit_results[qp_name] = FitResults(
                control_phase_correction=np.nan,
                target_phase_correction=np.nan,
                success=False,
            )

    return ds_fit, fit_results
