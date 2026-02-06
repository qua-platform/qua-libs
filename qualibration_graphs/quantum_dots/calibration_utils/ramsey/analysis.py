import logging
from dataclasses import dataclass, asdict
from typing import Tuple, Dict

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation


@dataclass
class RamseyDetuningFitParameters:
    """Stores the relevant Ramsey detuning experiment fit parameters for a single qubit."""

    freq_offset: float
    success: bool

    def to_dict(self):
        return asdict(self)


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results.

    Parameters
    ----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_detuning = f"\tFrequency offset to correct: {1e-6 * fit_results[q]['freq_offset']:.3f} MHz"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_detuning)


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, RamseyDetuningFitParameters]]:
    """
    Fit the frequency offset from parity difference oscillations as a function of detuning.

    The parity difference signal (pdiff) oscillates as a function of frequency detuning.
    The oscillation frequency depends on the fixed idle time: pdiff ~ cos(2*pi*f_offset*t_idle + ...).
    We fit the oscillation to extract the frequency offset from the peak of pdiff.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the raw data with pdiff and detuning dimensions.
    node : QualibrationNode
        The calibration node containing parameters and qubits.

    Returns
    -------
    Tuple[xr.Dataset, Dict[str, RamseyDetuningFitParameters]]
        - Dataset with fit results merged in
        - Dictionary of fit parameters for each qubit
    """
    qubits = node.namespace["qubits"]

    ds_fit = ds.copy()
    fit_results = {}

    for qubit in qubits:
        qubit_pdiff = ds[f"pdiff_{qubit.name}"]

        try:
            fit = fit_oscillation(qubit_pdiff, "detuning")

            fitted_freq = float(fit.sel(fit_vals="f"))
            fitted_offset_pos = float(fit.sel(fit_vals="offset"))
            fitted_a = float(fit.sel(fit_vals="a"))
            fitted_phi = float(fit.sel(fit_vals="phi"))

            # The peak of pdiff corresponds to the frequency offset
            # where the qubit is maximally driven (resonance condition)
            # The center of the oscillation pattern gives the detuning correction
            detuning_values = ds.detuning.values
            fitted_curve = oscillation(detuning_values, fitted_a, fitted_freq, fitted_phi, fitted_offset_pos)

            # Find the detuning value at the peak of pdiff
            peak_idx = np.argmax(fitted_curve)
            freq_offset = float(detuning_values[peak_idx])

            ds_fit[f"pdiff_fit_{qubit.name}"] = xr.DataArray(
                fitted_curve,
                dims=["detuning"],
                coords={"detuning": ds.detuning},
            )

            fit_results[qubit.name] = RamseyDetuningFitParameters(
                freq_offset=freq_offset,
                success=True,
            )
        except Exception:
            fit_results[qubit.name] = RamseyDetuningFitParameters(
                freq_offset=0.0,
                success=False,
            )

    node.outcomes = {
        qubit.name: ("successful" if fit_results[qubit.name].success else "failed")
        for qubit in qubits
    }

    return ds_fit, fit_results
