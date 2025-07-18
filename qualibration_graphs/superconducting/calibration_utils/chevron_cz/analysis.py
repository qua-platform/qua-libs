import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis.fitting import fit_oscillation_decay_exp, oscillation_decay_exp
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit

from quam.components.quantum_components import qubit


def rabi_chevron_model(ft, J, f0, a, offset):
    f, t = ft
    J = J
    det = (f - f0) / 2
    # g = offset+a * np.sin(2*np.pi*np.sqrt(J**2 + (w-w0)**2) * t)**2*np.exp(-tau*np.abs((w-w0)))
    g = offset + a * (J**2) / (J**2 + det**2) * np.sin(2 * np.pi * np.sqrt(J**2 + det**2) * t) ** 2

    return g.ravel()


def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    da_target = ds_qp.state_target
    exp_data = da_target.values
    detuning = da_target.detuning[0]
    time = da_target.time * 1e-9
    t, f = np.meshgrid(time, detuning)
    initial_guess = (1e9 / init_length, init_detuning[0], -1, 1.0)
    fdata = np.vstack((f.ravel(), t.ravel()))
    tdata = exp_data.ravel()
    popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
    J = popt[0]
    f0 = popt[1]
    a = popt[2]
    offset = popt[3]

    return J, f0, a, offset


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    J: float  # Rabi frequency
    f0: float  # Frequency at which the Rabi oscillation is maximum
    cz_len: int  # Length of the CZ gate in nanoseconds
    cz_amp: float  # Amplitude of the CZ gate in volts


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    pass


def fit_chevron_cz(ds, dim):
    def fit_routine(ds_qp):
        # ds_qp is a Dataset for a single qubit_pair
        amp_guess = ds_qp.state_target.max("time") - ds_qp.state_target.min("time")
        flux_amp_idx = int(amp_guess.argmax())
        flux_amp = float(ds_qp.amp_full[0][flux_amp_idx])
        fit_data = fit_oscillation_decay_exp(ds_qp.state_target.isel(amplitude=flux_amp_idx), "time")
        flux_time = int(1 / fit_data.sel(fit_vals="f"))

        amplitudes = flux_amp
        detunings = -(flux_amp**2) * ds_qp.quad_term_control
        lengths = flux_time - flux_time % 4 + 4

        t = ds_qp.time * 1e-9
        f = ds_qp.detuning
        t, f = np.meshgrid(t, f)
        J, f0, a, offset = fit_rabi_chevron(ds_qp, lengths * 2, detunings.values)
        # initial_guess = (1e9 / lengths / 2, detunings, -1, 1.0)
        # initial_model = rabi_chevron_model((f, t), *initial_guess).reshape(len(ds_qp.amplitude), len(ds_qp.time))
        # data_fitted = rabi_chevron_model((f, t), J, f0, a, offset).reshape(len(ds_qp.amplitude), len(ds_qp.time))
        detunings = f0
        amplitudes = np.sqrt(-detunings / ds_qp.quad_term_control)
        flux_time = int(1 / (2 * J) * 1e9)
        lengths = flux_time - flux_time % 4 + 4

        # Return as DataArray for stacking
        return xr.DataArray([J, f0, a, offset], dims=["fit_vals"])

    # Use groupby-apply pattern
    fit_res = ds.groupby("qubit_pair").apply(fit_routine)
    fit_res = fit_res.assign_coords(fit_vals=("fit_vals", ["J", "f0", "a", "offset"]))
    return fit_res


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["I_control", "Q_control"])

    def detuning(qp, amp):
        return -((amp * node.namespace["pulse_amplitudes"][qp.name]) ** 2) * qp.qubit_control.freq_vs_flux_01_quad_term

    def abs_amp(qp, amp):
        return amp * node.namespace["pulse_amplitudes"][qp.name]

    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]

    ds = ds.assign_coords(
        {"detuning": (["qubit_pair", "amplitude"], np.array([detuning(qp, ds.amplitude) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"amp_full": (["qubit_pair", "amplitude"], np.array([abs_amp(qp, ds.amplitude) for qp in qubit_pairs]))},
    )

    ds = ds.assign_coords(
        {
            "quad_term_control": (
                ["qubit_pair"],
                np.array([qp.qubit_control.freq_vs_flux_01_quad_term for qp in qubit_pairs]),
            )
        }
    )

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

    ds_fit_res = fit_chevron_cz(ds, "qubit_pair")

    ds_fit = xr.merge([ds, ds_fit_res.rename("fit")])

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Populate the FitParameters class with fitted values
    fit_results = {}
    for qp in fit.qubit_pair.values:
        J_val = fit.fit.sel(qubit_pair=qp, fit_vals="J").values.item()
        f0_val = fit.fit.sel(qubit_pair=qp, fit_vals="f0").values.item()
        cz_len_val = int(1 / (2 * J_val) * 1e9)
        cz_amp_val = np.sqrt(-f0_val / fit.quad_term_control.sel(qubit_pair=qp).values.item())

        # Determine success based on reasonable parameter ranges
        amp_min = fit.amp_full.sel(qubit_pair=qp).min().item()
        amp_max = fit.amp_full.sel(qubit_pair=qp).max().item()

        success = bool((10 < cz_len_val < 1000) and (amp_min <= cz_amp_val <= amp_max))

        fit_results[qp] = FitParameters(
            success=success,
            J=J_val,
            f0=f0_val,
            cz_len=cz_len_val,
            cz_amp=cz_amp_val,
        )

    fit = fit.assign_coords(
        {
            "cz_len": ("qubit_pair", [fit_results[qp].cz_len for qp in fit.qubit_pair.values]),
            "cz_amp": ("qubit_pair", [fit_results[qp].cz_amp for qp in fit.qubit_pair.values]),
        }
    )
    return fit, fit_results
