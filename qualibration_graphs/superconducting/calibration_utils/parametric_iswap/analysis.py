import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis.fitting import fit_oscillation_decay_exp, oscillation_decay_exp
from qualibration_libs.data import convert_IQ_to_V
from quam.components.quantum_components import qubit
from scipy.optimize import curve_fit


def rabi_chevron_model(f_t, J, f0, a, offset):
    """Model the Rabi chevron response for a driven two-level (or effective two-qubit CZ) system."""
    f, t = f_t
    det = (f - f0) / 2
    g = offset + a * (J**2) / (J**2 + det**2) * np.sin(2 * np.pi * np.sqrt(J**2 + det**2) * t) ** 2

    return g.ravel()


def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    """Fit a Rabi chevron dataset to extract coupling (J), resonance frequency (f0), amplitude, and offset."""
    if hasattr(ds_qp, "state_control"):
        data = ds_qp.state_control
    else:
        data = ds_qp.I_control

    try:
        da_control = data
        exp_data = da_control.values
        frequencies = da_control.frequencies_shifted.values
        durations = da_control.durations.values * 1e-9
        t, f = np.meshgrid(durations, frequencies)
        if isinstance(init_detuning, (np.ndarray, list)):
            init_detuning = init_detuning[0]
        initial_guess = (1e9 / (2 * init_length), init_detuning, -1, 1.0)
        fdata = np.vstack((f.ravel(), t.ravel()))
        tdata = exp_data.ravel()
        popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
        J = popt[0]
        f0 = popt[1]
        a = popt[2]
        offset = popt[3]
        return J, f0, a, offset
    except Exception as e:
        print(e)
        # Return NaN values if fitting fails
        return float("nan"), float("nan"), float("nan"), float("nan")


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

    for qp_name, fit_result in fit_results.items():
        # Support both dataclass instances and plain dictionaries (after asdict)
        def _get(field, default=np.nan):
            if hasattr(fit_result, field):
                return getattr(fit_result, field)
            if isinstance(fit_result, dict):
                return fit_result.get(field, default)
            return default

        success = bool(_get("success", False))
        cz_len_val = _get("cz_len", 0)
        f0_val = _get("f0", np.nan)

        s_qubit = f"Results for qubit pair {qp_name}: "
        s_qubit += "SUCCESS!\n" if success else "FAIL!\n"

        if isinstance(cz_len_val, (int, float)) and cz_len_val not in (None, np.nan):
            cz_len_str = f"\tOptimal duration: {int(cz_len_val)} ns"
        else:
            cz_len_str = "\tOptimal duration: N/A"

        if isinstance(f0_val, (int, float)) and not np.isnan(f0_val):
            f0_str = f"\tOptimal frequency: {f0_val / 1e6:.2f} MHz"
        else:
            f0_str = "\tOptimal frequency: N/A"

        log_callable(s_qubit + cz_len_str + "\n" + f0_str)


def fit_chevron_cz(ds, dim):
    def fit_routine(ds_qp):
        try:
            # ds_qp is a Dataset for a single qubit_pair
            initial_cz_len = 500  # ns
            initial_detuning = ds_qp.frequencies_shifted.isel(frequencies=ds_qp.frequencies.argmax() // 2).values

            J, f0, a, offset = fit_rabi_chevron(ds_qp, initial_cz_len * 2, initial_detuning)

            # Check if fitting produced valid results
            if np.isnan(J) or np.isnan(f0):
                # Return default/invalid values
                return xr.DataArray([float("nan"), float("nan"), float("nan"), float("nan")], dims=["fit_vals"])

            flux_time = int(1 / (2 * J) * 1e9)
            lengths = flux_time - flux_time % 4 + 4

            # Return as DataArray for stacking
            return xr.DataArray([J, f0, a, offset], dims=["fit_vals"])
        except Exception as e:
            print(e)
            # Return NaN values if any step fails
            return xr.DataArray([float("nan"), float("nan"), float("nan"), float("nan")], dims=["fit_vals"])

    # Use groupby-apply pattern
    fit_res = ds.groupby("qubit_pair").apply(fit_routine)
    fit_res = fit_res.assign_coords(fit_vals=("fit_vals", ["J", "f0", "a", "offset"]))
    return fit_res


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, qubit_pairs=node.namespace["qubit_pairs"], IQ_list=["I_control", "Q_control"])

    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]

    ds = ds.assign_coords(
        {
            "frequencies_shifted": (
                ("qubit_pair", "frequencies"),
                np.array(
                    [ds.frequencies + node.namespace["central_frequencies"][i] for i, qp in enumerate(qubit_pairs)]
                ),
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
        try:
            J_val = fit.fit.sel(qubit_pair=qp, fit_vals="J").values.item()
            f0_val = fit.fit.sel(qubit_pair=qp, fit_vals="f0").values.item()

            # Check if values are valid
            if np.isnan(J_val) or np.isnan(f0_val) or J_val <= 0:
                success = False
                cz_len_val = 0
                cz_amp_val = float("nan")
            else:
                try:
                    cz_len_val = int(1 / (2 * J_val) * 1e9)
                    cz_amp_val = node.parameters.modulation_amplitude

                    # Determine success based on reasonable parameter ranges
                    is_length_valid = 10 < cz_len_val < 1000
                    is_not_nan = not np.isnan(cz_amp_val)
                    success = bool(is_length_valid and is_not_nan)
                except Exception:
                    # If parameter calculation fails, mark as failed
                    success = False
                    cz_len_val = 0
                    cz_amp_val = float("nan")

            fit_results[qp] = FitParameters(
                success=success,
                J=J_val,
                f0=f0_val,
                cz_len=cz_len_val,
                cz_amp=cz_amp_val,
            )
        except Exception as e:
            # If any step fails, mark as failed
            fit_results[qp] = FitParameters(
                success=False,
                J=float("nan"),
                f0=float("nan"),
                cz_len=0,
                cz_amp=float("nan"),
            )

    fit = fit.assign_coords(
        {
            "cz_len": ("qubit_pair", [fit_results[qp].cz_len for qp in fit.qubit_pair.values]),
            "cz_amp": ("qubit_pair", [fit_results[qp].cz_amp for qp in fit.qubit_pair.values]),
        }
    )
    return fit, fit_results
