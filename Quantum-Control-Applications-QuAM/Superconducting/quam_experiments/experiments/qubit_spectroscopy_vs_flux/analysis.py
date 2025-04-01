import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit import fit_oscillation


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    resonator_frequency: float
    frequency_shift: float
    min_offset: float
    idle_offset: float
    dv_phi0: float
    phi0_current: float
    m_pH: float


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    Returns:
    --------
    None

    Example:
    --------
        >>> log_fitted_results(fit_results)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_idle_offset = f"\tidle offset: {fit_results[q]['idle_offset'] * 1e3:.0f} mV | "
        s_min_offset = f"min offset: {fit_results[q]['min_offset'] * 1e3:.0f} mV | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz)\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        logger.info(s_qubit + s_idle_offset + s_min_offset + s_freq + s_shift)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Add the current axis of each qubit to the dataset coordinates for plotting
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

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
    #    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    #     ds = fetch_results_as_xarray(
    #         job.result_handles, qubits, {"flux": dcs, "freq": dfs}
    #     )
    #     # Convert IQ data into volts
    #     ds = convert_IQ_to_V(ds, qubits)
    #     # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
    #     ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    #     # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
    #     ds = ds.assign_coords(
    #         {
    #             "freq_full": (
    #                 ["qubit", "freq"],
    #                 np.array([dfs + q.xy.RF_frequency for q in qubits]),
    #             )
    #         }
    #     )
    #     ds.freq_full.attrs["long_name"] = "Frequency"
    #     ds.freq_full.attrs["units"] = "GHz"
    # # Add the dataset to the node
    # node.results = {"ds": ds}
    #
    # # %% {Data_analysis}
    # # Find the resonance dips for each flux point
    # peaks = peaks_dips(ds.I, dim="freq", prominence_factor=5)
    # # Fit the result with a parabola
    # parabolic_fit_results = peaks.position.polyfit("flux", 2)
    # # Try to fit again with a smaller prominence factor (may need some adjustment)
    # if np.any(
    #     np.isnan(np.concatenate(parabolic_fit_results.polyfit_coefficients.values))
    # ):
    #     # Find the resonance dips for each flux point
    #     peaks = peaks_dips(ds.I, dim="freq", prominence_factor=4)
    #     # Fit the result with a parabola
    #     parabolic_fit_results = peaks.position.polyfit("flux", 2)
    # # Extract relevant fitted parameters
    # coeff = parabolic_fit_results.polyfit_coefficients
    # fitted = (
    #     coeff.sel(degree=2) * ds.flux**2
    #     + coeff.sel(degree=1) * ds.flux
    #     + coeff.sel(degree=0)
    # )
    # flux_shift = -coeff[1] / (2 * coeff[0])
    # freq_shift = (
    #     coeff.sel(degree=2) * flux_shift**2
    #     + coeff.sel(degree=1) * flux_shift
    #     + coeff.sel(degree=0)
    # )
    #
    # # Save fitting results
    # if node.parameters.load_data_id is None:
    #     fit_results = {}
    #     for q in qubits:
    #         fit_results[q.name] = {}
    #         if not np.isnan(flux_shift.sel(qubit=q.name).values):
    #             if q.z.flux_point == "independent":
    #                 offset = q.z.independent_offset
    #             elif q.z.flux_point == "joint":
    #                 offset = q.z.joint_offset
    #             else:
    #                 offset = 0.0
    #             print(
    #                 f"flux offset for qubit {q.name} is {offset*1e3 + flux_shift.sel(qubit = q.name).values*1e3:.0f} mV"
    #             )
    #             print(f"(shift of  {flux_shift.sel(qubit = q.name).values*1e3:.0f} mV)")
    #             print(
    #                 f"Drive frequency for {q.name} is {(freq_shift.sel(qubit = q.name).values + q.xy.RF_frequency)/1e9:.3f} GHz"
    #             )
    #             print(f"(shift of {freq_shift.sel(qubit = q.name).values/1e6:.0f} MHz)")
    #             print(
    #                 f"quad term for qubit {q.name} is {float(coeff.sel(degree = 2, qubit = q.name)/1e9):.3e} GHz/V^2 \n"
    #             )
    #             fit_results[q.name]["flux_shift"] = float(
    #                 flux_shift.sel(qubit=q.name).values
    #             )
    #             fit_results[q.name]["drive_freq"] = float(
    #                 freq_shift.sel(qubit=q.name).values
    #             )
    #             fit_results[q.name]["quad_term"] = float(
    #                 coeff.sel(degree=2, qubit=q.name)
    #             )
    #         else:
    #             print(f"No fit for qubit {q.name}")
    #             fit_results[q.name]["flux_shift"] = np.nan
    #             fit_results[q.name]["drive_freq"] = np.nan
    #             fit_results[q.name]["quad_term"] = np.nan
    #             fit_results[q.name]["success"] = False
    #     node.results["fit_results"] = fit_results
    # node.outcomes = {
    #     qubit_name: ("successful" if fit_result["success"] else "failed")
    #     for qubit_name, fit_result in node.results["fit_results"].items()
    # }
    # Find the minimum of each frequency line to follow the resonance vs flux
    # Extract the relevant fitted parameters
    fit_dataset, fit_results = None, None
    # fit_dataset, fit_results = _extract_relevant_fit_parameters(fit_results_ds, node)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the fit dataset and fit result dictionary."""
    pass

    # return fit, fit_results
