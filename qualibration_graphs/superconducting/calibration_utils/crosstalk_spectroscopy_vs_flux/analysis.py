import logging
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon

from .fitting.fit_linear import fit_linear, calculate_crosstalk_coefficient
from .program import get_flux_detuning_in_v


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubit pairs from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubit pairs.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    
    for target_qubit_name, target_qubit_results in fit_results.items():
        for aggressor_qubit_name, results in target_qubit_results.items():
            s_pair = f"Results for {aggressor_qubit_name} freq. vs. {aggressor_qubit_name} flux: "
            s_crosstalk = f"\tCrosstalk coefficient: {100*results['crosstalk_coefficient']:.2f}% | "
            s_slope = f"Slope: {results['linear_fit_slope']/1e6:.3f} MHz/V\n"

            if results["success"]:
                s_pair += " SUCCESS!\n"
            else:
                s_pair += " FAIL!\n"
            log_callable(s_pair + s_crosstalk + s_slope)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Process the raw dataset by converting IQ quadratures and adding relevant coordinates.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset containing I and Q quadratures
    node : QualibrationNode
        Node containing parameters and qubit information
        
    Returns:
    --------
    xr.Dataset
        Processed dataset with amplitude, phase, and additional coordinates
    """
    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
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


def fit_lorentzian_peaks(ds: xr.Dataset, node) -> xr.Dataset:
    """
    Fit Lorentzian peaks for each flux bias point to extract peak frequencies.

    Returns:
    --------
    New dataset with peak frequency data
    """
    from .fitting.fit_lorentzian import fit_lorentzian_for_each_detuning_fixed

    peak_data = []

    for target_qubit in node.namespace["target_qubits"]:
        for aggressor_qubit in node.namespace["aggressor_qubits"]:
            if target_qubit.name == aggressor_qubit.name:
                continue

            da = ds.sel(qubit=target_qubit.name).sel(aggressor=aggressor_qubit.name).IQ_abs

            # Fit Lorentzian
            peak_freq, peak_freq_err, flux_bias = fit_lorentzian_for_each_detuning_fixed(da)

            # Extract .data if they are DataArrays
            if isinstance(peak_freq, xr.DataArray):
                peak_freq = peak_freq.data
            if isinstance(peak_freq_err, xr.DataArray):
                peak_freq_err = peak_freq_err.data
            if isinstance(flux_bias, xr.DataArray):
                flux_bias = flux_bias.data

            # Small dataset for this pair
            pair_ds = xr.Dataset(
                {
                    "peak_frequencies": (("flux_bias",), peak_freq),
                    "peak_frequency_errors": (("flux_bias",), peak_freq_err)
                },
                coords={
                    "flux_bias": flux_bias,
                    "qubit": target_qubit.name,
                    "aggressor": aggressor_qubit.name
                }
            )

            peak_data.append(pair_ds)

    combined_ds = xr.concat(peak_data, dim="pair", join="outer").set_index(pair=["qubit", "aggressor"])

    return combined_ds


def fit_linear_crosstalk(peak_results: xr.Dataset, node: QualibrationNode) -> Dict:
    """
    Fit linear relationship between peak frequency and flux bias to extract crosstalk.
    
    Parameters:
    -----------
    peak_results : Dict
        Dictionary containing peak frequency results for each qubit pair
    node : QualibrationNode
        Node containing parameters including expected crosstalk
        
    Returns:
    --------
    Dict containing the relevant crosstalk spectroscopy experiment fit parameters for a single qubit pair
        crosstalk_coefficient: float
            Crosstalk coefficient (dimensionless)
        crosstalk_error: float
            Error in crosstalk coefficient
        linear_fit_slope: float
            Slope of linear fit (Hz/V)
        linear_fit_intercept: float
            Intercept of linear fit (Hz)
        success: bool
            Whether the fit was successful
    """

    fit_results = {str(target_qubit_name.data): {} for target_qubit_name in peak_results.qubit}

    for pair in peak_results.pair:
        peak_freq = peak_results.sel(pair=pair).peak_frequencies
        flux_bias = peak_freq.flux_bias

        target_qubit_name = str(pair.qubit.data)
        aggressor_qubit_name = str(pair.aggressor.data)

        try:
            # Fit linear relationship using the new fitting function
            slope, intercept, inlier_mask = fit_linear(flux_bias, peak_freq)

            # Calculate crosstalk coefficient using the new function
            target_qubit = node.machine.qubits[target_qubit_name]
            flux_detuning = get_flux_detuning_in_v(node.parameters, target_qubit)
            target_qubit_slope = get_target_qubit_slope_at_flux_detuning(target_qubit, flux_detuning)
            crosstalk_coefficient = calculate_crosstalk_coefficient(slope, target_qubit_slope)

            fit_results[target_qubit_name][aggressor_qubit_name] = dict(
                crosstalk_coefficient=crosstalk_coefficient,
                linear_fit_slope=slope,
                linear_fit_intercept=intercept,
                linear_fit_inlier_mask=inlier_mask,
                success=True
            )

        except Exception as e:
            logging.warning(f"Linear fit failed for {target_qubit_name} vs. {aggressor_qubit_name}: {e}")
            fit_results[target_qubit_name][aggressor_qubit_name] = dict(
                crosstalk_coefficient=np.nan,
                crosstalk_error=np.nan,
                linear_fit_slope=np.nan,
                linear_fit_intercept=np.nan,
                success=False
            )
    
    return fit_results


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict]:
    """
    Complete analysis pipeline: fit Lorentzian peaks and linear crosstalk relationship.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Processed dataset containing IQ_abs data
    node : QualibrationNode
        Node containing parameters
        
    Returns:
    --------
    Tuple[xr.Dataset, Dict]
        Dataset with analysis results and fit results dictionary
    """
    node.results["peak_results"] = peak_results = fit_lorentzian_peaks(ds, node)

    fit_results = fit_linear_crosstalk(peak_results, node)
    
    return ds, fit_results


def get_target_qubit_slope_at_flux_detuning(target_qubit: FluxTunableTransmon, flux_detuning_in_v: float):
    """
    Calculate the target qubit slope at some flux detuning

    Returns:
    --------
    float
        slope in Hz/V
    """
    return -2 * target_qubit.freq_vs_flux_01_quad_term * flux_detuning_in_v


def get_target_slope_from_parameter_ranges(parameters):
    """
    Calculate the target slope from parameter ranges.
    
    Parameters:
    -----------
    parameters : Parameters
        Node parameters containing flux and frequency ranges
        
    Returns:
    --------
    float
        Target slope in Hz/V
    """
    flux_span = parameters.flux_span_in_v
    freq_span = parameters.frequency_span_in_mhz * 1e6  # Convert to Hz
    return freq_span / flux_span
