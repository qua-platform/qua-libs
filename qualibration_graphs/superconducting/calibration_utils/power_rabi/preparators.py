"""
Data preparator for power Rabi experiments.

This module transforms raw power Rabi datasets and fit results into the format
expected by the standardized plotting framework.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import xarray as xr
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

# Constants for unit conversion
MV_PER_V = 1e3


def prepare_power_rabi_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None
) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
    """
    Prepare power Rabi data for standardized plotting.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits
        ds_fit: Optional fit results dataset
        
    Returns:
        Tuple of (prepared_raw_data, prepared_fit_data)
    """
    # Create a copy to avoid modifying original
    ds_prepared = ds_raw.copy()
    
    # Add derived coordinates for plotting (exactly like original)
    ds_prepared = ds_prepared.assign_coords(
        amp_mV=ds_raw.full_amp * MV_PER_V,
        amp_prefactor=ds_raw.amp_prefactor
    )
    
    # Determine data type and convert if needed
    if "I" in ds_raw:
        # For 1D power Rabi (single pulse), squeeze out the nb_of_pulses dimension
        if "nb_of_pulses" in ds_raw.dims and ds_raw.sizes["nb_of_pulses"] == 1:
            ds_prepared["I_mV"] = ds_raw.I.squeeze("nb_of_pulses") * MV_PER_V
        else:
            ds_prepared["I_mV"] = ds_raw.I * MV_PER_V
        data_source = "I"
        data_label = "Rotated I quadrature [mV]"
    elif "state" in ds_raw:
        # For 1D power Rabi (single pulse), squeeze out the nb_of_pulses dimension
        if "nb_of_pulses" in ds_raw.dims and ds_raw.sizes["nb_of_pulses"] == 1:
            ds_prepared["state"] = ds_raw.state.squeeze("nb_of_pulses")
        else:
            ds_prepared["state"] = ds_raw.state
        data_source = "state"
        data_label = "Qubit state"
    else:
        raise RuntimeError("Dataset must contain either 'I' or 'state' for power Rabi plotting")
    
    # Store metadata for plotting
    ds_prepared.attrs.update({
        "data_source": data_source,
        "data_label": data_label
    })
    
    # Process fit data if available
    ds_fit_prepared = None
    if ds_fit is not None:
        ds_fit_prepared = ds_fit.copy()
        
        # Add derived coordinates for fit data
        if "amp_mV" not in ds_fit_prepared.coords and "full_amp" in ds_fit_prepared:
            ds_fit_prepared = ds_fit_prepared.assign_coords(
                amp_mV=ds_fit_prepared["full_amp"] * MV_PER_V
            )
        
        # Add fitted curves for 1D power Rabi (exactly like original)
        try:
            from qualibration_libs.analysis import oscillation
            
            # Initialize fitted data arrays
            n_qubits = len(ds_fit_prepared.qubit)
            n_amp = len(ds_fit_prepared.amp_prefactor)
            
            fitted_data_array = np.zeros((n_qubits, n_amp))
            
            for i, qubit in enumerate(qubits):
                qubit_name = qubit.name
                if qubit_name in ds_fit_prepared.qubit.values:
                    fit_qubit = ds_fit_prepared.sel(qubit=qubit_name)
                    if hasattr(fit_qubit, "outcome") and fit_qubit.outcome.values == "successful":
                        try:
                            # Compute fitted curve exactly like original
                            fitted_data = oscillation(
                                fit_qubit.amp_prefactor.values,
                                fit_qubit.fit.sel(fit_vals="a").values,
                                fit_qubit.fit.sel(fit_vals="f").values,
                                fit_qubit.fit.sel(fit_vals="phi").values,
                                fit_qubit.fit.sel(fit_vals="offset").values,
                            )
                            
                            # Find qubit index in dataset
                            qubit_idx = list(ds_fit_prepared.qubit.values).index(qubit_name)
                            fitted_data_array[qubit_idx] = fitted_data
                            
                        except Exception as e:
                            print(f"Warning: Could not compute fitted curve for {qubit_name}: {e}")
            
            # Add fitted data to dataset based on measurement type
            if data_source == "I":
                ds_fit_prepared["fitted_data_mV"] = (
                    ("qubit", "amp_prefactor"), 
                    fitted_data_array * MV_PER_V
                )
            else:
                ds_fit_prepared["fitted_state"] = (
                    ("qubit", "amp_prefactor"), 
                    fitted_data_array
                )
                
        except ImportError:
            print("Warning: Could not import oscillation function for fits")
        
        # Add optimal amplitude for overlays (for 2D power rabi)
        if "opt_amp_prefactor" in ds_fit_prepared:
            # Convert optimal amplitude prefactor to mV (exactly like original)
            opt_amp_mV_values = []
            for qubit in qubits:
                qubit_name = qubit.name
                if qubit_name in ds_fit_prepared.qubit.values:
                    fit_qubit = ds_fit_prepared.sel(qubit=qubit_name)
                    raw_qubit = ds_prepared.sel(qubit=qubit_name)
                    
                    if hasattr(fit_qubit, "outcome") and fit_qubit.outcome.values == "successful":
                        try:
                            # Find mV value corresponding to opt_amp_prefactor (exact copy of original logic)
                            opt_amp_mV = float(
                                raw_qubit["full_amp"].sel(
                                    amp_prefactor=fit_qubit.opt_amp_prefactor, 
                                    method="nearest"
                                ).values
                            ) * MV_PER_V
                            opt_amp_mV_values.append(opt_amp_mV)
                        except Exception as e:
                            # Fallback to numpy method (exact copy of original)
                            amp_mV = raw_qubit['amp_mV'].values if 'amp_mV' in raw_qubit else raw_qubit['full_amp'].values * MV_PER_V
                            amp_prefactor = raw_qubit['amp_prefactor'].values
                            opt_amp_mV = float(amp_mV[np.argmin(np.abs(amp_prefactor - fit_qubit.opt_amp_prefactor.values))])
                            opt_amp_mV_values.append(opt_amp_mV)
                    else:
                        opt_amp_mV_values.append(0.0)  # Default for failed fits
                else:
                    opt_amp_mV_values.append(0.0)
            
            # Add to fit dataset
            ds_fit_prepared["opt_amp_mV"] = (("qubit",), np.array(opt_amp_mV_values))
    
    return ds_prepared, ds_fit_prepared


# Remove legacy fallback functions - use unified architecture instead