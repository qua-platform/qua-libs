"""
Data preparator for power Rabi experiments.

This module transforms raw power Rabi datasets and fit results into the format
expected by the standardized plotting framework.
"""

from typing import List, Optional, Tuple, Any
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
    
    # Add derived coordinates for plotting
    ds_prepared = ds_prepared.assign_coords(
        amp_mV=ds_raw.full_amp * MV_PER_V,
        amp_prefactor=ds_raw.amp_prefactor
    )
    
    # Determine data type and convert if needed
    if "I" in ds_raw:
        ds_prepared["I_mV"] = ds_raw.I * MV_PER_V
        data_source = "I"
        data_label = "Rotated I quadrature [mV]"
    elif "state" in ds_raw:
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
        
        # Add any necessary derived coordinates for fit data
        if "amp_mV" not in ds_fit_prepared.coords and "full_amp" in ds_fit_prepared:
            ds_fit_prepared = ds_fit_prepared.assign_coords(
                amp_mV=ds_fit_prepared["full_amp"] * MV_PER_V
            )
    
    return ds_prepared, ds_fit_prepared


def create_plotly_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: List[Any],
    ds_fit: Optional[xr.Dataset] = None,
) -> Any:
    """
    Create plotly figure using the original plotting logic to maintain exact visual output.
    This function replicates the behavior of plotly_plot_raw_data_with_fit.
    """
    # Import here to avoid circular import
    from calibration_utils.power_rabi.plotting import plotly_plot_raw_data_with_fit
    
    # Use the original plotting function to maintain exact visual output
    return plotly_plot_raw_data_with_fit(ds_raw, qubits, ds_fit)


def create_matplotlib_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon], 
    plot_configs: List[Any],
    ds_fit: Optional[xr.Dataset] = None,
) -> Any:
    """
    Create matplotlib figure using the original plotting logic to maintain exact visual output.
    This function replicates the behavior of plot_raw_data_with_fit.
    """
    # Import here to avoid circular import
    from calibration_utils.power_rabi.plotting import plot_raw_data_with_fit
    
    # Use the original plotting function to maintain exact visual output
    return plot_raw_data_with_fit(ds_raw, qubits, ds_fit)