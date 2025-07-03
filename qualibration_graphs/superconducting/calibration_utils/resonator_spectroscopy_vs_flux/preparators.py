"""
Data preparator for resonator spectroscopy vs flux experiments.

This module transforms raw flux sweep datasets and fit results into the format
expected by the standardized plotting framework.
"""

from typing import List, Optional, Tuple, Any
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# Constants for unit conversion  
GHZ_PER_HZ = 1e-9
MHZ_PER_HZ = 1e-6


def prepare_flux_sweep_data(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None
) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
    """
    Prepare resonator spectroscopy vs flux data for standardized plotting.
    
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
    freq_coord_name = "full_freq" if "full_freq" in ds_prepared else "freq_full"
    
    ds_prepared = ds_prepared.assign_coords(
        freq_GHz=ds_prepared[freq_coord_name] * GHZ_PER_HZ,
        detuning_MHz=ds_prepared.detuning * MHZ_PER_HZ
    )
    
    # Process fit data if available
    ds_fit_prepared = None
    if ds_fit is not None:
        ds_fit_prepared = ds_fit.copy()
        
        # Add any necessary derived coordinates for fit data
        if "freq_GHz" not in ds_fit_prepared.coords and freq_coord_name in ds_fit_prepared:
            ds_fit_prepared = ds_fit_prepared.assign_coords(
                freq_GHz=ds_fit_prepared[freq_coord_name] * GHZ_PER_HZ
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
    from calibration_utils.resonator_spectroscopy_vs_flux.plotting import plotly_plot_raw_data_with_fit
    
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
    from calibration_utils.resonator_spectroscopy_vs_flux.plotting import plot_raw_data_with_fit
    
    # Use the original plotting function to maintain exact visual output
    return plot_raw_data_with_fit(ds_raw, qubits, ds_fit)


def create_plotly_raw_figure(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    plot_configs: List[Any] = None,
) -> Any:
    """
    Create plotly figure for raw data only using the original plotting logic.
    This function replicates the behavior of plotly_plot_raw_data.
    """
    # Import here to avoid circular import
    from calibration_utils.resonator_spectroscopy_vs_flux.plotting import plotly_plot_raw_data
    
    # Use the original plotting function to maintain exact visual output
    return plotly_plot_raw_data(ds_raw, qubits)