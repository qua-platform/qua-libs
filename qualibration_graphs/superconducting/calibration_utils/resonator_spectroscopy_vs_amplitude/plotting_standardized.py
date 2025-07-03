"""
Standardized plotting for resonator spectroscopy vs amplitude (node 02c).

This module provides refactored plotting functions that use the enhanced PlotConfig 
system from qualibration-libs for consistent, maintainable plotting across all nodes.
"""

from typing import List
import xarray as xr
import plotly.graph_objects as go
from matplotlib.figure import Figure
from qualibration_libs.plotting.standard_plotter import (
    create_matplotlib_figure,
    create_specialized_plotly_figure
)
from qualibration_libs.plotting.configs import (
    STANDARD_AMPLITUDE_HEATMAP_CONFIG,
    HeatmapConfig,
    HeatmapTraceConfig,
    LayoutConfig,
    LineOverlayConfig,
    ColorbarConfig
)
from qualibration_libs.plotting.preparators import prepare_amplitude_sweep_data
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit_standardized(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset
) -> Figure:
    """
    Standardized matplotlib plotting for resonator spectroscopy vs amplitude with fits.
    
    This is a drop-in replacement for the original plot_raw_data_with_fit function
    that uses the enhanced PlotConfig system.
    
    Args:
        ds: Raw experimental dataset
        qubits: List of qubits to plot
        fits: Fit results dataset
        
    Returns:
        Matplotlib figure
    """
    # Prepare data using standardized preparator
    ds_prepared, ds_fit_prepared = prepare_amplitude_sweep_data(ds, qubits, fits)
    
    # Use the standard matplotlib plotter with amplitude heatmap config
    config = _create_matplotlib_amplitude_config()
    
    return create_matplotlib_figure(ds_prepared, qubits, [config], ds_fit_prepared)


def plotly_plot_raw_data_with_fit_standardized(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset
) -> go.Figure:
    """
    Standardized Plotly plotting for resonator spectroscopy vs amplitude with fits.
    
    This is a drop-in replacement for the original plotly_plot_raw_data_with_fit 
    function that uses the enhanced PlotConfig system.
    
    Args:
        ds_raw: Raw experimental dataset
        qubits: List of qubits to plot
        fits: Fit results dataset
        
    Returns:
        Plotly figure
    """
    # Prepare data using standardized preparator
    ds_prepared, ds_fit_prepared = prepare_amplitude_sweep_data(ds_raw, qubits, fits)
    
    # Create specialized amplitude heatmap configuration
    config = _create_amplitude_heatmap_config()
    
    return create_specialized_plotly_figure(
        ds_raw, qubits, config, ds_fit_prepared, ds_prepared
    )


def plotly_plot_raw_data_standardized(
    ds: xr.Dataset,
    qubits: List[AnyTransmon]
) -> go.Figure:
    """
    Standardized Plotly plotting for resonator spectroscopy vs amplitude (raw data only).
    
    This is a drop-in replacement for the original plotly_plot_raw_data function
    that uses the enhanced PlotConfig system.
    
    Args:
        ds: Raw experimental dataset
        qubits: List of qubits to plot
        
    Returns:
        Plotly figure
    """
    # Prepare data using standardized preparator
    ds_prepared, _ = prepare_amplitude_sweep_data(ds, qubits, None)
    
    # Create amplitude heatmap configuration without overlays
    config = _create_amplitude_heatmap_config(include_overlays=False)
    
    return create_specialized_plotly_figure(
        ds, qubits, config, None, ds_prepared
    )


def _create_matplotlib_amplitude_config():
    """Create a basic configuration for matplotlib amplitude plots."""
    from qualibration_libs.plotting.configs import PlotConfig, LayoutConfig, TraceConfig
    
    return PlotConfig(
        layout=LayoutConfig(
            title="Resonator spectroscopy vs power",
            x_axis_title="Frequency (GHz)",
            y_axis_title="Power (dBm)"
        ),
        traces=[
            TraceConfig(
                plot_type="heatmap",
                x_source="freq_GHz",
                y_source="power_dbm",
                z_source="IQ_abs_norm",
                name="Raw Data"
            )
        ]
    )


def _create_amplitude_heatmap_config(include_overlays: bool = True) -> HeatmapConfig:
    """
    Create a comprehensive amplitude heatmap configuration.
    
    Args:
        include_overlays: Whether to include fit overlays
        
    Returns:
        HeatmapConfig for amplitude sweep plots
    """
    # Main heatmap trace with custom hover template
    heatmap_trace = HeatmapTraceConfig(
        plot_type="heatmap",
        x_source="freq_GHz",
        y_source="power_dbm",
        z_source="IQ_abs_norm",
        name="Raw Data",
        colorscale="Viridis",
        colorbar=ColorbarConfig(
            title="|IQ|",
            x_offset=0.01,
            width=0.02,
            height_ratio=0.90,
            thickness=14
        ),
        hover_template=(
            "Freq [GHz]: %{x:.3f}<br>"
            "Power [dBm]: %{y:.2f}<br>"
            "Detuning [MHz]: %{customdata:.2f}<br>"
            "|IQ|: %{z:.3f}<extra>%{text}</extra>"
        ),
        custom_data_sources=["detuning_MHz"],
        zmin_percentile=2.0,
        zmax_percentile=98.0
    )
    
    # Fit overlays (if requested)
    overlays = []
    if include_overlays:
        overlays = [
            # Red dashed vertical line at resonator frequency
            LineOverlayConfig(
                orientation="vertical",
                condition_source="outcome",
                condition_value="successful",
                position_source="res_freq_GHz",
                line_style={"color": "#FF0000", "width": 2, "dash": "dash"}
            ),
            # Magenta horizontal line at optimal power
            LineOverlayConfig(
                orientation="horizontal",
                condition_source="outcome",
                condition_value="successful", 
                position_source="optimal_power",
                line_style={"color": "#FF00FF", "width": 2}
            )
        ]
    
    return HeatmapConfig(
        layout=LayoutConfig(
            title="Resonator Spectroscopy: Power vs Frequency" + (" (with fits)" if include_overlays else ""),
            x_axis_title="Frequency (GHz)",
            y_axis_title="Power (dBm)"
        ),
        traces=[heatmap_trace],
        overlays=overlays,
        subplot_spacing={"horizontal": 0.15, "vertical": 0.12}
    )


# Helper functions for migrating existing code
def create_amplitude_plot_config(
    title: str = "Resonator Spectroscopy: Power vs Frequency",
    include_fits: bool = True,
    colorbar_title: str = "|IQ|",
    spacing: dict = None
) -> HeatmapConfig:
    """
    Factory function for creating customized amplitude plot configurations.
    
    This provides a flexible way to create configurations while maintaining
    the standardized framework.
    
    Args:
        title: Plot title
        include_fits: Whether to include fit overlays
        colorbar_title: Title for the colorbar
        spacing: Custom subplot spacing
        
    Returns:
        Customized HeatmapConfig
    """
    if spacing is None:
        spacing = {"horizontal": 0.15, "vertical": 0.12}
    
    config = _create_amplitude_heatmap_config(include_overlays=include_fits)
    config.layout.title = title
    config.traces[0].colorbar.title = colorbar_title
    config.subplot_spacing = spacing
    
    return config


# Backward compatibility aliases - these can be used as drop-in replacements
plot_raw_data_with_fit = plot_raw_data_with_fit_standardized
plotly_plot_raw_data_with_fit = plotly_plot_raw_data_with_fit_standardized
plotly_plot_raw_data = plotly_plot_raw_data_standardized