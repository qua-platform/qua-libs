"""
Standardized plotting for resonator spectroscopy vs flux (node 02b).

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
    STANDARD_FLUX_HEATMAP_CONFIG,
    HeatmapConfig,
    HeatmapTraceConfig,
    LayoutConfig,
    LineOverlayConfig,
    MarkerOverlayConfig,
    DualAxisConfig,
    ColorbarConfig
)
from qualibration_libs.plotting.preparators import prepare_flux_sweep_data
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit_standardized(
    ds: xr.Dataset, 
    qubits: List[AnyTransmon], 
    fits: xr.Dataset
) -> Figure:
    """
    Standardized matplotlib plotting for resonator spectroscopy vs flux with fits.
    
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
    ds_prepared, ds_fit_prepared = prepare_flux_sweep_data(ds, qubits, fits)
    
    # Use the standard matplotlib plotter with flux heatmap config
    config = _create_matplotlib_flux_config()
    
    return create_matplotlib_figure(ds_prepared, qubits, [config], ds_fit_prepared)


def plotly_plot_raw_data_with_fit_standardized(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon], 
    fits: xr.Dataset
) -> go.Figure:
    """
    Standardized Plotly plotting for resonator spectroscopy vs flux with fits.
    
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
    ds_prepared, ds_fit_prepared = prepare_flux_sweep_data(ds_raw, qubits, fits)
    
    # Create specialized flux heatmap configuration
    config = _create_flux_heatmap_config()
    
    return create_specialized_plotly_figure(
        ds_raw, qubits, config, ds_fit_prepared, ds_prepared
    )


def plotly_plot_raw_data_standardized(
    ds: xr.Dataset,
    qubits: List[AnyTransmon]
) -> go.Figure:
    """
    Standardized Plotly plotting for resonator spectroscopy vs flux (raw data only).
    
    This is a drop-in replacement for the original plotly_plot_raw_data function
    that uses the enhanced PlotConfig system.
    
    Args:
        ds: Raw experimental dataset
        qubits: List of qubits to plot
        
    Returns:
        Plotly figure
    """
    # Prepare data using standardized preparator
    ds_prepared, _ = prepare_flux_sweep_data(ds, qubits, None)
    
    # Create flux heatmap configuration without overlays
    config = _create_flux_heatmap_config(include_overlays=False)
    
    return create_specialized_plotly_figure(
        ds, qubits, config, None, ds_prepared
    )


def _create_matplotlib_flux_config():
    """Create a basic configuration for matplotlib flux plots."""
    from qualibration_libs.plotting.configs import PlotConfig, LayoutConfig, TraceConfig
    
    return PlotConfig(
        layout=LayoutConfig(
            title="Resonator spectroscopy vs flux",
            x_axis_title="Flux (V)",
            y_axis_title="Freq (GHz)"
        ),
        traces=[
            TraceConfig(
                plot_type="heatmap",
                x_source="flux_bias",
                y_source="freq_GHz",
                z_source="IQ_abs",
                name="Raw Data"
            )
        ]
    )


def _create_flux_heatmap_config(include_overlays: bool = True) -> HeatmapConfig:
    """
    Create a comprehensive flux heatmap configuration.
    
    Args:
        include_overlays: Whether to include fit overlays
        
    Returns:
        HeatmapConfig for flux sweep plots
    """
    # Main heatmap trace with custom hover template
    heatmap_trace = HeatmapTraceConfig(
        plot_type="heatmap",
        x_source="flux_bias",
        y_source="freq_GHz", 
        z_source="IQ_abs",
        name="Raw Data",
        colorscale="Viridis",
        colorbar=ColorbarConfig(
            title="|IQ|",
            x_offset=0.03,
            width=0.02,
            height_ratio=0.90,
            thickness=14
        ),
        hover_template=(
            "Flux [V]: %{x:.3f}<br>"
            "Current [A]: %{customdata[1]:.6f}<br>"
            "Freq [GHz]: %{y:.3f}<br>"
            "Detuning [MHz]: %{customdata[0]:.2f}<br>"
            "|IQ|: %{z:.3f}<extra>%{text}</extra>"
        ),
        custom_data_sources=["detuning_MHz", "attenuated_current"]
    )
    
    # Fit overlays (if requested)
    overlays = []
    if include_overlays:
        overlays = [
            # Red dashed line at idle offset
            LineOverlayConfig(
                orientation="vertical",
                condition_source="outcome",
                condition_value="successful",
                position_source="idle_offset",
                line_style={"color": "#FF0000", "width": 2.5, "dash": "dash"}
            ),
            # Purple dashed line at flux minimum
            LineOverlayConfig(
                orientation="vertical",
                condition_source="outcome", 
                condition_value="successful",
                position_source="flux_min",
                line_style={"color": "#800080", "width": 2.5, "dash": "dash"}
            ),
            # Magenta X marker at sweet spot
            MarkerOverlayConfig(
                condition_source="outcome",
                condition_value="successful",
                x_source="idle_offset",
                y_source="sweet_spot_frequency_GHz",
                marker_style={"symbol": "x", "color": "#FF00FF", "size": 15}
            )
        ]
    
    return HeatmapConfig(
        layout=LayoutConfig(
            title="Resonator Spectroscopy: Flux vs Frequency" + (" (with fits)" if include_overlays else ""),
            x_axis_title="Flux bias [V]",
            y_axis_title="RF frequency [GHz]"
        ),
        traces=[heatmap_trace],
        overlays=overlays,
        subplot_spacing={"horizontal": 0.25, "vertical": 0.12},
        dual_axis=DualAxisConfig(
            enabled=True,
            top_axis_title="Current [A]",
            top_axis_source="attenuated_current",
            top_axis_format="{:.6f}",
            overlay_offset=100
        )
    )


# Backward compatibility aliases - these can be used as drop-in replacements
plot_raw_data_with_fit = plot_raw_data_with_fit_standardized
plotly_plot_raw_data_with_fit = plotly_plot_raw_data_with_fit_standardized  
plotly_plot_raw_data = plotly_plot_raw_data_standardized