"""
Standardized plotting for power Rabi experiments (node 04b).

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
    STANDARD_POWER_RABI_CONFIG,
    ChevronConfig,
    HeatmapTraceConfig,
    TraceConfig,
    LayoutConfig,
    LineOverlayConfig,
    DualAxisConfig,
    ColorbarConfig
)
from qualibration_libs.plotting.preparators import prepare_power_rabi_data
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit_standardized(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset
) -> Figure:
    """
    Standardized matplotlib plotting for power Rabi with fits.
    
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
    ds_prepared, ds_fit_prepared = prepare_power_rabi_data(ds, qubits, fits)
    
    # Determine plot type (1D vs 2D)
    is_2d = "nb_of_pulses" in ds.dims and ds.sizes.get("nb_of_pulses", 1) > 1
    
    if is_2d:
        config = _create_matplotlib_power_rabi_2d_config(ds_prepared)
    else:
        config = _create_matplotlib_power_rabi_1d_config(ds_prepared)
    
    return create_matplotlib_figure(ds_prepared, qubits, [config], ds_fit_prepared)


def plotly_plot_raw_data_with_fit_standardized(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset
) -> go.Figure:
    """
    Standardized Plotly plotting for power Rabi with fits.
    
    This is a drop-in replacement for the original plotly_plot_raw_data_with_fit 
    function that uses the enhanced PlotConfig system.
    
    Args:
        ds: Raw experimental dataset
        qubits: List of qubits to plot
        fits: Fit results dataset
        
    Returns:
        Plotly figure
    """
    # Prepare data using standardized preparator
    ds_prepared, ds_fit_prepared = prepare_power_rabi_data(ds, qubits, fits)
    
    # Determine plot type and create appropriate configuration
    is_2d = "nb_of_pulses" in ds.dims and ds.sizes.get("nb_of_pulses", 1) > 1
    
    if is_2d:
        config = _create_power_rabi_2d_config(ds_prepared)
    else:
        config = _create_power_rabi_1d_config(ds_prepared)
    
    return create_specialized_plotly_figure(
        ds, qubits, config, ds_fit_prepared, ds_prepared
    )


def _create_matplotlib_power_rabi_1d_config(ds: xr.Dataset):
    """Create matplotlib configuration for 1D power Rabi plots."""
    from qualibration_libs.plotting.configs import PlotConfig, LayoutConfig, TraceConfig
    
    # Determine data source from dataset attributes
    data_source = ds.attrs.get("data_source", "I")
    data_label = ds.attrs.get("data_label", "Rotated I quadrature [mV]")
    
    y_source = f"{data_source}_mV" if data_source == "I" else data_source
    
    return PlotConfig(
        layout=LayoutConfig(
            title="Power Rabi",
            x_axis_title="Pulse amplitude [mV]",
            y_axis_title=data_label
        ),
        traces=[
            TraceConfig(
                plot_type="scatter",
                x_source="amp_mV",
                y_source=y_source,
                name="Raw Data"
            )
        ],
        fit_traces=[
            TraceConfig(
                plot_type="scatter",
                x_source="amp_mV",
                y_source="fitted_data_mV",
                name="Fit",
                mode="lines",
                style={"color": "#FF0000"}
            )
        ]
    )


def _create_matplotlib_power_rabi_2d_config(ds: xr.Dataset):
    """Create matplotlib configuration for 2D power Rabi plots.""" 
    from qualibration_libs.plotting.configs import PlotConfig, LayoutConfig, TraceConfig
    
    # Determine data source from dataset attributes
    data_source = ds.attrs.get("data_source", "I")
    
    return PlotConfig(
        layout=LayoutConfig(
            title="Power Rabi",
            x_axis_title="Pulse amplitude [mV]",
            y_axis_title="Number of pulses"
        ),
        traces=[
            TraceConfig(
                plot_type="heatmap",
                x_source="amp_mV",
                y_source="nb_of_pulses",
                z_source=data_source,
                name="Raw Data"
            )
        ]
    )


def _create_power_rabi_1d_config(ds: xr.Dataset) -> ChevronConfig:
    """
    Create a comprehensive 1D power Rabi configuration.
    
    Args:
        ds: Prepared dataset
        
    Returns:
        ChevronConfig for 1D power Rabi plots
    """
    # Determine data source from dataset attributes
    data_source = ds.attrs.get("data_source", "I")
    data_label = ds.attrs.get("data_label", "Rotated I quadrature [mV]")
    
    y_source = f"{data_source}_mV" if data_source == "I" else data_source
    
    # Raw data trace with custom hover
    raw_trace = TraceConfig(
        plot_type="scatter",
        x_source="amp_mV",
        y_source=y_source,
        name="Raw Data",
        mode="lines+markers",
        style={"color": "#1f77b4"},
        hover_template="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata[0]:.3f}<br>%{y:.3f} mV<extra></extra>",
        custom_data_sources=["amp_prefactor"]
    )
    
    # Fit trace
    fit_trace = TraceConfig(
        plot_type="scatter",
        x_source="amp_mV",
        y_source="fitted_data_mV",
        name="Fit",
        mode="lines",
        style={"color": "#FF0000", "width": 2}
    )
    
    return ChevronConfig(
        layout=LayoutConfig(
            title="Power Rabi",
            x_axis_title="Pulse amplitude [mV]",
            y_axis_title=data_label
        ),
        traces=[raw_trace],
        fit_traces=[fit_trace],
        subplot_spacing={"horizontal": 0.1, "vertical": 0.2},
        dual_axis=DualAxisConfig(
            enabled=True,
            top_axis_title="Amplitude prefactor",
            top_axis_source="amp_prefactor",
            top_axis_format="{:.2f}",
            overlay_offset=100
        )
    )


def _create_power_rabi_2d_config(ds: xr.Dataset) -> ChevronConfig:
    """
    Create a comprehensive 2D power Rabi (chevron) configuration.
    
    Args:
        ds: Prepared dataset
        
    Returns:
        ChevronConfig for 2D power Rabi plots
    """
    # Determine data source from dataset attributes
    data_source = ds.attrs.get("data_source", "I")
    
    # Heatmap trace for chevron pattern
    heatmap_trace = HeatmapTraceConfig(
        plot_type="heatmap",
        x_source="amp_mV",
        y_source="nb_of_pulses",
        z_source=data_source,
        name="Raw Data",
        colorscale="Viridis",
        colorbar=ColorbarConfig(
            title="|IQ|" if data_source == "I" else "State",
            thickness=14
        ),
        hover_template=(
            "Amplitude: %{x:.3f} mV<br>"
            "Prefactor: %{customdata:.3f}<br>"
            "Pulses: %{y}<br>"
            "Value: %{z:.3f}<extra>%{text}</extra>"
        ),
        custom_data_sources=["amp_prefactor"],
        zmin_percentile=2.0,
        zmax_percentile=98.0
    )
    
    # Fit overlay (vertical line at optimal amplitude)
    overlays = [
        LineOverlayConfig(
            orientation="vertical",
            condition_source="outcome",
            condition_value="successful",
            position_source="opt_amp_prefactor",  # This will be converted to mV in the plotter
            line_style={"color": "#FF0000", "width": 2, "dash": "solid"}
        )
    ]
    
    return ChevronConfig(
        layout=LayoutConfig(
            title="Power Rabi",
            x_axis_title="Pulse amplitude [mV]",
            y_axis_title="Number of pulses"
        ),
        traces=[heatmap_trace],
        overlays=overlays,
        subplot_spacing={"horizontal": 0.1, "vertical": 0.2},
        dual_axis=DualAxisConfig(
            enabled=True,
            top_axis_title="Amplitude prefactor",
            top_axis_source="amp_prefactor",
            top_axis_format="{:.2f}",
            overlay_offset=100
        )
    )


# Helper functions for migrating existing code
def create_power_rabi_plot_config(
    ds: xr.Dataset,
    title: str = "Power Rabi",
    include_fits: bool = True,
    colorbar_title: str = None
) -> ChevronConfig:
    """
    Factory function for creating customized power Rabi configurations.
    
    This provides a flexible way to create configurations while maintaining
    the standardized framework.
    
    Args:
        ds: Dataset to analyze for plot type
        title: Plot title
        include_fits: Whether to include fit overlays
        colorbar_title: Custom colorbar title
        
    Returns:
        Customized ChevronConfig
    """
    is_2d = "nb_of_pulses" in ds.dims and ds.sizes.get("nb_of_pulses", 1) > 1
    
    if is_2d:
        config = _create_power_rabi_2d_config(ds)
        if not include_fits:
            config.overlays = []
    else:
        config = _create_power_rabi_1d_config(ds)
        if not include_fits:
            config.fit_traces = []
    
    config.layout.title = title
    
    if colorbar_title and is_2d:
        config.traces[0].colorbar.title = colorbar_title
    
    return config


def determine_data_source_and_label(ds: xr.Dataset) -> tuple[str, str]:
    """
    Determine the data source and label for power Rabi plots.
    
    Args:
        ds: Dataset to analyze
        
    Returns:
        Tuple of (data_source, data_label)
    """
    if "I" in ds.data_vars:
        return "I", "Rotated I quadrature [mV]"
    elif "state" in ds.data_vars:
        return "state", "Qubit state"
    else:
        raise RuntimeError("Dataset must contain either 'I' or 'state' for power Rabi plotting")


# Backward compatibility aliases - these can be used as drop-in replacements
plot_raw_data_with_fit = plot_raw_data_with_fit_standardized
plotly_plot_raw_data_with_fit = plotly_plot_raw_data_with_fit_standardized