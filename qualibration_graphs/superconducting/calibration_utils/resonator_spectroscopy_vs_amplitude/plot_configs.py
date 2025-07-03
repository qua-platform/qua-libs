"""
Plot configurations for resonator spectroscopy vs amplitude (power) experiments.

This module defines PlotConfig objects that specify how to visualize
amplitude sweep data using the standardized plotting framework.
"""

from qualibration_libs.plotting.standard_plotter import PlotConfig, TraceConfig, LayoutConfig


# Configuration for amplitude vs power heatmap plot
amplitude_vs_power_config = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy vs Power",
        x_axis_title="Frequency (GHz)", 
        y_axis_title="Power (dBm)"
    ),
    traces=[
        TraceConfig(
            plot_type="heatmap",
            x_source="freq_GHz",
            y_source="power_dbm",
            z_source="IQ_abs_norm",
            name="Raw Data",
            style={"colorscale": "Viridis"}
        )
    ],
    fit_traces=[
        # Fit overlays can be added here if needed
        # TraceConfig for vertical line at resonance frequency
        # TraceConfig for horizontal line at optimal power
    ]
)