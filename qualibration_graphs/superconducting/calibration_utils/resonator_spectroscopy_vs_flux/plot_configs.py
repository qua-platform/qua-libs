"""
Plot configurations for resonator spectroscopy vs flux experiments.

This module defines PlotConfig objects that specify how to visualize
flux sweep data using the standardized plotting framework.
"""

from qualibration_libs.plotting.standard_plotter import PlotConfig, TraceConfig, LayoutConfig


# Configuration for flux vs frequency heatmap plot
flux_vs_frequency_config = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy vs Flux",
        x_axis_title="Flux bias [V]",
        y_axis_title="RF frequency [GHz]"
    ),
    traces=[
        TraceConfig(
            plot_type="heatmap",
            x_source="flux_bias",
            y_source="freq_GHz", 
            z_source="IQ_abs",
            name="Raw Data",
            style={"colorscale": "Viridis"}
        )
    ],
    fit_traces=[
        # Fit overlays can be added here if needed
        # TraceConfig for vertical lines at idle_offset, flux_min
        # TraceConfig for marker at sweet spot
    ]
)