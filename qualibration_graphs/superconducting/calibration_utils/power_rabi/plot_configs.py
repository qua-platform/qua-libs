"""
Plot configurations for power Rabi experiments.

This module defines PlotConfig objects that specify how to visualize
power Rabi data using the standardized plotting framework.
"""

from qualibration_libs.plotting.standard_plotter import PlotConfig, TraceConfig, LayoutConfig


# Configuration for power Rabi plot (handles both 1D and 2D cases)
power_rabi_config = PlotConfig(
    layout=LayoutConfig(
        title="Power Rabi",
        x_axis_title="Pulse amplitude [mV]",
        y_axis_title="Signal [mV]"  # Will be updated based on data type
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",  # Will be changed to heatmap for 2D data
            x_source="amp_mV",
            y_source="I_mV",  # Will be determined dynamically
            name="Raw Data",
            mode="lines+markers",
            style={"color": "#1f77b4"}
        )
    ],
    fit_traces=[
        TraceConfig(
            plot_type="scatter",
            x_source="amp_mV",
            y_source="fitted_data_mV", 
            name="Fit",
            mode="lines",
            style={"color": "#FF0000", "width": 2}
        )
    ]
)