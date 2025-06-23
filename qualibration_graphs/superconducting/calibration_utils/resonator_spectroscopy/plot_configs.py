from qualibration_libs.plotting.standard_plotter import (LayoutConfig,
                                                         PlotConfig,
                                                         TraceConfig)

# Configuration for the phase vs. frequency plot
phase_vs_freq_config = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy (Phase)",
        x_axis_title="RF Frequency [GHz]",
        y_axis_title="Phase [rad]",
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            name="Raw Phase",
            x_source="full_freq_ghz",
            y_source="phase",
            custom_data_sources=["detuning_mhz"],
            hover_template="<b>Freq</b>: %{x:.4f} GHz<br>" +
                           "<b>Detuning</b>: %{customdata[0]:.2f} MHz<br>" +
                           "<b>Phase</b>: %{y:.3f} rad<extra></extra>",
            style={"color": "#1f77b4"},
        )
    ],
)

# Configuration for the amplitude vs. frequency plot, including the fit
amplitude_vs_freq_config = PlotConfig(
    layout=LayoutConfig(
        title="Resonator Spectroscopy (Amplitude + Fit)",
        x_axis_title="RF Frequency [GHz]",
        y_axis_title="R = sqrt(I² + Q²) [mV]",
    ),
    traces=[
        TraceConfig(
            plot_type="scatter",
            name="Raw Amplitude",
            x_source="full_freq_ghz",
            y_source="iq_abs_mv",
            custom_data_sources=["detuning_mhz"],
            hover_template="<b>Freq</b>: %{x:.4f} GHz<br>" +
                           "<b>Detuning</b>: %{customdata[0]:.2f} MHz<br>" +
                           "<b>Amplitude</b>: %{y:.3f} mV<extra></extra>",
            style={"color": "#1f77b4"},
        )
    ],
    fit_traces=[
        TraceConfig(
            plot_type="line",
            name="Fit",
            x_source="full_freq_ghz",
            y_source="fitted_curve_mv",
            mode="lines",
            style={"color": "#FF0000", "dash": "dash"},
            hover_template="<b>Fit</b><extra></extra>"
        )
    ]
) 