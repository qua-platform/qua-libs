from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from calibration_utils.cz_conditional_phase.analysis import FitResults
from qualibration_libs.core import BatchableList
import xarray as xr


def plot_raw_data_with_fit(
    fit_results: xr.Dataset,
    qubit_pairs: BatchableList,
) -> plt.Figure:
    """
    Plot the CZ phase calibration data showing phase difference vs amplitude with fit.

    Parameters:
    -----------
    fit_results : xr.Dataset
        Fit results for each qubit pair
        Optimal amplitudes for each qubit pair
    qubit_pairs : BatchableList
        List of qubit pairs

    Returns:
    --------
    plt.Figure
        The generated figure
    """
    n_pairs = len(qubit_pairs)
    cols = min(4, n_pairs)  # Max 4 columns
    rows = (n_pairs + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[i]
        qp_name = qp.name
        fit_result = fit_results.sel(qubit_pair=qp_name)

        # Plot phase difference data
        fit_result.phase_diff.plot.line(ax=ax, x="amp_full")

        # Plot fitted curve if available
        if fit_result.success and not np.all(np.isnan(fit_result.fitted_curve)):
            ax.plot(fit_result.phase_diff.amp_full, fit_result.fitted_curve)

        # Mark optimal point
        ax.plot([fit_result.optimal_amplitude], [0.5], marker="o", color="red")
        ax.axhline(y=0.5, color="red", linestyle="--", lw=0.5)
        ax.axvline(x=fit_result.optimal_amplitude, color="red", linestyle="--", lw=0.5)

        # Add secondary x-axis for detuning in MHz
        def amp_to_detuning_MHz(amp):
            return -(amp**2) * qp.qubit_control.freq_vs_flux_01_quad_term / 1e6  # Convert Hz to MHz

        def detuning_MHz_to_amp(detuning_MHz):
            return np.sqrt(-detuning_MHz * 1e6 / qp.qubit_control.freq_vs_flux_01_quad_term)

        secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
        secax.set_xlabel("Detuning (MHz)")

        ax.set_title(qp_name)
        ax.set_xlabel("Amplitude (V)")
        ax.set_ylabel("Phase difference")

    # Hide unused axes
    for i in range(n_pairs, len(axes)):
        axes[i].axis("off")

    fig.suptitle("CZ phase calibration (phase difference + fit)")
    fig.tight_layout()

    return fig
