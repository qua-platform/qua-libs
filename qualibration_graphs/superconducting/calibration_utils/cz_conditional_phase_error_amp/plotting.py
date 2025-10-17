from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_conditional_phase_error_amp.analysis import FitResults
from qualibration_libs.core import BatchableList


def plot_raw_data_with_fit(
    fit_results: xr.Dataset,
    qubit_pairs: BatchableList,
) -> plt.Figure:
    """Plot phase difference as 2D heatmap (number_of_operations vs amplitude) with optimal amplitude line.

    For each qubit pair we display:
      - pcolormesh of phase_diff (dims: number_of_operations x amp)
      - vertical line at optimal_amplitude
      - horizontal dashed line at phase=0.5 reference (shown via colorbar context)
      - secondary x-axis with detuning (MHz)
    """
    n_pairs = len(qubit_pairs)
    cols = min(4, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[i]
        qp_name = qp.name
        fr = fit_results.sel(qubit_pair=qp_name)

        phase = fr.phase_diff  # dims: number_of_operations, amp
        # Coordinates
        amps = (fr.amp_full if "amp_full" in fr.coords else fr.amp).values
        n_ops = fr.number_of_operations.values if "number_of_operations" in phase.dims else np.arange(phase.sizes[0])

        # Create mesh
        X, Y = np.meshgrid(amps, n_ops)
        pcm = ax.pcolormesh(
            X,
            Y,
            phase.values,
            cmap="twilight_shifted",
            shading="auto",
            vmin=0.0,
            vmax=1.0,
        )
        ax.axvline(fr.optimal_amplitude.item(), color="lime", lw=2, label="optimal")

        # Secondary x-axis: detuning (MHz)
        quad = qp.qubit_control.freq_vs_flux_01_quad_term

        def amp_to_detuning_MHz(a):
            return -(a**2) * quad / 1e6

        def detuning_MHz_to_amp(d):
            return np.sqrt(np.maximum(0, -d * 1e6 / quad))

        secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
        secax.set_xlabel("Detuning (MHz)")

        ax.set_title(qp_name)
        ax.set_xlabel("Amplitude (V)")
        ax.set_ylabel("# CZ operations")
        ax.legend(loc="upper right", fontsize=8)
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.85)
        cbar.set_label("Phase diff (2π units)")

    # Hide unused axes
    for j in range(n_pairs, len(axes)):
        axes[j].axis("off")

    fig.suptitle("CZ conditional phase error amp: phase diff heatmaps")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
