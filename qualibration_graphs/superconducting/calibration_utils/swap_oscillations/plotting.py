import xarray as xr
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

from calibration_utils.swap_oscillations.analysis import FitParameters
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: List[AnyTransmon],
    ds_fit: xr.Dataset,
    fit_results: Dict[str, FitParameters],
) -> plt.Figure:
    """
    Plot control and target state chevrons with fitted overlays and fringe lines.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing state_control and state_target.
    qubit_pairs : list of AnyTransmon
        Qubit pairs used in the experiment.
    ds_fit : xr.Dataset
        Dataset with fitted 1D oscillation curves.
    fit_results : dict of FitParameters
        Fitting results with J, f0, amplitude, duration, etc.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(qubit_pairs)
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(15, 4 * n))

    if n == 1:
        axs = np.array([[axs[0], axs[1]]])

    for i, qp in enumerate(qubit_pairs):
        name = qp.name
        ds_qp = ds.sel(qubit=name)
        detuning = ds.detuning.sel(qubit=name)
        amp_full = ds.amp_full.sel(qubit=name)

        for col, var in enumerate(["state_control", "state_target"]):
            ax = axs[i][col]
            da = ds_qp[var].assign_coords(detuning_MHz=detuning * 1e-6)
            im = da.plot(ax=ax, x="idle_time", y="detuning_MHz", add_colorbar=False)
            plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.2, aspect=30)

            fp = fit_results.get(name)
            if fp and fp.success:
                ax.axhline(fp.detuning * 1e-6, linestyle="--", color="k", lw=0.5)
                ax.axvline(fp.optimal_length, linestyle="--", color="k", lw=0.5)

                f_eff = np.sqrt(fp.J**2 + (detuning - fp.detuning) ** 2)
                for n_line in range(1, 10):
                    ax.plot(n_line * 0.5 / f_eff * 1e9, detuning * 1e-6, "r-", lw=0.3)

            ax.set_title(f"{name} - {var.replace('state_', '')}")
            ax.set_xlabel("Idle time [ns]")
            ax.set_ylabel("Detuning [MHz]")

            # Twin axis for flux amplitude
            ax2 = ax.twinx()
            amp_range = np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term)
            ax2.set_ylim(amp_range.min(), amp_range.max())
            ax2.set_ylabel("Flux amplitude [V]")
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()

    fig.tight_layout()
    fig.suptitle("SWAP Oscillations - Control and Target States", y=1.02)
    return fig
