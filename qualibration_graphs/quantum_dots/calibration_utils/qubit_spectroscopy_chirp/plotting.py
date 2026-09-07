from typing import List, Optional
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from calibration_utils.common_utils.plot_style import apply_qubit_outcome_style, empty_figure
from qualang_tools.units import unit
from qualibration_libs.analysis import lorentzian_peak

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List,
    fits: xr.Dataset = None,
    threshold_results: dict = None,
    signal_threshold: float = None,
):
    """
    Plot the chirped spectroscopy state trace with optional threshold and peak-fit overlays.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``state(qubit, detuning)``.
    qubits : list
        A list of qubits to plot.
    fits : xr.Dataset, optional
        The dataset containing the peak-fit parameters (from ``fit_raw_data``).
    threshold_results : dict, optional
        Per-qubit threshold fit results (as dicts).
    signal_threshold : float, optional
        The signal threshold value to draw as a horizontal line.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    qubit_names = [str(v) for v in ds.qubit.values]
    qubits_by_name = {getattr(q, "name", f"Q{i}"): q for i, q in enumerate(qubits)}
    n = len(qubit_names)
    if n == 0:
        return empty_figure("No qubit data available for chirp spectroscopy.")
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax = axes[0, i]
        fit = fits.sel(qubit=qname) if fits is not None else None
        thr = threshold_results.get(qname) if threshold_results else None
        plot_individual_data_with_fit(
            ax,
            ds,
            qname,
            rf_frequency=qubits_by_name[qname].xy.RF_frequency,
            fit=fit,
            threshold_result=thr,
            signal_threshold=signal_threshold,
        )

    fig.suptitle("Chirp qubit spectroscopy")
    fig.tight_layout()
    return fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit_name: str,
    rf_frequency: float,
    fit: xr.Dataset = None,
    threshold_result: dict = None,
    signal_threshold: float = None,
):
    """
    Plot one qubit's state trace with optional peak-fit and threshold overlays.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        Dataset containing ``state(qubit, detuning)``.
    qubit_name : str
        The qubit name to plot.
    rf_frequency : float
        The RF carrier frequency for this qubit (Hz), used to compute full frequency axis.
    fit : xr.Dataset, optional
        The dataset slice for this qubit from ``fit_raw_data`` (has ``pdiff``, ``full_freq``).
    threshold_result : dict, optional
        Threshold fit result for this qubit.
    signal_threshold : float, optional
        Threshold level to draw as a horizontal line.
    """
    if "state" not in ds.data_vars:
        ax.text(0.5, 0.5, f"No data for {qubit_name}", transform=ax.transAxes, ha="center")
        apply_qubit_outcome_style(ax, qubit_name, None, subtitle="No data")
        return

    detuning = ds.detuning.values
    signal = ds.state.sel(qubit=qubit_name, drop=True).transpose("detuning").values.astype(float)
    full_freq_GHz = (detuning + rf_frequency) / u.GHz

    # Primary x-axis: RF frequency in GHz
    ax.plot(full_freq_GHz, signal)
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("State")
    success = threshold_result.get("success") if threshold_result is not None else None
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Chirp spectroscopy")

    # Secondary x-axis: detuning in MHz
    ax2 = ax.twiny()
    detuning_MHz = detuning / u.MHz
    ax2.plot(detuning_MHz, signal, alpha=0)  # invisible — sets the x-scale only
    ax2.set_xlabel("Detuning [MHz]")
    ax2.set_title("")

    # # Threshold overlay
    # if signal_threshold is not None:
    #     ax2.axhline(
    #         signal_threshold,
    #         color="grey",
    #         ls=":",
    #         lw=1,
    #         alpha=0.7,
    #         label=f"threshold = {signal_threshold}",
    #     )
    # if threshold_result is not None and threshold_result.get("success"):
    #     center_mhz = threshold_result["relative_freq"] / u.MHz
    #     ax2.axvline(
    #         center_mhz,
    #         color="green",
    #         ls="--",
    #         lw=1.5,
    #         alpha=0.8,
    #         label=f"threshold centre = {center_mhz:.1f} MHz",
    #     )

    # Peak-fit Lorentzian overlay (uses ds_fit which has pdiff + fit params)
    if fit is not None and "amplitude" in fit:
        try:
            fitted_data = lorentzian_peak(
                ds.detuning,
                float(fit.amplitude.values),
                float(fit.position.values),
                float(fit.width.values) / 2,
                float(fit.base_line.mean().values),
            )
            ax2.plot(detuning_MHz, fitted_data, "r--", label="Lorentzian fit")
        except Exception:
            pass

    ax2.legend(fontsize=7, loc="upper right")


def plot_all(
    ds: xr.Dataset,
    qubits: List,
    *,
    fits: Optional[xr.Dataset] = None,
    threshold_results: Optional[dict] = None,
    signal_threshold: Optional[float] = None,
    show: bool = True,
) -> dict[str, Figure]:
    """Build and return all 08a chirp spectroscopy figures."""
    figures = {
        "qubit_spectroscopy_chirp": plot_raw_data_with_fit(
            ds,
            qubits,
            fits=fits,
            threshold_results=threshold_results,
            signal_threshold=signal_threshold,
        )
    }
    if show:
        plt.show()
    return figures
