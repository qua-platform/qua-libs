from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from calibration_utils.common_utils.plot_style import apply_qubit_outcome_style, empty_figure
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubits: List,
    fit_results: dict | None = None,
):
    """
    Plot the qubit spectroscopy state traces with fitted curves.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``state(qubit, detuning)`` and optional raw ``I`` / ``Q`` traces.
    qubits : list
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters and ``fit_curve``.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    qubit_names = [str(v) for v in ds_fit.qubit.values]
    n = len(qubit_names)
    if n == 0:
        return empty_figure("No qubit data available for qubit spectroscopy.")
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax = axes[0, i]
        fit = ds_fit.sel(qubit=qname)
        plot_individual_data_with_fit(ax, ds_fit, qname, fit, fit_results=(fit_results or {}).get(qname))

    fig.suptitle("Qubit spectroscopy")
    fig.tight_layout()
    return fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit_name: str,
    fit: xr.Dataset = None,
    fit_results: dict | None = None,
):
    """
    Plot one qubit's state trace with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        Dataset containing ``state(qubit, detuning)``.
    qubit_name : str
        The qubit name to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    """
    if fit is None or "state" not in fit.data_vars:
        ax.text(0.5, 0.5, f"No data for {qubit_name}", transform=ax.transAxes, ha="center")
        apply_qubit_outcome_style(ax, qubit_name, None, subtitle="No data")
        return

    (fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz).state).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("State")
    success = fit_results.get("success") if fit_results is not None else None
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Qubit spectroscopy")

    ax2 = ax.twiny()
    (fit.assign_coords(detuning_MHz=fit.detuning / u.MHz).state).plot(ax=ax2, x="detuning_MHz", label="")
    ax2.set_xlabel("Detuning [MHz]")

    if fit is not None and "fit_curve" in fit.data_vars:
        ax2.plot(
            fit.detuning / u.MHz,
            fit.fit_curve.values,
            "r--",
            label="fit",
        )
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def _plot_raw_iq_traces(ds: xr.Dataset, qubits: List) -> Figure:
    """Plot the averaged raw I and Q traces versus detuning for each qubit."""
    qubit_names = [str(v) for v in ds.qubit.values]
    n = len(qubit_names)
    if n == 0:
        return empty_figure("No IQ data available for qubit spectroscopy.")
    fig_iq, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for idx, qname in enumerate(qubit_names):
        ax = axes[0, idx]
        i_vals = ds.I.sel(qubit=qname, drop=True).transpose("detuning").values.astype(float)
        q_vals = ds.Q.sel(qubit=qname, drop=True).transpose("detuning").values.astype(float)
        detuning_mhz = ds["detuning"].values / 1e6
        ax_q = ax.twinx()
        (line_i,) = ax.plot(detuning_mhz, i_vals, color="C0", label="I", zorder=3)
        (line_q,) = ax_q.plot(detuning_mhz, q_vals, color="C1", label="Q", zorder=2)
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("I", color="C0")
        ax_q.set_ylabel("Q", color="C1")
        ax.tick_params(axis="y", labelcolor="C0")
        ax_q.tick_params(axis="y", labelcolor="C1")
        apply_qubit_outcome_style(ax, qname, None, subtitle="IQ vs frequency")
        ax.legend(handles=[line_i, line_q])

    fig_iq.suptitle("Raw IQ signal")
    fig_iq.tight_layout()
    return fig_iq


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List,
    fit_results: dict | None = None,
) -> dict[str, Figure]:
    """Build and return all 08b spectroscopy figures."""
    figures = {
        "qubit_spectroscopy": plot_raw_data_with_fit(
            ds_fit,
            qubits,
            fit_results=fit_results,
        ),
        "iq_scatter": _plot_raw_iq_traces(ds_fit, qubits),
    }
    return figures
