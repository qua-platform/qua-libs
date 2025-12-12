from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_control_data(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset, node: QualibrationNode):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    fig, axs = plt.subplots(nrows=len(qubit_pairs), ncols=1, figsize=(15, 9))
    for ii, qp in enumerate(qubit_pairs):
        ax = axs[ii] if len(qubit_pairs) > 1 else axs

        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_control.sel(qubit_pair=qp.name)
        else:
            values_to_plot = ds.Q_control.sel(qubit_pair=qp.name)
        values_to_plot.plot(ax=ax, cmap="viridis", y="idle_time", x="flux_coupler")
        qubit_pair = node.machine.qubit_pairs[qp.name]
        ax.set_title(f"{qp.name}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize=10)
    fig.suptitle("Control")
    fig.tight_layout()

    return fig


def plot_target_data(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset, node: QualibrationNode):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    fig, axs = plt.subplots(nrows=len(qubit_pairs), ncols=1, figsize=(15, 9))
    for ii, qp in enumerate(qubit_pairs):
        ax = axs[ii] if len(qubit_pairs) > 1 else axs

        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_target.sel(qubit_pair=qp.name)
        else:
            values_to_plot = ds.Q_target.sel(qubit_pair=qp.name)
        values_to_plot.plot(ax=ax, cmap="viridis", y="idle_time", x="flux_coupler")
        qubit_pair = node.machine.qubit_pairs[qp.name]
        ax.set_title(f"{qp.name}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize=10)
    fig.suptitle("Target")
    fig.tight_layout()

    return fig


def plot_domain_frequency(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset, node: QualibrationNode):
    fig, axs = plt.subplots(nrows=len(qubit_pairs), ncols=1, figsize=(15, 9))

    for ii, qp in enumerate(qubit_pairs):
        ax = axs[ii] if len(qubit_pairs) > 1 else axs
        target_gate_time = 50  # ns (fallback)
        (1e3 * ds.dominant_frequency.sel(qubit_pair=qp.name)).plot(ax=ax, marker=".", ls="None", x="flux_coupler")
        qubit_pair = node.machine.qubit_pairs[qp.name]
        ax.axvline(x=qubit_pair.coupler.decouple_offset, color="black", label="Decouple offset")
        ax.axvline(
            x=fits[qp.name]["coupler_flux_pulse"],
            color="red",
            lw=0.5,
            ls="--",
            label=f"Optimal frequency for {target_gate_time}ns pulse",
        )
        ax.axvline(
            x=fits[qp.name]["coupler_flux_min"] - qubit_pair.coupler.decouple_offset, color="green", lw=0.5, ls="--"
        )
        ax.set_title(f"{qp.name}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize=10)
        ax.set_xlabel("Flux Coupler")
        ax.set_ylabel("Frequency (MHz)")
        ax.legend()
    fig.suptitle("Target")
    fig.tight_layout()

    return fig


def plot_jeff_vs_flux(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset, node: QualibrationNode):
    fig, axs = plt.subplots(nrows=len(qubit_pairs), ncols=1, figsize=(8, 5 * len(qubit_pairs)))
    if len(qubit_pairs) == 1:
        axs = [axs]

    for i, qp_name in enumerate(ds.qubit_pair.values):
        ax = axs[i]
        ds_qp = ds.sel(qubit_pair=qp_name)
        phi_vals = ds_qp.flux_coupler.values
        jeff_raw = ds_qp[f"jeff_raw_{qp_name}"].values
        fit_mask = ds_qp[f"fit_mask_{qp_name}"].values
        jeff_smooth = ds_qp[f"jeff_smooth_{qp_name}"].values

        ax.plot(phi_vals[~fit_mask], jeff_raw[~fit_mask], "o", color="blue", alpha=0.5, label="Flat signal (J = 0)")
        ax.plot(phi_vals[fit_mask], jeff_raw[fit_mask], "o", color="gold", alpha=0.6, label="Extracted $J_{eff}$")
        ax.plot(phi_vals, jeff_smooth, "-", color="orange", linewidth=2, label="Smoothed $J_{eff}$")
        ax.axvline(phi_vals[np.argmin(jeff_smooth)], color="red", linestyle="--", label="Min $J_{eff}$ point")

        flux_at_target = fits[qp_name]["flux_at_target"]
        ax.axvline(flux_at_target, color="green", linestyle="--", label="CZ gate amplitude")

        ax.set_xlabel("Flux Bias (arb. units)")
        ax.set_ylabel("Effective Coupling $J_{eff}$ (MHz)")
        ax.set_title(f"Extracted Coupling vs Flux Bias for {qp_name}")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    return fig
