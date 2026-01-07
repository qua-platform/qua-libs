from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.JAZZ_ZZ.analysis import FitResults
from qualibration_libs.core import BatchableList


def plot_raw_data_with_fit(
    fit_results: xr.Dataset,
    qubit_pairs: BatchableList,
    node=None,
) -> plt.Figure:
    """
    Plot the JAZZ_ZZ data showing effective coupling J_eff vs flux bias with optimal point.

    Parameters:
    -----------
    fit_results : xr.Dataset
        Fit results for each qubit pair containing jeff_raw, jeff_smooth, fit_mask, optimal_amplitude
    qubit_pairs : BatchableList
        List of qubit pairs
    node : QualibrationNode, optional
        Node containing parameters like artificial_detuning_mhz

    Returns:
    --------
    plt.Figure
        The generated figure
    """
    n_pairs = len(qubit_pairs)
    cols = min(4, n_pairs)  # Max 4 columns
    rows = (n_pairs + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[i]
        qp_name = qp.name
        fit_result = fit_results.sel(qubit_pair=qp_name)

        # Get flux bias values and coupling data
        flux_bias = fit_result.amp.values
        jeff_raw = fit_result.jeff_raw.values
        jeff_smooth = fit_result.jeff_smooth.values
        fit_mask = fit_result.fit_mask.values.astype(bool)

        # Get artificial detuning from parameters
        artificial_detuning = node.parameters.artificial_detuning_mhz if node else 1.0

        # Plot flat traces (failed fits) - Blue
        plt.sca(ax)
        # ax.plot(
        #     flux_bias[~fit_mask],
        #     jeff_raw[~fit_mask] - artificial_detuning,
        #     "o",
        #     color="blue",
        #     alpha=0.5,
        #     label="Flat signal (J = 0)",
        # )

        # Plot valid fits - Gold
        ax.plot(
            flux_bias[fit_mask],
            np.abs(jeff_raw[fit_mask] - artificial_detuning),
            "o",
            color="gold",
            alpha=0.6,
            label="Extracted $J_{eff}$",
        )

        # Plot smoothed fit - Orange
        if np.any(fit_mask):
            ax.plot(
                flux_bias[fit_mask],
                np.abs(jeff_smooth[fit_mask] - artificial_detuning),
                "-",
                color="orange",
                linewidth=2,
                label="Smoothed $J_{eff}$",
            )

        # Mark optimal amplitude (minimum coupling)
        if not np.isnan(fit_result.optimal_amplitude.values):
            ax.axvline(
                x=fit_result.optimal_amplitude.values,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Optimal amplitude",
            )

        ax.set_xlabel("Flux Bias (arb. units)")
        ax.set_ylabel("Effective Coupling $|J_{eff}|$ (MHz)")
        ax.set_title(f"JAZZ_ZZ Coupling vs Flux Bias - {qp_name}")
        ax.grid(True)
        ax.legend()

    # Hide unused axes
    for i in range(n_pairs, len(axes)):
        axes[i].axis("off")

    fig.suptitle("JAZZ_ZZ Effective Coupling Extraction")
    fig.tight_layout()

    return fig


def plot_oscillation_data(
    ds_raw: xr.Dataset,
    qubit_pairs: BatchableList,
    amp_indices: list = None,
) -> plt.Figure:
    """
    Plot raw oscillation data for selected amplitudes to show the time evolution.

    Parameters:
    -----------
    ds_raw : xr.Dataset
        Raw dataset containing state_target oscillations
    qubit_pairs : BatchableList
        List of qubit pairs
    amp_indices : list, optional
        List of amplitude indices to plot. If None, plots a few representative ones.

    Returns:
    --------
    plt.Figure
        The generated figure showing oscillation traces
    """
    n_pairs = len(qubit_pairs)
    cols = min(4, n_pairs)
    rows = (n_pairs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[i]
        qp_name = qp.name
        qp_data = ds_raw.sel(qubit_pair=qp_name)

        qp_data.state_target.plot(ax=ax, x="amp")

        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("State Target")
        ax.set_title(f"JAZZ_ZZ Oscillations - {qp_name}")

    # Hide unused axes
    for i in range(n_pairs, len(axes)):
        axes[i].axis("off")

    fig.suptitle("JAZZ_ZZ Raw Oscillation Data")
    fig.tight_layout()

    return fig


def plot_decay_rate_data(
    fit_results: xr.Dataset,
    qubit_pairs: BatchableList,
    node=None,
) -> plt.Figure:
    """
    Plot the decay time constant (τ = 1/γ) vs coupler amplitude for JAZZ_ZZ data.

    Parameters:
    -----------
    fit_results : xr.Dataset
        Fit results for each qubit pair containing tau_raw, tau_smooth, fit_mask, max_decay_time
    qubit_pairs : BatchableList
        List of qubit pairs
    node : QualibrationNode, optional
        Node containing parameters

    Returns:
    --------
    plt.Figure
        The generated figure showing decay time constant vs coupler amplitude
    """
    n_pairs = len(qubit_pairs)
    cols = min(4, n_pairs)  # Max 4 columns
    rows = (n_pairs + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, qp in enumerate(qubit_pairs):
        ax = axes[i]
        qp_name = qp.name
        fit_result = fit_results.sel(qubit_pair=qp_name)

        # Get flux bias values and decay time data
        flux_bias = fit_result.amp.values
        tau_raw = fit_result.tau_raw.values
        tau_smooth = fit_result.tau_smooth.values
        fit_mask = fit_result.fit_mask.values.astype(bool)

        # Plot flat traces (failed fits) - Blue
        plt.sca(ax)
        # ax.plot(
        #     flux_bias[~fit_mask],
        #     tau_raw[~fit_mask],
        #     "o",
        #     color="blue",
        #     alpha=0.5,
        #     label="Failed fits (τ = 0)",
        # )

        # Plot valid fits - Gold
        valid_tau_mask = fit_mask & (tau_raw > 0)
        if np.any(valid_tau_mask):
            ax.plot(
                flux_bias[valid_tau_mask],
                tau_raw[valid_tau_mask],
                "o",
                color="gold",
                alpha=0.6,
                label="Extracted decay time",
            )

        # Plot smoothed fit - Orange
        if np.any(valid_tau_mask):
            ax.plot(
                flux_bias[valid_tau_mask],
                tau_smooth[valid_tau_mask],
                "-",
                color="orange",
                linewidth=2,
                label="Smoothed decay time",
            )

        # Mark maximum decay time
        if not np.isnan(fit_result.max_decay_time.values):
            max_tau = fit_result.max_decay_time.values
            max_tau_amp = fit_result.max_decay_time_amplitude.values
            ax.axvline(
                x=max_tau_amp,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Max τ amplitude (τ={max_tau:.3f} µs)",
            )

            # Mark the maximum decay time point
            ax.plot(
                max_tau_amp,
                max_tau,
                "ro",
                markersize=8,
                label="Maximum decay time",
            )

        # Also mark optimal amplitude for reference
        if not np.isnan(fit_result.optimal_amplitude.values):
            ax.axvline(
                x=fit_result.optimal_amplitude.values,
                color="green",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label="Optimal coupling amplitude",
            )

        ax.set_xlabel("Coupler Amplitude (arb. units)")
        ax.set_ylabel("Decay Time Constant τ (µs)")
        ax.set_title(f"JAZZ_ZZ Decay Time vs Coupler Amplitude - {qp_name}")
        ax.grid(True)
        ax.legend()

    # Hide unused axes
    for i in range(n_pairs, len(axes)):
        axes[i].axis("off")

    fig.suptitle("JAZZ_ZZ Decay Time Constant Analysis")

    return fig
