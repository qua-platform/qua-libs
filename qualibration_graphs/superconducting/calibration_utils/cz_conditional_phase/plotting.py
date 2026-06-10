from typing import Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.cz_iswap_flux_bootstrap.parameters import get_moving_qubit
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
) -> Figure:
    """Plot phase difference vs amplitude with fit for every qubit pair on a chip-topology grid.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fit dataset (must contain ``phase_diff``, ``fitted_curve``,
        ``optimal_amplitude``, ``success``).
    qubit_pairs : list
        Qubit pair objects used for grid placement.

    Returns
    -------
    Figure
        Matplotlib figure with one phase-diff panel per pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    qp_map = {qp.name: qp for qp in qubit_pairs}
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        plot_individual_data_with_fit(ax, ds_fit, qp_name, qp_map[qp_name])

    grid.fig.suptitle("CZ phase calibration — phase difference vs amplitude")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
):
    """Plot phase difference + fit curve for one qubit pair.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds_fit : xr.Dataset
        Fit dataset containing ``phase_diff``, ``fitted_curve``,
        ``optimal_amplitude``, and ``success`` for this pair.
    qp_name : str
        Qubit pair name used to select data.
    qp : qubit pair object
        Used to compute the secondary detuning axis via the moving qubit.
    """
    fit = ds_fit.sel(qubit_pair=qp_name)

    fit.phase_diff.plot.line(ax=ax, x="amp_full")

    success = bool(fit.success) if hasattr(fit, "success") else False
    if success and not np.all(np.isnan(fit.fitted_curve.values)):
        ax.plot(fit.phase_diff.amp_full, fit.fitted_curve, color="orange", lw=1.5, label="fit")

    opt_amp = float(fit.optimal_amplitude)
    ax.axvline(opt_amp, color="red", ls="--", lw=0.8)
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.scatter([opt_amp], [0.5], color="red", zorder=5, label="optimal")
    ax.legend(fontsize=8)

    mq = get_moving_qubit(qp)

    def amp_to_detuning_MHz(amp):
        return -(amp**2) * mq.freq_vs_flux_01_quad_term / 1e6

    def detuning_MHz_to_amp(detuning_MHz):
        return np.sqrt(-detuning_MHz * 1e6 / mq.freq_vs_flux_01_quad_term)

    secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
    secax.set_xlabel("Detuning (MHz)")

    title_suffix = "fit OK" if success else "fit failed"
    ax.set_title(f"{qp_name} — {title_suffix}")
    ax.set_xlabel("Amplitude (V)")
    ax.set_ylabel("Phase difference")


def plot_moving_qubit_populations(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
) -> Figure:
    """Plot moving-qubit g/e/f populations vs amplitude for every pair on a chip-topology grid.

    The moving qubit (the one flux-pulsed during the CZ gate) is the source of
    leakage to the |f⟩ state.  The moving qubit is resolved via
    ``cz_iswap_flux_bootstrap.parameters.get_moving_qubit`` from the pair
    frequencies and anharmonicity, so this works for both control-as-moving
    and target-as-moving architectures.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fit dataset (must contain ``g_state_moving``, ``e_state_moving``,
        ``f_state_moving``, ``optimal_amplitude``).
    qubit_pairs : list
        Qubit pair objects used for grid placement.

    Returns
    -------
    Figure
        Matplotlib figure with one population panel per pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    qp_map = {qp.name: qp for qp in qubit_pairs}
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        plot_individual_moving_qubit_populations(ax, ds_fit, qp_name, qp_map[qp_name])

    grid.fig.suptitle("CZ phase calibration — moving qubit populations")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_moving_qubit_populations(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
):
    """Plot g/e/f moving-qubit populations vs amplitude for one qubit pair.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds_fit : xr.Dataset
        Fit dataset containing ``g_state_moving``, ``e_state_moving``,
        ``f_state_moving``, and ``optimal_amplitude``.
    qp_name : str
        Qubit pair name used to select data.
    qp : qubit pair object
        Used to compute the secondary detuning axis via the moving qubit.
    """
    fit = ds_fit.sel(qubit_pair=qp_name)

    data_g = fit.g_state_moving.sel(control_axis=1).mean(dim="frame")
    data_e = fit.e_state_moving.sel(control_axis=1).mean(dim="frame")
    data_f = fit.f_state_moving.sel(control_axis=1).mean(dim="frame")

    amps = fit.amp_full.values if "amp_full" in fit.coords else fit.amp.values
    ax.plot(amps, data_g, label="g", color="steelblue")
    ax.plot(amps, data_e, label="e", color="tomato")
    ax.plot(amps, data_f, label="f (leakage)", color="seagreen")

    opt_amp = float(fit.optimal_amplitude)
    ax.axvline(opt_amp, color="red", ls="--", lw=0.8, label="optimal")
    ax.axhline(0.0, color="grey", ls=":", lw=0.5)
    ax.axhline(1.0, color="grey", ls=":", lw=0.5)
    ax.legend(fontsize=8)

    mq = get_moving_qubit(qp)

    def amp_to_detuning_MHz(amp):
        return -(amp**2) * mq.freq_vs_flux_01_quad_term / 1e6

    def detuning_MHz_to_amp(detuning_MHz):
        return np.sqrt(-detuning_MHz * 1e6 / mq.freq_vs_flux_01_quad_term)

    secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
    secax.set_xlabel("Detuning (MHz)")

    ax.set_title(f"{qp_name} — moving qubit: {mq.name}")
    ax.set_xlabel("Amplitude (V)")
    ax.set_ylabel("Moving qubit population")
