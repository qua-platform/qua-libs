"""Plotting functions for JAZZ_ZZ calibration results."""

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

_FIG_TITLE_PREFIX = "ZZ coupling off - JAZZ"


def _subplot_title(ds: xr.Dataset, qp_name: str) -> str:
    if "measured_qubit_name" in ds.coords:
        measured_name = str(ds.measured_qubit_name.sel(qubit_pair=qp_name).values)
        return f"{qp_name}\nMeasured: {measured_name}"
    return qp_name


def _pairs_with_successful_fits(ds_fit: xr.Dataset, qubit_pairs) -> list:
    return [qp for qp in qubit_pairs if bool(ds_fit.sel(qubit_pair=qp.name).success.values)]


def plot_effective_coupling(ds_fit: xr.Dataset, qubit_pairs) -> Figure | None:
    """Effective coupling $|J_{eff}|$ vs coupler flux for each pair.

    The subplot layout is computed automatically from the qubit-pair
    grid locations using :class:`~calibration_utils.pair_grid.QubitPairGrid`.
    Pairs where all per-amplitude fits failed are skipped. Returns ``None`` if
    there is nothing to plot.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing ``jeff_raw``, ``jeff_smooth``, ``fit_mask``,
        ``optimal_amplitude``, ``measured_qubit_name``, and ``artificial_detuning``.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure or None
    """
    valid_pairs = _pairs_with_successful_fits(ds_fit, qubit_pairs)
    if not valid_pairs:
        return None

    artificial_detuning = float(ds_fit.artificial_detuning)
    g_names, qp_names = grid_pair_names(valid_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        fit_result = ds_fit.sel(qubit_pair=qp_name)
        flux_bias = fit_result.amp.values
        jeff_raw = fit_result.jeff_raw.values
        jeff_smooth = fit_result.jeff_smooth.values
        fit_mask = fit_result.fit_mask.values.astype(bool)

        ax.plot(
            flux_bias[fit_mask],
            np.abs(jeff_raw[fit_mask] - artificial_detuning),
            "o",
            color="gold",
            alpha=0.6,
            label="Extracted $J_{eff}$",
        )
        ax.plot(
            flux_bias[fit_mask],
            np.abs(jeff_smooth[fit_mask] - artificial_detuning),
            "-",
            color="orange",
            linewidth=2,
            label="Smoothed $J_{eff}$",
        )
        if not np.isnan(fit_result.optimal_amplitude.values):
            ax.axvline(
                x=fit_result.optimal_amplitude.values,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Optimal amplitude",
            )

        ax.set_title(_subplot_title(ds_fit, qp_name))
        ax.set_xlabel("Coupler flux (V)")
        ax.set_ylabel(r"Effective coupling $|J_{eff}|$ (MHz)")
        ax.grid(True)
        ax.legend(fontsize="small")

    grid.fig.suptitle(f"{_FIG_TITLE_PREFIX} — $J_{{eff}}$ vs coupler flux")
    grid.fig.tight_layout()
    return grid.fig


def plot_decay_rate_data(ds_fit: xr.Dataset, qubit_pairs) -> Figure | None:
    """Decay time constant τ vs coupler flux for each pair.

    The subplot layout is computed automatically from the qubit-pair
    grid locations using :class:`~calibration_utils.pair_grid.QubitPairGrid`.
    Pairs with no valid decay-time extraction are skipped. Returns ``None`` if
    there is nothing to plot.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing ``tau_raw``, ``tau_smooth``, ``fit_mask``,
        ``max_decay_time``, ``max_decay_time_amplitude``, and ``measured_qubit_name``.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure or None
    """
    valid_pairs = [
        qp for qp in qubit_pairs if np.any(ds_fit.sel(qubit_pair=qp.name).tau_raw.values > 0)
    ]
    if not valid_pairs:
        return None

    g_names, qp_names = grid_pair_names(valid_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        fit_result = ds_fit.sel(qubit_pair=qp_name)
        flux_bias = fit_result.amp.values
        tau_raw = fit_result.tau_raw.values
        tau_smooth = fit_result.tau_smooth.values
        fit_mask = fit_result.fit_mask.values.astype(bool)
        valid_tau_mask = fit_mask & (tau_raw > 0)

        ax.plot(
            flux_bias[valid_tau_mask],
            tau_raw[valid_tau_mask],
            "o",
            color="gold",
            alpha=0.6,
            label="Extracted decay time",
        )
        ax.plot(
            flux_bias[valid_tau_mask],
            tau_smooth[valid_tau_mask],
            "-",
            color="orange",
            linewidth=2,
            label="Smoothed decay time",
        )

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
            ax.plot(
                max_tau_amp,
                max_tau,
                "ro",
                markersize=8,
                label="Maximum decay time",
            )

        if not np.isnan(fit_result.optimal_amplitude.values):
            ax.axvline(
                x=fit_result.optimal_amplitude.values,
                color="green",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label="Optimal coupling amplitude",
            )

        ax.set_title(_subplot_title(ds_fit, qp_name))
        ax.set_xlabel("Coupler flux (V)")
        ax.set_ylabel("Decay time constant τ (µs)")
        ax.grid(True)
        ax.legend(fontsize="small")

    grid.fig.suptitle(f"{_FIG_TITLE_PREFIX} — decay vs coupler flux")
    grid.fig.tight_layout()
    return grid.fig


def plot_raw_data(ds: xr.Dataset, qubit_pairs) -> Figure:
    """2D heatmap of measured signal vs (coupler flux, time) for each pair.

    The subplot layout is computed automatically from the qubit-pair
    grid locations using :class:`~calibration_utils.pair_grid.QubitPairGrid`.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing ``state_target`` or ``I_target``, and
        ``measured_qubit_name``.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure
    """
    data_var = "state_target" if "state_target" in ds.data_vars else "I_target"
    g_names, qp_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        ds[data_var].sel(qubit_pair=qp_name).squeeze().plot(ax=ax, x="amp", y="time")
        ax.set_xlabel("Coupler flux (V)")
        ax.set_ylabel("Time (ns)")
        ax.set_title(_subplot_title(ds, qp_name))
    grid.fig.suptitle(f"{_FIG_TITLE_PREFIX} — raw data")
    grid.fig.tight_layout()
    return grid.fig


def plot_fit_data(ds_fit: xr.Dataset, qubit_pairs) -> Figure | None:
    """2D heatmap of the fitted oscillation for each pair.

    Skips pairs where all fits failed (all ``fitted_state`` values are NaN)
    and returns ``None`` if there is nothing to plot.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing ``fitted_state`` and ``measured_qubit_name``.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure or None
    """
    valid_pairs = _pairs_with_successful_fits(ds_fit, qubit_pairs)
    if not valid_pairs:
        return None

    g_names, qp_names = grid_pair_names(valid_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        ds_fit.fitted_state.sel(qubit_pair=qp_name).squeeze().plot(ax=ax, x="amp", y="time")
        ax.set_xlabel("Coupler flux (V)")
        ax.set_ylabel("Time (ns)")
        ax.set_title(_subplot_title(ds_fit, qp_name))
    grid.fig.suptitle(f"{_FIG_TITLE_PREFIX} — fit")
    grid.fig.tight_layout()
    return grid.fig
