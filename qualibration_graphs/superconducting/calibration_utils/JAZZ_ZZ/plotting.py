"""Plotting functions for JAZZ_ZZ calibration results."""

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

_FIG_TITLE_PREFIX = "ZZ coupling off (JAZZ)"


def _subplot_title(ds: xr.Dataset, qp_name: str) -> str:
    if "measured_qubit_name" in ds.coords:
        measured_name = str(ds.measured_qubit_name.sel(qubit_pair=qp_name).values)
        return f"{qp_name}\nMeasured: {measured_name}"
    return qp_name


def _pairs_with_successful_fits(ds_fit: xr.Dataset, qubit_pairs) -> list:
    return [qp for qp in qubit_pairs if bool(ds_fit.sel(qubit_pair=qp.name).success.values)]


def plot_effective_coupling(ds_fit: xr.Dataset, qubit_pairs, log_y: bool = False) -> Figure | None:
    """Residual ZZ coupling |ζ| vs coupler flux for each pair.

    Following arXiv:2402.18926 (Li et al., Sec. III.1):
      - ``jeff_raw`` stores ωm_code = ζ + ωb_eff, the total oscillation frequency
        measured vs t_single (one coupler pulse duration).
      - The true ZZ coupling is ζ = jeff_raw − artificial_detuning.
      - The plotted quantity is |ζ| = |jeff_raw − artificial_detuning|.
      - The optimal amplitude is the coupler bias where |ζ| is minimised (ζ ≈ 0).

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
    log_y : bool
        If ``True``, use a logarithmic y-axis (only points with positive residual
        coupling are shown).

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

        # |ζ| = |ωm_code − ωb_eff| = |jeff_raw − artificial_detuning|
        residual_raw = np.abs(jeff_raw[fit_mask] - artificial_detuning)
        residual_smooth = np.abs(jeff_smooth[fit_mask] - artificial_detuning)
        flux_plot = flux_bias[fit_mask]

        if log_y:
            positive = residual_raw > 0
            if np.any(positive):
                ax.plot(
                    flux_plot[positive],
                    residual_raw[positive],
                    "o",
                    color="gold",
                    alpha=0.6,
                    label="Extracted residual",
                )
            positive_smooth = residual_smooth > 0
            if np.any(positive_smooth):
                ax.plot(
                    flux_plot[positive_smooth],
                    residual_smooth[positive_smooth],
                    "-",
                    color="orange",
                    linewidth=2,
                    label="Smoothed residual",
                )
            ax.set_yscale("log")
        else:
            ax.plot(
                flux_plot,
                residual_raw,
                "o",
                color="gold",
                alpha=0.6,
                label="Extracted residual",
            )
            ax.plot(
                flux_plot,
                residual_smooth,
                "-",
                color="orange",
                linewidth=2,
                label="Smoothed residual",
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
        ax.set_ylabel(r"Residual ZZ coupling $|\zeta|$ (MHz)")
        ax.grid(True, which="both" if log_y else "major")
        ax.legend(fontsize="small")

    grid.fig.suptitle(f"{_FIG_TITLE_PREFIX} — residual ZZ coupling |ζ| vs coupler flux")
    grid.fig.tight_layout()
    return grid.fig


def plot_decay_rate_data(ds_fit: xr.Dataset, qubit_pairs, log_y: bool = False) -> Figure | None:
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
    log_y : bool
        If ``True``, use a logarithmic y-axis (only τ > 0 are shown).

    Returns
    -------
    Figure or None
    """
    valid_pairs = [qp for qp in qubit_pairs if np.any(ds_fit.sel(qubit_pair=qp.name).tau_raw.values > 0)]
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
        flux_plot = flux_bias[valid_tau_mask]
        tau_raw_plot = tau_raw[valid_tau_mask]
        tau_smooth_plot = tau_smooth[valid_tau_mask]

        if log_y:
            smooth_mask = tau_smooth_plot > 0
            if np.any(valid_tau_mask):
                ax.plot(
                    flux_plot,
                    tau_raw_plot,
                    "o",
                    color="gold",
                    alpha=0.6,
                    label="Extracted decay time",
                )
            if np.any(smooth_mask):
                ax.plot(
                    flux_plot[smooth_mask],
                    tau_smooth_plot[smooth_mask],
                    "-",
                    color="orange",
                    linewidth=2,
                    label="Smoothed decay time",
                )
            ax.set_yscale("log")
        else:
            ax.plot(
                flux_plot,
                tau_raw_plot,
                "o",
                color="gold",
                alpha=0.6,
                label="Extracted decay time",
            )
            ax.plot(
                flux_plot,
                tau_smooth_plot,
                "-",
                color="orange",
                linewidth=2,
                label="Smoothed decay time",
            )

        if not np.isnan(fit_result.max_decay_time.values):
            max_tau = fit_result.max_decay_time.values
            max_tau_amp = fit_result.max_decay_time_amplitude.values
            if not log_y or max_tau > 0:
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
        ax.grid(True, which="both" if log_y else "major")
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
        Raw dataset containing ``state_measured`` or ``I_measured``, and
        ``measured_qubit_name``.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure
    """
    data_var = "state_measured" if "state_measured" in ds.data_vars else "I_measured"
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
