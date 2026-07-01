import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def _mark_fitted_peak(
    ax: Axes,
    frames: np.ndarray,
    fit_curve: np.ndarray,
    fitted_phase: float,
    color: str,
    channel_label: str,
) -> None:
    """Mark the fitted-sine maximum."""
    if not np.any(np.isfinite(fit_curve)):
        return
    i_max = int(np.nanargmax(fit_curve))
    peak_frame = float(frames[i_max])
    peak_val = float(fit_curve[i_max])
    ax.axvline(
        peak_frame,
        color=color,
        ls="--",
        lw=1.5,
        alpha=0.85,
        zorder=3,
        label=f"{channel_label} peak @ {peak_frame:.4f} (φ={fitted_phase:.4f})",
    )
    ax.plot(
        peak_frame,
        peak_val,
        "*",
        ms=12,
        mew=1.2,
        zorder=5,
        markerfacecolor="white",
        markeredgecolor=color,
    )


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    qubit_pairs: list,
    ds_fit: xr.Dataset = None,
) -> Figure:
    """Plot raw Ramsey oscillations with fitted curves for every qubit pair on a chip-topology grid.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset containing ``state_control`` / ``state_target`` or
        ``I_control`` / ``I_target`` variables with a ``frame`` dimension.
    qubit_pairs : list
        Qubit pair objects used for grid placement.
    ds_fit : xr.Dataset, optional
        Fit dataset containing ``fitted_control``, ``fitted_target``, and ``success``.
        If None, only the raw data is shown.

    Returns
    -------
    Figure
        Matplotlib figure with one panel per qubit pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        plot_individual_data_with_fit(ax, ds_raw, qp_name, ds_fit)

    grid.fig.suptitle("CZ 1Q phase compensation")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds_raw: xr.Dataset,
    qp_name: str,
    ds_fit: xr.Dataset = None,
):
    """Plot raw Ramsey oscillations and fitted curves for one qubit pair.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds_raw : xr.Dataset
        Raw dataset with ``state_control`` / ``state_target`` or
        ``I_control`` / ``I_target`` variables.
    qp_name : str
        Qubit pair name used to select data.
    ds_fit : xr.Dataset, optional
        Fit dataset containing ``fitted_control``, ``fitted_target``, and ``success``.
        If None or fit failed, only raw data is shown.
    """
    qp_data = ds_raw.sel(qubit_pair=qp_name)

    if "state_control" in ds_raw.data_vars:
        qp_data.state_control.plot(ax=ax, marker="o", linestyle="", color="blue", label="Control")
        qp_data.state_target.plot(ax=ax, marker="o", linestyle="", color="red", label="Target")
        ylabel = "Measured State"
    else:
        qp_data.I_control.sel(control_target="c").plot(
            ax=ax, marker="o", linestyle="", color="blue", label="Control"
        )
        qp_data.I_target.sel(control_target="t").plot(
            ax=ax, marker="o", linestyle="", color="red", label="Target"
        )
        ylabel = "Rotated I Quadrature (V)"

    if ds_fit is not None:
        qp_fit = ds_fit.sel(qubit_pair=qp_name)
        frames = qp_data.frame.values
        if bool(qp_fit.success.values):
            if "fitted_control" in ds_fit.data_vars:
                fit_c = np.asarray(qp_fit.fitted_control.values, dtype=float)
                ax.plot(frames, fit_c, color="blue", alpha=0.5)
                if "fitted_control_phase" in ds_fit.data_vars:
                    _mark_fitted_peak(
                        ax,
                        frames,
                        fit_c,
                        float(qp_fit.fitted_control_phase.values),
                        "blue",
                        "Control",
                    )
            if "fitted_target" in ds_fit.data_vars:
                fit_t = np.asarray(qp_fit.fitted_target.values, dtype=float)
                ax.plot(frames, fit_t, color="red", alpha=0.5)
                if "fitted_target_phase" in ds_fit.data_vars:
                    _mark_fitted_peak(
                        ax,
                        frames,
                        fit_t,
                        float(qp_fit.fitted_target_phase.values),
                        "red",
                        "Target",
                    )

    ax.set_title(qp_name)
    ax.set_xlabel(r"Virtual-Z frame [$2\pi$]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
