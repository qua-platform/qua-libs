from typing import Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
    qubit_roles_map: Optional[dict] = None,
) -> Figure:
    """Plot phase-diff heatmap (# operations × amplitude) for every pair on a chip-topology grid.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fit dataset containing ``phase_diff``, ``optimal_amplitude``.
    qubit_pairs : list
        Qubit pair objects used for grid placement.

    Returns
    -------
    Figure
        Matplotlib figure with one heatmap panel per pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    qp_map = {qp.name: qp for qp in qubit_pairs}
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        qubit_role_obj = qubit_roles_map.get(qp_name) if qubit_roles_map else None
        plot_individual_data_with_fit(ax, ds_fit, qp_name, qp_map[qp_name], qubit_role_obj=qubit_role_obj)

    grid.fig.suptitle("CZ conditional phase error amplification - phase difference")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
    qubit_role_obj=None,
):
    """Plot phase-diff heatmap for one qubit pair.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds_fit : xr.Dataset
        Fit dataset containing ``phase_diff`` and ``optimal_amplitude``.
    qp_name : str
        Qubit pair name used to select data.
    qp : qubit pair object
        Used to compute the secondary detuning axis via the moving qubit.
    """
    fr = ds_fit.sel(qubit_pair=qp_name)
    phase = fr.phase_diff  # dims: number_of_operations, amp

    amps = (fr.amp_full if "amp_full" in fr.coords else fr.amp).values
    n_ops = fr.number_of_operations.values if "number_of_operations" in phase.dims else np.arange(phase.sizes[0])

    X, Y = np.meshgrid(amps, n_ops)
    pcm = ax.pcolormesh(X, Y, phase.values, cmap="twilight_shifted", shading="auto", vmin=0.0, vmax=1.0)
    ax.axvline(fr.optimal_amplitude.item(), color="lime", lw=2, label="optimal")
    ax.legend(loc="upper right", fontsize=8)

    if qubit_role_obj is not None:
        quad = qubit_role_obj.moving.freq_vs_flux_01_quad_term

        def amp_to_detuning_MHz(a):
            return -(a**2) * quad / 1e6

        def detuning_MHz_to_amp(d):
            return np.sqrt(np.maximum(0, -d * 1e6 / quad))

        secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
        secax.set_xlabel("Detuning (MHz)")

    ax.figure.colorbar(pcm, ax=ax, shrink=0.85).set_label("Phase diff (2π units)")
    ax.set_title(qp_name)
    ax.set_xlabel("Amplitude (V)")
    ax.set_ylabel("# CZ operations")


def plot_leakage_qubit_populations(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
    qubit_roles_map: Optional[dict] = None,
) -> Figure:
    """Plot leakage-qubit g/f populations vs # operations for every pair on a chip-topology grid.

    Populations are shown at the optimal amplitude for each pair. The leakage qubit
    (always the higher-frequency qubit) is determined via ``QubitRoles.resolve``
    and the correct set of streams (``_moving`` or ``_stationary``) is selected.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fit dataset containing ``g/e/f_state_moving``, ``g/e/f_state_stationary``,
        ``optimal_amplitude``.
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
        qubit_role_obj = qubit_roles_map.get(qp_name) if qubit_roles_map else None
        plot_individual_leakage_qubit_populations(ax, ds_fit, qp_name, qp_map[qp_name], qubit_role_obj=qubit_role_obj)

    grid.fig.suptitle("CZ conditional phase error amplification — leakage qubit populations")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_leakage_qubit_populations(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
    qubit_role_obj=None,
):
    """Plot leakage-qubit g/f populations vs # operations for one pair at its optimal amplitude.

    Selects ``g/f_state_moving`` or ``g/f_state_stationary`` depending on which
    qubit is the leakage qubit (the higher-frequency qubit).

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds_fit : xr.Dataset
        Fit dataset containing ``g/e/f_state_moving``, ``g/e/f_state_stationary``,
        and ``optimal_amplitude``.
    qp_name : str
        Qubit pair name used to select data.
    qp : qubit pair object
        Used to resolve qubit roles and label the plot.
    """
    fr = ds_fit.sel(qubit_pair=qp_name)
    n_ops = fr.number_of_operations.values if "number_of_operations" in fr.dims else None

    roles = qubit_role_obj
    lq = roles.leakage if roles is not None else None
    mq = roles.moving if roles is not None else None
    leakage_key = "moving" if (roles is not None and roles.leakage is roles.moving) else "stationary"

    try:
        data_g = (
            ds_fit[f"g_state_{leakage_key}"]
            .sel(qubit_pair=qp_name, control_axis=1)
            .sel(amp=fr.optimal_amplitude, method="nearest")
            .mean(dim="frame")
        )
        data_e = (
            ds_fit[f"e_state_{leakage_key}"]
            .sel(qubit_pair=qp_name, control_axis=1)
            .sel(amp=fr.optimal_amplitude, method="nearest")
            .mean(dim="frame")
        )
        data_f = (
            ds_fit[f"f_state_{leakage_key}"]
            .sel(qubit_pair=qp_name, control_axis=1)
            .sel(amp=fr.optimal_amplitude, method="nearest")
            .mean(dim="frame")
        )
        ax.plot(n_ops, data_g, label="g", color="steelblue")
        ax.plot(n_ops, data_e, label="e", color="tomato")
        ax.plot(n_ops, data_f, label="f (leakage)", color="seagreen")
        ax.legend(fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"Plot failed:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)

    if lq is not None and mq is not None:
        ax.set_title(f"{qp_name} \n leakage qubit: {lq.name}")
    else:
        ax.set_title(qp_name)
    ax.set_xlabel("# CZ operations")
    ax.set_ylabel("Leakage qubit state fractions")
