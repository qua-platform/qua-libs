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
        qubit_role_obj = qubit_roles_map.get(qp_name) if qubit_roles_map else None
        plot_individual_data_with_fit(ax, ds_fit, qp_name, qp_map[qp_name], qubit_role_obj=qubit_role_obj)

    grid.fig.suptitle("CZ phase calibration — phase difference vs amplitude")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
    qubit_role_obj=None,
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

    mq = qubit_role_obj.moving if qubit_role_obj is not None else None

    if mq is not None:
        def amp_to_detuning_MHz(amp):
            return -(amp**2) * mq.freq_vs_flux_01_quad_term / 1e6

        def detuning_MHz_to_amp(detuning_MHz):
            return np.sqrt(-detuning_MHz * 1e6 / mq.freq_vs_flux_01_quad_term)

        secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
        secax.set_xlabel("Detuning (MHz)")

    ax.set_title(f"{qp_name}")
    ax.set_xlabel("Amplitude (V)")
    ax.set_ylabel("Phase difference")


def plot_leakage_qubit_populations(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
    qubit_roles_map: Optional[dict] = None,
) -> Figure:
    """Plot leakage-qubit g/e/f populations vs amplitude for every pair on a chip-topology grid.

    The leakage qubit (always the higher-frequency qubit, whose |1⟩→|2⟩ transition
    is resonant at the |11⟩↔|20⟩ crossing) is resolved via ``QubitRoles.resolve``
    (.leakage) from qubit frequencies and anharmonicity, so this works regardless of
    which qubit is moving.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fit dataset (must contain ``g/e/f_state_moving`` and ``g/e/f_state_stationary``,
        plus ``optimal_amplitude``).
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

    grid.fig.suptitle("CZ phase calibration — leakage qubit populations")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_leakage_qubit_populations(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
    qubit_role_obj=None,
):
    """Plot g/e/f populations of the leakage qubit vs amplitude for one qubit pair.

    Populations for both the moving and stationary qubits are stored in the dataset.
    This function selects the correct set (``_moving`` or ``_stationary``) by
    checking whether ``roles.leakage`` is the moving or stationary qubit.

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
        Used to resolve qubit roles and compute the secondary detuning axis.
    """
    fit = ds_fit.sel(qubit_pair=qp_name)

    roles = qubit_role_obj
    lq = roles.leakage if roles is not None else None
    mq = roles.moving if roles is not None else None
    leakage_key = "moving" if (roles is not None and roles.leakage is roles.moving) else "stationary"

    data_g = fit[f"g_state_{leakage_key}"].sel(control_axis=1).mean(dim="frame")
    data_e = fit[f"e_state_{leakage_key}"].sel(control_axis=1).mean(dim="frame")
    data_f = fit[f"f_state_{leakage_key}"].sel(control_axis=1).mean(dim="frame")

    amps = fit.amp_full.values if "amp_full" in fit.coords else fit.amp.values
    ax.plot(amps, data_g, label="g", color="steelblue")
    ax.plot(amps, data_e, label="e", color="tomato")
    ax.plot(amps, data_f, label="f (leakage)", color="seagreen")

    opt_amp = float(fit.optimal_amplitude)
    ax.axvline(opt_amp, color="red", ls="--", lw=0.8, label="optimal")
    ax.axhline(0.0, color="grey", ls=":", lw=0.5)
    ax.axhline(1.0, color="grey", ls=":", lw=0.5)
    ax.legend(fontsize=8)

    if mq is not None:
        def amp_to_detuning_MHz(amp):
            return -(amp**2) * mq.freq_vs_flux_01_quad_term / 1e6

        def detuning_MHz_to_amp(detuning_MHz):
            return np.sqrt(-detuning_MHz * 1e6 / mq.freq_vs_flux_01_quad_term)

        secax = ax.secondary_xaxis("top", functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
        secax.set_xlabel("Detuning (MHz)")

    if lq is not None and mq is not None:
        ax.set_title(f"{qp_name} \n leakage qubit: {lq.name}")
    else:
        ax.set_title(qp_name)
    ax.set_xlabel("Amplitude (V)")
    ax.set_ylabel("Leakage qubit population")
