import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.cz_iswap_flux_bootstrap.parameters import moving_qubit
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
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
        plot_individual_data_with_fit(ax, ds_fit, qp_name, qp_map[qp_name])

    grid.fig.suptitle("CZ conditional phase error amplification \n phase difference")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
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

    quad = moving_qubit(qp).freq_vs_flux_01_quad_term

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


def plot_moving_qubit_populations(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
) -> Figure:
    """Plot moving-qubit g/f populations vs # operations for every pair on a chip-topology grid.

    Populations are shown at the optimal amplitude for each pair.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fit dataset containing ``g_state_moving``, ``f_state_moving``,
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
        plot_individual_moving_qubit_populations(ax, ds_fit, qp_name, qp_map[qp_name])

    grid.fig.suptitle("CZ conditional phase error amplification \n moving qubit populations")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_moving_qubit_populations(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    qp,
):
    """Plot moving-qubit g/f populations vs # operations for one pair at its optimal amplitude.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds_fit : xr.Dataset
        Fit dataset containing ``g_state_moving``, ``f_state_moving``,
        ``optimal_amplitude``.
    qp_name : str
        Qubit pair name used to select data.
    qp : qubit pair object
        Used to label the moving qubit.
    """
    fr = ds_fit.sel(qubit_pair=qp_name)
    n_ops = fr.number_of_operations.values if "number_of_operations" in fr.dims else None

    try:
        data_g = (
            ds_fit.g_state_moving.sel(qubit_pair=qp_name, control_axis=1)
            .sel(amp=fr.optimal_amplitude, method="nearest")
            .mean(dim="frame")
        )
        data_f = (
            ds_fit.f_state_moving.sel(qubit_pair=qp_name, control_axis=1)
            .sel(amp=fr.optimal_amplitude, method="nearest")
            .mean(dim="frame")
        )
        ax.plot(n_ops, data_g, label="g", color="steelblue")
        ax.plot(n_ops, data_f, label="f (leakage)", color="seagreen")
        ax.legend(fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"Plot failed:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)

    mq = moving_qubit(qp)
    ax.set_title(f"{qp_name} — moving qubit: {mq.name}")
    ax.set_xlabel("# CZ operations")
    ax.set_ylabel("Moving qubit state fractions")
