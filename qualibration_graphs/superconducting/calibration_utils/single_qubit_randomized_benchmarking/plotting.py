from typing import List
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from calibration_utils.common_utils.leakage_rb import LeakageFit
from calibration_utils.single_qubit_randomized_benchmarking.analysis import format_leakage_rate

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Single qubit randomized benchmarking")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    # Fitted decay
    fitted = decay_exp(
        fit.depths,
        fit.fit_data.sel(fit_vals="a"),
        fit.fit_data.sel(fit_vals="offset"),
        fit.fit_data.sel(fit_vals="decay"),
    )
    if hasattr(fit, "pg"):  # measure_leakage: g/e/f readout
        data = fit.pg
        label = "P(g) (survival)"
    elif hasattr(fit, "state"):
        data = fit.state
        label = "P(g) (survival)"
    elif hasattr(fit, "I"):
        data = fit.I
        label = "I quadrature [mV]"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    data_std = data.std(dim="nb_of_sequences") / np.sqrt(ds.nb_of_sequences.size)
    ax.errorbar(
        fit.depths,
        fit.averaged_data,
        yerr=data_std,
        fmt=".",
        capsize=2,
        elinewidth=0.5,
    )
    ax.grid("all")
    ax.set_title(qubit["qubit"], pad=22)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel(label)
    ax.plot(fit.depths, fitted, "r--", label="fit")
    ax.text(
        0.15,
        0.9,
        f"1Q RB fidelity = {100*(1 - float(fit.error_per_gate.values)):.3f}%",
        transform=ax.transAxes,
    )


def plot_leakage(
    ds_leakage: xr.Dataset,
    qubits: List[AnyTransmon],
    leakage_results: dict,
    *,
    log_x: bool = False,
    reset_type: str = "",
) -> Figure:
    """Plot the leaked population P(f) against circuit depth with its fit, one panel per qubit.

    ``leakage_results`` is ``node.results["leakage_results"]``: the ``asdict`` of each qubit's
    ``LeakageResult``.
    """
    grid = QubitGrid(ds_leakage, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        result = leakage_results[name]
        fit = LeakageFit(**result["per_clifford"])
        fr = ds_leakage.sel(qubit=name)
        depths = fr.depths.values
        line = ax.errorbar(
            depths,
            fr.leaked.values,
            yerr=np.nan_to_num(fr.leaked_stderr.values),
            fmt="o",
            ms=4,
            capsize=2,
            label=f"L1 {format_leakage_rate(fit.l1, fit.l1_stderr)}",
        )
        if fit.success:
            dense = (np.geomspace if log_x else np.linspace)(depths.min(), depths.max(), 200)
            ax.plot(dense, fit.curve(dense), "-", color=line[0].get_color(), lw=1)
        subtitle = "\nFAILED: g/e/f readout suspect (not saved)" if result["failure"] else ""
        ax.set_title(f"{name}{subtitle}", fontsize=9)
        ax.set_xlabel("Circuit depth (Cliffords)")
        ax.set_ylabel("Leaked population P(f)")
        if log_x:
            ax.set_xscale("log")
        ax.legend(fontsize=7)

    reset = f"{reset_type} reset, " if reset_type else ""
    grid.fig.suptitle(f"Single qubit leakage randomized benchmarking\n({reset}g/e/f readout)")
    grid.fig.tight_layout()
    return grid.fig
