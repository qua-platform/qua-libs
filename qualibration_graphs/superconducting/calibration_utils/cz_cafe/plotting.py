"""Plotting module for Context Aware Fidelity Estimation (CAFE) of a CZ gate."""

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

_VARIANT_COLORS = {"cafe": "C0", "decaf": "C1"}


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
    title_prefix: str = "CZ CAFE",
) -> Figure:
    """Plot the state-averaged return probability vs cycle count with fits and error budget.

    Each subplot shows one qubit pair: the data with binomial error bars, the Eq. (B9) fit
    (solid), the quadratic cross-check (dotted) and a text box with the error budget, for
    every measured variant.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds_fit, qubit["qubit"])
    grid.fig.suptitle(f"{title_prefix} — average gate fidelity vs cycle repetitions")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds_fit: xr.Dataset, qp_name: str) -> None:
    """Plot one qubit pair's CAFE curves and error budget."""
    fr = ds_fit.sel(qubit_pair=qp_name)
    lines = []
    for variant in fr.variant.values:
        variant = str(variant)
        color = _VARIANT_COLORS.get(variant, "C2")
        sel = fr.sel(variant=variant)
        ax.errorbar(
            sel.depth.values,
            sel.fidelity.values,
            yerr=sel.fidelity_std.values,
            fmt="o",
            ms=4,
            color=color,
            label=variant.upper(),
        )
        ax.plot(sel.depth_fine.values, sel.fit_curve.values, "-", color=color, lw=1.5)
        ax.plot(sel.depth_fine.values, sel.quadratic_curve.values, ":", color=color, lw=1.2)
        status = "" if bool(sel.fit_success) else " (fit failed)"
        lines.append(
            f"{variant.upper()}{status}: 1-F={1 - float(sel.fit_fidelity):.2e}\n"
            f"  incoh={float(sel.fit_incoherent_error):.2e}, coh={float(sel.fit_coherent_error):.2e}"
        )
    ax.text(
        0.03,
        0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_title(qp_name)
    ax.set_xlabel("Cycle repetitions n")
    ax.set_ylabel("Average gate fidelity")
    ax.legend(loc="upper right", fontsize=8)


def plot_leakage(ds_fit: xr.Dataset, qubit_pairs: list, title_prefix: str = "CZ CAFE") -> Figure:
    """Plot the state-averaged |f> population of both qubits vs cycle count."""
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(grid):
        fr = ds_fit.sel(qubit_pair=qubit["qubit"])
        for variant in fr.variant.values:
            sel = fr.sel(variant=variant)
            for name, marker in (("f_control", "o"), ("f_target", "s")):
                ax.plot(
                    sel.depth.values,
                    sel[name].mean(dim="state").values,
                    marker=marker,
                    ls="-",
                    ms=4,
                    label=f"{str(variant).upper()} {name.split('_')[1]}",
                )
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Cycle repetitions n")
        ax.set_ylabel(r"$P_{|f\rangle}$")
        ax.legend(fontsize=8)
    grid.fig.suptitle(f"{title_prefix} — leakage population vs cycle repetitions")
    grid.fig.tight_layout()
    return grid.fig
