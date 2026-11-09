from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

__all__ = ["plot_raw_data_with_fit", "plot_stark_shift_with_fit", "plot_photon_number_with_fit"]

# The curve for each prepared state, keyed by the state the qubit was prepared in
STATE_COLORS = {"g": "C0", "e": "C3"}
STATE_LABELS = {"g": "|0>", "e": "|1>"}


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, initial_state: str) -> Figure:
    """
    Plots the 2D resonator-qubit spectroscopy map for one prepared state, with the extracted ac Stark shift and
    its Lorentzian fit overlaid. This is the equivalent of panel (b) or (c) of Fig. 2 in
    https://link.aps.org/doi/10.1103/PhysRevApplied.23.024055.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the processed data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    initial_state : str
        The prepared state whose panel should be plotted, "g" or "e".

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    ds_state = ds.sel(initial_state=initial_state)
    fits_state = fits.sel(initial_state=initial_state)
    grid = QubitGrid(ds_state, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_raw_data_with_fit(ax, ds_state, qubit, fits_state.loc[qubit], initial_state)

    grid.fig.suptitle(
        f"2D resonator-qubit spectroscopy, qubit prepared in {STATE_LABELS.get(initial_state, initial_state)}"
    )
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_raw_data_with_fit(
    ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset, initial_state: str
):
    """Plots the 2D map of a single qubit on the given axis, with the ac Stark shift and its fit on top."""
    ds_qubit = ds.loc[qubit].assign_coords(
        resonator_detuning_MHz=ds.resonator_detuning / u.MHz,
        qubit_detuning_MHz=ds.qubit_detuning / u.MHz,
    )
    ds_qubit.flip_signal.plot(
        ax=ax,
        add_colorbar=False,
        x="resonator_detuning_MHz",
        y="qubit_detuning_MHz",
        robust=True,
    )
    chi = None
    if fit is not None:
        if "chi" in fit and np.isfinite(fit.chi.values) and fit.chi.values != 0:
            chi = float(fit.chi.values)
        # The ac Stark shift extracted at each resonator drive frequency, used as a sensor of the photon number
        ax.plot(
            fit.resonator_detuning / u.MHz,
            fit.stark_shift / u.MHz,
            ".",
            color="w",
            markersize=3,
            label="ac Stark shift",
        )
        if bool(fit.success.values):
            ax.plot(
                fit.resonator_detuning / u.MHz,
                fit.stark_shift_fitted / u.MHz,
                "-",
                color=STATE_COLORS.get(initial_state, "r"),
                linewidth=1.5,
                label="fit",
            )
            ax.axvline(
                float(fit.fit_results.sel(fit_vals="center").values) / u.MHz,
                linestyle="dashed",
                linewidth=1,
                color=STATE_COLORS.get(initial_state, "r"),
            )
        ax.legend(loc="lower left", fontsize=7)

    title = qubit["qubit"]
    if fit is not None and bool(fit.success.values) and chi is not None and chi != 0:
        offset = float(fit.fit_results.sel(fit_vals="offset").values)
        depth = float(fit.fit_results.sel(fit_vals="depth").values)
        n_bar_max = depth / (2 * chi)
        title += f"\npeak " + r"$\bar{n}$" + f" = {n_bar_max:.1f}"

        # Secondary y-axis on the right mapping qubit detuning directly to mean photon number n_bar
        # from stark_shift = offset + 2 * chi * n_bar (Eq. 6 & 7)
        secax = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y_mhz: (y_mhz * u.MHz - offset) / (2 * chi),
                lambda n: (n * 2 * chi + offset) / u.MHz,
            ),
        )
        secax.set_ylabel(r"Mean photon number $\bar{n}$")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Resonator drive detuning (MHz)")
    ax.set_ylabel("Qubit pulse detuning (MHz)")


def plot_stark_shift_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """
    Plots the extracted ac Stark shift versus the resonator drive frequency for every prepared state, together
    with the fitted Lorentzians. This is the equivalent of panel (d) of Fig. 2 in
    https://link.aps.org/doi/10.1103/PhysRevApplied.23.024055: the horizontal distance between the two means is
    2 * chi and their common width is kappa.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the processed data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds.isel(initial_state=0), [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_stark_shift_with_fit(ax, qubit, fits.loc[qubit])

    grid.fig.suptitle("ac Stark shift vs resonator drive frequency")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_stark_shift_with_fit(ax: Axes, qubit: dict[str, str], fit: xr.Dataset):
    """Plots both ac Stark shift Lorentzians of a single qubit on the given axis."""
    centers = {}
    depths = {}
    for state in [str(s) for s in fit.initial_state.values]:
        sel = fit.sel(initial_state=state)
        color = STATE_COLORS.get(state, "k")
        label = STATE_LABELS.get(state, state)
        ax.plot(sel.resonator_detuning / u.MHz, sel.stark_shift / u.MHz, ".", color=color, label=label)
        if bool(sel.success.values):
            ax.plot(sel.resonator_detuning / u.MHz, sel.stark_shift_fitted / u.MHz, "-", color=color, linewidth=1.5)
            centers[state] = float(sel.fit_results.sel(fit_vals="center").values)
            depths[state] = float(sel.fit_results.sel(fit_vals="depth").values)
            ax.axvline(centers[state] / u.MHz, linestyle="dashed", linewidth=1, color=color)

    title = qubit["qubit"]
    chi = None
    if "chi" in fit and np.isfinite(fit.chi.values) and fit.chi.values != 0:
        chi = float(fit.chi.values)
    elif "g" in centers and "e" in centers:
        # chi is half the horizontal distance between the two means, kappa their common width
        chi = 0.5 * (centers["e"] - centers["g"])

    if chi is not None and chi != 0:
        kappa = float(fit.fit_results.sel(fit_vals="kappa").mean().values)
        if "n_bar_max" in fit and np.isfinite(fit.n_bar_max.values).any():
            valid_n_bars = fit.n_bar_max.values[np.isfinite(fit.n_bar_max.values)]
            n_bar_max = float(np.mean(valid_n_bars)) if len(valid_n_bars) > 0 else None
        elif depths:
            n_bar_max = float(np.mean([d / (2 * chi) for d in depths.values()]))
        else:
            n_bar_max = None

        if n_bar_max is not None:
            drive_amp_sq = n_bar_max * kappa / 4
            title += (
                f"\nchi = {chi / u.MHz:.3f} MHz, kappa = {kappa / u.MHz:.3f} MHz, "
                + r"$\bar{n}$"
                + f" = {n_bar_max:.1f}, |A|^2 = {drive_amp_sq:.2e} Hz"
            )
        else:
            title += f"\nchi = {chi / u.MHz:.3f} MHz, kappa = {kappa / u.MHz:.3f} MHz"

        # Secondary y-axis on the right mapping ac Stark shift (MHz) to mean photon number n_bar
        # from stark_shift = 2 * chi * n_bar (Eq. 6 & 7)
        secax = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y_mhz: (y_mhz * u.MHz) / (2 * chi),
                lambda n: (n * 2 * chi) / u.MHz,
            ),
        )
        secax.set_ylabel(r"Mean photon number $\bar{n}$")
    elif "g" in centers and "e" in centers:
        chi = 0.5 * (centers["e"] - centers["g"])
        kappa = float(fit.fit_results.sel(fit_vals="kappa").mean().values)
        title += f"\nchi = {chi / u.MHz:.3f} MHz, kappa = {kappa / u.MHz:.3f} MHz"

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Resonator drive detuning (MHz)")
    ax.set_ylabel("ac Stark shift (MHz)")
    ax.legend(loc="best", fontsize=7)


def plot_photon_number_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """
    Plots the mean photon number n_bar versus resonator drive detuning for every prepared state, together
    with the fitted curves from Eq. (6) of https://link.aps.org/doi/10.1103/PhysRevApplied.23.024055:
        n_bar = (4 * |A|^2 / kappa) / (1 + 4 * delta^2 / kappa^2).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the processed data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds.isel(initial_state=0), [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_photon_number_with_fit(ax, qubit, fits.loc[qubit])

    grid.fig.suptitle(r"Mean photon number $\bar{n}$ vs resonator drive frequency")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_photon_number_with_fit(ax: Axes, qubit: dict[str, str], fit: xr.Dataset):
    """Plots both mean photon number curves of a single qubit on the given axis."""
    centers = {}
    depths = {}
    chi = None
    if "chi" in fit and np.isfinite(fit.chi.values) and fit.chi.values != 0:
        chi = float(fit.chi.values)

    for state in [str(s) for s in fit.initial_state.values]:
        sel = fit.sel(initial_state=state)
        if bool(sel.success.values):
            centers[state] = float(sel.fit_results.sel(fit_vals="center").values)
            depths[state] = float(sel.fit_results.sel(fit_vals="depth").values)

    if chi is None and "g" in centers and "e" in centers:
        chi = 0.5 * (centers["e"] - centers["g"])

    for state in [str(s) for s in fit.initial_state.values]:
        sel = fit.sel(initial_state=state)
        color = STATE_COLORS.get(state, "k")
        label = STATE_LABELS.get(state, state)
        if chi is not None and chi != 0:
            if "n_bar" in sel and np.isfinite(sel.n_bar.values).any():
                n_bar_data = sel.n_bar
                n_bar_curve = sel.n_bar_fitted if "n_bar_fitted" in sel else None
            else:
                offset = float(sel.fit_results.sel(fit_vals="offset").values) if bool(sel.success.values) else 0.0
                n_bar_data = (sel.stark_shift - offset) / (2 * chi)
                n_bar_curve = (sel.stark_shift_fitted - offset) / (2 * chi) if bool(sel.success.values) else None

            ax.plot(sel.resonator_detuning / u.MHz, n_bar_data, ".", color=color, label=label)
            if bool(sel.success.values) and n_bar_curve is not None:
                ax.plot(sel.resonator_detuning / u.MHz, n_bar_curve, "-", color=color, linewidth=1.5)
                ax.axvline(centers[state] / u.MHz, linestyle="dashed", linewidth=1, color=color)

    title = qubit["qubit"]
    if chi is not None and chi != 0 and depths:
        kappa = float(fit.fit_results.sel(fit_vals="kappa").mean().values)
        if "n_bar_max" in fit and np.isfinite(fit.n_bar_max.values).any():
            valid_n_bars = fit.n_bar_max.values[np.isfinite(fit.n_bar_max.values)]
            n_bar_max = float(np.mean(valid_n_bars)) if len(valid_n_bars) > 0 else None
        else:
            n_bar_max = float(np.mean([d / (2 * chi) for d in depths.values()]))

        if n_bar_max is not None:
            drive_amp_sq = n_bar_max * kappa / 4
            title += (
                f"\npeak "
                + r"$\bar{n}$"
                + f" = {n_bar_max:.1f}, |A|^2 = {drive_amp_sq:.2e} Hz"
                f"\n(chi = {chi / u.MHz:.3f} MHz, kappa = {kappa / u.MHz:.3f} MHz)"
            )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Resonator drive detuning (MHz)")
    ax.set_ylabel(r"Mean photon number $\bar{n}$")
    ax.legend(loc="best", fontsize=7)
