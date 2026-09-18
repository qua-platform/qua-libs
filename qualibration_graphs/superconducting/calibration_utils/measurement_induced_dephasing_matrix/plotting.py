"""Plotting utilities for the measurement-induced dephasing matrix experiment.

The figures reproduce and extend Fig. 6 of Phys. Rev. Applied 23, 054089 (arXiv:2412.14853):
  * :func:`plot_contrast_with_fit` is panel (a), the relative echo contrast versus the relative
    readout amplitude;
  * :func:`plot_dephasing_matrix` is panel (b), the dephasing-rate colour map;
  * :func:`plot_stark_shift_matrix` is the same map for the AC-Stark shift, which is the other
    observable the swept-phase echo gives. It only exists when the probe was played in the first
    half of the echo alone;
  * :func:`plot_phase_oscillations` is the underlying raw data, used for debugging.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _decay_model(xi: np.ndarray, c0: float, gamma: float, tau_p: float) -> np.ndarray:
    """The fitted contrast decay c(xi) = c0 * exp(-Gamma * tau_p * xi**2)."""
    return c0 * np.exp(-gamma * tau_p * xi**2)


def _plot_pair(ax, ds_fit: xr.Dataset, qubit_name: str, driven: str, label=None, colour=None, marker_size=4):
    """Plot the relative contrast of one (measured qubit, driven resonator) pair with its fit.

    Returns the colour used, so that the same qubit keeps the same colour in the inset.
    """
    pair = ds_fit.sel(qubit=qubit_name, driven_resonator=driven)
    xi = pair.xi.values
    line = ax.plot(xi, pair.contrast_relative.values, "o", ms=marker_size, label=label, color=colour)[0]

    gamma = float(pair.Gamma)
    if np.isfinite(gamma):
        xi_dense = np.linspace(0, xi.max(), 201)
        ax.plot(
            xi_dense,
            _decay_model(xi_dense, 1.0, gamma, float(pair.tau_p)),
            "--",
            color=line.get_color(),
            lw=1,
        )
    return line.get_color()


def plot_contrast_with_fit(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """Plot the relative echo contrast c/c0 against the relative readout amplitude xi.

    One panel per driven resonator, one curve per measured qubit, with the fitted exponential
    overlaid. The vertical axis is logarithmic because the diagonal element decays by orders of
    magnitude over the swept range.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``.
    qubits : list of AnyTransmon
        The measured qubits, used only for their names.

    Returns
    -------
    Figure
        The matplotlib figure containing the plots.
    """
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    num_panels = len(driven_names)
    num_columns = min(num_panels, 2)
    num_rows = int(np.ceil(num_panels / num_columns))
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(7 * num_columns, 4.5 * num_rows), squeeze=False)

    for panel_index, driven in enumerate(driven_names):
        ax = axes.flat[panel_index]
        colours = {}
        for qubit in qubits:
            colours[qubit.name] = _plot_pair(ax, ds_fit, qubit.name, driven, label=qubit.name)
        ax.set_yscale("log")
        ax.set_xlabel("Relative readout amplitude $\\xi$")
        ax.set_ylabel("Relative echo contrast $c/c_0$")
        ax.set_title(f"Driving {driven}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # The diagonal element decays by orders of magnitude while the crosstalk barely moves, so on
        # the shared logarithmic axis the off-diagonal curves collapse onto c/c0 = 1. They are
        # repeated in an inset on a linear, auto-scaled axis, as in the inset of Fig. 6(a).
        off_diagonal_qubits = [qubit for qubit in qubits if qubit.name != driven]
        if not off_diagonal_qubits:
            continue
        inset = ax.inset_axes([0.45, 0.12, 0.5, 0.42])
        for qubit in off_diagonal_qubits:
            _plot_pair(inset, ds_fit, qubit.name, driven, colour=colours[qubit.name], marker_size=3)
        inset.set_title("off-diagonal only", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(alpha=0.3)

    for empty_index in range(num_panels, axes.size):
        axes.flat[empty_index].axis("off")

    fig.suptitle("Measurement-induced dephasing: echo contrast vs readout amplitude")
    fig.tight_layout()
    return fig


def _readable_text_colour(image, value: float) -> str:
    """Return black or white, whichever stays legible on the cell colour holding ``value``.

    Both colour maps run from a pale low end to a dark high end, so a single fixed text colour is
    unreadable at one end of the scale. The choice is made on the perceived luminance of the cell.
    """
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "k" if luminance > 0.55 else "w"


def _cell_label(value: float, error: float, resolved: bool, reason: str, scale: float, resolved_sigma: float) -> str:
    """Build the text of one matrix cell.

    A resolved cell is labelled with its value and its error. An unresolved cell is labelled with the
    upper bound on its magnitude instead, because its fitted value is consistent with zero and
    printing it as a number invites reading noise as a measurement. A rejected cell keeps its value
    for inspection and names what it failed, so that the figure answers on its own why an element was
    thrown out.
    """
    if not np.isfinite(value):
        return "no fit"
    if reason:
        return f"{value * scale:.3g}\n{reason}"
    if resolved:
        return f"{value * scale:.3g}\n$\\pm${error * scale:.2g}"
    return f"< {(abs(value) + resolved_sigma * error) * scale:.2g}"


def _rejection_reasons(chi2: np.ndarray, max_reduced_chi2: float, residual=None, max_residual=None) -> np.ndarray:
    """Return the short reason each element was rejected, or an empty string when it was kept.

    Naming the failed check on the figure saves going back to the dataset to find out whether an
    element was dropped because its fit does not describe the data or because its phase could not be
    unwrapped, which call for completely different responses.
    """
    reasons = np.full(chi2.shape, "", dtype=object)
    for index in np.ndindex(chi2.shape):
        if residual is not None and np.isfinite(residual[index]) and residual[index] > max_residual:
            reasons[index] = f"unwrap {residual[index]:.2f} rad"
        elif np.isfinite(chi2[index]) and chi2[index] > max_reduced_chi2:
            reasons[index] = f"$\\chi^2_\\nu$ = {chi2[index]:.1f}"
    return reasons


def _cell_labels(values, errors, resolved, reasons, is_diagonal, diagonal_scale, resolved_sigma):
    """Build the per-cell text of a whole matrix, scaling the diagonal onto its own units."""
    labels = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        labels[index] = _cell_label(
            values[index],
            errors[index],
            bool(resolved[index]),
            reasons[index],
            diagonal_scale if is_diagonal[index] else 1.0,
            resolved_sigma,
        )
    return labels


def _matrix_figure(
    qubit_names: List[str],
    driven_names: List[str],
    off_values: np.ndarray,
    diagonal_values: np.ndarray,
    labels: np.ndarray,
    rejected: np.ndarray,
    off_cmap: str,
    off_label: str,
    diagonal_label: str,
    title: str,
    off_norm: Optional[Normalize] = None,
) -> Figure:
    """Draw a dephasing-style matrix with one colour scale for the diagonal and one for the rest.

    The self (diagonal) and the crosstalk (off-diagonal) elements differ by several orders of
    magnitude, so a single colour scale would saturate. Following Fig. 6(b) of Phys. Rev. Applied 23,
    054089, both are drawn on one set of axes as two superimposed layers, each masked to its own
    cells and carrying its own colour bar.

    Cells whose value is masked in ``off_values`` or ``diagonal_values`` are drawn transparent. That
    is used to leave unresolved elements uncoloured, so the colour of the figure only ever reflects
    what was actually measured.
    """
    fig, ax = plt.subplots(figsize=(1.1 * len(driven_names) + 6, 1.1 * len(qubit_names) + 2.5))

    # The colour bar axes are placed explicitly rather than stacked automatically, so that the
    # off-diagonal scale - the actual crosstalk result - sits closest to the matrix and each bar has
    # room for its own label. The pad of the outer bar must clear the label of the inner one.
    divider = make_axes_locatable(ax)
    images = {}
    for key, values, cmap_name, label, norm, pad in (
        ("off", off_values, off_cmap, off_label, off_norm, 0.15),
        ("diagonal", diagonal_values, "Greys", diagonal_label, None, 1.05),
    ):
        # Masked cells are drawn fully transparent so the two layers do not hide one another.
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0.0)
        images[key] = ax.imshow(values, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        colour_axes = divider.append_axes("right", size="4%", pad=pad)
        fig.colorbar(images[key], cax=colour_axes, label=label)

    ax.set_xticks(range(len(driven_names)), driven_names)
    ax.set_yticks(range(len(qubit_names)), qubit_names)
    ax.set_xlabel("Resonator driven")
    ax.set_ylabel("Qubit measured")
    # Grid lines on the cell boundaries, to separate the two colour scales visually. A neutral grey
    # is used because both colour maps are pale at their low end.
    ax.set_xticks(np.arange(len(driven_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(qubit_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="0.6", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    for row in range(len(qubit_names)):
        for column in range(len(driven_names)):
            on_diagonal = row == column and qubit_names[row] == driven_names[column]
            values = diagonal_values if on_diagonal else off_values
            value = values[row, column]
            coloured = value is not np.ma.masked and np.isfinite(value)
            ax.text(
                column,
                row,
                labels[row, column],
                ha="center",
                va="center",
                color=_readable_text_colour(images["diagonal" if on_diagonal else "off"], value) if coloured else "k",
                fontsize=7.5,
            )
            if rejected[row, column]:
                ax.add_patch(
                    Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, hatch="///", edgecolor="0.45", linewidth=0)
                )

    ax.set_title(title)
    fig.tight_layout()
    return fig


def _matrix_inputs(ds_fit: xr.Dataset, prefix: str):
    """Return the value, error, resolved and rejected matrices of one observable as plain arrays."""
    order = ("qubit", "driven_resonator")
    is_diagonal = ds_fit.is_diagonal.transpose(*order).values
    values = ds_fit[prefix].transpose(*order).values
    errors = ds_fit[f"{prefix}_error"].transpose(*order).values
    resolved = ds_fit[f"{prefix}_resolved"].transpose(*order).values.astype(bool)
    rejected = ds_fit[f"{prefix}_rejected"].transpose(*order).values.astype(bool)
    # A rejected element is not a measurement, so it is not treated as resolved anywhere.
    return is_diagonal, values, errors, resolved & ~rejected, rejected


def plot_dephasing_matrix(ds_fit: xr.Dataset, node=None, resolved_sigma: float = 2.0) -> Figure:
    """Plot the measurement-induced dephasing rates as a matrix with two colour bars.

    Only elements that were resolved, meaning larger than ``resolved_sigma`` times their own standard
    error, are given a colour. An element consistent with zero is left uncoloured and labelled with
    the upper bound on its magnitude, and an element whose straight-line fit was rejected is hatched.
    The colour of the figure therefore shows the crosstalk that was actually measured, rather than
    the noise of the elements that were not.

    The off-diagonal is coloured by magnitude: the sign of a dephasing rate only distinguishes a
    physical result from an unphysical one, and an unphysical rate large enough to be resolved is a
    problem worth seeing whichever way it points.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``.
    node : QualibrationNode, optional
        Node whose parameters hold the thresholds, used only to name on each rejected cell which
        check it failed. Without it a rejected cell is still hatched but says nothing.
    resolved_sigma : float
        Number of standard errors an element must exceed to be shown as measured. Default is 2.

    Returns
    -------
    Figure
        The matplotlib figure containing the dephasing matrix.
    """
    qubit_names = [str(name) for name in ds_fit.qubit.values]
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    is_diagonal, gamma, gamma_error, resolved, rejected = _matrix_inputs(ds_fit, "Gamma")

    show_off = ~is_diagonal & resolved
    off_values = np.ma.masked_invalid(np.where(show_off, np.abs(gamma), np.nan))
    diagonal_values = np.ma.masked_invalid(np.where(is_diagonal, gamma * 1e-6, np.nan))

    # Anchor the magnitude scale at zero. Left to auto-scale over the resolved cells alone it would
    # spread its full colour range over whatever narrow band those happen to span, which makes two
    # similar rates look wildly different.
    off_norm = Normalize(vmin=0.0, vmax=float(off_values.max()) if off_values.count() else 1.0)

    reasons = _rejection_reasons(
        ds_fit.Gamma_chi2.transpose("qubit", "driven_resonator").values,
        node.parameters.max_reduced_chi2 if node is not None else np.inf,
    )
    labels = _cell_labels(gamma, gamma_error, resolved, reasons, is_diagonal, 1e-6, resolved_sigma)

    return _matrix_figure(
        qubit_names,
        driven_names,
        off_values,
        diagonal_values,
        labels,
        rejected,
        off_cmap="Reds",
        off_label="Off-diagonal dephasing rate (Hz)",
        diagonal_label="Diagonal dephasing rate (MHz)",
        title="Measurement-induced dephasing matrix",
        off_norm=off_norm,
    )


def plot_stark_shift_matrix(ds_fit: xr.Dataset, node=None, resolved_sigma: float = 2.0) -> Figure:
    """Plot the AC-Stark shift matrix, the coherent counterpart of the dephasing matrix.

    Each element is the frequency shift the measured qubit acquires while its neighbour's resonator
    is probed at the calibrated readout amplitude. Because that shift is linear in the cross-Kerr
    coupling between the qubit and the driven resonator, while the dephasing rate is quadratic in it,
    this matrix resolves crosstalk on pairs whose dephasing sits below the noise floor.

    The sign is physical here, so the off-diagonal uses a diverging colour scale centred on zero.
    Unresolved and rejected cells are marked as in :func:`plot_dephasing_matrix`. The diagonal is
    routinely rejected, because the self-Stark shift is large enough that the phase wraps between
    consecutive amplitude points and cannot be unwrapped; the off-diagonal, which is the result this
    figure exists for, is unaffected.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``. It must carry the Stark-shift variables,
        which are absent when the probe was played in both halves of the echo.
    node : QualibrationNode, optional
        Node whose parameters hold the thresholds, used only to name on each rejected cell which
        check it failed. Without it a rejected cell is still hatched but says nothing.
    resolved_sigma : float
        Number of standard errors an element must exceed to be shown as measured. Default is 2.

    Returns
    -------
    Figure
        The matplotlib figure containing the Stark shift matrix.
    """
    qubit_names = [str(name) for name in ds_fit.qubit.values]
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    is_diagonal, shift, shift_error, resolved, rejected = _matrix_inputs(ds_fit, "stark_shift")

    show_off = ~is_diagonal & resolved
    off_values = np.ma.masked_invalid(np.where(show_off, shift, np.nan))
    diagonal_values = np.ma.masked_invalid(np.where(is_diagonal, shift * 1e-6, np.nan))

    # Centre the diverging scale on zero so that the sign of the shift is readable from the colour.
    largest = float(np.abs(off_values).max()) if off_values.count() else 1.0
    off_norm = Normalize(vmin=-largest, vmax=largest)

    order = ("qubit", "driven_resonator")
    reasons = _rejection_reasons(
        ds_fit.stark_shift_chi2.transpose(*order).values,
        node.parameters.max_reduced_chi2 if node is not None else np.inf,
        ds_fit.stark_shift_unwrap_residual.transpose(*order).values,
        node.parameters.max_phase_residual_in_rad if node is not None else np.inf,
    )
    labels = _cell_labels(shift, shift_error, resolved, reasons, is_diagonal, 1e-6, resolved_sigma)

    return _matrix_figure(
        qubit_names,
        driven_names,
        off_values,
        diagonal_values,
        labels,
        rejected,
        off_cmap="RdBu_r",
        off_label="Off-diagonal AC-Stark shift at $\\xi = 1$ (Hz)",
        diagonal_label="Diagonal AC-Stark shift at $\\xi = 1$ (MHz)",
        title="AC-Stark shift matrix",
        off_norm=off_norm,
    )


def _plot_stark_pair(ax, ds_fit: xr.Dataset, qubit_name: str, driven: str, label=None, colour=None):
    """Plot the unwrapped Stark phase of one pair against xi**2, with its fitted straight line.

    The unwrapping repeats what the fit did, so that the points shown are the points fitted. Returns
    the colour used, so that the same qubit keeps the same colour in the inset.
    """
    from .analysis import _progressive_unwrap

    pair = ds_fit.sel(qubit=qubit_name, driven_resonator=driven)
    # The same noise-floor cut as the fit, so the plot shows the points the fit actually used.
    usable = (pair.contrast > pair.contrast_error).values
    xi_squared = pair.xi.values[usable] ** 2
    if xi_squared.size < 2:
        return colour
    order = np.argsort(xi_squared)
    xi_squared = xi_squared[order]
    phase, _ = _progressive_unwrap(xi_squared, pair.oscillation_phase.values[usable][order])
    line = ax.plot(xi_squared, phase, "o", ms=4, label=label, color=colour)[0]

    shift = float(pair.stark_shift)
    if np.isfinite(shift):
        slope = shift * 2 * np.pi * float(pair.tau_p)
        intercept = np.mean(phase - slope * xi_squared)
        ax.plot(xi_squared, intercept + slope * xi_squared, "--", lw=1, color=line.get_color())
    return line.get_color()


def plot_stark_phase(ds_fit: xr.Dataset) -> Figure:
    """Plot the unwrapped AC-Stark phase against the squared readout amplitude, with its fit.

    This is the figure behind the Stark shift matrix, the counterpart of
    :func:`plot_contrast_with_fit`: a straight line here is the evidence that the phase shift really
    is proportional to the photon number, and curvature is the evidence that the driven resonator has
    left its linear range.

    The diagonal is swept over a much smaller amplitude range than the off-diagonal, so on a shared
    axis its points pile up against zero while its far larger phase stretches the vertical scale and
    flattens every other curve. It is therefore drawn in its own inset, auto-scaled to its own range.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``, carrying the Stark-shift variables.

    Returns
    -------
    Figure
        The matplotlib figure containing one panel per driven resonator.
    """
    qubit_names = [str(name) for name in ds_fit.qubit.values]
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    num_columns = min(len(driven_names), 2)
    num_rows = int(np.ceil(len(driven_names) / num_columns))
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(7 * num_columns, 4.5 * num_rows), squeeze=False)

    for panel_index, driven in enumerate(driven_names):
        ax = axes.flat[panel_index]
        colours = {}
        for qubit_name in qubit_names:
            if qubit_name == driven:
                continue
            colours[qubit_name] = _plot_stark_pair(ax, ds_fit, qubit_name, driven, label=qubit_name)
        ax.set_xlabel("Squared relative readout amplitude $\\xi^2$")
        ax.set_ylabel("Unwrapped echo phase [rad]")
        ax.set_title(f"Driving {driven}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        if driven in qubit_names:
            inset = ax.inset_axes([0.14, 0.62, 0.34, 0.33])
            _plot_stark_pair(inset, ds_fit, driven, driven)
            inset.set_title(f"{driven} itself", fontsize=7)
            inset.tick_params(labelsize=6)
            inset.grid(alpha=0.3)

    for empty_index in range(len(driven_names), axes.size):
        axes.flat[empty_index].axis("off")

    fig.suptitle("AC-Stark phase vs squared readout amplitude")
    fig.tight_layout()
    return fig


def plot_phase_oscillations(ds_fit: xr.Dataset, node) -> Figure:
    """Plot every phase oscillation together with its fitted sinusoid.

    This is the raw data behind the contrast and phase extraction: one panel per (measured qubit,
    driven resonator) pair, each holding the oscillations for all swept amplitudes. It is the figure
    to look at when a fit misbehaves. The drawn sinusoids use the biased contrast, which is the
    amplitude that was actually fitted to these points; the debiased one that feeds the dephasing fit
    would sit below the data at the noise floor.

    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset produced by ``fit_raw_data``.
    node : QualibrationNode
        Node whose parameters determine which signal was measured.

    Returns
    -------
    Figure
        The matplotlib figure containing the grid of oscillations.
    """
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    qubit_names = [str(name) for name in ds_fit.qubit.values]
    driven_names = [str(name) for name in ds_fit.driven_resonator.values]
    phases = ds_fit.phase.values
    phases_dense = np.linspace(phases.min(), phases.max(), 201)

    fig, axes = plt.subplots(
        len(qubit_names),
        len(driven_names),
        figsize=(3.2 * len(driven_names), 2.6 * len(qubit_names)),
        squeeze=False,
        sharex=True,
    )
    colours = plt.cm.viridis(np.linspace(0, 1, ds_fit.sizes["xi_idx"]))

    for row, qubit_name in enumerate(qubit_names):
        for column, driven in enumerate(driven_names):
            ax = axes[row][column]
            pair = ds_fit.sel(qubit=qubit_name, driven_resonator=driven)
            for xi_index in range(ds_fit.sizes["xi_idx"]):
                point = pair.isel(xi_idx=xi_index)
                ax.plot(phases, point[signal_name].values, ".", ms=3, color=colours[xi_index])
                fitted = float(point.oscillation_offset) + float(point.contrast_raw) * np.cos(
                    phases_dense - float(point.oscillation_phase)
                )
                ax.plot(phases_dense, fitted, "-", lw=0.8, color=colours[xi_index])
            ax.set_title(f"{qubit_name} | {driven}", fontsize=9)
            if row == len(qubit_names) - 1:
                ax.set_xlabel("Final pulse phase [rad]")
            if column == 0:
                ax.set_ylabel(signal_name)

    fig.suptitle("Echo phase oscillations (colour: relative readout amplitude)")
    fig.tight_layout()
    return fig
