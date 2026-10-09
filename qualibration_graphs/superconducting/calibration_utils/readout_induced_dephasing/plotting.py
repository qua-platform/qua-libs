"""Plotting helpers for cross-Ramsey readout-induced dephasing."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


def plot_results(ds_fit: xr.Dataset) -> dict[str, object]:
    """Plot representative Ramsey fringes, phase curves, and crosstalk matrices."""
    aggressors = list(map(str, ds_fit.aggressor_qubit.values))
    victims = list(map(str, ds_fit.qubit.values))
    pairs = [(a, v) for a in aggressors for v in victims if a != v]
    fig, axes = plt.subplots(max(1, len(pairs)), 2, squeeze=False, figsize=(12, max(3.5, 3.5 * len(pairs))))
    frame_degrees = 360 * ds_fit.frame.values
    cut_targets = np.linspace(float(ds_fit.amp_factor.min()), float(ds_fit.amp_factor.max()), 4)
    cut_colors = ("#2C6EAA", "#E17C05", "#4C956C", "#8F5DA2")
    if "state" in ds_fit:
        signal_name, signal_scale, signal_label = "state", 1.0, "Excited-state probability"
    elif "I" in ds_fit:
        signal_name, signal_scale, signal_label = "I", 1e3, "Rotated I (mV)"
    else:
        raise ValueError("Fitted dataset must contain either 'state' or 'I'")

    for row, (aggressor, victim) in enumerate(pairs):
        selection = dict(aggressor_qubit=aggressor, qubit=victim)
        signal = signal_scale * ds_fit[signal_name].sel(**selection)
        fringe_fit = signal_scale * ds_fit.fringe_fit.sel(**selection)
        for target, color in zip(cut_targets, cut_colors):
            raw_cut = signal.sel(amp_factor=target, method="nearest")
            fitted_cut = fringe_fit.sel(amp_factor=target, method="nearest")
            actual_amplitude = float(raw_cut.amp_factor)
            axes[row, 0].plot(
                frame_degrees,
                raw_cut,
                "o",
                ms=4,
                color=color,
                label=rf"$\xi={actual_amplitude:.1f}$ data",
            )
            axes[row, 0].plot(frame_degrees, fitted_cut, "-", lw=2, color=color)
        axes[row, 0].set(
            ylabel=signal_label,
            xlabel=r"Second $\pi/2$ analysis phase $\theta$ (deg)",
            xlim=(0, 360),
            xticks=np.arange(0, 361, 60),
            title=f"Selected Ramsey line cuts ({aggressor} → {victim})",
        )
        axes[row, 0].grid(alpha=0.22)
        axes[row, 0].legend(ncol=2, fontsize=8, loc="best")

        axes[row, 1].plot(ds_fit.amp_factor, np.rad2deg(ds_fit.fringe_phase_rad.sel(selection)), "o")
        axes[row, 1].plot(ds_fit.amp_factor, np.rad2deg(ds_fit.quadratic_phase_fit_rad.sel(selection)))
        axes[row, 1].set(
            xlabel="Readout amplitude factor",
            ylabel="Ramsey phase (deg)",
            title=f"Ramsey phase ({aggressor} → {victim})",
        )
        axes[row, 1].grid(alpha=0.22)
    fig.tight_layout()

    nominal = ds_fit.sel(amp_factor=float(ds_fit.attrs.get("report_amp_factor", 1.0)), method="nearest")
    fig_matrix, matrix_axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    matrix_specs = [
        (
            100 * nominal.phase_flip_probability,
            r"Phase-flip probability $P_\phi$ (%)",
            lambda value: f"{value:.4f}%",
        ),
        (
            nominal.coherent_rotation_deg,
            r"Coherent Z rotation $\Delta\phi$ (deg)",
            lambda value: f"{value:.1f}°",
        ),
    ]
    for ax, (matrix, colorbar_label, formatter) in zip(matrix_axes, matrix_specs):
        values = matrix.transpose("aggressor_qubit", "qubit").values
        finite = values[np.isfinite(values)]
        upper = float(np.nanmax(finite)) * 1.05 if finite.size else 1.0
        norm = Normalize(vmin=0.0, vmax=upper or 1.0)
        image = ax.imshow(np.ma.masked_invalid(values), cmap="RdPu", norm=norm, aspect="equal")
        ax.set_xticks(range(len(victims)), victims, rotation=45)
        ax.set_yticks(range(len(aggressors)), [f"RO {name}" for name in aggressors])
        ax.set(xlabel="Victim qubit Qj", ylabel="Readout line Qi")
        for (row, column), value in np.ndenumerate(values):
            if np.isfinite(value):
                text_color = "white" if float(norm(value)) > 0.62 else "black"
                ax.text(column, row, formatter(value), ha="center", va="center", color=text_color)
            else:
                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="#E5E5E5",
                        edgecolor="#B0B0B0",
                        linewidth=1.0,
                    )
                )
                ax.text(column, row, "N/A", ha="center", va="center", color="#666666")
        colorbar = fig_matrix.colorbar(image, ax=ax, location="top", fraction=0.055, pad=0.06)
        colorbar.set_label(colorbar_label, labelpad=8)
    fig_matrix.suptitle(
        "Configured readout amplitude "
        + rf"($\xi={float(ds_fit.attrs.get('report_amp_factor', 1.0)):.1f}$)"
        + "\nGray diagonal: self-readout is not measured"
    )
    return {"pair_fits": fig, "crosstalk_matrices": fig_matrix}
