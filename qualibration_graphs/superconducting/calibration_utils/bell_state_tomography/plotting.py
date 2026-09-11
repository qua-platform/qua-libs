"""Plotting module for Bell state tomography calibration."""

from typing import Callable, Dict, Iterable, Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

_IDEAL_BELL_RHO = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
_STATE_LABELS = ["00", "01", "10", "11"]
_MITIGATION_LABELS = {
    "kron": "Uncorrelated confusion matrix correction",
    "joint": "Correlated confusion matrix correction",
}
_FIDELITY_KEYS = {"kron": "fidelity_kron", "joint": "fidelity_joint"}
_PURITY_KEYS = {"kron": "purity_kron", "joint": "purity_joint"}


def plot_3d_hist_with_frame_real(rho: np.ndarray, ideal_dat: np.ndarray, ax: plt.Axes) -> None:
    """Plot the real part of the density matrix as a 3D city/bar chart."""
    _plot_3d_bars(ax, np.real(rho), vmin=-0.5, vmax=0.5)


def plot_3d_hist_with_frame_imag(rho: np.ndarray, ideal_dat: np.ndarray, ax: plt.Axes) -> None:
    """Plot the imaginary part of the density matrix as a 3D city/bar chart."""
    _plot_3d_bars(ax, np.imag(rho), vmin=-0.1, vmax=0.1)


def _plot_3d_bars(ax: plt.Axes, data: np.ndarray, vmin: float, vmax: float) -> None:
    """Create 3D bar plot for a 4x4 matrix."""
    dx = dy = 0.8
    x_pos = np.arange(4)
    y_pos = np.arange(4)
    xx, yy = np.meshgrid(x_pos, y_pos)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = data.flatten()

    z_base = np.where(zz >= 0, 0, zz)
    dz = np.abs(zz)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu")
    colors = cmap(norm(zz))

    ax.bar3d(xx, yy, z_base, dx, dy, dz, color=colors, shade=True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks(np.arange(4) + 0.4)
    ax.set_yticks(np.arange(4) + 0.4)
    ax.set_xticklabels(_STATE_LABELS)
    ax.set_yticklabels(_STATE_LABELS)


def _methods_to_plot(
    plot_level: Literal["full", "minimal"],
    fit_results: Mapping[str, Mapping[str, object]],
    rhos_by_method: Mapping[str, Mapping[str, np.ndarray]],
    pair_names: Iterable[str],
) -> tuple[str, ...]:
    """Return mitigation methods to plot for the requested plot level."""
    if plot_level == "full":
        return ("kron", "joint")

    for method_name in ("joint", "kron"):
        rhos = rhos_by_method.get(method_name, {})
        fidelity_key = _FIDELITY_KEYS[method_name]
        if any(
            pair_name in rhos and fit_results.get(pair_name, {}).get(fidelity_key) is not None
            for pair_name in pair_names
        ):
            return (method_name,)
    return ()


def _metric_fn(
    fit_results: Mapping[str, Mapping[str, object]],
    metric_key: str,
) -> Callable[[str], float]:
    """Build a per-pair metric accessor for one mitigation method."""

    def _lookup(qp_name: str) -> float:
        return float(fit_results.get(qp_name, {}).get(metric_key, 0.0))

    return _lookup


def plot_individual_3d_city(
    ax: Axes,
    qp_name: str,
    rhos: Dict[str, np.ndarray],
    plot_fn: Callable,
    fidelity_fn: Callable[[str], float],
    purity_fn: Callable[[str], float],
    *,
    show_fidelity: bool,
) -> None:
    """Plot one qubit-pair 3D density-matrix city chart."""
    if qp_name not in rhos:
        ax.text2D(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(qp_name)
        return

    plot_fn(rhos[qp_name], _IDEAL_BELL_RHO, ax)
    if show_fidelity:
        title = f"{qp_name}\nFidelity: {fidelity_fn(qp_name):.3f}, Purity: {purity_fn(qp_name):.3f}"
    else:
        title = f"{qp_name} - Imaginary"
    ax.set_title(title)


def plot_individual_rho(
    ax: Axes,
    qp_name: str,
    rhos: Dict[str, np.ndarray],
    *,
    is_real: bool,
    fidelity_fn: Callable[[str], float],
    purity_fn: Callable[[str], float],
) -> None:
    """Plot one qubit-pair 2D density matrix (real or imaginary part)."""
    if qp_name not in rhos:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(qp_name)
        return

    vmin, vmax = (-0.5, 0.5) if is_real else (-0.1, 0.1)
    rho = np.real(rhos[qp_name]) if is_real else np.imag(rhos[qp_name])
    ax.pcolormesh(rho, vmin=vmin, vmax=vmax, cmap="RdBu")
    for i in range(4):
        for j in range(4):
            color = "k" if np.abs(rho[i, j]) < 0.1 else "w"
            ax.text(
                i + 0.5,
                j + 0.5,
                f"{rho[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    ax.set_title(f"{qp_name}\nFidelity: {fidelity_fn(qp_name):.3f}, Purity: {purity_fn(qp_name):.3f}")
    ax.set_xlabel("Computational basis")
    ax.set_ylabel("Computational basis")
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_xticklabels(_STATE_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(_STATE_LABELS)


def plot_individual_pauli(
    ax: Axes,
    qp_name: str,
    paulis_data: Dict[str, xr.Dataset],
) -> None:
    """Plot one qubit-pair Pauli expectation bar chart."""
    if qp_name not in paulis_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(qp_name)
        return

    pauli_arr = paulis_data[qp_name]["pauli"]
    values = pauli_arr.values
    labels = pauli_arr.coords["pauli_op"].values
    bar_patches = ax.bar(range(len(values)), values)
    ax.set_xlabel("Pauli Operators")
    ax.set_ylabel("Value")
    ax.set_title(qp_name)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    for bar_patch in bar_patches:
        height = bar_patch.get_height()
        ax.text(
            bar_patch.get_x() + bar_patch.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )


def _plot_method_figures(
    method_name: str,
    rhos: Mapping[str, np.ndarray],
    paulis_data: Mapping[str, xr.Dataset],
    fit_results: Mapping[str, Mapping[str, object]],
    grid_names,
    pair_names: list[str],
    *,
    plot_level: Literal["full", "minimal"],
) -> Dict[str, Figure]:
    """Create chip-grid figures for one mitigation method."""
    mitigation_label = _MITIGATION_LABELS[method_name]
    fidelity_fn = _metric_fn(fit_results, _FIDELITY_KEYS[method_name])
    purity_fn = _metric_fn(fit_results, _PURITY_KEYS[method_name])
    prefix = f"figure_{method_name}"
    figures: Dict[str, Figure] = {}

    city_real_grid = QubitPairGrid(grid_names, pair_names, size=5, projection="3d")
    for ax, qubit in grid_iter(city_real_grid):
        plot_individual_3d_city(
            ax,
            qubit["qubit"],
            dict(rhos),
            plot_3d_hist_with_frame_real,
            fidelity_fn,
            purity_fn,
            show_fidelity=True,
        )
    city_real_grid.fig.suptitle(
        f"Bell state tomography - Real part (3D city plots)\n({mitigation_label})",
        y=0.98,
    )
    city_real_grid.fig.subplots_adjust(top=0.82, bottom=0.08, hspace=0.6, wspace=0.35)
    figures[f"{prefix}_city_real"] = city_real_grid.fig

    city_imag_grid = QubitPairGrid(grid_names, pair_names, size=5, projection="3d")
    for ax, qubit in grid_iter(city_imag_grid):
        plot_individual_3d_city(
            ax,
            qubit["qubit"],
            dict(rhos),
            plot_3d_hist_with_frame_imag,
            fidelity_fn,
            purity_fn,
            show_fidelity=False,
        )
    city_imag_grid.fig.suptitle(
        f"Bell state tomography - Imaginary part (3D city plots)\n({mitigation_label})",
        y=0.98,
    )
    city_imag_grid.fig.subplots_adjust(top=0.82, bottom=0.08, hspace=0.5, wspace=0.3)
    figures[f"{prefix}_city_imag"] = city_imag_grid.fig

    if plot_level == "full":
        rho_real_grid = QubitPairGrid(grid_names, pair_names)
        for ax, qubit in grid_iter(rho_real_grid):
            plot_individual_rho(
                ax,
                qubit["qubit"],
                dict(rhos),
                is_real=True,
                fidelity_fn=fidelity_fn,
                purity_fn=purity_fn,
            )
        rho_real_grid.fig.suptitle(f"Bell state tomography (real part)\n({mitigation_label})")
        rho_real_grid.fig.tight_layout()
        figures[f"{prefix}_rho_real"] = rho_real_grid.fig

        rho_imag_grid = QubitPairGrid(grid_names, pair_names)
        for ax, qubit in grid_iter(rho_imag_grid):
            plot_individual_rho(
                ax,
                qubit["qubit"],
                dict(rhos),
                is_real=False,
                fidelity_fn=fidelity_fn,
                purity_fn=purity_fn,
            )
        rho_imag_grid.fig.suptitle(f"Bell state tomography (imaginary part)\n({mitigation_label})")
        rho_imag_grid.fig.tight_layout()
        figures[f"{prefix}_rho_imag"] = rho_imag_grid.fig

        pauli_grid = QubitPairGrid(grid_names, pair_names)
        for ax, qubit in grid_iter(pauli_grid):
            plot_individual_pauli(
                ax,
                qubit["qubit"],
                dict(paulis_data),
            )
        pauli_grid.fig.suptitle(f"Bell state tomography - Pauli expectations\n({mitigation_label})")
        pauli_grid.fig.tight_layout()
        figures[f"{prefix}_paulis"] = pauli_grid.fig

    return figures


def plot_bell_state_tomography(
    rhos_by_method: Mapping[str, Mapping[str, np.ndarray]],
    paulis_by_method: Mapping[str, Mapping[str, xr.Dataset]],
    qubit_pairs: list,
    fit_results: Mapping[str, Mapping[str, object]],
    *,
    plot_level: Literal["full", "minimal"] = "minimal",
) -> Dict[str, Figure]:
    """Create Bell state tomography figures on chip-topology grids.

    Parameters
    ----------
    rhos_by_method : Mapping[str, Mapping[str, np.ndarray]]
        Density matrices keyed by mitigation method (``kron``, ``joint``) and pair name.
    paulis_by_method : Mapping[str, Mapping[str, xr.Dataset]]
        Pauli expectation datasets keyed by mitigation method and pair name.
    qubit_pairs : list
        Qubit pair objects used for grid placement on the chip layout.
    fit_results : Mapping[str, Mapping[str, object]]
        Serialized fit results per pair (from ``fit_raw_data``), used for subplot
        fidelity and purity annotations.
    plot_level : {"minimal", "full"}, optional
        ``minimal`` (default): 3D city plots for correlated mitigation if available,
        otherwise uncorrelated. ``full``: both methods plus 2D density-matrix heatmaps
        and Pauli expectation bar charts.

    Returns
    -------
    Dict[str, Figure]
        Figure handles keyed by mitigation method and plot type, e.g.
        ``figure_joint_city_real``, ``figure_joint_city_imag``, and with
        ``plot_level="full"`` also ``figure_kron_rho_real``, ``figure_joint_paulis``, etc.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    methods = _methods_to_plot(plot_level, fit_results, rhos_by_method, pair_names)

    figures: Dict[str, Figure] = {}
    for method_name in methods:
        rhos = rhos_by_method.get(method_name, {})
        paulis_data = paulis_by_method.get(method_name, {})
        if not rhos:
            continue
        figures.update(
            _plot_method_figures(
                method_name,
                rhos,
                paulis_data,
                fit_results,
                grid_names,
                pair_names,
                plot_level=plot_level,
            )
        )

    return figures
