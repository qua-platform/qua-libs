"""Plotting module for Bell state tomography calibration."""

from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

_IDEAL_BELL_RHO = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
_STATE_LABELS = ["00", "01", "10", "11"]


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
    ax.set_xlabel("Pauli Operators")
    ax.set_ylabel("Pauli Operators")
    ax.set_xticks(np.arange(4) + 0.5)
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_xticklabels(_STATE_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(_STATE_LABELS)


def plot_individual_pauli(ax: Axes, qp_name: str, paulis_data: Dict[str, xr.Dataset]) -> None:
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


def plot_bell_state_tomography(
    rhos: Dict[str, np.ndarray],
    paulis_data: Dict[str, xr.Dataset],
    qubit_pairs: list,
    node=None,
) -> Dict[str, Figure]:
    """Create Bell state tomography figures on chip-topology grids.

    Parameters
    ----------
    rhos : dict[str, np.ndarray]
        Density matrix per qubit pair.
    paulis_data : dict[str, xr.Dataset]
        Pauli expectation values per qubit pair.
    qubit_pairs : list
        Qubit pair objects used for grid placement.
    node : optional
        Node for fidelity/purity results in subplot labels.

    Returns
    -------
    dict[str, Figure]
        ``figure_city_real``, ``figure_city_imag``, ``figure_rho_real``,
        ``figure_rho_imag``, and ``figure_paulis``.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    results = getattr(node, "results", {}) if node is not None else {}

    def _fidelity(qp_name: str) -> float:
        return float(results.get(f"{qp_name}_fidelity", 0.0))

    def _purity(qp_name: str) -> float:
        return float(results.get(f"{qp_name}_purity", 0.0))

    figures: Dict[str, Figure] = {}

    city_real_grid = QubitPairGrid(grid_names, pair_names, size=5, projection="3d")
    for ax, qubit in grid_iter(city_real_grid):
        plot_individual_3d_city(
            ax,
            qubit["qubit"],
            rhos,
            plot_3d_hist_with_frame_real,
            _fidelity,
            _purity,
            show_fidelity=True,
        )
    city_real_grid.fig.suptitle(f"Bell state tomography - Real part (3D city plots)", y=0.98)
    city_real_grid.fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.6, wspace=0.35)
    figures["figure_city_real"] = city_real_grid.fig

    city_imag_grid = QubitPairGrid(grid_names, pair_names, size=5, projection="3d")
    for ax, qubit in grid_iter(city_imag_grid):
        plot_individual_3d_city(
            ax,
            qubit["qubit"],
            rhos,
            plot_3d_hist_with_frame_imag,
            _fidelity,
            _purity,
            show_fidelity=False,
        )
    city_imag_grid.fig.suptitle("Bell state tomography - Imaginary part (3D city plots)", y=0.98)
    city_imag_grid.fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.5, wspace=0.3)
    figures["figure_city_imag"] = city_imag_grid.fig

    rho_real_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(rho_real_grid):
        plot_individual_rho(
            ax,
            qubit["qubit"],
            rhos,
            is_real=True,
            fidelity_fn=_fidelity,
            purity_fn=_purity,
        )
    rho_real_grid.fig.suptitle("Bell state tomography (real part)")
    rho_real_grid.fig.tight_layout()
    figures["figure_rho_real"] = rho_real_grid.fig

    rho_imag_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(rho_imag_grid):
        plot_individual_rho(
            ax,
            qubit["qubit"],
            rhos,
            is_real=False,
            fidelity_fn=_fidelity,
            purity_fn=_purity,
        )
    rho_imag_grid.fig.suptitle("Bell state tomography (imaginary part)")
    rho_imag_grid.fig.tight_layout()
    figures["figure_rho_imag"] = rho_imag_grid.fig

    pauli_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(pauli_grid):
        plot_individual_pauli(ax, qubit["qubit"], paulis_data)
    pauli_grid.fig.suptitle("Bell state tomography - Pauli expectations")
    pauli_grid.fig.tight_layout()
    figures["figure_paulis"] = pauli_grid.fig

    return figures
