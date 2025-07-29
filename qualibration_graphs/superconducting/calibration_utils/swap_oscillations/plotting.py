import xarray as xr
from typing import List
import matplotlib.pyplot as plt
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names


def plot_control_state(ds: xr.Dataset, qubit_pairs: List):
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        qp = next(qp for qp in qubit_pairs if qp.name == qubit_pair['qubit'])
        plot = ds.assign_coords(detuning_MHz=1e-6 * ds.detuning).state_control.sel(qubit=qubit_pair['qubit']).plot(ax=ax, x='time', y='detuning_MHz', add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.set_title(qubit_pair["qubit"])
        add_overlay(ax, qubit_pair, ds, qp)
    return grid


def plot_target_state(ds: xr.Dataset, qubit_pairs: List):
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        qp = next(qp for qp in qubit_pairs if qp.name == qubit_pair['qubit'])
        plot = ds.assign_coords(detuning_MHz=1e-6 * ds.detuning).state_target.sel(qubit=qubit_pair['qubit']).plot(ax=ax, x='time', y='detuning_MHz', add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.set_title(qubit_pair["qubit"])
        add_overlay(ax, qubit_pair, ds, qp)
    return grid


def add_overlay(ax, qubit_pair, ds, qp):
    fp = ds.results["fit_results"][qubit_pair['qubit']]
    J = fp.J
    detuning = fp.detuning
    optimal_length = fp.optimal_length

    ax.plot(
        [optimal_length],
        [detuning * 1e-6],
        marker='.',
        color='red'
    )
    ax.axhline(y=detuning * 1e-6, color='k', linestyle='--', lw=0.5)
    ax.axvline(x=optimal_length, color='k', linestyle='--', lw=0.5)

    f_eff = np.sqrt(J**2 + (ds.detuning.sel(qubit=qubit_pair['qubit']) - detuning)**2)
    for n in range(10):
        ax.plot(
            n * 0.5 / f_eff * 1e9,
            ds.detuning.sel(qubit=qubit_pair['qubit']) * 1e-6,
            color='red', lw=0.3
        )

    ax2 = ax.twinx()
    detuning_range = ds.detuning.sel(qubit=qubit_pair['qubit'])
    amp_full_range = np.sqrt(-detuning_range / qp.qubit_control.freq_vs_flux_01_quad_term)
    ax2.set_ylim(amp_full_range.min(), amp_full_range.max())
    ax2.set_ylabel('Flux amplitude [V]')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()