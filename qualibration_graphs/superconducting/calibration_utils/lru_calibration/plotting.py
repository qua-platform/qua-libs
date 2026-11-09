"""Plotting for the f-LRU calibration."""

from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_config.lru_components import LRU_TRANSITIONS

from .analysis import lru_transition

#: Floor of the log-scale P(f) axis, so empty bins (P = 0) stay on the plot.
PF_FLOOR = 1e-4
#: Colour map of the rising population in the LRU maps.
POPULATION_CMAPS = {"pg": "Blues", "pe": "Greens", "pf": "Reds"}


def _amplitude_label(amplitude: xr.DataArray) -> str:
    """Axis label from the amplitude coordinate's attributes (fraction of full scale for the XY pump)."""
    return f"{amplitude.attrs.get('long_name', 'LRU amplitude')} ({amplitude.attrs.get('units', 'fraction of full scale')})"


def plot_lru_maps(
    ds: xr.Dataset, qubits: List[AnyTransmon], ds_fit: xr.Dataset, duration: float, population: Optional[str] = None
) -> Figure:
    """One population versus LRU amplitude and drive frequency for one LRU duration, with the fitted line.

    By default the population is the one the dataset's transition fills: P(g) for |f0> -> |g1>, P(e) for
    |f0> -> |e1> (see ``lru_transition``). ``population`` ("pg", "pe" or "pf") plots another one instead,
    for display only. White dots are the per-amplitude P(f) minima used in the fit, the orange curve the fitted
    ``d0 + c2 * A**2`` line and the star the optimum on that line.
    """
    info = LRU_TRANSITIONS[lru_transition(ds)]
    population = population or info.population
    population_label = f"P({population[1]})"
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        data = ds.sel(qubit=name, duration=duration)
        fit = ds_fit.sel(qubit=name, duration=duration)
        amplitude = data.amplitude.values
        rf = float(data.frequency.values[0] - data.detuning.values[0])
        mesh = ax.pcolormesh(
            amplitude,
            data.frequency.values / 1e9,
            data[population].transpose("detuning", "amp_prefactor").values,
            shading="nearest",
            cmap=POPULATION_CMAPS[population],
            vmin=0,
            vmax=1,
        )
        used = fit.ridge_used.values
        ax.plot(
            amplitude[used], (rf + fit.ridge_detuning.values[used]) / 1e9, "o", color="white", ms=3, mec="k", mew=0.3
        )
        ax.plot(amplitude, (rf + fit.line_detuning.values) / 1e9, color="tab:orange", lw=1.5)
        if np.isfinite(fit.opt_amplitude.values):
            ia = int(np.argmin(np.abs(amplitude - float(fit.opt_amplitude.values))))
            ax.plot(amplitude[ia], (rf + fit.line_detuning.values[ia]) / 1e9, "*", color="tab:red", ms=10)
        ax.set_ylim(data.frequency.values.min() / 1e9, data.frequency.values.max() / 1e9)
        ax.set_title(name)
        ax.set_xlabel(_amplitude_label(data.amplitude))
        ax.set_ylabel(f"{data.frequency.attrs.get('long_name', 'Drive frequency')} (GHz)")
        grid.fig.colorbar(mesh, ax=ax, label=population_label)

    grid.fig.suptitle(f"f-LRU {info.label}: {population_label} for a {duration:.0f} ns pulse")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_pf_along_line(ds: xr.Dataset, qubits: List[AnyTransmon], ds_fit: xr.Dataset, fit_results: Dict) -> Figure:
    """P(f) along the fitted resonance line versus LRU amplitude, one curve per LRU duration.

    The dashed line marks the selected optimum (duration and amplitude of lowest P(f)).
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        fit = ds_fit.sel(qubit=name)
        amplitude = fit.amplitude.values
        for duration in fit.duration.values:
            pf_line = fit.pf_line.sel(duration=duration).values
            ax.semilogy(amplitude, np.clip(pf_line, PF_FLOOR, None), "o-", ms=3, label=f"tau = {duration:.0f} ns")
        result = fit_results[name]
        if result["success"]:
            ax.axvline(result["amplitude"], color="k", ls="--", lw=1)
            ax.annotate(
                f"A = {result['amplitude']:.3f}\ntau = {result['duration_ns']:.0f} ns\nP(f) = {result['pf_min']:.2g}",
                xy=(result["amplitude"], max(result["pf_min"], PF_FLOOR)),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_ylim(PF_FLOOR / 2, 1.5)
        ax.set_title(name)
        ax.set_xlabel(_amplitude_label(fit.amplitude))
        ax.set_ylabel("Population |f>")
        ax.legend(fontsize=7)

    grid.fig.suptitle(f"f-LRU {LRU_TRANSITIONS[lru_transition(ds)].label}: P(f) along the fitted resonance line")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


#: At most this many shots are drawn per qubit in the IQ debug figure.
MAX_PLOTTED_SHOTS = 50_000


def plot_iq_blobs(ds: xr.Dataset, qubits: List[AnyTransmon], iq_info: Dict) -> Figure:
    """Debug mode (``return_iq_blobs``): all raw shots in the IQ plane, with the stored and refit centers.

    The shaded regions are the states ``readout_state_gef`` assigns in real time (nearest stored center in
    L1 distance); crosses are the stored centers and plus signs the centers refit on this run's shots.
    Axes use node 15's scaling (raw units * 2**12 / readout length) so the figure compares directly.
    """
    by_name = {q.name: q for q in qubits}
    colors = {"g": "tab:blue", "e": "tab:orange", "f": "tab:green"}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        q = by_name[name]
        scale = 2**12 / q.resonator.operations["readout"].length * 1e3  # raw demod units -> mV, as in node 15
        shots = ds.sel(qubit=name)
        i_shots, q_shots = shots.I.values.ravel() * scale, shots.Q.values.ravel() * scale
        centers = np.asarray(q.resonator.gef_centers, dtype=float) * scale

        i_all, q_all = np.concatenate([i_shots, centers[:, 0]]), np.concatenate([q_shots, centers[:, 1]])
        pad_i, pad_q = 0.1 * np.ptp(i_all), 0.1 * np.ptp(q_all)
        ii, qq = np.meshgrid(
            np.linspace(i_all.min() - pad_i, i_all.max() + pad_i, 300),
            np.linspace(q_all.min() - pad_q, q_all.max() + pad_q, 300),
        )
        regions = (np.abs(ii[..., None] - centers[:, 0]) + np.abs(qq[..., None] - centers[:, 1])).argmin(axis=-1)
        ax.contourf(ii, qq, regions, levels=[-0.5, 0.5, 1.5, 2.5], colors=list(colors.values()), alpha=0.15)

        if len(i_shots) > MAX_PLOTTED_SHOTS:
            keep = np.random.default_rng(0).choice(len(i_shots), MAX_PLOTTED_SHOTS, replace=False)
            i_shots, q_shots = i_shots[keep], q_shots[keep]
        ax.plot(i_shots, q_shots, ".", ms=1, color="k", alpha=0.3)
        refit = np.asarray(iq_info[name]["centers_refit"]) * scale
        for (label, color), (ci, cq), (ri, rq) in zip(colors.items(), centers, refit, strict=True):
            ax.plot(ci, cq, "X", ms=12, color=color, mec="k", label=f"stored {label} center")
            ax.plot(ri, rq, "P", ms=12, color=color, mec="w", label=f"refit {label} center")

        stored_p, refit_p = iq_info[name]["populations_stored"], iq_info[name]["populations_refit"]
        ax.set_title(
            f"{name}: g/e/f = {stored_p[0]:.2f}/{stored_p[1]:.2f}/{stored_p[2]:.2f} (stored centers), "
            f"{refit_p[0]:.2f}/{refit_p[1]:.2f}/{refit_p[2]:.2f} (refit)"
        )
        ax.set_xlabel("I (mV, node 15 scaling)")
        ax.set_ylabel("Q (mV, node 15 scaling)")
        ax.legend(fontsize=7)

    grid.fig.suptitle("f-LRU debug: readout_GEF shots after |f> preparation and LRU pulse")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig
