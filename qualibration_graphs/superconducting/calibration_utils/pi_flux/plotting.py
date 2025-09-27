from typing import Dict, List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_pi_flux(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: Dict[str, object] | None = None):
    figures: Dict[str, plt.Figure] = {}

    # Raw
    if ("state" in ds.data_vars and ("freq_full" in ds.coords or "freq" in ds.coords)) or "IQ_abs" in ds.data_vars or "I" in ds.data_vars:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            if "state" in ds.data_vars and "freq_full" in ds.coords:
                im = ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state.plot(
                    ax=ax, add_colorbar=False, x="time", y="freq_GHz"
                )
                ax.set_ylabel("Freq (GHz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Qubit State")
            elif "state" in ds.data_vars and "freq" in ds.coords:
                im = ds.assign_coords(freq_GHz=ds.freq / 1e9).loc[qubit].state.plot(
                    ax=ax, add_colorbar=False, x="time", y="freq_GHz"
                )
                ax.set_ylabel("Freq (GHz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Qubit State")
            elif "IQ_abs" in ds.data_vars:
                im = ds.loc[qubit].IQ_abs.plot(ax=ax, add_colorbar=False, x="time", y="detuning")
                ax.set_ylabel("Detuning (Hz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Amplitude (V)")
            elif "I" in ds.data_vars:
                im = ds.loc[qubit].I.plot(ax=ax, add_colorbar=False, x="time", y="detuning")
                ax.set_ylabel("Detuning (Hz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("I Quadrature (V)")
        grid.fig.suptitle("Qubit spectroscopy vs time after flux pulse")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_raw"] = grid.fig

    # Center frequencies
    if "center_freqs" in ds.data_vars:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            cf = ds.loc[qubit].center_freqs
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(cf.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y / 1e9)
            ax.set_ylabel("Freq (GHz)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
        grid.fig.suptitle("Qubit frequency shift vs time after flux pulse")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_freqs_shift"] = grid.fig

        # Log-scale version
        grid_log = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_log):
            cf = ds.loc[qubit].center_freqs
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(cf.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y / 1e9)
            ax.set_ylabel("Freq (GHz)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.set_xscale('log')
            ax.grid(True)
        grid_log.fig.suptitle("Qubit frequency shift vs time after flux pulse")
        grid_log.fig.set_size_inches(15, 9)
        grid_log.fig.tight_layout()
        figures["figure_freqs_shift_log"] = grid_log.fig

    # Flux and fits
    if "flux_response" in ds.data_vars:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid):
            qb_flux = ds.loc[qubit].flux_response
            try:
                final_val = float(qb_flux.isel(time=-1).values)
            except Exception:
                final_val = np.nan
            if np.isfinite(final_val) and abs(final_val) > 1e-12:
                qb_flux_plot = qb_flux / final_val
            else:
                max_abs = float(np.nanmax(np.abs(qb_flux.values)))
                qb_flux_plot = qb_flux / max_abs if max_abs > 0 else qb_flux
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(qb_flux_plot.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y)
            if fit_results and fit_results.get(qubit["qubit"], None) is not None:
                res = fit_results[qubit["qubit"]]
                # Support both dict-like and dataclass result
                best_a_dc = res["best_a_dc"] if isinstance(res, dict) else res.best_a_dc
                components = res["best_components"] if isinstance(res, dict) else res.best_components
                t_data = ds.loc[qubit].time.values
                y_fit = np.ones_like(t_data) * best_a_dc
                for amp, tau in components:
                    y_fit += amp * np.exp(-(t_data - t_data[0]) / tau)
                if np.isfinite(final_val) and abs(final_val) > 1e-12:
                    y_fit = y_fit / final_val
                else:
                    if max_abs > 0:
                        y_fit = y_fit / max_abs
                ax.plot(t_data, y_fit, "r-", linewidth=2)
            ax.set_ylabel("Flux (normalized)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.grid(True)
        grid.fig.suptitle("Flux response vs time")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_flux_response"] = grid.fig

        # Log-scale version
        grid_log = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, qubit in grid_iter(grid_log):
            qb_flux = ds.loc[qubit].flux_response
            try:
                final_val = float(qb_flux.isel(time=-1).values)
            except Exception:
                final_val = np.nan
            if np.isfinite(final_val) and abs(final_val) > 1e-12:
                qb_flux_plot = qb_flux / final_val
            else:
                max_abs = float(np.nanmax(np.abs(qb_flux.values)))
                qb_flux_plot = qb_flux / max_abs if max_abs > 0 else qb_flux
            x = np.asarray(ds.loc[qubit].time.values, dtype=float)
            y = np.asarray(qb_flux_plot.values, dtype=float)
            y = np.squeeze(y)
            if y.ndim == 2:
                if y.shape[-1] == x.size:
                    y = y[0, :]
                elif y.shape[0] == x.size:
                    y = y[:, 0]
                else:
                    y = y.reshape(-1)[: x.size]
            ax.plot(x, y)
            ax.set_ylabel("Flux (normalized)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.set_xscale('log')
            ax.grid(True)
        grid_log.fig.suptitle("Flux response vs time")
        grid_log.fig.set_size_inches(15, 9)
        grid_log.fig.tight_layout()
        figures["figure_flux_response_log"] = grid_log.fig

    return figures

