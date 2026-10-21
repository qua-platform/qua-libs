"""Plotting helpers for the readout quantum efficiency node."""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_efficiency_vs_frequency(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plot the headline result: the fitted quantum efficiency versus readout frequency.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset carrying ``eta`` and ``eta_error``.
    qubits : list of AnyTransmon
        Qubits to plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the plots.
    """
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        detuning_mhz = fit.detuning.values / u.MHz
        ax.errorbar(
            detuning_mhz,
            fit.eta.values,
            yerr=fit.eta_error.values,
            fmt="o-",
            capsize=3,
            label=r"$\eta$ (global fit)",
        )
        ax.axvline(0, color="k", linestyle="dashed", label="current readout frequency")
        ax.set_xlabel("Readout detuning [MHz]")
        ax.set_ylabel(r"Quantum efficiency $\eta$")
        ax.set_title(qubit["qubit"])
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    grid.fig.suptitle(r"Readout quantum efficiency $\eta = a^2\sigma_m^2/2$ vs readout frequency")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout(rect=(0, 0.06, 1, 1))
    return grid.fig


def plot_efficiency_map(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plot the point-by-point efficiency ``SNR^2/(4 beta)`` over the frequency-amplitude grid.

    This map is a diagnostic rather than the result: in the linear regime eta does not depend
    on the measurement amplitude, so the map should be flat along the amplitude axis. Structure
    along that axis means the sweep left the regime where eta is defined (or that the low
    amplitude points are simply too noisy, the ratio being 0/0 at eps = 0).

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset carrying ``eta_pointwise``.
    qubits : list of AnyTransmon
        Qubits to plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the plots.
    """
    single_frequency = ds_fit.sizes["detuning"] == 1
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        # eps = 0 carries no information here (0/0), so it is dropped from the map.
        fit = fit.where(fit.amp_prefactor > 0, drop=True)
        if single_frequency:
            # pcolormesh cannot infer cell edges from a single column, so a run without a
            # frequency sweep shows the same quantity as a line against amplitude.
            ax.plot(fit.amp_prefactor.values, fit.eta_pointwise.isel(detuning=0).values, "o-")
            ax.set_xlabel("Amplitude prefactor")
            ax.set_ylabel(r"$\eta$")
        else:
            mesh = ax.pcolormesh(
                fit.detuning.values / u.MHz,
                fit.amp_prefactor.values,
                fit.eta_pointwise.transpose("amp_prefactor", "detuning").values,
                shading="nearest",
            )
            grid.fig.colorbar(mesh, ax=ax, label=r"$\eta$")
            ax.set_xlabel("Readout detuning [MHz]")
            ax.set_ylabel("Amplitude prefactor")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle(r"Point-by-point $\eta = \mathrm{SNR}^2/(4\beta)$ (flat vs amplitude = linear regime)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_snr_and_dephasing(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plot ``SNR^2`` and ``2 beta`` against ``eps^2`` at the frequency where eta peaks.

    Both are linear in ``eps^2`` in the linear regime, and their ratio is what fixes eta, so
    this is the figure that shows whether the fit is trustworthy. Points excluded from the
    linear-range fit are drawn hollow.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset.
    qubits : list of AnyTransmon
        Qubits to plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the plots.
    """
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        eta = fit.eta.values
        if not np.isfinite(eta).any():
            ax.set_title(f"{qubit['qubit']} (no fit)")
            continue
        best = int(np.nanargmax(np.where(np.isfinite(eta), eta, -np.inf)))
        at_best = fit.isel(detuning=best)
        eps_squared = at_best.amp_prefactor.values**2
        mask = at_best.fit_mask.values

        # The fitted lines must come from the same estimators the analysis uses, or the figure
        # can look wrong while the number is right (and vice versa). SNR is fitted against eps,
        # not eps^2, so its slope is squared to be drawn on these axes.
        snr_slope = float(at_best.snr_slope)
        beta_slope = 1 / (2 * float(at_best.sigma_m) ** 2) if np.isfinite(at_best.sigma_m) else np.nan
        for values, slope, label, color in (
            (at_best.snr.values**2, snr_slope**2, r"$\mathrm{SNR}^2$", "tab:blue"),
            (2 * at_best.beta.values, 2 * beta_slope, r"$2\beta$", "tab:red"),
        ):
            ax.plot(eps_squared[mask], values[mask], "o", color=color, label=label)
            ax.plot(eps_squared[~mask], values[~mask], "o", color=color, markerfacecolor="none")
            ax.plot(eps_squared, slope * eps_squared, "-", color=color, alpha=0.6)

        ax.set_xlabel(r"$\epsilon^2$ (amplitude prefactor squared)")
        ax.set_ylabel("Signal / dephasing")
        ax.set_title(f"{qubit['qubit']}: $\\eta$={eta[best]:.3f} at {at_best.detuning.values / u.MHz:+.2f} MHz")
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    grid.fig.suptitle(r"$\mathrm{SNR}^2$ and $2\beta$ vs $\epsilon^2$ (hollow = outside the fitted linear range)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout(rect=(0, 0.06, 1, 1))
    return grid.fig


def plot_raw_fringes(ds: xr.Dataset, ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plot the Ramsey fringes versus the azimuthal angle of the second pi/2 pulse.

    One curve per measurement amplitude, at the frequency where eta peaks. The shrinking
    amplitude is the measurement-induced dephasing; the sideways shift is the AC-Stark phase.

    Parameters
    ----------
    ds : xr.Dataset
        Processed raw dataset carrying ``state_corrected``.
    ds_fit : xr.Dataset
        Fitted dataset, used to pick the frequency to display.
    qubits : list of AnyTransmon
        Qubits to plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        eta = ds_fit.sel(qubit=qubit["qubit"]).eta.values
        best = int(np.nanargmax(np.where(np.isfinite(eta), eta, -np.inf))) if np.isfinite(eta).any() else 0
        fringes = ds.state_corrected.sel(qubit=qubit["qubit"]).isel(detuning=best)
        for amplitude in fringes.amp_prefactor.values:
            ax.plot(
                fringes.phase.values,
                fringes.sel(amp_prefactor=amplitude).values,
                "o-",
                label=f"$\\epsilon$={amplitude:.2f}",
            )
        ax.set_xlabel("Phase of the second $\\pi/2$ pulse [rad]")
        ax.set_ylabel("P(excited)")
        ax.set_title(qubit["qubit"])
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    grid.fig.suptitle("Ramsey fringes with the measurement pulse embedded")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout(rect=(0, 0.06, 1, 1))
    return grid.fig


def plot_fringe_amplitude_vs_power(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plot the Ramsey fringe amplitude against the measurement pulse power.

    This is the dephasing half of the experiment in its most direct form: the fringe amplitude is
    ``|rho01(eps)|``, and the paper's model has it decaying as ``exp(-eps^2 / (2 sigma_m^2))``,
    i.e. exponentially in POWER rather than in amplitude. The fitted decay from the same
    ``sigma_m`` that enters eta is overlaid, so a curve bending away from it at the top of the
    range is the sweep leaving the linear regime.

    The x axis is the absolute power at the output port, ``readout_power_dbm``, so it can be read
    against the readout power set elsewhere in the calibration chain. Being logarithmic in power,
    it drops the ``eps = 0`` reference point (at -inf dBm) and turns the model's straight line
    into an exponential-of-power curve.

    When several frequencies were measured, the curve at the frequency where eta peaks is drawn
    with its error bars and fit, the others as faint lines for context.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset carrying ``fringe_amplitude``, ``fringe_amplitude_error``, ``sigma_m``
        and the ``readout_power_dbm`` coordinate.
    qubits : list of AnyTransmon
        Qubits to plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the plots.
    """
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        eta = fit.eta.values
        best = int(np.nanargmax(np.where(np.isfinite(eta), eta, -np.inf))) if np.isfinite(eta).any() else 0
        eps = fit.amp_prefactor.values
        # Absolute power at the output port: the configured readout pulse power (port full scale
        # + waveform amplitude) scaled by the swept prefactor, i.e. base_dbm + 20*log10(eps).
        # eps = 0 has no dBm representation and is NaN here, so it drops out of the plot.
        power_dbm = fit.readout_power_dbm.values
        # base_dbm, recovered from any finite point, places the fitted curve on the same axis.
        finite = np.flatnonzero(np.isfinite(power_dbm) & (eps > 0))
        base_dbm = power_dbm[finite[0]] - 20 * np.log10(eps[finite[0]]) if finite.size else np.nan

        for di in range(fit.sizes["detuning"]):
            if di == best:
                continue
            ax.plot(power_dbm, fit.fringe_amplitude.isel(detuning=di).values, "-", color="0.8", linewidth=0.8)

        at_best = fit.isel(detuning=best)
        ax.errorbar(
            power_dbm,
            at_best.fringe_amplitude.values,
            yerr=at_best.fringe_amplitude_error.values,
            fmt="o",
            capsize=3,
            color="tab:blue",
            label="measured",
        )
        sigma_m = float(at_best.sigma_m)
        if np.isfinite(sigma_m) and sigma_m > 0 and np.isfinite(base_dbm):
            reference = float(at_best.fringe_amplitude.values[np.argmin(eps)])
            # Dense in eps, then mapped to dBm: log-spaced from the smallest measured prefactor,
            # since eps = 0 sits at -inf dBm.
            eps_dense = np.geomspace(eps[finite[0]], eps.max(), 200)
            ax.plot(
                base_dbm + 20 * np.log10(eps_dense),
                reference * np.exp(-(eps_dense**2) / (2 * sigma_m**2)),
                "-",
                color="tab:red",
                label=r"$|\rho_{01}(0)|e^{-\epsilon^2/2\sigma_m^2}$",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Readout power [dBm]")
        ax.set_ylabel("Ramsey fringe amplitude")
        ax.set_title(
            f"{qubit['qubit']} at {at_best.detuning.values / u.MHz:+.2f} MHz"
            if fit.sizes["detuning"] > 1
            else qubit["qubit"]
        )
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    grid.fig.suptitle("Ramsey fringe amplitude vs readout power (measurement-induced dephasing)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout(rect=(0, 0.06, 1, 1))
    return grid.fig


def plot_stark_phase(ds_fit: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plot the fitted AC-Stark phase over the frequency-amplitude grid.

    This is the deterministic phase the measurement photons impart, the quantity the phase
    sweep exists to separate from the dephasing. It is not used to compute eta, but it maps the
    photon number and is a useful cross-check on the frequency axis.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset carrying ``stark_phase``.
    qubits : list of AnyTransmon
        Qubits to plot.

    Returns
    -------
    Figure
        Matplotlib figure containing the plots.
    """
    single_frequency = ds_fit.sizes["detuning"] == 1
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = ds_fit.sel(qubit=qubit["qubit"])
        # Referenced to the eps = 0 phase so that only the photon-induced shift is shown.
        phase = np.unwrap(fit.stark_phase.transpose("amp_prefactor", "detuning").values, axis=0)
        phase = phase - phase[0]
        if single_frequency:
            # Same fallback as the efficiency map: one column cannot be drawn as a mesh.
            ax.plot(fit.amp_prefactor.values, phase[:, 0], "o-")
            ax.set_xlabel("Amplitude prefactor")
            ax.set_ylabel("Stark phase [rad]")
        else:
            mesh = ax.pcolormesh(fit.detuning.values / u.MHz, fit.amp_prefactor.values, phase, shading="nearest")
            grid.fig.colorbar(mesh, ax=ax, label="Stark phase [rad]")
            ax.set_xlabel("Readout detuning [MHz]")
            ax.set_ylabel("Amplitude prefactor")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("AC-Stark phase relative to the zero-amplitude reference")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig
