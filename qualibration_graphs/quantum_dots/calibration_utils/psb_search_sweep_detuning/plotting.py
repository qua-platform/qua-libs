"""Plotting for the PSB sweep-detuning analysis.

Produces a single-panel figure per sensor showing:

* IQ amplitude vs detuning (scatter + line).
* Sigmoid fit overlaid.
* Transition detuning and optimal readout detuning marked.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _get_sensor_keys_from_ds(ds: xr.Dataset) -> List[str]:
    """Extract sensor keys from ``I_<key>`` data-variable names."""
    i_vars = sorted([v for v in ds.data_vars if v.startswith("I_")])
    return [v.replace("I_", "") for v in i_vars]


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    dot_pairs: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot PSB sensor signal vs detuning with sigmoid fit.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with ``I_<key>`` / ``Q_<key>`` variables and
        ``detuning`` coordinate.
    ds_fit : xr.Dataset or None
        Unused — kept for API consistency with other plotting modules.
    dot_pairs : list
        Dot-pair objects (used only for count).
    fit_results : dict
        Sensor key → fit-result dict as returned by
        :func:`~.analysis.fit_raw_data`.
    """
    sensor_keys = _get_sensor_keys_from_ds(ds)
    if not sensor_keys:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    detuning = np.asarray(ds.detuning.values, dtype=float)
    # Display in mV if the span is below 1 V
    if abs(detuning[-1] - detuning[0]) < 1:
        det_display = detuning * 1e3
        det_unit = "mV"
    else:
        det_display = detuning
        det_unit = "V"

    n_sensors = len(sensor_keys)
    fig, axes = plt.subplots(
        n_sensors,
        1,
        figsize=(10, 3.5 * n_sensors),
        squeeze=False,
    )

    for si, key in enumerate(sensor_keys):
        ax = axes[si, 0]
        fr = fit_results.get(key, {})
        diag = fr.get("_diag", {})
        fitted_curve = diag.get("fitted_curve")
        eps0 = fr.get("transition_detuning", np.nan)
        opt = fr.get("optimal_readout_detuning", np.nan)
        contrast = fr.get("contrast", np.nan)
        success = fr.get("success", False)

        # Compute amplitude from I / Q
        i_var = f"I_{key}"
        q_var = f"Q_{key}"
        I = np.asarray(ds[i_var].values, dtype=float) if i_var in ds.data_vars else np.zeros(len(detuning))
        Q = np.asarray(ds[q_var].values, dtype=float) if q_var in ds.data_vars else np.zeros(len(detuning))
        amplitude = np.sqrt(I**2 + Q**2)

        # Data
        ax.plot(det_display, amplitude, "-", color="C0", lw=0.8, alpha=0.7)
        ax.scatter(
            det_display,
            amplitude,
            c="C0",
            s=8,
            alpha=0.5,
            zorder=3,
            label="Amplitude",
        )

        # Fit curve
        if fitted_curve is not None:
            ax.plot(
                det_display,
                fitted_curve,
                "-",
                color="C1",
                lw=1.5,
                alpha=0.9,
                label="Sigmoid fit",
            )

        # Mark transition and optimal readout detuning
        if success and np.isfinite(eps0):
            eps0_disp = eps0 * 1e3 if det_unit == "mV" else eps0
            ax.axvline(
                eps0_disp,
                color="C3",
                ls="--",
                lw=1,
                alpha=0.7,
                label="Transition",
            )
        if success and np.isfinite(opt):
            opt_disp = opt * 1e3 if det_unit == "mV" else opt
            ax.axvline(
                opt_disp,
                color="C2",
                ls="--",
                lw=1,
                alpha=0.7,
                label="Optimal readout",
            )

        ax.set_xlabel(f"Detuning ({det_unit})")
        ax.set_ylabel("Sensor signal (a.u.)")

        # Title with fit parameters
        title = f"{key}"
        if success and np.isfinite(eps0):
            if abs(eps0) < 1:
                title += f"  |  ε₀ = {eps0 * 1e3:.2f} mV"
            else:
                title += f"  |  ε₀ = {eps0:.4f} V"
            if np.isfinite(contrast):
                title += f",  contrast = {contrast:.4f}"
        elif not success:
            title += "  |  fit failed"
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("PSB sweep detuning — sigmoid fit", fontsize=12)
    fig.tight_layout()
    return fig
