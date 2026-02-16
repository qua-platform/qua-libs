"""Plotting for the Ramsey detuning-sweep parity-difference analysis.

Produces a two-column figure per qubit:

1. **Left** — parity-difference vs detuning with the linear-cosine fit
   overlaid and the resonance position marked.
2. **Right** — fit residuals vs detuning.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _get_qubit_names_from_ds(ds: xr.Dataset) -> List[str]:
    """Extract qubit names from ``pdiff_<name>`` data-variable keys."""
    pdiff_vars = [
        v for v in ds.data_vars
        if v.startswith("pdiff_") and not v.endswith("_fit")
    ]
    return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot detuning-sweep Ramsey analysis for each qubit.

    Layout (per qubit row):

    * Column 1 — parity-difference vs detuning with cosine fit and
      vertical marker at the fitted resonance δ₀.
    * Column 2 — residuals (data − fit) vs detuning.

    Parameters
    ----------
    ds : xr.Dataset
        Raw (or fitted) dataset containing ``pdiff_<qubit>`` and a
        ``detuning`` coordinate.
    ds_fit : xr.Dataset or None
        Unused — kept for API consistency with 10a/10c plotting.
    qubits : list
        Qubit objects (used only for count).
    fit_results : dict
        Qubit name → fit-result dict as returned by
        :func:`~.analysis.fit_raw_data`.
    """
    qubit_names = _get_qubit_names_from_ds(ds)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    detuning_hz = np.asarray(ds.detuning.values, dtype=float)
    detuning_mhz = detuning_hz * 1e-6

    for i, qname in enumerate(qubit_names):
        fr = fit_results.get(qname, {})
        diag = fr.get("_diag", {})
        fitted_curve = diag.get("fitted_curve")
        freq_offset = fr.get("freq_offset", np.nan)
        contrast = fr.get("contrast", np.nan)
        success = fr.get("success", False)

        pdiff_var = f"pdiff_{qname}"
        if pdiff_var in ds.data_vars:
            pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        else:
            pdiff = np.full(len(detuning_hz), np.nan)

        # ── Left panel: data + fit ──────────────────────────────────────
        ax_data = axes[i, 0]
        ax_data.plot(detuning_mhz, pdiff, "b-", lw=0.8, alpha=0.7)
        ax_data.scatter(
            detuning_mhz, pdiff, c="b", s=8, alpha=0.5, zorder=3, label="Data",
        )

        if fitted_curve is not None and not np.all(np.isnan(fitted_curve)):
            ax_data.plot(
                detuning_mhz, fitted_curve, "r-", lw=1.5, alpha=0.9,
                label="Linear cosine fit",
            )

        if success and np.isfinite(freq_offset):
            ax_data.axvline(
                freq_offset * 1e-6, color="k", ls="--", lw=1.0,
                label=f"δ₀ = {freq_offset * 1e-6:.4f} MHz",
            )

        ax_data.set_xlabel("Detuning (MHz)")
        ax_data.set_ylabel("Parity difference")
        ax_data.set_ylim(-0.05, 1.05)

        title = f"{qname}"
        if success:
            title += f"  |  δ₀={freq_offset * 1e-6:.4f} MHz, A={contrast:.3f}"
        ax_data.set_title(title)
        ax_data.legend(loc="upper right", fontsize=7)

        # ── Right panel: residuals ──────────────────────────────────────
        ax_res = axes[i, 1]
        if fitted_curve is not None and not np.all(np.isnan(fitted_curve)):
            residuals = pdiff - fitted_curve
            ax_res.scatter(
                detuning_mhz, residuals, c="gray", s=8, alpha=0.6,
            )
            ax_res.axhline(0, color="k", ls="-", lw=0.5)
            ax_res.set_ylabel("Residual")
        else:
            ax_res.text(
                0.5, 0.5, "No fit data", transform=ax_res.transAxes, ha="center",
            )

        ax_res.set_xlabel("Detuning (MHz)")
        ax_res.set_title(f"{qname} — residuals")

    fig.suptitle("Ramsey detuning sweep (parity diff) — linear cosine decomposition")
    fig.tight_layout()
    return fig
