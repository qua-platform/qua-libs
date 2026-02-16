"""Plotting for the two-τ Ramsey detuning-sweep parity-difference analysis.

Produces a two-row figure per qubit:

1. **Top** — short-τ trace: parity-difference vs detuning with the
   joint fit overlaid and the resonance position marked.
2. **Bottom** — long-τ trace: same, showing the narrower fringes and
   reduced amplitude from T₂* decay.
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
    """Plot two-τ Ramsey detuning-sweep analysis for each qubit.

    Layout (per qubit): two rows showing short-τ and long-τ traces,
    each with data, joint fit, and resonance marker.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with ``pdiff_<qubit>``, ``detuning``, and ``tau``
        coordinates.
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

    detuning_hz = np.asarray(ds.detuning.values, dtype=float)
    detuning_mhz = detuning_hz * 1e-6
    tau_ns = np.asarray(ds.tau.values, dtype=float)
    n_tau = len(tau_ns)
    n_qubits = len(qubit_names)

    nrows = n_qubits * n_tau
    fig, axes = plt.subplots(
        nrows, 1, figsize=(10, 3.5 * nrows), squeeze=False,
    )

    colors = ["C0", "C2", "C4", "C6"]
    fit_colors = ["C1", "C3", "C5", "C7"]

    row = 0
    for qi, qname in enumerate(qubit_names):
        fr = fit_results.get(qname, {})
        diag = fr.get("_diag", {})
        fitted_curves = diag.get("fitted_curves")
        freq_offset = fr.get("freq_offset", np.nan)
        contrast = fr.get("contrast", np.nan)
        t2_star = fr.get("t2_star", np.nan)
        success = fr.get("success", False)

        pdiff_var = f"pdiff_{qname}"
        if pdiff_var in ds.data_vars:
            pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        else:
            pdiff = np.full((n_tau, len(detuning_hz)), np.nan)

        for ti in range(n_tau):
            ax = axes[row, 0]
            tau_label = f"τ = {tau_ns[ti]:.0f} ns"
            c = colors[ti % len(colors)]
            fc = fit_colors[ti % len(fit_colors)]

            trace = pdiff[ti] if pdiff.ndim == 2 else pdiff
            ax.plot(detuning_mhz, trace, f"-", color=c, lw=0.8, alpha=0.7)
            ax.scatter(
                detuning_mhz, trace, c=c, s=8, alpha=0.5,
                zorder=3, label="Data",
            )

            if fitted_curves is not None:
                fit_trace = fitted_curves[ti]
                ax.plot(
                    detuning_mhz, fit_trace, "-", color=fc, lw=1.5,
                    alpha=0.9, label="Joint DE fit",
                )

            if success and np.isfinite(freq_offset):
                ax.axvline(
                    freq_offset * 1e-6, color="k", ls="--", lw=1.0,
                    label=f"δ₀ = {freq_offset * 1e-6:.4f} MHz",
                )

            ax.set_xlabel("Detuning (MHz)")
            ax.set_ylabel("Parity difference")
            ax.set_ylim(-0.05, 1.05)

            title = f"{qname} — {tau_label}"
            if success and ti == 0:
                title += (
                    f"  |  δ₀={freq_offset * 1e-6:.4f} MHz, "
                    f"A₀={contrast:.3f}, T₂*={t2_star:.0f} ns"
                )
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=7)
            row += 1

    fig.suptitle(
        "Ramsey detuning sweep (parity diff) — joint two-τ DE fit",
        fontsize=13,
    )
    fig.tight_layout()
    return fig
