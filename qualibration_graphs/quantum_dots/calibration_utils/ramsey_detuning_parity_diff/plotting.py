"""Plotting for the two-τ Ramsey detuning-sweep conditional-readout analysis.

Produces a two-row figure per qubit:

1. **Top** — short-τ trace: analysis signal vs detuning with the
   per-trace cosine fit overlaid, resonance marker, and fitted
   frequency annotated.
2. **Bottom** — long-τ trace: same, showing the faster oscillations
   and reduced amplitude from T₂* decay.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.common_utils.parity_streams import get_parity_item_names


def _get_qubit_names_from_ds(
    ds: xr.Dataset,
    qubits: List[Any],
    analysis_signal: str,
) -> List[str]:
    return get_parity_item_names(
        ds,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p2_given_p1_0",
) -> "plt.Figure":
    """Plot two-τ Ramsey detuning-sweep analysis for each qubit.

    Layout (per qubit): two rows showing short-τ and long-τ traces,
    each with data, per-trace cosine fit, and resonance marker.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with ``{analysis_signal}_<qubit>``, ``detuning``, and ``tau``
        coordinates.
    ds_fit : xr.Dataset or None
        Unused — kept for API consistency with 10a/10c plotting.
    qubits : list
        Qubit objects (used only for count).
    fit_results : dict
        Qubit name → fit-result dict as returned by
        :func:`~.analysis.fit_raw_data`.
    analysis_signal : str
        Data variable prefix for the plotted signal (same as ``node.parameters.analysis_signal``).
    """
    qubit_names = _get_qubit_names_from_ds(ds, qubits, analysis_signal)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    detuning_hz = np.asarray(ds.detuning.values, dtype=float)
    detuning_mhz = detuning_hz * 1e-6
    tau_ns = np.asarray(ds.tau.values, dtype=float)
    n_tau = len(tau_ns)
    n_qubits = len(qubit_names)

    fig, axes = plt.subplots(
        n_qubits,
        n_tau,
        figsize=(7 * n_tau, 4 * n_qubits),
        squeeze=False,
    )

    data_colors = ["C0", "C2"]
    fit_colors = ["C1", "C3"]

    for qi, qname in enumerate(qubit_names):
        fr = fit_results.get(qname, {})
        diag = fr.get("_diag", {})
        fitted_curves = diag.get("fitted_curves")
        trace_fits = diag.get("_traces", [])
        freq_offset = fr.get("freq_offset", np.nan)
        t2_star = fr.get("t2_star", np.nan)
        gamma = fr.get("decay_rate", np.nan)
        success = fr.get("success", False)

        signal_var = f"{analysis_signal}_{qname}"
        if signal_var in ds.data_vars:
            sig_da = ds[signal_var]
            if "tau" in sig_da.dims and "detuning" in sig_da.dims:
                signal_2d = sig_da.transpose("tau", "detuning").values.astype(float)
            else:
                signal_2d = np.asarray(sig_da.values, dtype=float)
                if signal_2d.shape == (len(detuning_hz), n_tau):
                    signal_2d = signal_2d.T
        else:
            signal_2d = np.full((n_tau, len(detuning_hz)), np.nan)

        for ti in range(n_tau):
            ax = axes[qi, ti]
            tau_label = f"τ = {tau_ns[ti]:.0f} ns"
            c = data_colors[ti % len(data_colors)]
            fc = fit_colors[ti % len(fit_colors)]

            trace = signal_2d[ti] if signal_2d.ndim == 2 else signal_2d
            ax.plot(detuning_mhz, trace, "-", color=c, lw=0.8, alpha=0.7)
            ax.scatter(
                detuning_mhz,
                trace,
                c=c,
                s=8,
                alpha=0.5,
                zorder=3,
                label="Data",
            )

            if fitted_curves is not None:
                fit_trace = fitted_curves[ti]
                ax.plot(
                    detuning_mhz,
                    fit_trace,
                    "-",
                    color=fc,
                    lw=1.5,
                    alpha=0.9,
                    label="Cosine fit",
                )

            if success and np.isfinite(freq_offset):
                ax.axvline(
                    freq_offset * 1e-6,
                    color="k",
                    ls="--",
                    lw=1.0,
                    label=f"δ₀ = {freq_offset * 1e-6:.3f} MHz",
                )

            ax.set_xlabel("Detuning (MHz)")
            ax.set_ylabel(analysis_signal)
            ax.set_ylim(-0.05, 1.05)

            tf = trace_fits[ti] if ti < len(trace_fits) else {}
            a_i = tf.get("amplitude", np.nan)
            f_i = tf.get("osc_freq", np.nan)
            d0_i = tf.get("delta0", np.nan)
            title = f"{qname} — {tau_label}"
            if np.isfinite(a_i):
                title += f"  |  A={a_i:.3f}"
            if np.isfinite(f_i):
                tau_eff = f_i * 1e9
                title += f", f={f_i:.2e} Hz⁻¹ (τ_eff≈{tau_eff:.0f} ns)"
            if np.isfinite(d0_i):
                title += f", δ₀_i={d0_i * 1e-6:.3f} MHz"
            ax.set_title(title, fontsize=9)
            ax.legend(loc="upper right", fontsize=7)

    # Super-title with joint results
    suptitle = "Ramsey detuning sweep — independent fits + joint extraction"
    if success:
        parts = [f"δ₀={freq_offset * 1e-6:.3f} MHz"]
        if np.isfinite(gamma):
            parts.append(f"γ={gamma:.4f} 1/ns")
        if np.isfinite(t2_star):
            parts.append(f"T₂*={t2_star:.0f} ns")
        suptitle += f"\n{', '.join(parts)}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig
