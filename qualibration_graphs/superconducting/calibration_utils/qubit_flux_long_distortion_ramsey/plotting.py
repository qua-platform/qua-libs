"""Plotting utilities for Ramsey-based qubit flux long distortion visualizations."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import xarray as xr

# Shared fit overlay (identical to 17a)
from calibration_utils.qubit_flux_long_distortion_qubitspec.plotting import plot_fit

# ---------------------------------------------------------------------------
# Default figures (always produced by ``plot_raw_data_with_fit``)
# ---------------------------------------------------------------------------


def annotate_branch_risk(fig, ds: xr.Dataset) -> bool:
    """Stamp a warning on ``fig`` if any qubit has branch-aliasing risk metadata."""
    if "branch_risk_code" not in ds:
        return False
    qnames = [str(n) for n in ds["branch_risk_code"].coords["qubit"].values]
    lines = []
    for nm in qnames:
        code = int(ds["branch_risk_code"].sel(qubit=nm).values)
        if code < 1:
            continue
        sw = float(ds["branch_sig_swing"].sel(qubit=nm).values)
        rs = float(ds["branch_ref_span"].sel(qubit=nm).values)
        tag = "HIGH" if code >= 2 else "marginal"
        lines.append(f"{nm}: {tag} — phase swing {sw:.2f}×2π, ref span {rs:.2f}×2π")
    if not lines:
        return False
    msg = (
        "⚠ BRANCH-ALIASING RISK: per-point phase→flux inversion (np.round) may be unreliable\n"
        + "\n".join(lines)
        + "\n(true phase approaches/exceeds one 2π window; see _compute_flux_response)"
    )
    fig.text(
        0.5,
        0.01,
        msg,
        ha="center",
        va="bottom",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round", facecolor="crimson", alpha=0.9),
    )
    try:
        fig.subplots_adjust(bottom=0.22)
    except Exception:
        pass
    return True


def plot_signal_phase(ds: xr.Dataset, qubits, log_scale: bool = False):
    """Plot extracted Ramsey signal phase vs delay for each qubit.

    Analogue of qubitspec ``plot_center_freqs`` — the intermediate before
    phase→flux inversion.
    """
    if "signal_phase" not in ds:
        return None
    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    times = ds.time.values

    for ax, qname in zip(axes[0], names):
        phase = ds.sel(qubit=qname).signal_phase.values
        ax.plot(times, phase, marker="o", ms=4, lw=1.2)
        if log_scale:
            ax.set_xscale("log")
            ax.grid(True, which="both")
        else:
            ax.grid(True)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Ramsey phase (rad)")
        ax.set_title(qname)

    scale = " (log scale)" if log_scale else ""
    fig.suptitle(f"Ramsey signal phase vs time after flux pulse{scale}")
    fig.tight_layout()
    return fig


def plot_flux_response(ds: xr.Dataset, qubits, log_scale: bool = False):
    """Plot flux step response vs time for each qubit."""
    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    times = ds.time.values

    for ax, qname in zip(axes[0], names):
        fr = ds.sel(qubit=qname).flux_response.values
        ax.plot(times, fr, lw=1.5)
        if log_scale:
            ax.set_xscale("log")
            ax.grid(True, which="both")
        else:
            ax.grid(True)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Flux response (V)")
        ax.set_title(qname)

    scale = " (log scale)" if log_scale else ""
    fig.suptitle(f"Flux response vs time after flux pulse{scale}")
    fig.tight_layout()
    annotate_branch_risk(fig, ds)
    return fig


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Dict,
    *,
    debug: bool = False,
    ds_raw: Optional[xr.Dataset] = None,
) -> Dict[str, plt.Figure]:
    """Default 17b figures: signal phase, flux response, and exponential fit.

    Parameters
    ----------
    debug :
        If True, also add raw Ramsey signal and reference amp-sweep plots
        when ``ds_raw`` is provided.
    ds_raw :
        Raw/processed acquisition dataset (needed for debug figures).

    Returns
    -------
    dict
        Always: ``signal_phase_{linear,log}`` (if present),
        ``flux_response_{linear,log}``, ``fitted_data``.
        With ``debug=True``: may also include ``raw_data_*``, ``ref_data``,
        ``ref_phase_cal``.
    """
    figures: Dict[str, plt.Figure] = {
        "signal_phase_linear": plot_signal_phase(ds, qubits, log_scale=False),
        "signal_phase_log": plot_signal_phase(ds, qubits, log_scale=True),
        "flux_response_linear": plot_flux_response(ds, qubits, log_scale=False),
        "flux_response_log": plot_flux_response(ds, qubits, log_scale=True),
        "fitted_data": plot_fit(ds, qubits, fit_results),
    }
    figures = {k: v for k, v in figures.items() if v is not None}

    if not debug:
        return figures

    for key, fig in {
        "raw_data_linear": plot_raw_signal(ds_raw, qubits, log_scale=False),
        "raw_data_log": plot_raw_signal(ds_raw, qubits, log_scale=True),
        "ref_data": plot_ref_data(ds_raw, qubits),
        "ref_phase_cal": plot_ref_phase_cal(ds, qubits),
    }.items():
        if fig is not None:
            figures[key] = fig

    return figures


# ---------------------------------------------------------------------------
# Debug figures (``debug_plots=True``)
# ---------------------------------------------------------------------------


def plot_raw_signal(ds_raw: Optional[xr.Dataset], qubits, log_scale: bool = False):
    """Raw Ramsey signal (state or I) vs time per qubit."""
    if ds_raw is None:
        return None
    signal_key = "state" if "state" in ds_raw.data_vars else "I"
    if signal_key not in ds_raw.data_vars:
        return None

    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)

    for ax, qname in zip(axes[0], names):
        da = ds_raw[signal_key].sel(qubit=qname)
        if "frame" in da.dims:
            da = da.mean("frame")
        da.plot(ax=ax, xscale="log" if log_scale else "linear")
        ax.set_title(qname)
        ax.grid(True)

    scale = " (log scale)" if log_scale else ""
    fig.suptitle(f"Raw Ramsey signal vs time{scale}")
    fig.tight_layout()
    return fig


def plot_ref_data(ds_raw: Optional[xr.Dataset], qubits) -> Optional[plt.Figure]:
    """Reference Ramsey amplitude-sweep signal (state_ref / I_ref)."""
    if ds_raw is None:
        return None
    ref_key = "state_ref" if "state_ref" in ds_raw.data_vars else "I_ref"
    if ref_key not in ds_raw.data_vars:
        return None
    fg_ref = ds_raw[ref_key].plot(x="a", col="qubit")
    return fg_ref.fig


def plot_ref_phase_cal(ds: xr.Dataset, qubits) -> Optional[plt.Figure]:
    """Reference phase-vs-amp calibration curve attached to ``ds`` during analysis."""
    if "ref_phase_cal" not in ds:
        return None
    names = [q.name for q in qubits]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    amps = ds["ref_phase_cal"].coords["a"].values

    for ax, qname in zip(axes[0], names):
        phases = ds["ref_phase_cal"].sel(qubit=qname).values
        ax.plot(amps, phases, marker=".", lw=1.2)
        ax.set_xlabel("Ramsey flux amp (V)")
        ax.set_ylabel("Phase (rad)")
        ax.set_title(qname)
        ax.grid(True)

    fig.suptitle("Reference Ramsey phase vs flux amplitude")
    fig.tight_layout()
    return fig
