from __future__ import annotations

from typing import Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_pf_posterior(
    ds: xr.Dataset,
    qubit_names: Iterable[str],
    v_f: np.ndarray,
    tau_ns: np.ndarray,
    fit_results: Optional[Mapping[str, Mapping[str, object]]] = None,
    *,
    dim_repetition: str = "repetition",
) -> plt.Figure:
    """Plot posterior P(f | data) averaged over repetitions as a function of tau and f."""
    names = list(qubit_names)
    n = len(names)
    fig, axes = plt.subplots(
        nrows=max(1, n),
        ncols=1,
        figsize=(8, 3.5 * max(1, n)),
        squeeze=False,
    )

    for ax, (qi, qname) in zip(axes.ravel(), enumerate(names, start=1)):
        pf_name = f"Pf{qi}"
        if pf_name not in ds.data_vars:
            ax.set_title(f"{qname}: no {pf_name}")
            continue

        da = ds[pf_name]
        if dim_repetition in da.dims:
            da = da.mean(dim=dim_repetition)
        if "qubit" in da.dims:
            da = da.sel(qubit=qname, drop=True)

        vals = np.asarray(da.values, dtype=float)
        if vals.ndim == 1:
            vals = vals[np.newaxis, :]

        im = ax.imshow(
            vals,
            aspect="auto",
            origin="lower",
            extent=(float(v_f[0]), float(v_f[-1]), float(tau_ns[0]), float(tau_ns[-1])),
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pf")

        if fit_results and qname in fit_results and fit_results[qname].get("success"):
            f_map = fit_results[qname]["map_frequency"]
            ax.axvline(float(f_map), color="w", linestyle="--", linewidth=1.0, label="MAP f")
            ax.legend(loc="upper right")

        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("idle time (ns)")
        ax.set_title(f"{qname} posterior")

    plt.tight_layout()
    return fig


def plot_pf_posterior_single_rep_with_map_track(
    ds: xr.Dataset,
    qubit_name: str,
    v_f: np.ndarray,
    tau_ns: np.ndarray,
    *,
    all_qubit_names: Optional[Iterable[str]] = None,
    repetition_index: int = 0,
    dim_repetition: str = "repetition",
) -> plt.Figure:
    """Posterior P(f|data) for one repetition (no averaging) with argmax f(τ) overlaid."""
    if all_qubit_names is not None:
        names = list(all_qubit_names)
        qi = names.index(qubit_name) + 1
    else:
        qi = 1

    pf_name = f"Pf{qi}"
    if pf_name not in ds.data_vars:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.set_title(f"{qubit_name}: no {pf_name}")
        return fig

    da = ds[pf_name]
    if "qubit" in da.dims:
        da = da.sel(qubit=qubit_name, drop=True)
    if dim_repetition in da.dims:
        da = da.isel({dim_repetition: repetition_index})

    vals = np.asarray(da.values, dtype=float)
    if vals.ndim == 1:
        vals = vals[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(
        vals,
        aspect="auto",
        origin="lower",
        extent=(float(v_f[0]), float(v_f[-1]), float(tau_ns[0]), float(tau_ns[-1])),
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pf")

    j_max = np.nanargmax(vals, axis=1)
    f_track = np.asarray(v_f, dtype=float)[j_max]
    tau_y = np.asarray(tau_ns, dtype=float)
    ax.plot(f_track, tau_y, color="cyan", linewidth=1.5, label="argmax f(τ)")
    ax.legend(loc="upper right")

    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("idle time (ns)")
    ax.set_title(f"{qubit_name} posterior (rep {repetition_index})")
    plt.tight_layout()
    return fig


def plot_pf_posterior_single_rep_all_qubits_with_map_track(
    ds: xr.Dataset,
    qubit_names: Iterable[str],
    v_f: np.ndarray,
    tau_ns: np.ndarray,
    *,
    repetition_index: int = 0,
    dim_repetition: str = "repetition",
) -> plt.Figure:
    """One figure: each row is ``plot_pf_posterior_single_rep_with_map_track`` for that qubit."""
    names = list(qubit_names)
    n = len(names)
    fig, axes = plt.subplots(
        nrows=max(1, n),
        ncols=1,
        figsize=(8, 3.5 * max(1, n)),
        squeeze=False,
    )

    for ax, qname in zip(axes.ravel(), names):
        qi = names.index(qname) + 1
        pf_name = f"Pf{qi}"
        if pf_name not in ds.data_vars:
            ax.set_title(f"{qname}: no {pf_name}")
            continue

        da = ds[pf_name]
        if "qubit" in da.dims:
            da = da.sel(qubit=qname, drop=True)
        if dim_repetition in da.dims:
            da = da.isel({dim_repetition: repetition_index})

        vals = np.asarray(da.values, dtype=float)
        if vals.ndim == 1:
            vals = vals[np.newaxis, :]

        im = ax.imshow(
            vals,
            aspect="auto",
            origin="lower",
            extent=(float(v_f[0]), float(v_f[-1]), float(tau_ns[0]), float(tau_ns[-1])),
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pf")

        j_max = np.nanargmax(vals, axis=1)
        f_track = np.asarray(v_f, dtype=float)[j_max]
        tau_y = np.asarray(tau_ns, dtype=float)
        ax.plot(f_track, tau_y, color="cyan", linewidth=1.5, label="argmax f(τ)")
        ax.legend(loc="upper right")

        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("idle time (ns)")
        ax.set_title(f"{qname} posterior (rep {repetition_index})")

    plt.tight_layout()
    return fig
