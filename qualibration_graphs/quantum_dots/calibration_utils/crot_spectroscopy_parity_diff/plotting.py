"""Plotting utilities for CROT spectroscopy (joint-outcome / conditional expectation).

Generates a figure with two rows per qubit pair:
  Row 1: Drive the target qubit and read it out. Side-by-side 2-D colour maps of
         the analysis signal vs (exchange, frequency) with the control (spectator)
         in |↓⟩ (no x180, left) and |↑⟩ (with x180, right).
  Row 2: The symmetric experiment — drive the control qubit and read it out, with
         the target (spectator) in |↓⟩ (left) and |↑⟩ (right).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _analysis_signal_colorbar_label(analysis_signal: str) -> str:
    if analysis_signal == "E_p2_given_p1_0":
        return "E[p2 | p1=0]"
    if analysis_signal == "E_p2_given_p1_1":
        return "E[p2 | p1=1]"
    return analysis_signal


def _select_measured_qubit(da: xr.DataArray, label: str) -> xr.DataArray | None:
    """Return the slice for ``measured_qubit == label``.

    Falls back to positional indexing when the coordinate carries no labels,
    and returns the array unchanged if the dimension is absent (legacy data).
    Returns ``None`` if the requested label cannot be resolved.
    """
    if "measured_qubit" not in da.dims:
        return da if label == "target" else None
    coord = da.coords.get("measured_qubit")
    if coord is not None and label in set(np.atleast_1d(coord.values).tolist()):
        return da.sel(measured_qubit=label)
    # No usable labels — fall back to position (target first, control second).
    pos = 0 if label == "target" else 1
    if pos < da.sizes["measured_qubit"]:
        return da.isel(measured_qubit=pos)
    return None


def _draw_map_row(
    fig: plt.Figure,
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    da: xr.DataArray,
    freq_ghz: np.ndarray,
    exchange: np.ndarray,
    *,
    drive_label: str,
    spectator_label: str,
    z_label: str,
) -> None:
    """Draw the |↓⟩ / |↑⟩ colour-map pair for one (driven, measured) qubit."""
    spectator_x180_vals = da.coords["spectator_x180"].values
    idx_no = int(np.argmin(spectator_x180_vals))
    idx_with = int(np.argmax(spectator_x180_vals))
    signal_no_x180 = da.isel(spectator_x180=idx_no).values
    signal_with_x180 = da.isel(spectator_x180=idx_with).values

    vmin = min(np.nanmin(signal_no_x180), np.nanmin(signal_with_x180))
    vmax = max(np.nanmax(signal_no_x180), np.nanmax(signal_with_x180))

    for ax, signal, state in (
        (ax_left, signal_no_x180, "↓⟩ (no x180)"),
        (ax_right, signal_with_x180, "↑⟩ (with x180)"),
    ):
        im = ax.pcolormesh(
            freq_ghz, exchange, signal, shading="auto", vmin=vmin, vmax=vmax
        )
        ax.set_xlim(freq_ghz.min(), freq_ghz.max())
        ax.set_ylim(exchange.min(), exchange.max())
        ax.set_title(
            f"drive {drive_label} — {spectator_label} |{state}"
        )
        ax.set_xlabel(f"{drive_label} ESR frequency (GHz)")
        ax.set_ylabel("Exchange voltage (V)")
        fig.colorbar(im, ax=ax, label=z_label)


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
) -> plt.Figure:
    """Create a multi-panel CROT spectroscopy figure.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Dataset with ``spectator_x180``, ``exchange``, ``esr_frequency``
        coordinates and ``{analysis_signal}_{pair}`` variables (after joint-stream processing).
    ds_fit : xr.Dataset or None
        Fitted peak positions (``f0_down_<pair>``, ``f0_up_<pair>``).
    qubit_pairs : list
        Qubit pair objects (each must have a ``.name`` attribute).
    fit_results : dict
        Per-pair fit results from :func:`~.analysis.fit_raw_data`.
    analysis_signal : str
        Which variable to plot (must match analysis / node parameters).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_pairs = len(qubit_pairs)
    if n_pairs == 0:
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, "No qubit pairs", transform=ax.transAxes, ha="center", va="center")
        return fig
    fig, axes = plt.subplots(
        n_pairs * 2,
        2,
        figsize=(12, 5 * n_pairs),
        squeeze=False,
    )

    exchange = ds_raw.coords["exchange"].values
    freqs = ds_raw.coords["esr_frequency"].values
    freq_ghz = freqs / 1e9
    z_label = _analysis_signal_colorbar_label(analysis_signal)

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        var_name = f"{analysis_signal}_{name}"

        target_label = getattr(getattr(qp, "qubit_target", None), "name", "target")
        control_label = getattr(getattr(qp, "qubit_control", None), "name", "control")

        # Row 1: drive + measure the target (spectator = control).
        # Row 2: drive + measure the control (spectator = target).
        ax_target_no = axes[pair_idx * 2, 0]
        ax_target_yes = axes[pair_idx * 2, 1]
        ax_control_no = axes[pair_idx * 2 + 1, 0]
        ax_control_yes = axes[pair_idx * 2 + 1, 1]

        if var_name not in ds_raw.data_vars:
            for ax in (ax_target_no, ax_target_yes, ax_control_no, ax_control_yes):
                ax.set_title(f"{name} — no data")
            continue

        da = ds_raw[var_name]

        da_target = _select_measured_qubit(da, "target")
        da_control = _select_measured_qubit(da, "control")

        if da_target is not None:
            _draw_map_row(
                fig, ax_target_no, ax_target_yes, da_target, freq_ghz, exchange,
                drive_label=target_label, spectator_label=control_label, z_label=z_label,
            )
        else:
            for ax in (ax_target_no, ax_target_yes):
                ax.set_title(f"{name} — no target data")

        if da_control is not None:
            _draw_map_row(
                fig, ax_control_no, ax_control_yes, da_control, freq_ghz, exchange,
                drive_label=control_label, spectator_label=target_label, z_label=z_label,
            )
        else:
            for ax in (ax_control_no, ax_control_yes):
                ax.set_title(f"{name} — no control data")

    fig.suptitle(
        f"CROT Spectroscopy — {_analysis_signal_colorbar_label(analysis_signal)}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
