"""Plotting utilities for CROT spectroscopy parity-diff measurement.

Generates a figure with two rows per qubit pair:
  Row 1: Side-by-side 2-D colour maps of parity difference vs (exchange, frequency)
         for control_x180 = False (left) and True (right).
  Row 2: Extracted peak positions f_↓ and f_↑ vs exchange voltage, with the
         exchange coupling J = |f_↑ − f_↓| annotated.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
) -> plt.Figure:
    """Create a multi-panel CROT spectroscopy figure.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``control_x180``, ``exchange``, ``esr_frequency``
        coordinates and ``pdiff_<pair>`` variables.
    ds_fit : xr.Dataset or None
        Fitted peak positions (``f0_down_<pair>``, ``f0_up_<pair>``).
    qubit_pairs : list
        Qubit pair objects (each must have a ``.name`` attribute).
    fit_results : dict
        Per-pair fit results from :func:`~.analysis.fit_raw_data`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs * 2,
        2,
        figsize=(12, 5 * n_pairs),
        squeeze=False,
    )

    exchange = ds_raw.coords["exchange"].values
    freqs = ds_raw.coords["esr_frequency"].values
    freq_ghz = freqs / 1e9

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        var_name = f"pdiff_{name}"

        ax_map_no = axes[pair_idx * 2, 0]
        ax_map_yes = axes[pair_idx * 2, 1]
        ax_fit_left = axes[pair_idx * 2 + 1, 0]
        ax_fit_right = axes[pair_idx * 2 + 1, 1]

        if var_name not in ds_raw.data_vars:
            for ax in (ax_map_no, ax_map_yes, ax_fit_left, ax_fit_right):
                ax.set_title(f"{name} — no data")
            continue

        pdiff = ds_raw[var_name].values
        pdiff_no_x180 = pdiff[0]
        pdiff_with_x180 = pdiff[1]

        vmin = min(np.nanmin(pdiff_no_x180), np.nanmin(pdiff_with_x180))
        vmax = max(np.nanmax(pdiff_no_x180), np.nanmax(pdiff_with_x180))

        im0 = ax_map_no.pcolormesh(freq_ghz, exchange, pdiff_no_x180, shading="auto", vmin=vmin, vmax=vmax)
        ax_map_no.set_title(f"{name} — control |↓⟩ (no x180)")
        ax_map_no.set_xlabel("ESR frequency (GHz)")
        ax_map_no.set_ylabel("Exchange voltage (V)")
        fig.colorbar(im0, ax=ax_map_no, label="P_diff")

        im1 = ax_map_yes.pcolormesh(freq_ghz, exchange, pdiff_with_x180, shading="auto", vmin=vmin, vmax=vmax)
        ax_map_yes.set_title(f"{name} — control |↑⟩ (with x180)")
        ax_map_yes.set_xlabel("ESR frequency (GHz)")
        ax_map_yes.set_ylabel("Exchange voltage (V)")
        fig.colorbar(im1, ax=ax_map_yes, label="P_diff")

        r = fit_results.get(name, {})

        if ds_fit is not None:
            f0_down_key = f"f0_down_{name}"
            f0_up_key = f"f0_up_{name}"
            if f0_down_key in ds_fit.data_vars and f0_up_key in ds_fit.data_vars:
                f0_down_ghz = ds_fit[f0_down_key].values / 1e9
                f0_up_ghz = ds_fit[f0_up_key].values / 1e9
                exchange_fit = ds_fit.coords["exchange"].values

                ax_map_no.plot(f0_down_ghz, exchange_fit, "w--", lw=1.5, label="f_↓ fit")
                ax_map_yes.plot(f0_up_ghz, exchange_fit, "w--", lw=1.5, label="f_↑ fit")
                ax_map_no.legend(loc="best", fontsize=7)
                ax_map_yes.legend(loc="best", fontsize=7)

                ax_fit_left.plot(exchange_fit, f0_down_ghz, "o-", ms=3, label="f_↓")
                ax_fit_left.plot(exchange_fit, f0_up_ghz, "s-", ms=3, label="f_↑")
                ax_fit_left.set_xlabel("Exchange voltage (V)")
                ax_fit_left.set_ylabel("Resonance frequency (GHz)")
                ax_fit_left.set_title(f"{name} — peak positions vs exchange")
                ax_fit_left.legend(loc="best", fontsize=8)

                splitting_mhz = np.abs(f0_up_ghz - f0_down_ghz) * 1e3
                ax_fit_right.plot(exchange_fit, splitting_mhz, "d", ms=3, color="C2", label="data")
                ax_fit_right.set_xlabel("Exchange voltage (V)")
                ax_fit_right.set_ylabel("J coupling (MHz)")
                ax_fit_right.set_title(f"{name} — exchange coupling J")

                em = r.get("exchange_model")
                if em and em.get("success"):
                    v_fine = np.linspace(exchange_fit[0], exchange_fit[-1], 200)
                    j_fit_mhz = (em["J_0"] / 1e6) * np.exp((v_fine - em["V_ref"]) / em["lever_arm"])
                    ax_fit_right.plot(
                        v_fine,
                        j_fit_mhz,
                        "-",
                        color="C3",
                        lw=1.5,
                        label=(f"fit: J₀={em['J_0']/1e6:.2f} MHz, " f"λ={em['lever_arm']*1e3:.1f} mV"),
                    )
                    ax_fit_right.legend(loc="best", fontsize=7)

                if r.get("success"):
                    opt_idx = r.get("optimal_exchange_idx", 0)
                    j_mhz = r["exchange_coupling_J"] / 1e6
                    ax_fit_right.axvline(exchange_fit[opt_idx], ls=":", color="k", alpha=0.5)
                    ax_fit_right.annotate(
                        f"J = {j_mhz:.3f} MHz",
                        xy=(exchange_fit[opt_idx], splitting_mhz[opt_idx]),
                        xytext=(10, 10),
                        textcoords="offset points",
                        fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="k"),
                    )
        else:
            for ax in (ax_fit_left, ax_fit_right):
                ax.text(0.5, 0.5, "No fit data", transform=ax.transAxes, ha="center", va="center")

    fig.suptitle("CROT Spectroscopy — Parity Difference", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
