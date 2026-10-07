"""Plots for the CZ unitary reconstruction: one figure per sub-experiment, one row per qubit pair."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure


def _figure(num_pairs: int, title: str):
    fig, axes = plt.subplots(num_pairs, 3, figsize=(15, 3.8 * num_pairs), squeeze=False)
    fig.suptitle(title)
    return fig, axes


def _status(fit_result: Dict) -> str:
    return "SUCCESS" if fit_result["success"] else "FAIL"


def plot_meadd_phi(ds_fit: xr.Dataset, fit_results: Dict[str, Dict]) -> Figure:
    """MEADD-ϕ: raw diagonal readings, the same with the (-1)^n flips removed, and the determinant phase fit."""
    fig, axes = _figure(ds_fit.sizes["qubit_pair"], "MEADD-ϕ: CZ, then X⊗X")
    for row, qp_name in zip(axes, ds_fit.qubit_pair.values):
        d = ds_fit.sel(qubit_pair=qp_name)
        r = fit_results[qp_name]
        n = d.cz_pairs.values
        flip = (-1.0) ** n
        for ax, factor in ((row[0], 1.0), (row[1], flip)):
            ax.plot(n, factor * d.phi_control_x, "o-", label="control ⟨X⟩ (prep +0)")
            ax.plot(n, factor * d.phi_control_y, "s-", label="control ⟨Y⟩ (prep +0)")
            ax.plot(n, factor * d.phi_target_x, "o--", label="target ⟨X⟩ (prep 0+)")
            ax.plot(n, factor * d.phi_target_y, "s--", label="target ⟨Y⟩ (prep 0+)")
            ax.set_xlabel("CZ pairs n")
        row[0].set_title(f"{qp_name}: diagonal readings")
        row[1].set_title("diagonal readings × (-1)^n")
        row[0].legend(fontsize=7)
        row[2].plot(n, d.phi_det_phase, "o", label="data")
        row[2].plot(n, d.phi_det_phase_fit, "-", label="fit, slope 2δϕ")
        row[2].set_title(f"δϕ = {1e3 * (r['phi'] - np.pi):.2f} ± {1e3 * r['phi_err']:.2f} mrad ({_status(r)})")
        row[2].set_xlabel("CZ pairs n")
        row[2].set_ylabel("unwrapped arg det M_n (rad)")
        row[2].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_meadd_theta(ds_fit: xr.Dataset, fit_results: Dict[str, Dict]) -> Figure:
    """MEADD-θ: odd-parity Bloch vector for both decoupling flavours, and the signed rotation angles."""
    fig, axes = _figure(ds_fit.sizes["qubit_pair"], "MEADD-θ: start |10⟩, CZ without corrections, then X⊗X or Y⊗X")
    for row, qp_name in zip(axes, ds_fit.qubit_pair.values):
        d = ds_fit.sel(qubit_pair=qp_name)
        r = fit_results[qp_name]
        n = d.cz_pairs.values
        for ax, name, label in ((row[0], "xx", "X⊗X"), (row[1], "yx", "Y⊗X")):
            ax.plot(n, d[f"theta_{name}_x"], "o-", label="x (X_odd)")
            ax.plot(n, d[f"theta_{name}_y"], "s-", label="y (Y_odd)")
            ax.plot(n, d[f"theta_{name}_z"], "^-", label="z along start pole")
            ax.set_title(f"{qp_name}: odd-parity Bloch vector, {label}")
            ax.set_xlabel("CZ pairs n")
            ax.legend(fontsize=7)
        for name, slope_label in (("xx", "4θcosχ"), ("yx", "4θsinχ")):
            (line,) = row[2].plot(n, d[f"theta_{name}_alpha"], "o", label=f"α_n {name.upper()}")
            row[2].plot(n, d[f"theta_{name}_alpha_fit"], "-", color=line.get_color(), label=f"fit, slope {slope_label}")
        row[2].set_title(
            f"θ = {1e3 * r['theta']:.2f} ± {1e3 * r['theta_err']:.2f} mrad\nχ = {r['chi']:.2f} rad, "
            f"kept ≥ {r['kept_fraction_min']:.2f} ({_status(r)})"
        )
        row[2].set_xlabel("CZ pairs n")
        row[2].set_ylabel("signed angle α_n (rad)")
        row[2].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_floquet(ds_fit: xr.Dataset, fit_results: Dict[str, Dict]) -> Figure:
    """Floquet: raw control-qubit readings, determinant phase (γ) and |10⟩ eigenphase (Ω, gives ζ)."""
    fig, axes = _figure(ds_fit.sizes["qubit_pair"], "Floquet: CZ repeated alone")
    for row, qp_name in zip(axes, ds_fit.qubit_pair.values):
        d = ds_fit.sel(qubit_pair=qp_name)
        r = fit_results[qp_name]
        ncz = d.ncz_floquet.values
        row[0].plot(ncz, d.floquet_control_x, "o-", label="control ⟨X⟩ (prep +0)")
        row[0].plot(ncz, d.floquet_control_y, "s-", label="control ⟨Y⟩ (prep +0)")
        row[0].plot(ncz, d.floquet_offdiag_01, "^--", label="|M₀₁|")
        row[0].plot(ncz, d.floquet_offdiag_10, "v--", label="|M₁₀|")
        row[0].set_title(f"{qp_name}: raw readings")
        row[0].legend(fontsize=7)
        row[1].plot(ncz, d.floquet_det_phase, "o", label="data")
        row[1].plot(ncz, d.floquet_det_phase_fit, "-", label="fit, slope -2γ")
        row[1].set_title(f"γ = {1e3 * r['gamma']:.2f} ± {1e3 * r['gamma_err']:.2f} mrad")
        row[1].set_ylabel("unwrapped arg det M_n (rad)")
        row[1].legend(fontsize=7)
        row[2].plot(ncz, d.floquet_eigphase, "o", label="data")
        row[2].plot(ncz, d.floquet_eigphase_fit, "-", label="fit, slope Ω")
        row[2].set_title(f"ζ = {1e3 * r['zeta']:.2f} ± {1e3 * r['zeta_err']:.2f} mrad ({_status(r)})")
        row[2].set_ylabel("|10⟩ eigenphase (rad)")
        row[2].legend(fontsize=7)
        for ax in row:
            ax.set_xlabel("number of CZ gates")
    fig.tight_layout()
    return fig
