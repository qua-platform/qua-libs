"""Geometric CZ error-amplification analysis.

The experiment measures two Ramsey quadratures for the target qubit while the
control qubit is prepared in |0> or |1>. Repeating the raw CPhase gate N times
amplifies the conditional phase in the parity I/Q plane.

The **central panel** shows wrapped **Δφ = φ1 − φ0** (before applying cosine).
The **green 1D curve** uses **cos(Δφ)**: mean over ``N`` and the same
``_analyse_single_qubit`` fit as power-Rabi error amplification (2D surface
``cos(Δφ)(V, N)``) to pick ``optimal_amplitude``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from calibration_utils.power_rabi.error_amplification_analysis import (
    _analyse_single_qubit,
)

logger = logging.getLogger(__name__)


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
) -> tuple[xr.Dataset, dict[str, dict[str, float | bool]]]:
    """Run chevron analysis: Δφ map + cos(Δφ) fit for the optimum and green curve.

    Returns
    -------
    ds_fit
        ``chevron_signal`` = wrapped Δφ (2D). ``mean_signal`` / ``mean_signal_fit``
        are ⟨cos(Δφ)⟩_N and the DE model curve from :func:`_analyse_single_qubit`.
    fit_results
        Per pair: ``optimal_amplitude`` (V) and ``success``.
    """
    amplitudes = ds_raw.coords["exchange_amplitude"].values.astype(np.float64)
    num_gates = ds_raw.coords["num_cphase_gates"].values.astype(np.float64)

    fit_results: dict[str, dict[str, float | bool]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s; skipping.", var_name, qp.name)
            fit_results[qp.name] = {"optimal_amplitude": 0.0, "success": False}
            continue

        data = ds_raw[var_name]
        d0 = data.sel(control_state=0).values
        d1 = data.sel(control_state=1).values
        i0, q0 = d0[0, :, :], d0[1, :, :]
        i1, q1 = d1[0, :, :], d1[1, :, :]

        phi0_all = np.arctan2(
            q0 - quadrature_signal_center, i0 - quadrature_signal_center
        )
        phi1_all = np.arctan2(
            q1 - quadrature_signal_center, i1 - quadrature_signal_center
        )
        dphi_all = phi1_all - phi0_all
        dphi_wrapped = np.arctan2(np.sin(dphi_all), np.cos(dphi_all))

        cos_signal = np.cos(dphi_wrapped)
        chevron_res = _analyse_single_qubit(
            cos_signal.T, amplitudes, num_gates.astype(float)
        )
        optimal_amplitude = float(chevron_res.get("opt_amp", np.nan))

        diag = chevron_res.get("_diag")
        mean_signal_fit: np.ndarray | None = None
        if isinstance(diag, dict) and diag.get("mean_signal_fit") is not None:
            mean_signal_fit = np.asarray(diag["mean_signal_fit"]).ravel()

        mean_cos = np.nanmean(cos_signal, axis=1)
        success = bool(np.isfinite(optimal_amplitude))

        fit_results[qp.name] = {
            "optimal_amplitude": float(optimal_amplitude) if success else 0.0,
            "success": success,
        }

        fit_vars[f"chevron_signal_{qp.name}"] = xr.DataArray(
            dphi_wrapped,
            dims=["exchange_amplitude", "num_cphase_gates"],
            attrs={"long_name": "Δφ (wrapped)", "units": "rad"},
        )
        fit_vars[f"mean_signal_fit_{qp.name}"] = xr.DataArray(
            (
                np.asarray(mean_signal_fit)
                if mean_signal_fit is not None
                and len(mean_signal_fit) == len(amplitudes)
                else np.full(len(amplitudes), np.nan)
            ).astype(float),
            dims=["exchange_amplitude"],
            attrs={"long_name": "Chevron 1D fit (cos)", "units": ""},
        )
        fit_vars[f"mean_signal_{qp.name}"] = xr.DataArray(
            mean_cos.astype(float),
            dims=["exchange_amplitude"],
            attrs={"long_name": "⟨cos(Δφ)⟩_N", "units": ""},
        )

    ds_fit = xr.Dataset(
        fit_vars,
        coords={
            "exchange_amplitude": amplitudes,
            "num_cphase_gates": num_gates,
        },
    )
    return ds_fit, fit_results
