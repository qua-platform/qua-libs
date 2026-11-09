"""Residual iSWAP error-amplification analysis.

The node repeats a fixed raw CZ/CPhase block inside two DD cycles:

* ``X⊗X`` selects the odd-subspace swap component proportional to
  ``theta * cos(chi)``.
* ``Y⊗X`` selects the complementary component proportional to
  ``theta * sin(chi)``.

Only population transfer in the odd-parity subspace is measured in v1, so the
analysis reports absolute component magnitudes and the combined residual
``theta``.  The sign/quadrant of the swap axis requires Bell-basis odd-subspace
measurements and is intentionally left out.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.optimize import differential_evolution

logger = logging.getLogger(__name__)


@dataclass
class ComponentFit:
    """Fit result for one DD-selected iSWAP component."""

    theta_abs: float = 0.0
    offset: float = 0.0
    contrast: float = 0.0
    decay_rate: float = 0.0
    rmse: float = 0.0
    span: float = 0.0
    success: bool = False


def _transfer_model(
    n_cycles: np.ndarray,
    theta_abs: float,
    offset: float,
    contrast: float,
    decay_rate: float,
) -> np.ndarray:
    """Population transfer for repeated small swap-angle amplification."""
    n = np.asarray(n_cycles, dtype=np.float64)
    envelope = np.exp(-decay_rate * n)
    return offset + contrast * envelope * np.sin(2.0 * theta_abs * n) ** 2


def _fit_component(
    n_cycles: np.ndarray,
    transfer: np.ndarray,
    *,
    max_theta_rad: float,
    min_fit_contrast: float,
) -> tuple[ComponentFit, np.ndarray]:
    """Fit one transfer trace to the repeated-cycle transfer model."""
    n = np.asarray(n_cycles, dtype=np.float64)
    y = np.asarray(transfer, dtype=np.float64)
    valid = np.isfinite(n) & np.isfinite(y)
    n = n[valid]
    y = y[valid]
    if len(n) < 2:
        fit = ComponentFit(success=False)
        return fit, np.full_like(n_cycles, np.nan, dtype=np.float64)

    span = float(np.ptp(y))
    offset0 = float(np.nanmean(y))
    if span < min_fit_contrast:
        y_fit = np.full_like(np.asarray(n_cycles, dtype=np.float64), offset0)
        fit = ComponentFit(
            theta_abs=0.0,
            offset=offset0,
            contrast=0.0,
            decay_rate=0.0,
            rmse=float(np.sqrt(np.mean((y - offset0) ** 2))),
            span=span,
            success=True,
        )
        return fit, y_fit

    n_span = max(float(np.max(n) - np.min(n)), 1.0)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    offset_lo = max(-0.25, y_min - span)
    offset_hi = min(1.25, y_max + span)
    contrast_hi = max(0.05, min(1.5, 3.0 * span))
    theta_hi = max(float(max_theta_rad), 1e-6)
    gamma_hi = 10.0 / n_span

    bounds = [
        (0.0, theta_hi),
        (offset_lo, offset_hi),
        (0.0, contrast_hi),
        (0.0, gamma_hi),
    ]

    def objective(params: np.ndarray) -> float:
        theta_abs, offset, contrast, decay_rate = params
        pred = _transfer_model(n, theta_abs, offset, contrast, decay_rate)
        return float(np.sum((pred - y) ** 2))

    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=1000,
            tol=1e-10,
            polish=True,
            popsize=20,
        )
        theta_abs, offset, contrast, decay_rate = map(float, result.x)
        y_fit_valid = _transfer_model(n, theta_abs, offset, contrast, decay_rate)
        rmse = float(np.sqrt(np.mean((y_fit_valid - y) ** 2)))
        success = bool(np.isfinite(theta_abs) and np.isfinite(rmse))
    except Exception:
        logger.debug("iSWAP component fit failed; using zero-angle fallback.")
        theta_abs = 0.0
        offset = offset0
        contrast = 0.0
        decay_rate = 0.0
        rmse = float(np.sqrt(np.mean((y - offset0) ** 2)))
        success = False

    y_fit = _transfer_model(
        np.asarray(n_cycles, dtype=np.float64),
        theta_abs,
        offset,
        contrast,
        decay_rate,
    )
    fit = ComponentFit(
        theta_abs=theta_abs,
        offset=offset,
        contrast=contrast,
        decay_rate=decay_rate,
        rmse=rmse,
        span=span,
        success=success,
    )
    return fit, y_fit


def _axis_transfer(
    data: xr.DataArray, *, dd_axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return average transfer and the two directional transfer traces."""
    selected = data.sel(dd_axis=dd_axis)
    transfer_from_10 = np.asarray(
        selected.sel(initial_state=0).values, dtype=np.float64
    )
    target_survival_from_01 = np.asarray(
        selected.sel(initial_state=1).values,
        dtype=np.float64,
    )
    transfer_from_01 = 1.0 - target_survival_from_01
    transfer_mean = np.nanmean(
        np.stack([transfer_from_10, transfer_from_01], axis=0),
        axis=0,
    )
    return transfer_mean, transfer_from_10, transfer_from_01


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    max_theta_rad: float = 0.25,
    min_fit_contrast: float = 1e-4,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Extract residual iSWAP angle from fixed-gate error amplification."""
    n_cycles = ds_raw.coords["num_cphase_cycles"].values.astype(np.float64)

    fit_results: dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s; skipping.", var_name, qp.name)
            fit_results[qp.name] = {
                "theta_x": 0.0,
                "theta_y": 0.0,
                "theta_iswap_abs": 0.0,
                "success": False,
            }
            continue

        data = ds_raw[var_name]
        transfer_x, transfer_10_x, transfer_01_x = _axis_transfer(data, dd_axis=0)
        transfer_y, transfer_10_y, transfer_01_y = _axis_transfer(data, dd_axis=1)

        fit_x, fit_curve_x = _fit_component(
            n_cycles,
            transfer_x,
            max_theta_rad=max_theta_rad,
            min_fit_contrast=min_fit_contrast,
        )
        fit_y, fit_curve_y = _fit_component(
            n_cycles,
            transfer_y,
            max_theta_rad=max_theta_rad,
            min_fit_contrast=min_fit_contrast,
        )

        theta_x = float(fit_x.theta_abs)
        theta_y = float(fit_y.theta_abs)
        theta_abs = float(np.hypot(theta_x, theta_y))
        success = bool(fit_x.success and fit_y.success and np.isfinite(theta_abs))

        fit_results[qp.name] = {
            "theta_x": theta_x,
            "theta_y": theta_y,
            "theta_iswap_abs": theta_abs,
            "fit_x": asdict(fit_x),
            "fit_y": asdict(fit_y),
            "success": success,
        }

        fit_vars[f"transfer_x_{qp.name}"] = xr.DataArray(
            transfer_x,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "X⊗X selected transfer", "units": ""},
        )
        fit_vars[f"transfer_y_{qp.name}"] = xr.DataArray(
            transfer_y,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "Y⊗X selected transfer", "units": ""},
        )
        fit_vars[f"transfer_x_fit_{qp.name}"] = xr.DataArray(
            fit_curve_x,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "X⊗X transfer fit", "units": ""},
        )
        fit_vars[f"transfer_y_fit_{qp.name}"] = xr.DataArray(
            fit_curve_y,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "Y⊗X transfer fit", "units": ""},
        )
        fit_vars[f"transfer_10_x_{qp.name}"] = xr.DataArray(
            transfer_10_x,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "|10>→|01> transfer, X⊗X", "units": ""},
        )
        fit_vars[f"transfer_01_x_{qp.name}"] = xr.DataArray(
            transfer_01_x,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "|01>→|10> transfer, X⊗X", "units": ""},
        )
        fit_vars[f"transfer_10_y_{qp.name}"] = xr.DataArray(
            transfer_10_y,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "|10>→|01> transfer, Y⊗X", "units": ""},
        )
        fit_vars[f"transfer_01_y_{qp.name}"] = xr.DataArray(
            transfer_01_y,
            dims=["num_cphase_cycles"],
            attrs={"long_name": "|01>→|10> transfer, Y⊗X", "units": ""},
        )

    ds_fit = xr.Dataset(
        fit_vars,
        coords={"num_cphase_cycles": n_cycles},
    )
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log residual iSWAP fit results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r.get("success") else "FAILED"
        _log(
            f"{name}: residual iSWAP theta={r.get('theta_iswap_abs', 0.0):.6g} rad, "
            f"theta_x={r.get('theta_x', 0.0):.6g}, "
            f"theta_y={r.get('theta_y', 0.0):.6g} [{status}]"
        )
