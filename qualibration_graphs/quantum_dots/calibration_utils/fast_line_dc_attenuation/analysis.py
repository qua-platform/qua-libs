"""Fast-line DC attenuation analysis.

Determines the ratio between the fast-line (AC / square-wave) amplitude
and the DC voltage scale by locating transitions in the sensor response
as a function of the DC sweep.

Two regimes:
* **Sensor dot swept** — the sensor dot's own Coulomb peak splits into
  two peaks separated by the square-wave amplitude.  We use ``peaks_dips``
  to find both peaks directly.
* **Non-sensor component swept** — the sensor is parked on-peak and
  sees step-like transitions (the Coulomb peak moves past it).  We
  take the numerical derivative and use ``peaks_dips`` on the resulting
  double peak to extract the step positions.

In both cases the DC separation of the two features divided by the
known square-wave amplitude gives the AC-to-DC attenuation ratio.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import xarray as xr
from scipy.signal import savgol_filter

from qualibrate.core import QualibrationNode
from qualibration_libs.analysis import peaks_dips

_logger = logging.getLogger(__name__)

SWEEP_DIM = "dc_values"


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Fit parameters for a single component / sensor pair."""

    peak_position_1: float
    peak_position_2: float
    dc_separation: float
    attenuation_ratio: float
    success: bool


# ── Helpers ───────────────────────────────────────────────────────────────────


def _smooth_derivative(y: np.ndarray, x: np.ndarray, window: int = 0) -> np.ndarray:
    """Smoothed numerical derivative via Savitzky-Golay."""
    if window <= 0:
        window = max(5, len(y) // 10)
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) - 1)
    if window < 5:
        return np.gradient(y, x)
    return savgol_filter(y, window_length=window, polyorder=3, deriv=1, delta=float(np.mean(np.diff(x))))


def _find_two_peaks(da: xr.DataArray, dim: str) -> Tuple[float, float, bool]:
    """Use peaks_dips to locate the two most prominent peaks.

    Returns ``(pos1, pos2, success)`` with ``pos1 < pos2``.
    """
    fit1 = peaks_dips(da, dim=dim, prominence_factor=3, number=1)
    fit2 = peaks_dips(da, dim=dim, prominence_factor=3, number=2)

    pos1 = float(fit1.position.values)
    pos2 = float(fit2.position.values)

    if np.isnan(pos1) or np.isnan(pos2):
        return np.nan, np.nan, False

    if pos1 > pos2:
        pos1, pos2 = pos2, pos1

    return pos1, pos2, True


def _analyse_sensor_dot_trace(
    signal: np.ndarray,
    dc_values: np.ndarray,
) -> Dict[str, Any]:
    """Sensor-dot component: expect two Coulomb peaks → find both via peaks_dips."""
    da = xr.DataArray(signal, dims=[SWEEP_DIM], coords={SWEEP_DIM: dc_values})
    pos1, pos2, success = _find_two_peaks(da, SWEEP_DIM)
    return {"peak1": pos1, "peak2": pos2, "success": success}


def _analyse_non_sensor_trace(
    signal: np.ndarray,
    dc_values: np.ndarray,
) -> Dict[str, Any]:
    """Non-sensor component: expect steps → derivative gives two peaks."""
    deriv = _smooth_derivative(signal, dc_values)
    deriv_abs = np.abs(deriv)

    da = xr.DataArray(deriv_abs, dims=[SWEEP_DIM], coords={SWEEP_DIM: dc_values})
    pos1, pos2, success = _find_two_peaks(da, SWEEP_DIM)
    return {"peak1": pos1, "peak2": pos2, "success": success, "derivative": deriv}


# ── Public API ────────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Analyse every (component, sensor) pair and return attenuation ratios.

    Returns ``(ds_fit, fit_results)`` where *fit_results* maps
    ``"<comp>__<sensor>"`` to a dict with :class:`FitParameters` fields
    plus private keys for plotting.
    """
    components: List[str] = node.namespace["components"]
    sensors = node.namespace["sensor_names"]
    square_wave_amplitude: float = node.parameters.square_wave_amplitude

    sensor_dot_names = set(node.machine.sensor_dots.keys()) if hasattr(node.machine, "sensor_dots") else set()

    fit_results: Dict[str, Dict[str, Any]] = {}

    for comp in components:
        dc_values = np.asarray(
            node.namespace["dc_list_values"][node.machine.get_component(comp).physical_channel.name],
            dtype=float,
        )
        is_sensor_dot = comp in sensor_dot_names

        for sensor in sensors:
            key = f"{comp}__{sensor.name}"

            I = np.asarray(ds[f"I_{comp}_sensor_{sensor.name}"].values, dtype=float)
            Q = np.asarray(ds[f"Q_{comp}_sensor_{sensor.name}"].values, dtype=float)
            signal = np.sqrt(I**2 + Q**2)

            if is_sensor_dot:
                result = _analyse_sensor_dot_trace(signal, dc_values)
            else:
                result = _analyse_non_sensor_trace(signal, dc_values)

            peak1, peak2 = result["peak1"], result["peak2"]
            dc_sep = abs(peak2 - peak1) if result["success"] else np.nan
            ratio = (
                dc_sep / (2 * square_wave_amplitude) if (result["success"] and square_wave_amplitude > 0) else np.nan
            )

            fp = FitParameters(
                peak_position_1=peak1,
                peak_position_2=peak2,
                dc_separation=dc_sep,
                attenuation_ratio=ratio,
                success=result["success"],
            )
            fit_results[key] = asdict(fp)
            fit_results[key]["_derivative"] = result.get("derivative")
            fit_results[key]["_is_sensor_dot"] = is_sensor_dot
            fit_results[key]["_dc_values"] = dc_values
            fit_results[key]["_signal"] = signal
            fit_results[key]["_I"] = I
            fit_results[key]["_Q"] = Q

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted results for all component / sensor pairs."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for key, r in fit_results.items():
        peak1 = r.get("peak_position_1", np.nan)
        peak2 = r.get("peak_position_2", np.nan)
        dc_sep = r.get("dc_separation", np.nan)
        ratio = r.get("attenuation_ratio", np.nan)
        success = r.get("success", False)
        is_sd = r.get("_is_sensor_dot", False)
        mode = "sensor-dot (double peak)" if is_sd else "non-sensor (double step)"
        msg = (
            f"Results for {key} [{mode}]: "
            f"peaks @ {peak1:.6f} V, {peak2:.6f} V, "
            f"separation={dc_sep:.6f} V, "
            f"AC/DC ratio={ratio:.4f}, "
            f"success={success}"
        )
        log_callable(msg)
