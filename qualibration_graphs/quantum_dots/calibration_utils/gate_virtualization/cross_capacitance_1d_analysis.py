"""Analysis functions for 1D cross-capacitance measurement (node 04).

Detects charge transition positions in paired 1D plunger sweeps and
computes cross-capacitance coefficients from the shift between them.

The primary detection method uses BayesianCP changepoint detection
to locate the transition, with a sigmoid refinement for sub-sample
accuracy.  A gradient-based fallback is provided when JAX is not
available.

See Volk et al., npj Quantum Information (2019) 5:29, Supplementary
Fig. S1 for the experimental procedure.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

try:
    import jax
    import jax.numpy as jnp
    from calibration_utils.bayesian_change_point import BayesianCP

    _HAS_BAYESIAN_CP = True
except ImportError:
    _HAS_BAYESIAN_CP = False


def _sigmoid(x: np.ndarray, x0: float, width: float, amp: float, offset: float) -> np.ndarray:
    """Fermi-function sigmoid: offset + amp / (1 + exp(-(x - x0) / width))."""
    return offset + amp / (1.0 + np.exp(-(x - x0) / width))


def detect_transition_position(
    voltage: np.ndarray,
    signal: np.ndarray,
    *,
    method: Literal["bayesian_cp", "gradient"] = "bayesian_cp",
    hazard: float = 1 / 50.0,
) -> Dict[str, Any]:
    """Find the voltage at which a charge transition occurs in a 1D sweep.

    Parameters
    ----------
    voltage : np.ndarray
        1-D array of sweep voltages.
    signal : np.ndarray
        1-D array of charge sensor response (same length as *voltage*).
    method : str
        Detection method: ``"bayesian_cp"`` (default) uses Bayesian
        changepoint detection followed by sigmoid refinement;
        ``"gradient"`` uses the derivative peak.
    hazard : float
        Hazard rate for the BayesianCP model (only used when
        method is ``"bayesian_cp"``).

    Returns
    -------
    dict
        ``{"position": float, "method": str, "success": bool, ...}``
        where ``position`` is the transition voltage.
    """
    voltage = np.asarray(voltage, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if method == "bayesian_cp" and _HAS_BAYESIAN_CP:
        try:
            return _detect_bayesian_cp(voltage, signal, hazard=hazard)
        except Exception:
            pass
    return _detect_gradient(voltage, signal)


def _detect_bayesian_cp(
    voltage: np.ndarray,
    signal: np.ndarray,
    *,
    hazard: float = 1 / 50.0,
) -> Dict[str, Any]:
    """Detect transition via BayesianCP + sigmoid refinement."""
    model = BayesianCP(hazard=hazard, standardize=True)
    cp_prob, _ = model.fit(jnp.asarray(signal))
    cp_prob = np.asarray(cp_prob)

    peak_idx = int(np.argmax(cp_prob[1:]) + 1)

    result = _refine_sigmoid(voltage, signal, peak_idx)
    result["method"] = "bayesian_cp"
    result["cp_prob"] = cp_prob
    return result


def _detect_gradient(voltage: np.ndarray, signal: np.ndarray) -> Dict[str, Any]:
    """Fallback: detect transition from the derivative peak."""
    grad = np.gradient(signal, voltage)
    peak_idx = int(np.argmax(np.abs(grad)))

    result = _refine_sigmoid(voltage, signal, peak_idx)
    result["method"] = "gradient"
    return result


def _refine_sigmoid(
    voltage: np.ndarray,
    signal: np.ndarray,
    initial_idx: int,
) -> Dict[str, Any]:
    """Refine transition position with a sigmoid fit around the initial estimate.

    Fits a window of +/- 20% of the sweep around the initial index.
    Falls back to the initial index if the fit fails.
    """
    n = len(voltage)
    window = max(10, n // 5)
    lo = max(0, initial_idx - window)
    hi = min(n, initial_idx + window)

    v_win = voltage[lo:hi]
    s_win = signal[lo:hi]

    x0_init = voltage[initial_idx]
    amp_init = float(s_win[-1] - s_win[0])
    width_init = float(np.abs(voltage[1] - voltage[0]) * 5)
    offset_init = float(s_win[0])

    try:
        popt, _ = curve_fit(
            _sigmoid,
            v_win,
            s_win,
            p0=[x0_init, width_init, amp_init, offset_init],
            maxfev=2000,
        )
        position = float(popt[0])
        if not (voltage[0] <= position <= voltage[-1]):
            position = float(voltage[initial_idx])
            success = False
        else:
            success = True
        return {
            "position": position,
            "success": success,
            "sigmoid_params": {"x0": popt[0], "width": popt[1], "amp": popt[2], "offset": popt[3]},
        }
    except (RuntimeError, ValueError):
        return {
            "position": float(voltage[initial_idx]),
            "success": False,
            "sigmoid_params": None,
        }


def extract_cross_capacitance_coefficient(
    ds_ref: xr.Dataset,
    ds_shifted: xr.Dataset,
    step_voltage: float,
    target_gate: str,
    perturb_gate: str,
    *,
    signal_var: str = "amplitude",
    sensor_idx: int = 0,
    method: Literal["bayesian_cp", "gradient"] = "bayesian_cp",
    hazard: float = 1 / 50.0,
) -> Dict[str, Any]:
    """Extract the cross-capacitance coefficient from paired 1D sweeps.

    Parameters
    ----------
    ds_ref : xr.Dataset
        Processed dataset from the reference (baseline) sweep.
    ds_shifted : xr.Dataset
        Processed dataset from the shifted sweep (perturbing gate
        stepped by *step_voltage*).
    step_voltage : float
        Voltage step applied to the perturbing gate (V).
    target_gate : str
        Name of the target plunger gate that was swept.
    perturb_gate : str
        Name of the perturbing gate that was stepped.
    signal_var : str
        Data variable to analyse (default ``"amplitude"``).
    sensor_idx : int
        Index along the ``sensors`` dimension if present.
    method : str
        Transition detection method (``"bayesian_cp"`` or ``"gradient"``).
    hazard : float
        Hazard rate for BayesianCP.

    Returns
    -------
    dict
        ``{"coefficient": float, "pos_ref": float, "pos_shifted": float,
          "target_gate": str, "perturb_gate": str,
          "ref_detection": dict, "shifted_detection": dict,
          "fit_params": {"success": bool, ...}}``
    """
    if abs(step_voltage) < 1e-12:
        raise ValueError("step_voltage must be non-zero.")

    sig_ref = ds_ref[signal_var]
    sig_shifted = ds_shifted[signal_var]

    if "sensors" in sig_ref.dims:
        sig_ref = sig_ref.isel(sensors=sensor_idx)
    if "sensors" in sig_shifted.dims:
        sig_shifted = sig_shifted.isel(sensors=sensor_idx)

    voltage = sig_ref.coords["x_volts"].values
    ref_signal = sig_ref.values
    shifted_signal = sig_shifted.values

    ref_det = detect_transition_position(voltage, ref_signal, method=method, hazard=hazard)
    shifted_det = detect_transition_position(voltage, shifted_signal, method=method, hazard=hazard)

    success = ref_det["success"] and shifted_det["success"]
    shift = shifted_det["position"] - ref_det["position"]
    coefficient = shift / step_voltage

    return {
        "coefficient": float(coefficient),
        "pos_ref": ref_det["position"],
        "pos_shifted": shifted_det["position"],
        "shift": float(shift),
        "target_gate": target_gate,
        "perturb_gate": perturb_gate,
        "ref_detection": ref_det,
        "shifted_detection": shifted_det,
        "fit_params": {
            "success": success,
            "reason": None if success else "one or both transition detections failed sigmoid refinement",
        },
    }
