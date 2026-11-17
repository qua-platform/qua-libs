import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class RingdownFit:
    """Stores the resonator ring-down fit of a single qubit, with the resonator rung up with the qubit in |g> and |e>"""

    tau_ground_ns: float
    tau_excited_ns: float
    depletion_time_ns: float
    success: bool


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def _fit_fringe(signal: np.ndarray, phases: np.ndarray):
    """Least-squares fit of offset + c cos(2 pi phase) + s sin(2 pi phase).

    Returns the contrast, the phase and the fitted fringe.
    """
    theta = 2 * np.pi * phases
    design = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])
    offset, cosine, sine = np.linalg.lstsq(design, signal, rcond=None)[0]
    return 2 * np.hypot(cosine, sine), np.arctan2(-sine, cosine), design @ [offset, cosine, sine]


def _exponential(t, amplitude, tau, asymptote):
    return amplitude * np.exp(-np.asarray(t) / tau) + asymptote


def _fit_decay(t: np.ndarray, phase: np.ndarray):
    """Fit the residual AC-Stark phase to amplitude * exp(-t / tau) + asymptote. Returns (popt, tau_error)."""
    try:
        popt, pcov = curve_fit(
            _exponential,
            t,
            phase,
            p0=(phase[0] - phase[-1], max(100.0, (t[-1] - t[0]) / 3), phase[-1]),
            bounds=([-4 * np.pi, 4.0, -20 * np.pi], [4 * np.pi, 20 * t[-1], 20 * np.pi]),
            maxfev=20000,
        )
        return popt, float(np.sqrt(np.diag(pcov))[1])
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return np.full(3, np.nan), np.nan


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, RingdownFit]]:
    """
    Fit the Ramsey fringe at every delay, then the decay of the Ramsey phase with the delay after the readout pulse.

    The residual photons left in the resonator after the readout pulse shift the qubit frequency, so the Ramsey
    phase decays with the delay as the resonator rings down. The depletion time is the longest delay, over the
    ground and excited preparations, after which the residual phase has decayed to ``residual_fraction``.

    Returns
    -------
    xr.Dataset
        Dataset with the fringe contrast and phase vs. delay, the phase fit, tau and the depletion time.
    dict[str, RingdownFit]
        The fit results of every qubit.
    """
    signal = ds.state if node.parameters.use_state_discrimination else ds.I
    t = ds.wait_time.values.astype(float)
    phases = ds.phase.values

    shape = (ds.sizes["qubit"], ds.sizes["init_state"], len(t))
    contrast, phase, phase_fit = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    tau, tau_error = np.full(shape[:2], np.nan), np.full(shape[:2], np.nan)
    for i, q in enumerate(ds.qubit.values):
        for k, init_state in enumerate(ds.init_state.values):
            for m in range(len(t)):
                trace = signal.sel(qubit=q, init_state=init_state).isel(wait_time=m).values
                contrast[i, k, m], phase[i, k, m], _ = _fit_fringe(trace, phases)
            phase[i, k] = np.unwrap(phase[i, k])
            popt, tau_error[i, k] = _fit_decay(t, phase[i, k])
            tau[i, k] = popt[1]
            phase_fit[i, k] = _exponential(t, *popt)

    dims = ("qubit", "init_state", "wait_time")
    ds_fit = xr.Dataset(
        {
            "contrast": (dims, contrast),
            "ramsey_phase": (dims, phase, {"long_name": "Ramsey phase", "units": "rad"}),
            "ramsey_phase_fit": (dims, phase_fit, {"long_name": "fitted Ramsey phase", "units": "rad"}),
            "tau": (dims[:2], tau, {"long_name": "resonator ring-down time", "units": "ns"}),
            "tau_error": (dims[:2], tau_error, {"long_name": "resonator ring-down time error", "units": "ns"}),
        },
        coords={"qubit": ds.qubit, "init_state": ds.init_state, "wait_time": ds.wait_time},
    )
    ds_fit["success"] = (np.isfinite(ds_fit.tau) & (ds_fit.tau_error < 0.5 * ds_fit.tau)).all("init_state")
    ds_fit["depletion_time"] = np.log(1 / node.parameters.residual_fraction) * ds_fit.tau.max("init_state")
    ds_fit.depletion_time.attrs = {"long_name": "resonator depletion time", "units": "ns"}
    # Each preparation's phase rescaled from its shortest delay (1) to its late-delay baseline (0), the mean of the last
    # quarter of the delays. This removes the 180 deg offset and the different amplitudes of |g> and |e> without relying
    # on the decay fit.
    tail = ds_fit.ramsey_phase.isel(wait_time=slice(-max(1, len(t) // 4), None)).mean("wait_time")
    ds_fit["normalized_phase"] = (ds_fit.ramsey_phase - tail) / (ds_fit.ramsey_phase.isel(wait_time=0) - tail)
    ds_fit.normalized_phase.attrs = {"long_name": "normalized Ramsey phase", "units": ""}

    fit_results = {
        q: RingdownFit(
            tau_ground_ns=float(ds_fit.tau.sel(qubit=q, init_state="g")),
            tau_excited_ns=float(ds_fit.tau.sel(qubit=q, init_state="e")),
            depletion_time_ns=float(ds_fit.depletion_time.sel(qubit=q)),
            success=bool(ds_fit.success.sel(qubit=q)),
        )
        for q in ds_fit.qubit.values
    }
    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict[str, dict], log_callable=None):
    """
    Logs the ring-down times and the depletion time of every qubit.

    Parameters:
    -----------
    fit_results : dict
        Fit results keyed by qubit name, as produced by ``asdict`` of ``RingdownFit``.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, fit in fit_results.items():
        log_callable(
            f"Resonator ring-down of {q} : tau(g) = {fit['tau_ground_ns']:.0f} ns, "
            f"tau(e) = {fit['tau_excited_ns']:.0f} ns, "
            f"depletion time = {fit['depletion_time_ns']:.0f} ns --> {'SUCCESS!' if fit['success'] else 'FAIL!'}"
        )
