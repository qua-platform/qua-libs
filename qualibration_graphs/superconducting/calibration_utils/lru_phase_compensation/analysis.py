"""Fit of the f-LRU phase compensation (node 26c).

The sequence is x90 - LRU (with the stored virtual Z ``c``) - virtual Z ``theta`` - x90, read as g/e.
With no residual phase the two x90 add up to x180, so P(e) peaks at theta = 0. A residual phase ``psi``
per LRU moves the peak: P(e) = offset + A cos(2 pi (theta + N psi)) for N LRUs, A > 0. Shifting the
correction by -psi cancels it: ``c_new = c - psi``, the convention of the CZ node 34a.

The frequency of the oscillation over theta is one per turn by construction, so the fit is linear in
(1, cos, sin) and gives the phase with its standard error directly.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

#: Smallest fitted Ramsey contrast A (P(e) units) for a usable phase. Below it the qubit has dephased
#: or the readout is not discriminating.
MIN_CONTRAST = 0.1
#: Largest standard error of the fitted phase, in turns, for a usable phase.
MAX_PHASE_STDERR = 0.02


@dataclass
class FitParameters:
    """Phase compensation result of one qubit; phases in units of 2 pi (turns)."""

    residual_phase: float
    """Residual phase per LRU with the correction stored before this run, wrapped to [-0.5, 0.5)."""
    residual_phase_stderr: float
    previous_phase_shift: float
    """``macros["lru"].phase_shift`` before the run (0 when no macro was stored)."""
    phase_shift: float
    """New correction ``previous_phase_shift - residual_phase``, in [0, 1)."""
    contrast: float
    success: bool
    fit_issues: Tuple[str, ...] = ()


def wrap_turns(phase):
    """Wrap a phase in turns to [-0.5, 0.5)."""
    return (np.asarray(phase) + 0.5) % 1 - 0.5


def fit_frame_oscillation(frames: np.ndarray, pe: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
    """Least-squares fit of pe = offset + A cos(2 pi (frame + psi)) with A >= 0.

    Returns:
        (psi in turns wrapped to [-0.5, 0.5), its standard error, A, offset, fitted curve).
    """
    x = 2 * np.pi * np.asarray(frames, dtype=float)
    y = np.asarray(pe, dtype=float)
    design = np.column_stack([np.ones_like(x), np.cos(x), np.sin(x)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    offset, c, s = coeffs
    curve = design @ coeffs
    # A cos(x + phi) = A cos(phi) cos(x) - A sin(phi) sin(x)
    amplitude = float(np.hypot(c, s))
    phi = float(np.arctan2(-s, c))
    dof = max(len(y) - 3, 1)
    sigma2 = float(np.sum((y - curve) ** 2) / dof)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    # d(phi) = (-c ds + s dc) / A^2 ... propagated from the (c, s) covariance
    grad = np.array([0.0, s, -c]) / max(amplitude**2, 1e-12)
    phi_stderr = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    return float(wrap_turns(phi / (2 * np.pi))), phi_stderr / (2 * np.pi), amplitude, float(offset), curve


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit P(e) against the frame for each qubit and compute the new ``phase_shift``."""
    qubits = node.namespace["qubits"]
    repetitions = node.parameters.num_lru_repetitions
    previous = node.namespace.get("previous_phase_shifts", {})
    frames = ds.frame.values
    results, curves = {}, []
    for qubit in qubits:
        pe = ds.state.sel(qubit=qubit.name).values
        psi, psi_stderr, contrast, _, curve = fit_frame_oscillation(frames, pe)
        residual, residual_stderr = psi / repetitions, psi_stderr / repetitions
        old = float(previous.get(qubit.name, 0.0))
        issues = []
        if contrast < MIN_CONTRAST:
            issues.append(f"Ramsey contrast {contrast:.3f} is below {MIN_CONTRAST}: no usable phase.")
        if residual_stderr > MAX_PHASE_STDERR:
            issues.append(f"Phase uncertainty {residual_stderr:.3f} turn is above {MAX_PHASE_STDERR} turn.")
        results[qubit.name] = FitParameters(
            residual_phase=float(wrap_turns(residual)),
            residual_phase_stderr=float(residual_stderr),
            previous_phase_shift=old,
            phase_shift=float((old - residual) % 1),
            contrast=float(contrast),
            success=not issues,
            fit_issues=tuple(issues),
        )
        curves.append(curve)

    names = [q.name for q in qubits]
    per_qubit = lambda key: ("qubit", [getattr(results[n], key) for n in names])  # noqa: E731
    ds_fit = ds.assign(
        fitted_state=(["qubit", "frame"], np.asarray(curves)),
        residual_phase=per_qubit("residual_phase"),
        residual_phase_stderr=per_qubit("residual_phase_stderr"),
        previous_phase_shift=per_qubit("previous_phase_shift"),
        phase_shift=per_qubit("phase_shift"),
        contrast=per_qubit("contrast"),
        success=per_qubit("success"),
    )
    ds_fit.residual_phase.attrs = {"long_name": "residual qubit phase per LRU", "units": "2π"}
    ds_fit.phase_shift.attrs = {"long_name": "LRU virtual-Z correction", "units": "2π"}
    return ds_fit, results


def log_fitted_results(fit_results: Dict[str, dict], log_callable=None) -> None:
    """Log the residual phase and the new correction of each qubit."""
    log_callable = log_callable or logging.getLogger(__name__).info
    for name, r in fit_results.items():
        status = "SUCCESS!" if r["success"] else "FAIL!"
        message = (
            f"Results for qubit {name}: {status}\n"
            f"\tResidual phase per LRU: {r['residual_phase'] * 360:+.2f} +/- {r['residual_phase_stderr'] * 360:.2f} deg"
            f" (correction before: {r['previous_phase_shift'] * 360:.2f} deg)\n"
            f"\tNew correction: {wrap_turns(r['phase_shift']) * 360:+.2f} deg ({r['phase_shift']:.5f} turn)\n"
            f"\tRamsey contrast: {r['contrast']:.3f}"
        )
        for issue in r["fit_issues"]:
            message += f"\n\t- {issue}"
        log_callable(message)
