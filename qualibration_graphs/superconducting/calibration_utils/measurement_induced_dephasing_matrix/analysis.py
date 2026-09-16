"""Analysis utilities for the measurement-induced dephasing matrix experiment.

The protocol follows Fig. 6 of Phys. Rev. Applied 23, 054089 (arXiv:2412.14853): a Hahn echo is
played on qubit Qi while a readout pulse of relative amplitude xi is inserted into the first half of
the echo on resonator Rj. The phase of the final pi/2 pulse is swept to reveal a coherent
oscillation whose contrast decays as

    c(xi) = c0 * exp(-Gamma * tau_p * xi**2)

with tau_p the duration of the probe (readout) pulse. Gamma is the measurement-induced dephasing
rate of the (Qi, Rj) pair; the diagonal holds the self-dephasing rates and the off-diagonal the
readout crosstalk.

The extraction is done in two stages:
  1. For every (qubit, driven resonator, xi) the oscillation is projected onto cos/sin of the swept
     phase by linear least squares. Because the phase is swept over exactly one turn the oscillation
     frequency is known, which makes this a closed-form fit that cannot fail to converge and that
     yields an analytic uncertainty on the contrast.
  2. ln(c) is fitted against xi**2 by weighted linear least squares, giving Gamma = -slope / tau_p
     and its standard error straight from the fit covariance.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

from .parameters import build_xi_values


@dataclass
class FitParameters:
    """Per measured-qubit summary of the measurement-induced dephasing matrix."""

    self_dephasing_hz: float
    self_dephasing_error_hz: float
    max_crosstalk_dephasing_hz: float
    worst_crosstalk_resonator: str
    success: bool


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ data to volts if needed and attach the per-pair ``xi`` coordinate.

    ``xi`` differs between the diagonal and the off-diagonal pairs, so it cannot be a dimension of
    the dataset. It is rebuilt from the node parameters (which are persisted alongside the data, so
    this also works when loading a historical dataset) and attached as a two-dimensional
    non-dimension coordinate over ``(qubit, driven_resonator, xi_idx)``.
    """
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    qubit_names = [str(q) for q in ds.qubit.values]
    xi_values = build_xi_values(qubit_names, node.parameters)
    ds = ds.assign_coords(
        xi=(
            ("qubit", "driven_resonator", "xi_idx"),
            xi_values,
            {"long_name": "relative readout amplitude", "units": ""},
        )
    )
    return ds


def _probe_durations_in_s(ds: xr.Dataset, node: QualibrationNode) -> xr.DataArray:
    """Return tau_p, the probe pulse duration of each driven resonator, in seconds."""
    durations = [
        node.machine.qubits[str(name)].resonator.operations["readout"].length * 1e-9
        for name in ds.driven_resonator.values
    ]
    return xr.DataArray(durations, coords={"driven_resonator": ds.driven_resonator}, dims="driven_resonator")


def _fit_phase_oscillations(signal: np.ndarray, phases_rad: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Extract the oscillation contrast and its uncertainty by linear least squares.

    The model is ``offset + A cos(phi) + B sin(phi)`` with the frequency fixed to one oscillation
    over the swept phase range, so the fit is linear in its parameters and solved in closed form for
    all pairs and all xi at once.

    Parameters
    ----------
    signal : np.ndarray
        Measured signal with the swept phase along the last axis.
    phases_rad : np.ndarray
        The swept phases, in radians.

    Returns
    -------
    contrast, contrast_error, offset, phase_offset : np.ndarray
        Each with the shape of ``signal`` minus its last axis. ``contrast`` is the peak-to-mean
        amplitude of the oscillation and ``phase_offset`` its phase, in radians.
    """
    design = np.stack([np.ones_like(phases_rad), np.cos(phases_rad), np.sin(phases_rad)], axis=-1)
    normal_inv = np.linalg.inv(design.T @ design)
    coefficients = signal @ (normal_inv @ design.T).T
    residuals = signal - coefficients @ design.T

    degrees_of_freedom = max(len(phases_rad) - design.shape[-1], 1)
    variance = np.sum(residuals**2, axis=-1) / degrees_of_freedom

    offset, cos_amp, sin_amp = coefficients[..., 0], coefficients[..., 1], coefficients[..., 2]
    contrast = np.hypot(cos_amp, sin_amp)
    # Propagate the parameter covariance onto sqrt(A^2 + B^2).
    cos_var, sin_var = variance * normal_inv[1, 1], variance * normal_inv[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        contrast_error = np.sqrt((cos_amp**2 * cos_var + sin_amp**2 * sin_var)) / contrast
    contrast_error = np.where(contrast > 0, contrast_error, np.nan)
    return contrast, contrast_error, offset, np.arctan2(sin_amp, cos_amp)


def _fit_dephasing_rates(
    contrast: np.ndarray,
    contrast_error: np.ndarray,
    xi: np.ndarray,
    probe_durations_s: np.ndarray,
    min_contrast_snr: float,
) -> Tuple[np.ndarray, ...]:
    """Fit ln(c) against xi**2 by weighted least squares, pair by pair.

    Points whose contrast has fallen into the noise floor are dropped: once the oscillation is fully
    dephased the fitted amplitude is a positive-biased noise estimate that would flatten the decay
    and bias Gamma downwards.

    Returns
    -------
    gamma, gamma_error, c0 : np.ndarray
        Dephasing rate in Hz, its standard error in Hz, and the extrapolated zero-amplitude contrast.
        Each has the shape of ``contrast`` minus its last axis.
    """
    shape = contrast.shape[:-1]
    gamma = np.full(shape, np.nan)
    gamma_error = np.full(shape, np.nan)
    c0 = np.full(shape, np.nan)

    for index in np.ndindex(shape):
        c = contrast[index]
        sigma_c = contrast_error[index]
        x = xi[index] ** 2
        valid = np.isfinite(c) & np.isfinite(sigma_c) & (c > 0) & (sigma_c > 0)
        valid &= c > min_contrast_snr * sigma_c
        if valid.sum() < 3:
            continue

        x, c, sigma_c = x[valid], c[valid], sigma_c[valid]
        y = np.log(c)
        # sigma_ln(c) = sigma_c / c, hence the weights below.
        weights = (c / sigma_c) ** 2

        design = np.stack([np.ones_like(x), x], axis=-1)
        normal = design.T @ (weights[:, None] * design)
        try:
            normal_inv = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            continue
        intercept, slope = normal_inv @ (design.T @ (weights * y))

        # Rescale the covariance by the reduced chi-squared so that the reported error also reflects
        # any scatter beyond the propagated shot noise.
        residuals = y - (intercept + slope * x)
        degrees_of_freedom = max(len(x) - 2, 1)
        chi2_reduced = np.sum(weights * residuals**2) / degrees_of_freedom
        slope_error = np.sqrt(normal_inv[1, 1] * max(chi2_reduced, 1.0))

        tau_p = probe_durations_s[index[1]]
        gamma[index] = -slope / tau_p
        gamma_error[index] = slope_error / tau_p
        c0[index] = np.exp(intercept)

    return gamma, gamma_error, c0


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Extract the measurement-induced dephasing matrix from the raw dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with dimensions ``(qubit, driven_resonator, xi_idx, phase)``.
    node : QualibrationNode
        Node whose parameters and machine drive the fit.

    Returns
    -------
    ds_fit : xr.Dataset
        The input dataset augmented with the per-xi contrast and the per-pair dephasing rates.
    fit_results : dict
        One :class:`FitParameters` per measured qubit.
    """
    signal = ds.state if node.parameters.use_state_discrimination else ds.I
    signal = signal.transpose("qubit", "driven_resonator", "xi_idx", "phase")

    contrast, contrast_error, offset, phase_offset = _fit_phase_oscillations(
        signal.values, ds.phase.values
    )

    pair_dims = ("qubit", "driven_resonator", "xi_idx")
    pair_coords = {dim: ds[dim] for dim in pair_dims}
    ds_fit = ds.assign(
        contrast=xr.DataArray(contrast, coords=pair_coords, dims=pair_dims),
        contrast_error=xr.DataArray(contrast_error, coords=pair_coords, dims=pair_dims),
        oscillation_offset=xr.DataArray(offset, coords=pair_coords, dims=pair_dims),
        oscillation_phase=xr.DataArray(phase_offset, coords=pair_coords, dims=pair_dims),
    )
    ds_fit["contrast"].attrs = {"long_name": "echo oscillation contrast", "units": ""}

    probe_durations = _probe_durations_in_s(ds, node)
    gamma, gamma_error, c0 = _fit_dephasing_rates(
        contrast,
        contrast_error,
        ds.xi.transpose(*pair_dims).values,
        probe_durations.values,
        node.parameters.min_contrast_snr,
    )

    matrix_dims = ("qubit", "driven_resonator")
    matrix_coords = {dim: ds[dim] for dim in matrix_dims}
    ds_fit = ds_fit.assign(
        Gamma=xr.DataArray(gamma, coords=matrix_coords, dims=matrix_dims),
        Gamma_error=xr.DataArray(gamma_error, coords=matrix_coords, dims=matrix_dims),
        c0=xr.DataArray(c0, coords=matrix_coords, dims=matrix_dims),
        tau_p=probe_durations,
    )
    ds_fit["Gamma"].attrs = {"long_name": "measurement-induced dephasing rate", "units": "Hz"}
    ds_fit["Gamma_error"].attrs = {"long_name": "dephasing rate error", "units": "Hz"}
    ds_fit["tau_p"].attrs = {"long_name": "probe pulse duration", "units": "s"}
    ds_fit["contrast_relative"] = ds_fit.contrast / ds_fit.c0
    ds_fit["contrast_relative"].attrs = {"long_name": "c / c0", "units": ""}

    return _extract_relevant_fit_parameters(ds_fit, node)


def _extract_relevant_fit_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Summarise the matrix into one result per measured qubit and assess the fit outcome."""
    is_diagonal = xr.DataArray(
        np.equal.outer(ds_fit.qubit.values, ds_fit.driven_resonator.values),
        coords={"qubit": ds_fit.qubit, "driven_resonator": ds_fit.driven_resonator},
        dims=("qubit", "driven_resonator"),
    )
    ds_fit = ds_fit.assign(is_diagonal=is_diagonal)

    crosstalk = ds_fit.Gamma.where(~is_diagonal)
    threshold = node.parameters.max_crosstalk_dephasing_in_hz
    # A NaN crosstalk element means its fit did not converge, which we do not want to pass silently.
    success = (crosstalk.max(dim="driven_resonator") < threshold) & (
        crosstalk.notnull().sum(dim="driven_resonator") == len(ds_fit.driven_resonator) - 1
    )
    ds_fit = ds_fit.assign(success=success)

    fit_results = {}
    for q in ds_fit.qubit.values:
        crosstalk_q = crosstalk.sel(qubit=q)
        if bool(crosstalk_q.notnull().any()):
            worst = str(crosstalk_q.idxmax(dim="driven_resonator").values)
            worst_value = float(crosstalk_q.max())
        else:
            worst, worst_value = "", np.nan
        fit_results[str(q)] = FitParameters(
            self_dephasing_hz=float(ds_fit.Gamma.sel(qubit=q, driven_resonator=q)),
            self_dephasing_error_hz=float(ds_fit.Gamma_error.sel(qubit=q, driven_resonator=q)),
            max_crosstalk_dephasing_hz=worst_value,
            worst_crosstalk_resonator=worst,
            success=bool(ds_fit.success.sel(qubit=q)),
        )
    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the self-dephasing rate and the worst crosstalk element for each measured qubit.

    Parameters
    ----------
    fit_results : dict
        Dictionary containing the fitted results for all measured qubits.
    log_callable : callable, optional
        Logging callable (e.g. ``logging.Logger.info``). If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qubit_name, result in fit_results.items():
        self_mhz = result["self_dephasing_hz"] * 1e-6
        self_error_mhz = result["self_dephasing_error_hz"] * 1e-6
        message = (
            f"{qubit_name}: self-dephasing = {self_mhz:.2f} +/- {self_error_mhz:.2f} MHz | "
            f"worst crosstalk = {result['max_crosstalk_dephasing_hz']:.1f} Hz "
            f"(driven by {result['worst_crosstalk_resonator']}) | "
            f"{'PASS' if result['success'] else 'FAIL'}"
        )
        log_callable(message)
