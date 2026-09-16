"""Analysis utilities for the measurement-induced dephasing matrix experiment.

The protocol follows Fig. 6 of Phys. Rev. Applied 23, 054089 (arXiv:2412.14853): a Hahn echo is
played on qubit Qi while a readout pulse of relative amplitude xi is played on resonator Rj, either
in the first half of the echo only or in both halves. The phase of the final pi/2 pulse is swept to
reveal a coherent oscillation, from which two independent observables are extracted.

The photons in Rj act on Qi through the cross-Kerr shift chi_ij, and they do so in two ways:

  * their mean number pulls the qubit frequency, which accumulates a deterministic AC-Stark phase
    linear in chi_ij. It shows up as a shift of the oscillation;
  * their fluctuations dephase the qubit at a rate quadratic in chi_ij. It shows up as a loss of
    oscillation contrast.

Because the Stark shift is linear in chi_ij and the dephasing quadratic, the Stark channel carries
signal over a far wider range of coupling strengths. Both are therefore fitted and reported:

    c(xi)   = c0 * exp(-Gamma * tau_probe * xi**2)
    phi(xi) = phi0 + 2*pi * delta_f * tau_probe * xi**2

with tau_probe the total time the driven resonator is probed during one echo. Gamma is the
measurement-induced dephasing rate of the (Qi, Rj) pair and delta_f the AC-Stark shift it would
experience at the calibrated readout amplitude. The diagonals hold the self-dephasing and the
self-Stark shift, the off-diagonals the readout crosstalk.

When the probe is played in both halves of the echo the x180 pulse refocuses the Stark phase, so
only the dephasing matrix is meaningful and only that one is computed.

The extraction is done in two stages:
  1. For every (qubit, driven resonator, xi) the oscillation is projected onto cos/sin of the swept
     phase by linear least squares. Because the phase is swept over exactly one turn the oscillation
     frequency is known, which makes this a closed-form fit that cannot fail to converge and that
     yields analytic uncertainties on the contrast and on the phase. The contrast is debiased for
     the noise floor, which an amplitude built as sqrt(A**2 + B**2) would otherwise overestimate.
  2. ln(c) is fitted against xi**2, and the unwrapped phase against xi**2, by weighted linear least
     squares. Each fit reports its reduced chi-squared, which gates whether the element is
     trustworthy, and its standard error, which decides whether the element is resolved at all.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

from .parameters import build_xi_values, total_probe_length_in_ns

# An element is called resolved when its magnitude exceeds this many standard errors. Below it the
# fit is consistent with zero and the element can only be reported as an upper bound.
RESOLVED_SIGMA = 2.0

# How far the xi = 0 contrasts of one row may disagree, in sigma, before it is worth reporting. They
# come from identical pulse sequences, so anything well outside shot noise points at the setup.
ZERO_AMPLITUDE_SPREAD_SIGMA = 5.0


@dataclass
class FitParameters:
    """Per measured-qubit summary of the measurement-induced dephasing matrix.

    ``max_crosstalk_dephasing_hz`` is the signed rate of the resolved off-diagonal element with the
    largest magnitude, or NaN when no element of that row is resolved. ``crosstalk_upper_bound_hz``
    is the largest ``|Gamma| + 2*sigma`` over the row, which bounds the crosstalk whether or not any
    element was resolved and is the right number to quote when nothing was resolved.

    ``zero_amplitude_spread_sigma`` is an internal consistency check rather than a result: see
    :func:`_zero_amplitude_spread`.

    ``max_crosstalk_stark_shift_hz`` and ``worst_stark_resonator`` describe the AC-Stark channel in
    the same way. They are NaN and empty when the probe was played in both halves of the echo, since
    the echo refocuses the Stark phase and the channel does not exist.
    """

    self_dephasing_hz: float
    self_dephasing_error_hz: float
    max_crosstalk_dephasing_hz: float
    max_crosstalk_dephasing_error_hz: float
    worst_crosstalk_resonator: str
    crosstalk_upper_bound_hz: float
    num_unresolved_crosstalk: int
    num_rejected_crosstalk: int
    zero_amplitude_spread_sigma: float
    self_stark_shift_hz: float
    max_crosstalk_stark_shift_hz: float
    worst_stark_resonator: str
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
    """Return tau_probe, the total probe time per echo for each driven resonator, in seconds.

    This has to match what the QUA program actually played, including any stretching asked for
    through ``readout_len_in_ns`` and the doubling when the probe sits in both halves of the echo,
    because both fitted slopes are divided by it. Both sides go through ``total_probe_length_in_ns``
    so they cannot disagree.
    """
    durations = [
        total_probe_length_in_ns(
            node.machine.qubits[str(name)].resonator.operations["readout"].length, node.parameters
        )
        * 1e-9
        for name in ds.driven_resonator.values
    ]
    return xr.DataArray(durations, coords={"driven_resonator": ds.driven_resonator}, dims="driven_resonator")


def _fit_phase_oscillations(signal: np.ndarray, phases_rad: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Extract the oscillation contrast and phase, with their uncertainties, by linear least squares.

    The model is ``offset + A cos(phi) + B sin(phi)`` with the frequency fixed to one oscillation
    over the swept phase range, so the fit is linear in its parameters and solved in closed form for
    all pairs and all xi at once.

    The amplitude ``sqrt(A**2 + B**2)`` is a biased estimate of the true contrast: it is positive
    even when the true contrast is zero, because it folds the noise on A and on B into itself. Left
    alone that bias flattens the tail of the decay and pulls the fitted dephasing rate downwards,
    which is exactly the regime the off-diagonal elements live in. Since the variances of A and B are
    known from the same fit, they are subtracted from the squared amplitude, which removes the bias
    to leading order. Points whose debiased square comes out negative have no resolvable contrast
    and are returned as NaN.

    Parameters
    ----------
    signal : np.ndarray
        Measured signal with the swept phase along the last axis.
    phases_rad : np.ndarray
        The swept phases, in radians.

    Returns
    -------
    contrast, contrast_error, raw_contrast, offset, phase, phase_error : np.ndarray
        Each with the shape of ``signal`` minus its last axis. ``contrast`` is the debiased
        peak-to-mean amplitude of the oscillation, ``raw_contrast`` the biased one kept for plotting,
        and ``phase`` the oscillation phase in radians.
    """
    design = np.stack([np.ones_like(phases_rad), np.cos(phases_rad), np.sin(phases_rad)], axis=-1)
    normal_inv = np.linalg.inv(design.T @ design)
    coefficients = signal @ (normal_inv @ design.T).T
    residuals = signal - coefficients @ design.T

    degrees_of_freedom = max(len(phases_rad) - design.shape[-1], 1)
    variance = np.sum(residuals**2, axis=-1) / degrees_of_freedom

    offset, cos_amp, sin_amp = coefficients[..., 0], coefficients[..., 1], coefficients[..., 2]
    raw_contrast = np.hypot(cos_amp, sin_amp)
    cos_var, sin_var = variance * normal_inv[1, 1], variance * normal_inv[2, 2]

    # Propagate the parameter covariance onto sqrt(A^2 + B^2). The raw amplitude is used as the
    # denominator because it is the better conditioned estimate of the magnitude; using the debiased
    # one would make the error diverge exactly where the debiasing matters.
    with np.errstate(divide="ignore", invalid="ignore"):
        contrast_error = np.sqrt(cos_amp**2 * cos_var + sin_amp**2 * sin_var) / raw_contrast
    contrast_error = np.where(raw_contrast > 0, contrast_error, np.nan)

    # Remove the noise floor folded into the amplitude by the square root.
    debiased_square = raw_contrast**2 - (cos_var + sin_var)
    contrast = np.sqrt(np.where(debiased_square > 0, debiased_square, np.nan))

    phase = np.arctan2(sin_amp, cos_amp)
    # For a small perturbation of a sinusoid the phase error is the relative amplitude error.
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_error = np.where(raw_contrast > 0, contrast_error / raw_contrast, np.nan)
    return contrast, contrast_error, raw_contrast, offset, phase, phase_error


def _weighted_line_fit(x: np.ndarray, y: np.ndarray, sigma_y: np.ndarray) -> Tuple[float, ...]:
    """Fit ``y = intercept + slope * x`` by weighted least squares.

    Returns ``(intercept, slope, slope_error, reduced_chi2)``, or NaNs when the system is singular.
    The reported slope error is inflated by the reduced chi-squared whenever that exceeds one, so
    that scatter beyond the propagated shot noise widens the error bar rather than hiding in it.
    """
    weights = 1.0 / sigma_y**2
    design = np.stack([np.ones_like(x), x], axis=-1)
    normal = design.T @ (weights[:, None] * design)
    try:
        normal_inv = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    intercept, slope = normal_inv @ (design.T @ (weights * y))

    residuals = y - (intercept + slope * x)
    degrees_of_freedom = max(len(x) - 2, 1)
    reduced_chi2 = float(np.sum(weights * residuals**2) / degrees_of_freedom)
    slope_error = float(np.sqrt(normal_inv[1, 1] * max(reduced_chi2, 1.0)))
    return float(intercept), float(slope), slope_error, reduced_chi2


def _valid_points(contrast: np.ndarray, contrast_error: np.ndarray, min_contrast_snr: float) -> np.ndarray:
    """Return the mask of amplitude points whose oscillation is resolved above the noise floor."""
    valid = np.isfinite(contrast) & np.isfinite(contrast_error) & (contrast > 0) & (contrast_error > 0)
    return valid & (contrast > min_contrast_snr * contrast_error)


def _fit_dephasing_rates(
    contrast: np.ndarray,
    contrast_error: np.ndarray,
    xi: np.ndarray,
    probe_durations_s: np.ndarray,
    min_contrast_snr: float,
    max_reduced_chi2: float,
) -> Tuple[np.ndarray, ...]:
    """Fit ln(c) against xi**2 by weighted least squares, pair by pair.

    Points whose contrast has fallen into the noise floor are dropped: once the oscillation is fully
    dephased the fitted amplitude carries no information and its logarithm is dominated by noise.

    Returns
    -------
    gamma, gamma_error, c0, chi2, rejected : np.ndarray
        Dephasing rate in Hz, its standard error in Hz, the extrapolated zero-amplitude contrast, the
        reduced chi-squared of the straight-line fit, and whether that chi-squared exceeds
        ``max_reduced_chi2``. Each has the shape of ``contrast`` minus its last axis.
    """
    shape = contrast.shape[:-1]
    gamma = np.full(shape, np.nan)
    gamma_error = np.full(shape, np.nan)
    c0 = np.full(shape, np.nan)
    chi2 = np.full(shape, np.nan)
    rejected = np.zeros(shape, dtype=bool)

    for index in np.ndindex(shape):
        valid = _valid_points(contrast[index], contrast_error[index], min_contrast_snr)
        if valid.sum() < 3:
            continue
        c, sigma_c = contrast[index][valid], contrast_error[index][valid]
        x = xi[index][valid] ** 2
        # sigma_ln(c) = sigma_c / c.
        intercept, slope, slope_error, reduced_chi2 = _weighted_line_fit(x, np.log(c), sigma_c / c)
        if not np.isfinite(slope):
            continue

        tau_probe = probe_durations_s[index[1]]
        gamma[index] = -slope / tau_probe
        gamma_error[index] = slope_error / tau_probe
        c0[index] = np.exp(intercept)
        chi2[index] = reduced_chi2
        rejected[index] = reduced_chi2 > max_reduced_chi2

    return gamma, gamma_error, c0, chi2, rejected


def _progressive_unwrap(x: np.ndarray, phase: np.ndarray) -> Tuple[np.ndarray, float]:
    """Unwrap a phase that is expected to be linear in ``x``, using the trend rather than the steps.

    ``np.unwrap`` assumes that consecutive samples move by less than pi, which fails here: the phase
    advances as xi**2 while xi is swept linearly, so the step between consecutive points grows
    quadratically and the last steps of a strongly shifted pair are several radians. Rejecting those
    pairs would throw away exactly the data the Stark channel exists to measure.

    Instead the line is grown point by point. The first three points, whose steps are the smallest of
    the sweep, are unwrapped conventionally to seed a straight line; every later point is then placed
    on the branch of 2*pi nearest to what that line predicts, and the line is refitted. This is
    unambiguous as long as the prediction is good to better than half a turn, which the returned
    largest residual reports.

    Parameters
    ----------
    x : np.ndarray
        Abscissa, sorted ascending. Here the squared relative readout amplitude.
    phase : np.ndarray
        Wrapped phase, in radians, in the same order.

    Returns
    -------
    unwrapped, largest_residual : np.ndarray, float
        The unwrapped phase and the largest distance, in radians, between a placed point and the
        prediction that placed it. A residual approaching pi means the branch choice was a coin toss
        and the result cannot be trusted.
    """
    unwrapped = np.unwrap(phase)
    if len(phase) < 4:
        return unwrapped, 0.0

    unwrapped[:3] = np.unwrap(phase[:3])
    largest_residual = 0.0
    for stop in range(3, len(phase)):
        # Least squares on the points placed so far, which is well conditioned from three points on.
        slope, intercept = np.polyfit(x[:stop], unwrapped[:stop], 1)
        predicted = intercept + slope * x[stop]
        # Place the point on the branch of 2*pi closest to the prediction.
        residual = np.remainder(phase[stop] - predicted + np.pi, 2 * np.pi) - np.pi
        unwrapped[stop] = predicted + residual
        largest_residual = max(largest_residual, abs(float(residual)))
    return unwrapped, largest_residual


def _fit_stark_shifts(
    phase: np.ndarray,
    phase_error: np.ndarray,
    contrast: np.ndarray,
    contrast_error: np.ndarray,
    xi: np.ndarray,
    probe_durations_s: np.ndarray,
    min_contrast_snr: float,
    max_reduced_chi2: float,
    max_phase_residual_rad: float,
) -> Tuple[np.ndarray, ...]:
    """Fit the unwrapped AC-Stark phase against xi**2 by weighted least squares, pair by pair.

    The photons in the driven resonator pull the measured qubit's frequency by an amount proportional
    to their mean number, hence to xi**2, and the probe sitting in one half of the echo only leaves
    that phase unrefocused. The fitted slope divided by ``2*pi*tau_probe`` is the frequency shift the
    qubit would see at the calibrated readout amplitude, xi = 1.

    Because this shift is linear in the cross-Kerr coupling while the dephasing rate is quadratic in
    it, this channel keeps a usable signal on pairs whose dephasing is far below the noise floor.

    The phase is only defined modulo one turn, so it is unwrapped along the amplitude axis by
    :func:`_progressive_unwrap`. An element is rejected when that unwrapping was ambiguous, meaning
    some point sat further than ``max_phase_residual_rad`` from the trend that placed it, or when the
    straight-line fit does not describe the data within the shot noise.

    Returns
    -------
    stark_hz, stark_error_hz, chi2, residual, rejected : np.ndarray
        The Stark shift at xi = 1 in Hz, its standard error in Hz, the reduced chi-squared of the
        straight-line fit, the largest unwrapping residual in radians, and whether the element failed
        either check. Each has the shape of ``phase`` minus its last axis.
    """
    shape = phase.shape[:-1]
    stark_hz = np.full(shape, np.nan)
    stark_error_hz = np.full(shape, np.nan)
    chi2 = np.full(shape, np.nan)
    residual = np.full(shape, np.nan)
    rejected = np.zeros(shape, dtype=bool)

    for index in np.ndindex(shape):
        # The phase of an oscillation that has been dephased away is meaningless, so the same noise
        # floor cut as the contrast fit is applied before unwrapping.
        valid = _valid_points(contrast[index], contrast_error[index], min_contrast_snr)
        valid &= np.isfinite(phase[index]) & np.isfinite(phase_error[index]) & (phase_error[index] > 0)
        if valid.sum() < 3:
            continue

        # Unwrapping has to follow increasing amplitude, which is also increasing accumulated phase.
        order = np.argsort(xi[index][valid])
        x = xi[index][valid][order] ** 2
        unwrapped, largest_residual = _progressive_unwrap(x, phase[index][valid][order])
        sigma_phi = phase_error[index][valid][order]

        intercept, slope, slope_error, reduced_chi2 = _weighted_line_fit(x, unwrapped, sigma_phi)
        if not np.isfinite(slope):
            continue

        tau_probe = probe_durations_s[index[1]]
        stark_hz[index] = slope / (2 * np.pi * tau_probe)
        stark_error_hz[index] = slope_error / (2 * np.pi * tau_probe)
        chi2[index] = reduced_chi2
        residual[index] = largest_residual
        rejected[index] = (reduced_chi2 > max_reduced_chi2) or (largest_residual > max_phase_residual_rad)

    return stark_hz, stark_error_hz, chi2, residual, rejected


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Extract the measurement-induced dephasing matrix from the raw dataset.

    The AC-Stark shift matrix is extracted alongside it, unless the probe was played in both halves
    of the echo, in which case the echo refocuses the Stark phase and only the dephasing matrix
    exists.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with dimensions ``(qubit, driven_resonator, xi_idx, phase)``.
    node : QualibrationNode
        Node whose parameters and machine drive the fit.

    Returns
    -------
    ds_fit : xr.Dataset
        The input dataset augmented with the per-xi contrast and phase and the per-pair rates.
    fit_results : dict
        One :class:`FitParameters` per measured qubit.
    """
    signal = ds.state if node.parameters.use_state_discrimination else ds.I
    signal = signal.transpose("qubit", "driven_resonator", "xi_idx", "phase")

    contrast, contrast_error, raw_contrast, offset, phase, phase_error = _fit_phase_oscillations(
        signal.values, ds.phase.values
    )

    pair_dims = ("qubit", "driven_resonator", "xi_idx")
    pair_coords = {dim: ds[dim] for dim in pair_dims}

    def pair_array(values):
        return xr.DataArray(values, coords=pair_coords, dims=pair_dims)

    ds_fit = ds.assign(
        contrast=pair_array(contrast),
        contrast_error=pair_array(contrast_error),
        contrast_raw=pair_array(raw_contrast),
        oscillation_offset=pair_array(offset),
        oscillation_phase=pair_array(phase),
        oscillation_phase_error=pair_array(phase_error),
    )
    ds_fit["contrast"].attrs = {"long_name": "echo oscillation contrast (noise debiased)", "units": ""}
    ds_fit["contrast_raw"].attrs = {"long_name": "echo oscillation contrast (biased)", "units": ""}

    probe_durations = _probe_durations_in_s(ds, node)
    xi_values = ds.xi.transpose(*pair_dims).values
    gamma, gamma_error, c0, gamma_chi2, gamma_rejected = _fit_dephasing_rates(
        contrast,
        contrast_error,
        xi_values,
        probe_durations.values,
        node.parameters.min_contrast_snr,
        node.parameters.max_reduced_chi2,
    )

    matrix_dims = ("qubit", "driven_resonator")
    matrix_coords = {dim: ds[dim] for dim in matrix_dims}

    def matrix_array(values):
        return xr.DataArray(values, coords=matrix_coords, dims=matrix_dims)

    ds_fit = ds_fit.assign(
        Gamma=matrix_array(gamma),
        Gamma_error=matrix_array(gamma_error),
        Gamma_chi2=matrix_array(gamma_chi2),
        Gamma_rejected=matrix_array(gamma_rejected),
        Gamma_resolved=matrix_array(np.abs(gamma) > RESOLVED_SIGMA * gamma_error),
        c0=matrix_array(c0),
        tau_p=probe_durations,
    )
    ds_fit["Gamma"].attrs = {"long_name": "measurement-induced dephasing rate", "units": "Hz"}
    ds_fit["Gamma_error"].attrs = {"long_name": "dephasing rate error", "units": "Hz"}
    ds_fit["tau_p"].attrs = {"long_name": "total probe duration per echo", "units": "s"}
    ds_fit["contrast_relative"] = ds_fit.contrast / ds_fit.c0
    ds_fit["contrast_relative"].attrs = {"long_name": "c / c0", "units": ""}

    # The echo refocuses the Stark phase when the probe is played in both halves, so there is no
    # Stark matrix to report in that case and it is left out of the dataset entirely.
    ds_fit.attrs["has_stark_matrix"] = int(not node.parameters.probe_in_both_halves)
    if not node.parameters.probe_in_both_halves:
        stark, stark_error, stark_chi2, stark_residual, stark_rejected = _fit_stark_shifts(
            phase,
            phase_error,
            contrast,
            contrast_error,
            xi_values,
            probe_durations.values,
            node.parameters.min_contrast_snr,
            node.parameters.max_reduced_chi2,
            node.parameters.max_phase_residual_in_rad,
        )
        ds_fit = ds_fit.assign(
            stark_shift=matrix_array(stark),
            stark_shift_error=matrix_array(stark_error),
            stark_shift_chi2=matrix_array(stark_chi2),
            stark_shift_unwrap_residual=matrix_array(stark_residual),
            stark_shift_rejected=matrix_array(stark_rejected),
            stark_shift_resolved=matrix_array(np.abs(stark) > RESOLVED_SIGMA * stark_error),
        )
        ds_fit["stark_shift"].attrs = {"long_name": "AC-Stark shift at xi = 1", "units": "Hz"}
        ds_fit["stark_shift_error"].attrs = {"long_name": "AC-Stark shift error", "units": "Hz"}

    return _extract_relevant_fit_parameters(ds_fit, node)


def _zero_amplitude_spread(ds_fit: xr.Dataset) -> xr.DataArray:
    """Return, per measured qubit, how far the xi = 0 contrasts of its row disagree, in sigma.

    At xi = 0 the driven resonator plays nothing, so the pulse sequence is identical whichever
    resonator was selected and every element of a row must return the same contrast. Any spread is
    therefore an instrumental effect and not physics: drift across the acquisition, a probe that is
    not silent at zero amplitude, or leakage from the driven element into the measured qubit. The
    spread is reported in units of its own uncertainty, as the largest deviation of a row from that
    row's mean, so that it can be compared against shot noise directly.

    This is a diagnostic on the data rather than a pass criterion. A large spread does not by itself
    bias the fitted rates, since a common factor on the contrast is absorbed by the intercept, but it
    does mean something in the setup is not what the sequence says it is.
    """
    at_zero = ds_fit.contrast.isel(xi_idx=0)
    error_at_zero = ds_fit.contrast_error.isel(xi_idx=0)
    deviation = np.abs(at_zero - at_zero.mean(dim="driven_resonator"))
    return (deviation / error_at_zero).max(dim="driven_resonator")


def _row_summary(values: xr.DataArray, errors: xr.DataArray, resolved: xr.DataArray, qubit) -> Tuple:
    """Return the worst resolved element of one row, its error and its driven resonator name.

    "Worst" is the resolved element of largest magnitude. The sign is kept, because a resolved
    negative dephasing rate is an unphysical fit that must stay visible rather than pass as low
    crosstalk. When no element of the row is resolved there is nothing to report and NaNs are
    returned, the row being described by its upper bound instead.
    """
    row = values.sel(qubit=qubit)
    row_resolved = row.where(resolved.sel(qubit=qubit).fillna(False).astype(bool))
    if not bool(row_resolved.notnull().any()):
        return np.nan, np.nan, ""
    worst = str(np.abs(row_resolved).idxmax(dim="driven_resonator").values)
    return float(row.sel(driven_resonator=worst)), float(errors.sel(qubit=qubit, driven_resonator=worst)), worst


def _extract_relevant_fit_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Summarise the matrix into one result per measured qubit and assess the fit outcome.

    A qubit passes when every off-diagonal element of its row has a usable fit and none of them is a
    resolved crosstalk above the threshold. Three things are distinguished, which the previous single
    magnitude test conflated:

      * an element whose fit did not converge or whose reduced chi-squared is too large is rejected,
        and a rejected element cannot certify anything, so the qubit fails;
      * an element smaller than twice its own error is unresolved. It is an upper bound rather than a
        measurement, so it is not judged against the threshold;
      * only a resolved element is compared with ``max_crosstalk_dephasing_in_hz``.
    """
    is_diagonal = xr.DataArray(
        np.equal.outer(ds_fit.qubit.values, ds_fit.driven_resonator.values),
        coords={"qubit": ds_fit.qubit, "driven_resonator": ds_fit.driven_resonator},
        dims=("qubit", "driven_resonator"),
    )
    ds_fit = ds_fit.assign(is_diagonal=is_diagonal)
    off_diagonal = ~is_diagonal

    gamma = ds_fit.Gamma.where(off_diagonal)
    gamma_error = ds_fit.Gamma_error.where(off_diagonal)
    rejected = ds_fit.Gamma_rejected.where(off_diagonal, False).fillna(False).astype(bool)
    # A rejected element is not a measurement, so it never counts as resolved and is never quoted as
    # the row's worst crosstalk; it fails the row on its own, through ``unusable`` below.
    resolved = ds_fit.Gamma_resolved.where(off_diagonal, False).fillna(False).astype(bool) & ~rejected
    # A failed fit is as unusable as one rejected on its chi-squared, so the two are merged here.
    unusable = rejected | (gamma.isnull() & off_diagonal)
    unresolved = off_diagonal & ~resolved & ~unusable

    # The bound that holds whether or not the element was resolved.
    upper_bound = (np.abs(gamma) + RESOLVED_SIGMA * gamma_error).where(~unusable)

    too_large = resolved & (np.abs(gamma) >= node.parameters.max_crosstalk_dephasing_in_hz)
    success = (too_large.sum(dim="driven_resonator") == 0) & (unusable.sum(dim="driven_resonator") == 0)
    ds_fit = ds_fit.assign(success=success)

    has_stark = bool(ds_fit.attrs.get("has_stark_matrix", 0))
    zero_amplitude_spread = _zero_amplitude_spread(ds_fit)
    ds_fit = ds_fit.assign(zero_amplitude_spread=zero_amplitude_spread)

    fit_results = {}
    for q in ds_fit.qubit.values:
        worst_value, worst_error, worst = _row_summary(gamma, gamma_error, resolved, q)
        bound_row = upper_bound.sel(qubit=q)
        if has_stark:
            stark = ds_fit.stark_shift.where(off_diagonal)
            stark_resolved = (
                ds_fit.stark_shift_resolved.where(off_diagonal, False).fillna(False).astype(bool)
                & ~ds_fit.stark_shift_rejected.where(off_diagonal, False).fillna(False).astype(bool)
            )
            stark_value, _, stark_worst = _row_summary(stark, ds_fit.stark_shift_error, stark_resolved, q)
            self_stark = float(ds_fit.stark_shift.sel(qubit=q, driven_resonator=q))
        else:
            stark_value, stark_worst, self_stark = np.nan, "", np.nan

        fit_results[str(q)] = FitParameters(
            self_dephasing_hz=float(ds_fit.Gamma.sel(qubit=q, driven_resonator=q)),
            self_dephasing_error_hz=float(ds_fit.Gamma_error.sel(qubit=q, driven_resonator=q)),
            max_crosstalk_dephasing_hz=worst_value,
            max_crosstalk_dephasing_error_hz=worst_error,
            worst_crosstalk_resonator=worst,
            crosstalk_upper_bound_hz=float(bound_row.max()) if bool(bound_row.notnull().any()) else np.nan,
            num_unresolved_crosstalk=int(unresolved.sel(qubit=q).sum()),
            num_rejected_crosstalk=int(unusable.sel(qubit=q).sum()),
            zero_amplitude_spread_sigma=float(zero_amplitude_spread.sel(qubit=q)),
            self_stark_shift_hz=self_stark,
            max_crosstalk_stark_shift_hz=stark_value,
            worst_stark_resonator=stark_worst,
            success=bool(ds_fit.success.sel(qubit=q)),
        )
    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the self-dephasing rate, the crosstalk and the AC-Stark shift of each measured qubit.

    The crosstalk is reported as the worst resolved element when there is one, and otherwise as the
    upper bound over the row, so that "nothing was resolved" never reads as "the crosstalk is zero".

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

        if result["worst_crosstalk_resonator"]:
            crosstalk = (
                f"worst crosstalk = {result['max_crosstalk_dephasing_hz']:.1f} "
                f"+/- {result['max_crosstalk_dephasing_error_hz']:.1f} Hz "
                f"(driven by {result['worst_crosstalk_resonator']})"
            )
        elif np.isfinite(result["crosstalk_upper_bound_hz"]):
            crosstalk = f"crosstalk unresolved, below {result['crosstalk_upper_bound_hz']:.1f} Hz"
        else:
            # Every element of the row was rejected or failed to fit, so there is not even a bound.
            crosstalk = "no usable crosstalk fit"

        message = (
            f"{qubit_name}: self-dephasing = {self_mhz:.2f} +/- {self_error_mhz:.2f} MHz | "
            f"{crosstalk} | {result['num_unresolved_crosstalk']} unresolved, "
            f"{result['num_rejected_crosstalk']} rejected"
        )
        if result["worst_stark_resonator"]:
            message += (
                f" | worst Stark shift = {result['max_crosstalk_stark_shift_hz']:.1f} Hz "
                f"(driven by {result['worst_stark_resonator']})"
            )
        message += f" | {'PASS' if result['success'] else 'FAIL'}"
        log_callable(message)

        # Reported separately because it says something about the setup rather than about the qubit.
        spread = result["zero_amplitude_spread_sigma"]
        if np.isfinite(spread) and spread > ZERO_AMPLITUDE_SPREAD_SIGMA:
            log_callable(
                f"{qubit_name}: the xi = 0 contrast differs by {spread:.1f} sigma across the row, "
                f"although that point plays no probe pulse and is the same sequence for every driven "
                f"resonator. Check for drift over the acquisition and for a probe that is not silent "
                f"at zero amplitude."
            )
