"""Analysis helpers for the readout quantum efficiency node.

The measurement follows Bultink et al., arXiv:1711.05336. Two sequences share one
(readout detuning, measurement-pulse amplitude) grid:

* a Ramsey sequence with the measurement pulse embedded in it, whose fringe amplitude
  gives the qubit coherence ``|rho01(eps)|``,
* a single-shot experiment on the same measurement pulse, which gives its ``SNR``.

The measurement-induced dephasing is defined by ``exp(-beta) = |rho01(eps)| / |rho01(0)|``
and the quantum efficiency by ``eta = SNR^2 / (4 beta)`` (Eq. 2 of the paper). Rather than
evaluating that ratio point by point -- it is 0/0 as ``eps -> 0`` -- the headline number per
frequency comes from the paper's global fits over the amplitude axis,

    SNR(eps)  = a * eps,
    |rho01|   = |rho01(0)| * exp(-eps^2 / (2 sigma_m^2))   i.e.  beta = eps^2 / (2 sigma_m^2),

which give ``eta = a^2 sigma_m^2 / 2``. Writing ``beta = s * eps^2`` with ``s = 1/(2 sigma_m^2)``
this is ``eta = a^2 / (4 s)``, the form used below. The point-by-point ratio is still computed
and plotted, as a check that the sweep stayed in the linear regime where eta is meaningful.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fitted quantum efficiency results for a single qubit."""

    eta_max: float
    """Largest quantum efficiency over the frequency axis."""
    detuning_at_eta_max: float
    """Readout detuning, in Hz relative to the current readout frequency, where eta peaks."""
    frequency_at_eta_max: float
    """Absolute readout frequency, in Hz, where eta peaks."""
    eta_at_current_setpoint: float
    """Quantum efficiency at the currently configured readout frequency (zero detuning)."""
    eta_error_at_eta_max: float
    """One-sigma uncertainty on eta_max propagated from the two linear fits."""
    linear_range_max_amp: float
    """Largest amplitude prefactor retained in the linear-regime fits."""
    num_points_in_fit: int
    """Number of non-zero amplitude points retained in the linear-regime fits."""
    reference_spread: float
    """Relative scatter of the eps=0 fringe amplitude across the frequency axis."""
    success: bool
    """Whether the fit produced a usable efficiency."""
    warnings: List[str] = field(default_factory=list)
    """Human-readable notes about anything the fit had to work around."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Log the fitted quantum efficiency for all qubits.

    Parameters
    ----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logging callable used to emit the results. If None, ``logging.getLogger(__name__).info`` is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        r = fit_results[q]
        s = f"Results for qubit {q}: {'SUCCESS!' if r['success'] else 'FAIL!'}\n"
        s += f"\teta at the current readout frequency: {r['eta_at_current_setpoint']:.4f}\n"
        s += (
            f"\tbest eta: {r['eta_max']:.4f} +/- {r['eta_error_at_eta_max']:.4f} "
            f"at {r['frequency_at_eta_max'] / 1e9:.6f} GHz "
            f"({r['detuning_at_eta_max'] / 1e6:+.3f} MHz)\n"
        )
        s += (
            f"\tlinear-regime fit used {r['num_points_in_fit']} amplitude points "
            f"up to a prefactor of {r['linear_range_max_amp']:.3f}\n"
        )
        for warning in r["warnings"]:
            s += f"\t[!] {warning}\n"
        log_callable(s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Convert the single-shot quadratures to volts, add the physical sweep coordinates and
    correct the Ramsey-half state probability for readout errors.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset holding ``state`` (Ramsey half) and ``Ig``/``Qg``/``Ie``/``Qe`` (SNR half).
    node : QualibrationNode
        The node object, used for the qubit list and their QUAM readout settings.

    Returns
    -------
    xr.Dataset
        The dataset with ``readout_amplitude``/``readout_power_dbm``/``full_freq`` coordinates
        and a ``state_corrected`` variable.
    """
    qubits = node.namespace["qubits"]
    if all(var in ds.data_vars for var in ["Ig", "Qg", "Ie", "Qe"]):
        ds = convert_IQ_to_V(ds, qubits, IQ_list=["Ig", "Qg", "Ie", "Qe"])

    # The measurement pulse amplitude in volts, and the absolute readout frequency.
    readout_amplitudes = np.array(
        [ds.amp_prefactor.values * q.resonator.operations["readout"].amplitude for q in qubits]
    )
    ds = ds.assign_coords(readout_amplitude=(["qubit", "amp_prefactor"], readout_amplitudes))
    ds.readout_amplitude.attrs = {"long_name": "measurement pulse amplitude", "units": "V"}

    # The same amplitude axis as an absolute power at the output port: the power of the readout
    # pulse as configured (full scale of the port + its waveform amplitude), scaled by the swept
    # prefactor. `get_output_power` handles both the MW-FEM and the IQ (gain) cases.
    base_power_dbm = np.array([q.resonator.get_output_power("readout") for q in qubits])
    with np.errstate(divide="ignore"):
        power_dbm = base_power_dbm[:, None] + 20 * np.log10(np.abs(ds.amp_prefactor.values)[None, :])
    # A prefactor of 0 has no dBm representation.
    power_dbm[~np.isfinite(power_dbm)] = np.nan
    ds = ds.assign_coords(readout_power_dbm=(["qubit", "amp_prefactor"], power_dbm))
    ds.readout_power_dbm.attrs = {"long_name": "measurement pulse power", "units": "dBm"}

    full_freq = np.array([ds.detuning.values + q.resonator.RF_frequency for q in qubits])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "readout frequency", "units": "Hz"}

    ds["state_corrected"] = _correct_readout_errors(ds.state, qubits)
    ds.state_corrected.attrs = {"long_name": "P(excited), readout-error corrected"}
    return ds


def _correct_readout_errors(state: xr.DataArray, qubits) -> xr.DataArray:
    """
    Undo readout assignment errors on the measured excited-state probability.

    ``iq_blobs`` stores the confusion matrix with rows indexed by the *prepared* state and
    columns by the *measured* one, so the measured distribution is ``p_meas = M.T @ p_true``
    and the correction is ``inv(M.T) @ p_meas``. As in the tomography analysis, negative
    entries are clipped away and the result renormalized. Qubits without a usable confusion
    matrix are passed through uncorrected.
    """
    corrected = []
    for q in qubits:
        p_meas = state.sel(qubit=q.name)
        matrix = getattr(q.resonator, "confusion_matrix", None)
        if matrix is None:
            corrected.append(p_meas)
            continue
        try:
            inverse = np.linalg.inv(np.asarray(matrix, dtype=float).T)
        except np.linalg.LinAlgError:
            logging.getLogger(__name__).warning(
                f"{q.name}: singular readout confusion matrix, using the uncorrected probability. "
                "Re-run node 16_iq_blobs or 15_readout_power_optimization."
            )
            corrected.append(p_meas)
            continue

        probabilities = np.stack([1.0 - p_meas.values, p_meas.values])  # shape (2, ...)
        flat = probabilities.reshape(2, -1)
        flat = inverse @ flat
        flat = np.clip(flat, 0.0, None)
        flat = flat / np.where(flat.sum(axis=0) == 0, 1.0, flat.sum(axis=0))
        corrected.append(xr.DataArray(flat[1].reshape(p_meas.shape), coords=p_meas.coords, dims=p_meas.dims))
    return xr.concat(corrected, dim=state.qubit)


def _fringe_amplitude_and_phase(probability: np.ndarray, phases: np.ndarray) -> Tuple[float, float]:
    """
    Extract the amplitude and phase of a fringe.

    The fringe period is set by the applied frame rotation, not by the qubit, so it is not a free
    parameter: the least-squares fit of ``c + A cos(phi - phi0)`` reduces to a projection onto
    ``exp(i phi)``. That projection is exact whenever the sampled phases are uniform over a whole
    number of turns -- one turn, two turns, any integer -- which is why the actual phase
    coordinate is passed in rather than reconstructed here.

    Parameters
    ----------
    probability : np.ndarray
        Excited-state probability, with the phase axis last.
    phases : np.ndarray
        The frame-rotation phases in radians, uniform over an integer number of turns.

    Returns
    -------
    tuple
        ``(A, phi0, sigma_A)``: the fringe amplitude, its phase offset (the AC-Stark phase) in
        radians, and the one-sigma uncertainty on the amplitude. The uncertainty comes from the
        scatter left after subtracting the fitted fringe, i.e. from the harmonics the model does
        not use, and it is what tells a resolved dephasing from a noise-floor one.
    """
    n = probability.shape[-1]
    coefficient = np.sum(probability * np.exp(1j * phases), axis=-1) * 2 / n
    amplitude, phase = np.abs(coefficient), np.angle(coefficient)
    offset = np.mean(probability, axis=-1)
    model = offset[..., None] + amplitude[..., None] * np.cos(phases - phase[..., None])
    point_noise = np.sqrt(np.sum((probability - model) ** 2, axis=-1) / max(n - 3, 1))
    return amplitude, phase, point_noise * np.sqrt(2 / n)


def _snr_from_shots(Ig, Qg, Ie, Qe) -> float:
    """
    Single-shot SNR of one measurement pulse, along the axis that separates the two blobs.

    Follows Eq. S5 of the paper: the numerator is the separation of the mean integrated
    voltages and the denominator is a *single* standard deviation along the same axis, common
    to both prepared states (here the pooled standard deviation of the two).

    Parameters
    ----------
    Ig, Qg, Ie, Qe : np.ndarray
        Single-shot quadratures for the qubit prepared in |0> and |1>.

    Returns
    -------
    float
        ``|mu_1 - mu_0| / sigma``.
    """
    separation = np.array([np.mean(Ie) - np.mean(Ig), np.mean(Qe) - np.mean(Qg)])
    distance = np.hypot(*separation)
    if distance == 0:
        return 0.0
    axis = separation / distance
    projected_g = Ig * axis[0] + Qg * axis[1]
    projected_e = Ie * axis[0] + Qe * axis[1]
    sigma = np.sqrt((np.var(projected_g) + np.var(projected_e)) / 2)
    if sigma == 0:
        return np.inf
    return distance / sigma


def _fit_through_origin(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Least-squares fit of ``y = slope * x`` with no intercept.

    Returns
    -------
    tuple of float
        ``(slope, standard error on the slope, largest relative residual)``.
    """
    denominator = np.sum(x**2)
    if denominator == 0:
        return np.nan, np.nan, np.inf
    slope = float(np.sum(x * y) / denominator)
    residuals = y - slope * x
    scale = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    max_relative_residual = float(np.max(np.abs(residuals)) / scale)
    dof = max(len(x) - 1, 1)
    slope_error = float(np.sqrt(np.sum(residuals**2) / dof / denominator))
    return slope, slope_error, max_relative_residual


def _select_linear_range(
    eps: np.ndarray,
    snr: np.ndarray,
    beta: np.ndarray,
    resolved: np.ndarray,
    max_amp_for_fit: Optional[float],
    linearity_rtol: float,
) -> np.ndarray:
    """
    Choose which non-zero amplitude points feed the efficiency fits.

    Only the largest amplitude is ever a candidate for removal: compression lives at the top of
    the range, whereas the smallest amplitudes carry almost no weight in either through-origin
    fit -- ``a = sum(eps SNR)/sum(eps^2)`` and ``s = sum(eps^2 beta)/sum(eps^4)`` both weight by
    the abscissa, so including a noisy eps -> 0 point costs essentially nothing. Trimming from
    the bottom, on the other hand, is what destroys the estimate, because the large amplitudes
    are where the signal is; the selection therefore never drops past half the points.

    The criterion compares the top point's ``SNR/eps`` and ``beta/eps^2`` -- both constant in the
    linear regime -- against the MEDIAN of the same ratios over the remaining resolved points.
    A median is used because the smallest amplitudes have wild ratios that are pure noise, and
    unresolved points (a dephasing indistinguishable from zero) are excluded from it outright.

    Returns
    -------
    np.ndarray
        Boolean mask over the amplitude axis.
    """
    mask = (eps > 0) & np.isfinite(snr) & np.isfinite(beta)
    if max_amp_for_fit is not None:
        mask &= eps <= max_amp_for_fit

    floor = max(3, int(np.ceil(0.5 * mask.sum())))
    while mask.sum() > floor:
        top = int(np.argmax(np.where(mask, eps, -np.inf)))
        others = mask & resolved
        others[top] = False
        if others.sum() < 2:
            break

        deviations = []
        for abscissa, ordinate in ((eps, snr), (eps**2, beta)):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(abscissa > 0, ordinate / abscissa, np.nan)
            reference = np.nanmedian(ratio[others])
            if not np.isfinite(reference) or reference == 0:
                return mask
            deviations.append(abs(ratio[top] / reference - 1))

        if max(deviations) <= linearity_rtol:
            break
        mask[top] = False
    return mask


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract the dephasing, the SNR and the quantum efficiency from the processed dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset from :func:`process_raw_dataset`.
    node : QualibrationNode
        The node object; reads ``max_amp_for_fit`` and ``linearity_rtol`` from its parameters.

    Returns
    -------
    tuple
        ``(ds_fit, fit_results)``; ``ds_fit`` carries the per-cell fringe amplitude, Stark
        phase, dephasing, SNR and point-by-point efficiency, plus the per-frequency fitted
        efficiency, and ``fit_results`` maps qubit name -> :class:`FitParameters`.
    """
    eps = ds.amp_prefactor.values
    qubits = [str(q) for q in ds.qubit.values]
    shape = (len(qubits), ds.sizes["detuning"], ds.sizes["amp_prefactor"])

    fringe_amplitude = np.zeros(shape)
    fringe_amplitude_error = np.zeros(shape)
    stark_phase = np.zeros(shape)
    snr = np.zeros(shape)
    for qi, qubit in enumerate(qubits):
        probability = ds.state_corrected.sel(qubit=qubit).transpose("detuning", "amp_prefactor", "phase").values
        (
            fringe_amplitude[qi],
            stark_phase[qi],
            fringe_amplitude_error[qi],
        ) = _fringe_amplitude_and_phase(probability, ds.phase.values)
        selection = ds.sel(qubit=qubit).transpose("shot", "detuning", "amp_prefactor", ...)
        for di in range(shape[1]):
            for ai in range(shape[2]):
                snr[qi, di, ai] = _snr_from_shots(
                    selection.Ig.values[:, di, ai],
                    selection.Qg.values[:, di, ai],
                    selection.Ie.values[:, di, ai],
                    selection.Qe.values[:, di, ai],
                )

    dims = ("qubit", "detuning", "amp_prefactor")
    coords = {"qubit": ds.qubit, "detuning": ds.detuning, "amp_prefactor": ds.amp_prefactor}
    ds_fit = xr.Dataset(
        {
            "fringe_amplitude": (dims, fringe_amplitude),
            "fringe_amplitude_error": (dims, fringe_amplitude_error),
            "stark_phase": (dims, stark_phase),
            "snr": (dims, snr),
        },
        coords=coords,
    )
    ds_fit = ds_fit.assign_coords(
        readout_amplitude=ds.readout_amplitude,
        readout_power_dbm=ds.readout_power_dbm,
        full_freq=ds.full_freq,
    )

    # Each frequency is normalized by its OWN eps = 0 fringe. The frequency loop is the outer
    # one, so that reference is measured immediately before the amplitudes it normalizes and is
    # drift-matched to them; pooling it across frequencies would instead average over the whole
    # run. Its scatter across frequency is kept as a drift monitor, since the eps = 0 fringe is
    # physically frequency independent and any structure in it is time, not physics.
    reference = ds_fit.fringe_amplitude.sel(amp_prefactor=0.0, drop=True)
    reference_spread = (reference.std(dim="detuning") / reference.mean(dim="detuning")).values

    reference_error = ds_fit.fringe_amplitude_error.sel(amp_prefactor=0.0, drop=True)
    ratio = (ds_fit.fringe_amplitude / reference).clip(min=1e-12)
    ds_fit["beta"] = -np.log(ratio)
    # beta = -ln(A/A0), so the two amplitude uncertainties add in quadrature as relative ones.
    ds_fit["beta_error"] = np.sqrt(
        (ds_fit.fringe_amplitude_error / ds_fit.fringe_amplitude) ** 2 + (reference_error / reference) ** 2
    )
    # A cell whose dephasing is within a few sigma of zero cannot say anything about eta: the
    # ratio is 0/0 there, and dividing anyway produces the wild outliers that used to dominate
    # the colour scale of the map (and, when the point selection kept only such cells, eta itself).
    ds_fit["beta_resolved"] = ds_fit.beta > 3 * ds_fit.beta_error
    ds_fit["eta_pointwise"] = ds_fit.snr**2 / (4 * ds_fit.beta.where(ds_fit.beta_resolved))

    ds_fit.fringe_amplitude.attrs = {"long_name": "Ramsey fringe amplitude"}
    ds_fit.stark_phase.attrs = {"long_name": "AC-Stark phase", "units": "rad"}
    ds_fit.snr.attrs = {"long_name": "SNR"}
    ds_fit.beta.attrs = {"long_name": "measurement-induced dephasing"}
    ds_fit.eta_pointwise.attrs = {"long_name": "quantum efficiency (point by point)"}

    return _fit_efficiency_per_frequency(ds_fit, ds, node, eps, reference_spread)


def _fit_efficiency_per_frequency(
    ds_fit: xr.Dataset,
    ds: xr.Dataset,
    node: QualibrationNode,
    eps: np.ndarray,
    reference_spread: np.ndarray,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Run the paper's two global fits at every frequency and summarise the result per qubit."""
    max_amp_for_fit = node.parameters.max_amp_for_fit
    linearity_rtol = node.parameters.linearity_rtol
    qubits = [str(q) for q in ds_fit.qubit.values]
    n_frequencies = ds_fit.sizes["detuning"]

    eta = np.full((len(qubits), n_frequencies), np.nan)
    eta_error = np.full((len(qubits), n_frequencies), np.nan)
    slope_a = np.full((len(qubits), n_frequencies), np.nan)
    sigma_m = np.full((len(qubits), n_frequencies), np.nan)
    fit_mask = np.zeros((len(qubits), n_frequencies, len(eps)), dtype=bool)

    for qi, qubit in enumerate(qubits):
        for di in range(n_frequencies):
            snr = ds_fit.snr.isel(qubit=qi, detuning=di).values
            beta = ds_fit.beta.isel(qubit=qi, detuning=di).values
            resolved = ds_fit.beta_resolved.isel(qubit=qi, detuning=di).values
            mask = _select_linear_range(eps, snr, beta, resolved, max_amp_for_fit, linearity_rtol)
            fit_mask[qi, di] = mask
            if mask.sum() < 2:
                continue
            a, a_error, _ = _fit_through_origin(eps[mask], snr[mask])
            s, s_error, _ = _fit_through_origin(eps[mask] ** 2, beta[mask])
            if not np.isfinite(a) or not np.isfinite(s) or s <= 0:
                continue
            slope_a[qi, di] = a
            sigma_m[qi, di] = np.sqrt(1 / (2 * s))
            eta[qi, di] = a**2 / (4 * s)
            # eta = a^2 / (4 s), so the relative errors add as (2 da/a) and (ds/s).
            if a != 0:
                relative_error = np.sqrt((2 * a_error / a) ** 2 + (s_error / s) ** 2)
                eta_error[qi, di] = abs(eta[qi, di]) * relative_error

    dims = ("qubit", "detuning")
    ds_fit["eta"] = (dims, eta)
    ds_fit["eta_error"] = (dims, eta_error)
    ds_fit["snr_slope"] = (dims, slope_a)
    ds_fit["sigma_m"] = (dims, sigma_m)
    ds_fit["fit_mask"] = (("qubit", "detuning", "amp_prefactor"), fit_mask)
    ds_fit.eta.attrs = {"long_name": "quantum efficiency"}
    ds_fit.eta_error.attrs = {"long_name": "quantum efficiency uncertainty"}

    detunings = ds_fit.detuning.values
    zero_index = int(np.argmin(np.abs(detunings)))
    fit_results = {}
    for qi, qubit in enumerate(qubits):
        warnings = []
        if reference_spread[qi] > 0.05:
            warnings.append(
                f"the eps=0 reference fringe varies by {100 * reference_spread[qi]:.1f}% across the "
                "frequency axis, which it cannot do physically - the frequency axis is the outer "
                "loop, so this is drift over the run showing up as apparent structure in eta(f). "
                "Each beta is still normalized by its own frequency's reference, so the cells "
                "themselves are sound; the comparison BETWEEN frequencies is what to distrust."
            )
        finite = np.isfinite(eta[qi])
        if not finite.any():
            fit_results[qubit] = FitParameters(
                eta_max=np.nan,
                detuning_at_eta_max=np.nan,
                frequency_at_eta_max=np.nan,
                eta_at_current_setpoint=np.nan,
                eta_error_at_eta_max=np.nan,
                linear_range_max_amp=np.nan,
                num_points_in_fit=0,
                reference_spread=float(reference_spread[qi]),
                success=False,
                warnings=warnings + ["no frequency yielded a usable efficiency fit."],
            )
            continue

        best = int(np.nanargmax(np.where(finite, eta[qi], -np.inf)))
        mask = fit_mask[qi, best]
        eta_best = float(eta[qi, best])
        # Taking the maximum over a long frequency axis is biased upwards: the winner tends to be
        # wherever the noise happened to fluctuate high, which is usually where the SNR is worst.
        # Say so unless the peak clears the typical value by more than its own error bar.
        typical = float(np.nanmedian(eta[qi]))
        if n_frequencies > 1 and eta_best - float(eta_error[qi, best]) <= typical:
            warnings.append(
                f"eta_max is not significant: {eta_best:.3f} +/- {eta_error[qi, best]:.3f} does not "
                f"clear the median over frequency ({typical:.3f}) by its own error bar, so the peak "
                "is likely a fluctuation. Read eta from eta_at_current_setpoint or from the trend "
                "in the eta-vs-frequency figure instead of from the maximum."
            )
        if eta_best > 1:
            warnings.append(
                f"eta = {eta_best:.3f} exceeds 1, which is unphysical: suspect the eps=0 reference, "
                "residual photons at the second pi/2 pulse, or a readout that is not in the linear regime."
            )
        # Only resolved cells can be non-monotonic in any meaningful sense; at the bottom of the
        # amplitude axis beta sits in its own noise and will reorder itself run to run.
        resolved_beta = ds_fit.beta.isel(qubit=qi, detuning=best).values[
            ds_fit.beta_resolved.isel(qubit=qi, detuning=best).values
        ]
        if resolved_beta.size >= 2 and not np.all(np.diff(resolved_beta) >= -1e-9):
            warnings.append("the dephasing is not monotonic in amplitude at the best frequency.")
        # eta is amplitude independent in the linear regime, so the spread of the point-by-point
        # values over the fitted cells says how well that holds. It is reported rather than acted
        # on: a global fit over a mildly curved range is still far better conditioned than any
        # attempt to find a perfectly linear sub-range.
        pointwise = ds_fit.eta_pointwise.isel(qubit=qi, detuning=best).values[mask]
        pointwise = pointwise[np.isfinite(pointwise)]
        if pointwise.size >= 2:
            spread = (np.max(pointwise) - np.min(pointwise)) / np.mean(pointwise)
            if spread > 0.3:
                warnings.append(
                    f"the point-by-point eta varies by {100 * spread:.0f}% over the fitted amplitude "
                    "range, so the response is not strictly linear there; eta is the global-fit "
                    "value over that range. Narrow max_amp_for_fit if you want a stricter regime."
                )

        fit_results[qubit] = FitParameters(
            eta_max=eta_best,
            detuning_at_eta_max=float(detunings[best]),
            frequency_at_eta_max=float(ds.full_freq.isel(qubit=qi, detuning=best).values),
            eta_at_current_setpoint=float(eta[qi, zero_index]),
            eta_error_at_eta_max=float(eta_error[qi, best]),
            linear_range_max_amp=float(np.max(eps[mask])) if mask.any() else np.nan,
            num_points_in_fit=int(mask.sum()),
            reference_spread=float(reference_spread[qi]),
            success=bool(np.isfinite(eta_best) and eta_best > 0),
            warnings=warnings,
        )
    return ds_fit, fit_results
