"""Notch-port circle fit of a readout resonance.

The fit follows the standard notch-port procedure: remove the cable delay, fit a circle to the
complex trace, fit the phase to extract the resonance frequency and the loaded quality factor, then
normalise the circle to separate the external from the internal quality factor.

Every frequency the module returns is in Hz, including the linewidths kappa.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, least_squares

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

__all__ = [
    "FitParameters",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
    "fit_circle",
    "fit_cable_delay",
    "fit_phase",
    "notch_port_fit",
]


@dataclass
class FitParameters:
    """Circle-fit results for a single resonator."""

    resonance_frequency: float
    """Fitted resonance frequency in Hz. Reported only; node 02a owns the readout frequency."""
    q_loaded: float
    q_external: float
    q_internal: float
    kappa_tot_hz: float
    kappa_ext_hz: float
    kappa_int_hz: float
    kappa_ext_over_kappa_tot: float
    """Hard upper bound on the collection efficiency of this resonator."""
    photon_lifetime_ns: float
    """1 / (2 pi kappa_tot), the energy decay time of the resonator."""
    depletion_time_over_lifetime: float
    """The stored depletion time expressed in photon lifetimes."""
    cable_delay_ns: float
    probe_power_dbm: float
    """Absolute power of the probe tone, which differs per qubit because the probe is a scale factor
    on each qubit's own readout amplitude."""
    impedance_mismatch_rad: float
    r_squared: float
    stored_kappa_hz: float
    """A kappa already present on the resonator, for comparison. NaN when there is none."""
    kappa_mismatch_fraction: float
    """Relative disagreement between the fitted and the stored kappa. NaN when there is no stored kappa."""
    excited_state_measured: bool
    """True when the resonance was also fitted with the qubit prepared in |1>, which is what yields
    chi. Every field below is NaN when it is False."""
    resonance_frequency_excited: float
    """Fitted resonance frequency with the qubit in |1>, in Hz."""
    chi_hz: float
    """Half the splitting between the |0> and |1> resonances, so the full splitting is 2 chi and the
    qubit Stark shift is 2 chi n_bar. Signed: negative when the |1> resonance sits below the |0> one,
    which is the usual case for a transmon below its resonator."""
    chi_over_kappa: float
    """|chi| / kappa_tot. Above about 0.3 the weak-dispersive formulas behind the photon calibration
    stop holding and the general steady-state expressions are needed instead."""
    kappa_tot_excited_hz: float
    kappa_ext_excited_hz: float
    kappa_int_excited_hz: float
    r_squared_excited: float
    stored_chi_hz: float
    """A chi already on the qubit, for comparison. NaN when there is none."""
    chi_mismatch_fraction: float
    """Relative disagreement between the fitted and the stored chi. NaN when there is no stored chi."""
    success: bool
    failure_reason: str


def fit_circle(z: np.ndarray) -> Tuple[complex, float]:
    """Algebraic least-squares fit of a circle to complex points.

    Solves x^2 + y^2 + A x + B y + C = 0 for (A, B, C).

    Returns
    -------
    center : complex
    radius : float
    """
    x = np.real(z)
    y = np.imag(z)
    a_matrix = np.column_stack([x, y, np.ones_like(x)])
    b_vector = -(x**2 + y**2)
    coefficients, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    a_coef, b_coef, c_coef = coefficients
    x_center = -a_coef / 2
    y_center = -b_coef / 2
    radius_squared = x_center**2 + y_center**2 - c_coef
    if radius_squared <= 0:
        raise ValueError("circle fit returned a non-positive radius")
    return complex(x_center, y_center), float(np.sqrt(radius_squared))


def _circle_deviation(tau: float, freq_offset: np.ndarray, s21: np.ndarray) -> np.ndarray:
    """Distance of every delay-corrected point from the best-fit circle."""
    z = s21 * np.exp(2j * np.pi * freq_offset * tau)
    try:
        center, radius = fit_circle(z)
    except (ValueError, np.linalg.LinAlgError):
        return np.full(z.shape, 1e3)
    return np.abs(z - center) - radius


def fit_cable_delay(freq_offset: np.ndarray, s21: np.ndarray, tau_guess: float) -> float:
    """Fit the cable delay, in seconds, from the measured complex trace.

    The phase slope across the span gives the starting point, and the delay is then refined by
    making the delay-corrected trace as circular as possible. `freq_offset` is the frequency
    relative to the centre of the span, so the constant phase it leaves behind is absorbed by the
    environment phase later in the fit.
    """
    phase = np.unwrap(np.angle(s21))
    # The resonance itself winds the phase by up to 2 pi, which would bias a slope fitted across the
    # whole span. Fit the slope on the wings instead, where only the cable contributes.
    magnitude = np.abs(s21)
    on_resonance = magnitude < np.median(magnitude) - 0.25 * (np.median(magnitude) - magnitude.min())
    wings = ~on_resonance
    if wings.sum() < max(8, 0.2 * phase.size):
        wings = np.ones_like(phase, dtype=bool)
    slope = np.polyfit(freq_offset[wings], phase[wings], 1)[0]
    # The correction multiplies by exp(+2 pi i f tau), so a phase falling with frequency means a
    # positive delay.
    tau_linear = -slope / (2 * np.pi)

    # Scan around the slope estimate before refining: the residual has local minima spaced by
    # roughly one over the span, so a gradient step alone can settle in the wrong one.
    span = float(freq_offset[-1] - freq_offset[0])
    candidates = tau_linear + np.linspace(-1.5, 1.5, 601) / span
    if np.isfinite(tau_guess):
        candidates = np.append(candidates, tau_guess)
    costs = [np.sum(_circle_deviation(tau, freq_offset, s21) ** 2) for tau in candidates]
    tau_best = float(candidates[int(np.argmin(costs))])

    result = least_squares(
        _circle_deviation,
        x0=tau_best,
        args=(freq_offset, s21),
        x_scale=[1 / span],
        diff_step=[1e-4],
    )
    return float(result.x[0]) if result.success else tau_best


def _phase_model(freq: np.ndarray, theta_0: float, q_loaded: float, f_r: float) -> np.ndarray:
    return theta_0 + 2 * np.arctan(2 * q_loaded * (1 - freq / f_r))


def fit_phase(
    freq: np.ndarray, z_centered: np.ndarray, f_r_guess: float, q_loaded_guess: float
) -> Tuple[float, float, float]:
    """Fit the phase of the centred circle against frequency.

    Returns
    -------
    f_r : float
        Resonance frequency in Hz.
    q_loaded : float
        Loaded quality factor, always positive.
    theta_0 : float
        Phase of the circle at resonance, used to locate the off-resonant point.
    """
    theta = np.unwrap(np.angle(z_centered))
    theta_0_guess = 0.5 * (theta[0] + theta[-1])
    popt, _ = curve_fit(
        _phase_model,
        freq,
        theta,
        p0=[theta_0_guess, q_loaded_guess, f_r_guess],
        maxfev=20000,
    )
    theta_0, q_loaded, f_r = popt
    if q_loaded < 0:
        # A negative Q_l only means the trace is travelled the other way round the circle.
        q_loaded = -q_loaded
        theta_0 = theta_0 + np.pi
    return float(f_r), float(q_loaded), float(np.angle(np.exp(1j * theta_0)))


def _guess_resonance(freq: np.ndarray, magnitude: np.ndarray) -> Tuple[float, float]:
    """Guess the resonance frequency and loaded Q from the magnitude dip."""
    index = int(np.argmin(magnitude))
    f_r_guess = float(freq[index])
    baseline = float(np.median(magnitude))
    half = 0.5 * (baseline + float(magnitude[index]))
    within = np.where(magnitude < half)[0]
    if within.size >= 2:
        fwhm = float(freq[within[-1]] - freq[within[0]])
    else:
        fwhm = float(freq[-1] - freq[0]) / 10
    fwhm = max(fwhm, float(np.abs(freq[1] - freq[0])))
    return f_r_guess, f_r_guess / fwhm


def notch_port_fit(freq: np.ndarray, s21: np.ndarray, tau_guess: float, fit_delay: bool = True) -> Dict[str, object]:
    """Run the full notch-port fit on one resonance.

    Parameters
    ----------
    freq : np.ndarray
        Absolute RF frequency of every point, in Hz.
    s21 : np.ndarray
        Complex transmission at every point.
    tau_guess : float
        Starting guess for the cable delay, in seconds.
    fit_delay : bool
        When False the cable delay is held at `tau_guess` instead of being fitted.

    Returns
    -------
    dict
        Keys: f_r, q_loaded, q_external, q_internal, kappa_tot_hz, kappa_ext_hz, kappa_int_hz,
        cable_delay_s, phi_rad, r_squared, plus the normalised trace `s21_normalised` and the
        fitted curve `s21_model` for plotting.
    """
    freq = np.asarray(freq, dtype=float)
    s21 = np.asarray(s21, dtype=complex)
    if freq.size < 8:
        raise ValueError("at least 8 frequency points are needed for a circle fit")
    if not np.all(np.isfinite(s21)):
        raise ValueError("the complex trace contains non-finite values")

    freq_offset = freq - float(np.mean(freq))
    tau = fit_cable_delay(freq_offset, s21, tau_guess) if fit_delay else float(tau_guess)
    z = s21 * np.exp(2j * np.pi * freq_offset * tau)

    center, radius = fit_circle(z)
    f_r_guess, q_loaded_guess = _guess_resonance(freq, np.abs(s21))
    f_r, q_loaded, theta_0 = fit_phase(freq, z - center, f_r_guess, q_loaded_guess)

    if not (freq[0] <= f_r <= freq[-1]):
        raise ValueError("the fitted resonance frequency lies outside the swept span")
    if q_loaded <= 0:
        raise ValueError("the fitted loaded quality factor is not positive")

    # The off-resonant point sits diametrically opposite the resonance on the circle. Dividing it
    # out removes the amplifier gain and the constant environment phase.
    off_resonant = center + radius * np.exp(1j * (theta_0 + np.pi))
    z_normalised = z / off_resonant
    center_normalised = center / off_resonant
    radius_normalised = radius / np.abs(off_resonant)

    # For an ideal notch the normalised circle has centre 1 - r e^{i phi} and radius r, so the
    # impedance-mismatch angle phi follows from the vector pointing from 1 to the centre.
    phi = float(np.angle(1 - center_normalised))
    if np.abs(np.cos(phi)) < 1e-6:
        raise ValueError("the impedance-mismatch angle is degenerate")
    q_external_magnitude = q_loaded / (2 * radius_normalised)
    q_external = q_external_magnitude / np.cos(phi)
    inverse_internal = 1 / q_loaded - 1 / q_external
    if inverse_internal <= 0:
        raise ValueError("the fit gives a non-positive internal quality factor")
    q_internal = 1 / inverse_internal

    model = 1 - (q_loaded / q_external_magnitude) * np.exp(1j * phi) / (1 + 2j * q_loaded * (freq / f_r - 1))
    residual = np.abs(z_normalised) - np.abs(model)
    total = np.abs(z_normalised) - np.mean(np.abs(z_normalised))
    r_squared = float(1 - np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) > 0 else 0.0

    return {
        "f_r": float(f_r),
        "q_loaded": float(q_loaded),
        "q_external": float(q_external),
        "q_internal": float(q_internal),
        "kappa_tot_hz": float(f_r / q_loaded),
        "kappa_ext_hz": float(f_r / q_external),
        "kappa_int_hz": float(f_r / q_internal),
        "cable_delay_s": float(tau),
        "phi_rad": phi,
        "r_squared": r_squared,
        "s21_normalised": z_normalised,
        "s21_model": model,
    }


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert the raw quadratures to volts and attach the absolute frequency axis.

    The complex transmission is rebuilt from `I` and `Q` where it is needed rather than stored, so
    that the dataset keeps to real dtypes and saves to HDF5 unchanged.
    """
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    full_freq = np.array([ds.detuning.values + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def _stored_kappa_hz(resonator) -> float:
    """Return a kappa already stored on the resonator, or NaN when there is none.

    A linewidth carried by a readout pulse comes first, because it was measured some other way and
    so is an independent check on this fit; the Drachma readout pulse stores one. Failing that, the
    kappa an earlier run of this node wrote is used, which then says how much kappa has moved since.
    """
    for operation in getattr(resonator, "operations", {}).values():
        kappa = getattr(operation, "resonator_kappa_hz", None)
        if kappa:
            return float(kappa)
    extras = getattr(resonator, "extras", None) or {}
    if extras.get("kappa_ext_hz") is not None and extras.get("kappa_int_hz") is not None:
        return float(extras["kappa_ext_hz"]) + float(extras["kappa_int_hz"])
    return float("nan")


def _fit_one_state(ds_q: xr.Dataset, tau_guess: float, params, state=None) -> Dict[str, object]:
    """Run the notch-port fit on one qubit's trace, optionally selecting a prepared state."""
    freq = np.asarray(ds_q.full_freq.values, dtype=float)
    quad_i = ds_q.I if state is None else ds_q.I.isel(state=state)
    quad_q = ds_q.Q if state is None else ds_q.Q.isel(state=state)
    s21 = np.asarray(quad_i.values, dtype=float) + 1j * np.asarray(quad_q.values, dtype=float)
    return notch_port_fit(freq, s21, tau_guess, fit_delay=params.fit_cable_delay)


def _stored_chi_hz(qubit) -> float:
    """A chi already on the qubit, for comparison. NaN when there is none."""
    chi = getattr(qubit, "chi", None)
    if chi is None or not np.isfinite(chi) or chi == 0:
        return float("nan")
    return float(chi)


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit a notch-port circle to each qubit's resonance.

    A qubit whose fit raises is reported as failed with the reason, and the remaining qubits are
    still fitted.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset carrying `I` and `Q` over the `detuning` axis and a `full_freq` coordinate.
    node : QualibrationNode
        Supplies `node.namespace["qubits"]` and the analysis parameters.

    Returns
    -------
    ds_fit : xr.Dataset
        The input dataset with per-qubit fit variables added.
    fit_results : dict[str, FitParameters]
    """
    qubits = node.namespace["qubits"]
    params = node.parameters
    probe_scale = node.parameters.probe_amplitude_scale
    has_excited = "state" in ds.dims and ds.sizes["state"] > 1

    fit_results: Dict[str, FitParameters] = {}
    traces: Dict[str, object] = {}
    for qubit in qubits:
        name = qubit.name
        resonator = qubit.resonator
        stored_kappa = _stored_kappa_hz(resonator)
        stored_chi = _stored_chi_hz(qubit)
        probe_power = probe_power_dbm(resonator, probe_scale)
        try:
            ds_q = ds.sel(qubit=name)
            tau_guess = float(resonator.time_of_flight) * 1e-9
            # The state axis is present whenever the node wrote one, even with a single entry, so the
            # ground-state trace is selected off it rather than taken whole.
            fit = _fit_one_state(ds_q, tau_guess, params, state=0 if "state" in ds.dims else None)
            fit_excited = _fit_one_state(ds_q, tau_guess, params, state=1) if has_excited else None
        except Exception as error:  # a single bad resonator must not stop the run
            fit_results[name] = _failed_fit(probe_power, stored_kappa, stored_chi, str(error))
            traces[name] = None
            continue
        traces[name] = (fit["s21_normalised"], fit["s21_model"])

        kappa_tot = fit["kappa_tot_hz"]
        lifetime_ns = 1e9 / (2 * np.pi * kappa_tot)
        mismatch = float("nan")
        if np.isfinite(stored_kappa) and stored_kappa > 0:
            mismatch = abs(kappa_tot - stored_kappa) / stored_kappa

        # chi is half the splitting between the two resonances, signed so that a |1> resonance below
        # the |0> one gives a negative chi, which is the convention the photon calibration reads.
        nan = float("nan")
        chi = nan if fit_excited is None else 0.5 * (fit_excited["f_r"] - fit["f_r"])
        chi_mismatch = nan
        if np.isfinite(chi) and np.isfinite(stored_chi) and stored_chi != 0:
            chi_mismatch = abs(chi - stored_chi) / abs(stored_chi)

        success = fit["r_squared"] >= params.min_r_squared
        reason = "" if success else f"R² {fit['r_squared']:.2f} below {params.min_r_squared}"
        if success and fit_excited is not None and fit_excited["r_squared"] < params.min_r_squared:
            success = False
            reason = f"the |1> resonance fitted with R² {fit_excited['r_squared']:.2f}, below {params.min_r_squared}"

        fit_results[name] = FitParameters(
            resonance_frequency=fit["f_r"],
            q_loaded=fit["q_loaded"],
            q_external=fit["q_external"],
            q_internal=fit["q_internal"],
            kappa_tot_hz=kappa_tot,
            kappa_ext_hz=fit["kappa_ext_hz"],
            kappa_int_hz=fit["kappa_int_hz"],
            kappa_ext_over_kappa_tot=fit["kappa_ext_hz"] / kappa_tot,
            photon_lifetime_ns=lifetime_ns,
            depletion_time_over_lifetime=float(resonator.depletion_time) / lifetime_ns,
            cable_delay_ns=fit["cable_delay_s"] * 1e9,
            probe_power_dbm=probe_power,
            impedance_mismatch_rad=fit["phi_rad"],
            r_squared=fit["r_squared"],
            stored_kappa_hz=stored_kappa,
            kappa_mismatch_fraction=mismatch,
            excited_state_measured=fit_excited is not None,
            resonance_frequency_excited=nan if fit_excited is None else fit_excited["f_r"],
            chi_hz=chi,
            chi_over_kappa=nan if not np.isfinite(chi) else abs(chi) / kappa_tot,
            kappa_tot_excited_hz=nan if fit_excited is None else fit_excited["kappa_tot_hz"],
            kappa_ext_excited_hz=nan if fit_excited is None else fit_excited["kappa_ext_hz"],
            kappa_int_excited_hz=nan if fit_excited is None else fit_excited["kappa_int_hz"],
            r_squared_excited=nan if fit_excited is None else fit_excited["r_squared"],
            stored_chi_hz=stored_chi,
            chi_mismatch_fraction=chi_mismatch,
            success=success,
            failure_reason=reason,
        )

    ds_fit = _attach_fit_to_dataset(ds, fit_results, traces)
    return ds_fit, fit_results


def probe_power_dbm(resonator, amplitude_scale: float, operation: str = "readout") -> float:
    """Absolute power of an operation played at a given amplitude scale, in dBm."""
    if amplitude_scale <= 0:
        return float("-inf")
    return float(resonator.get_output_power(operation) + 20 * np.log10(amplitude_scale))


def _failed_fit(probe_power: float, stored_kappa: float, stored_chi: float, reason: str) -> FitParameters:
    nan = float("nan")
    return FitParameters(
        resonance_frequency=nan,
        q_loaded=nan,
        q_external=nan,
        q_internal=nan,
        kappa_tot_hz=nan,
        kappa_ext_hz=nan,
        kappa_int_hz=nan,
        kappa_ext_over_kappa_tot=nan,
        photon_lifetime_ns=nan,
        depletion_time_over_lifetime=nan,
        cable_delay_ns=nan,
        probe_power_dbm=probe_power,
        impedance_mismatch_rad=nan,
        r_squared=nan,
        stored_kappa_hz=stored_kappa,
        kappa_mismatch_fraction=nan,
        excited_state_measured=False,
        resonance_frequency_excited=nan,
        chi_hz=nan,
        chi_over_kappa=nan,
        kappa_tot_excited_hz=nan,
        kappa_ext_excited_hz=nan,
        kappa_int_excited_hz=nan,
        r_squared_excited=nan,
        stored_chi_hz=stored_chi,
        chi_mismatch_fraction=nan,
        success=False,
        failure_reason=reason,
    )


def _attach_fit_to_dataset(
    ds: xr.Dataset, fit_results: Dict[str, FitParameters], traces: Dict[str, object]
) -> xr.Dataset:
    """Add the fit results to the dataset as per-qubit variables, for plotting and storage.

    The normalised trace and the fitted curve are stored as separate real and imaginary parts so
    that the dataset stays real-valued and saves to HDF5 unchanged.
    """
    names = list(ds.qubit.values)
    ds_fit = ds.copy()
    scalar_fields = [
        "resonance_frequency",
        "q_loaded",
        "q_external",
        "q_internal",
        "kappa_tot_hz",
        "kappa_ext_hz",
        "kappa_int_hz",
        "kappa_ext_over_kappa_tot",
        "photon_lifetime_ns",
        "cable_delay_ns",
        "probe_power_dbm",
        "impedance_mismatch_rad",
        "r_squared",
        "resonance_frequency_excited",
        "chi_hz",
        "chi_over_kappa",
        "kappa_tot_excited_hz",
        "kappa_ext_excited_hz",
        "kappa_int_excited_hz",
        "r_squared_excited",
    ]
    for field_name in scalar_fields:
        values = [getattr(fit_results[str(name)], field_name, float("nan")) for name in names]
        ds_fit[field_name] = xr.DataArray(values, coords={"qubit": ds.qubit}, dims="qubit")
    ds_fit["success"] = xr.DataArray(
        [bool(fit_results[str(name)].success) for name in names], coords={"qubit": ds.qubit}, dims="qubit"
    )

    num_points = ds.sizes["detuning"]
    empty = np.full(num_points, np.nan)
    for variable, index, source in (
        ("S21_normalised_I", 0, np.real),
        ("S21_normalised_Q", 0, np.imag),
        ("S21_model_I", 1, np.real),
        ("S21_model_Q", 1, np.imag),
    ):
        rows = []
        for name in names:
            trace = traces.get(str(name))
            rows.append(empty if trace is None else source(trace[index]))
        ds_fit[variable] = xr.DataArray(
            np.array(rows), coords={"qubit": ds.qubit, "detuning": ds.detuning}, dims=["qubit", "detuning"]
        )
    return ds_fit


def log_fitted_results(fit_results: Dict, log_callable=None) -> None:
    """Log kappa, the collection-efficiency ceiling and the probe power for every qubit.

    `fit_results` holds either FitParameters instances or the dictionaries they become once the node
    has stored them.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for name, result in fit_results.items():
        result = result if isinstance(result, dict) else result.__dict__
        if not result["success"]:
            log_callable(f"Results for qubit {name}: FAIL! {result['failure_reason']}")
            continue
        lines = [
            f"Results for qubit {name}: SUCCESS!",
            f"\tkappa_tot: {1e-6 * result['kappa_tot_hz']:.3f} MHz | "
            f"kappa_ext: {1e-6 * result['kappa_ext_hz']:.3f} MHz | "
            f"kappa_int: {1e-6 * result['kappa_int_hz']:.3f} MHz",
            f"\tkappa_ext / kappa_tot: {result['kappa_ext_over_kappa_tot']:.3f} " f"(ceiling on collection efficiency)",
            f"\tQ_l: {result['q_loaded']:.3g} | Q_c: {result['q_external']:.3g} | "
            f"Q_i: {result['q_internal']:.3g} at {result['probe_power_dbm']:.1f} dBm",
            f"\tfitted cable delay: {result['cable_delay_ns']:.1f} ns | R²: {result['r_squared']:.3f}",
            f"\tfitted resonance: {1e-9 * result['resonance_frequency']:.6f} GHz (reported only, not written to state)",
            f"\tphoton lifetime: {result['photon_lifetime_ns']:.1f} ns | "
            f"stored depletion time is {result['depletion_time_over_lifetime']:.2f} lifetimes",
        ]
        if np.isfinite(result["kappa_mismatch_fraction"]):
            lines.append(
                f"\tstored kappa: {1e-6 * result['stored_kappa_hz']:.3f} MHz, "
                f"disagreement {100 * result['kappa_mismatch_fraction']:.0f}%"
            )
        if result.get("excited_state_measured"):
            lines.append(
                f"\tchi: {1e-6 * result['chi_hz']:+.4f} MHz (|1> resonance at "
                f"{1e-9 * result['resonance_frequency_excited']:.6f} GHz, R² {result['r_squared_excited']:.3f})"
            )
            verdict = (
                "weak-dispersive, the simple formulas hold"
                if result["chi_over_kappa"] < 0.3
                else "ABOVE 0.3, the weak-dispersive formulas no longer hold"
            )
            lines.append(f"\t|chi| / kappa: {result['chi_over_kappa']:.3f} ({verdict})")
            lines.append(
                f"\tkappa_tot in |1>: {1e-6 * result['kappa_tot_excited_hz']:.3f} MHz | "
                f"kappa_ext in |1>: {1e-6 * result['kappa_ext_excited_hz']:.3f} MHz"
            )
            if np.isfinite(result["chi_mismatch_fraction"]):
                lines.append(
                    f"\tstored chi: {1e-6 * result['stored_chi_hz']:+.4f} MHz, "
                    f"disagreement {100 * result['chi_mismatch_fraction']:.0f}%"
                )
        log_callable("\n".join(lines))
