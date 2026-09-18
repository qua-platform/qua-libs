"""Convert a Stark tone amplitude into an intra-resonator photon number.

The node measures a Ramsey fringe at every combination of tone amplitude and free evolution time.
At each amplitude the accumulated fringe phase is fitted against the free evolution time with a free
intercept: the slope is the Stark shift Delta_omega and the intercept absorbs everything that does
not scale with the time the tone was on, which is the phase picked up during the two pi/2 pulses and
any residual fill or ring-down. The same fit on the log of the contrast ratio gives the
measurement-induced dephasing Gamma_d. The photon number follows as Delta_omega / 2 chi, and is
fitted linearly against the tone power in mW.

Frequencies read from and written to state are in Hz. Angular quantities are used
inside the formulas and are named accordingly.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

__all__ = [
    "FitParameters",
    "MissingResonatorLinewidthError",
    "critical_photon_number",
    "dressed_resonances",
    "fit_fringe",
    "fit_raw_data",
    "gamma_per_photon",
    "kappa_tot_hz_from_extras",
    "log_fitted_results",
    "photons_per_mw_fit",
    "power_dbm_to_mw",
    "process_raw_dataset",
    "slope_with_intercept",
    "tone_power_dbm",
    "transmon_anharmonicity",
]

_NAN = float("nan")


class MissingResonatorLinewidthError(RuntimeError):
    """Raised when kappa is absent from a resonator's extras.

    Without kappa there is no independent check on the photon number, and the
    sequence itself needs it to size the steady-state pad, so the node stops rather than quoting a
    number derived from an assumed linewidth.
    """


@dataclass
class FitParameters:
    """Photon calibration results for a single qubit."""

    photons_per_mw: float
    """The linear constant c in n_bar = c * P_mW, from the Stark phase."""
    single_photon_power_dbm: float
    """The power at which n_bar = 1, the same constant read as a reference level."""
    n_bar_at_operating_amplitude: float
    """Photon number at the qubit's own operating readout amplitude, i.e. amplitude scale 1.0. The
    amplitude sweep does not reach there, because the fringe collapses first, so this is the fitted
    line extended to it."""
    operating_power_dbm: float
    chi_hz: float
    kappa_tot_hz: float
    chi_over_kappa: float
    """|chi| / kappa. The formulas here assume the weak-dispersive limit; above about 0.3 the two
    dressed resonances are separated by less than a linewidth."""

    critical_photon_number: float
    """The photon number at which the dispersive approximation breaks down, derived from the stored
    chi, detuning and anharmonicity rather than measured here."""
    coupling_g_hz: float
    """The qubit-resonator coupling implied by the same three numbers. Reported as a sanity check on
    the stored chi: a g far from the 50 to 200 MHz a planar chip is built for means chi is wrong."""
    anharmonicity_sign_flipped: bool
    """True when the stored anharmonicity was positive and its sign was imposed negative, per
    A transmon is negatively anharmonic, and the sign matters more than the value."""
    chi_sign_flipped: bool
    """True when, after the anharmonicity sign was settled, chi still implied a negative g squared."""
    critical_power_dbm: float
    """The tone power at which n_bar reaches the critical photon number. Unlike the critical photon
    number itself this needs the measured calibration, so it exists only for a successful fit."""
    critical_amplitude_scale: float
    n_bar_over_critical: float

    gamma_d_over_delta_omega: float
    """Measured slope of the induced dephasing against the Stark shift."""
    expected_gamma_ratio: float
    """What that slope should be, from chi, kappa and the tone detuning, with no approximation in
    chi/kappa. At zero detuning it can never exceed 1, and detuning only lowers it, so a measured
    value above 1 means something other than measurement backaction is decohering the qubit."""
    gamma_ratio_disagreement: float
    n_bar_from_gamma_at_operating: float
    """The photon number at the operating amplitude derived from the dephasing instead of the phase.
    The same check as the ratio above, in the unit the node exists to report."""
    n_bar_gamma_over_phase: float
    """Ratio of the two photon numbers. One is the answer; two agreeing is the evidence."""

    phase_linearity_r_squared: float
    """R² of the accumulated phase against the free evolution time, over the amplitudes that carry
    real signal. This is what makes the number falsifiable: a phase that does not grow with the time
    the tone was on is not a Stark phase."""
    gamma_linearity_r_squared: float
    linear_fit_r_squared: float
    """R² of n_bar against tone power."""
    max_phase_step_rad: float
    """Largest growth of the fringe phase between adjacent free evolution times. The phase is
    unwrapped along that axis, which holds only while the step stays below pi."""
    min_usable_free_evolution_count: int
    """Fewest surviving free evolution times among the amplitudes that contributed."""
    number_splitting_phase_limit_ns: float
    """Longest free evolution time at which the qubit still cannot resolve one photon from the next,
    i.e. where 2 chi tau reaches the configured limit. Times past it are dropped, because the
    apparent phase there follows n_bar sin(2 chi tau) rather than 2 chi n_bar tau."""
    num_times_within_linear_regime: int
    max_tau_over_photon_lifetime: float
    """Longest free evolution time that kept a usable fringe at any amplitude, in resonator photon
    lifetimes. Below about one, a resonator transient grows almost linearly with time too, so the
    linearity check above cannot tell the two apart and a straight line proves nothing."""
    num_contributing_amplitudes: int

    fringe_phase_inverted: bool
    """True when the measured fringe phase ran opposite to the assumed sign convention and was
    flipped so that the photon number comes out positive."""
    tone_frequency_hz: float
    tone_detuning_hz: float
    """Tone frequency minus the midpoint of the two dressed resonances. NaN when neither node 23a's
    measured resonances nor a bare resonator frequency is in state, in which case the expected
    Gamma_d ratio is evaluated at zero detuning."""
    detuning_source: str
    steady_state_pad_ns: float
    steady_state_fraction: float
    """Fraction of the steady-state photon number the pad reaches."""
    depletion_over_lifetime: float
    """The depletion wait in photon lifetimes, i.e. how many residual photons are still in the
    resonator when the readout starts."""

    success: bool
    failure_reason: str


# --------------------------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------------------------- #
def fit_fringe(phases_in_turns: np.ndarray, signal: np.ndarray) -> Tuple[float, float, float]:
    """Fit one fringe, returning its contrast, phase and offset.

    The model is `offset + contrast * cos(2 pi phase + fringe_phase)`. The frequency is known,
    because the phase axis is the phase of the second pi/2 pulse, so the fit is an ordinary linear
    least squares on the basis [1, cos, sin]. That has no starting guess to get wrong and stays
    stable when the contrast has almost collapsed.

    Returns
    -------
    contrast : float
    fringe_phase : float
        In radians, wrapped to (-pi, pi].
    offset : float
    """
    phases_in_turns = np.asarray(phases_in_turns, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if phases_in_turns.size < 3:
        raise ValueError("at least 3 phase points are needed to fit a fringe")
    angle = 2 * np.pi * phases_in_turns
    design = np.column_stack([np.ones_like(angle), np.cos(angle), np.sin(angle)])
    coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
    offset, cosine_term, sine_term = coefficients
    contrast = float(np.hypot(cosine_term, sine_term))
    fringe_phase = float(np.arctan2(-sine_term, cosine_term))
    return contrast, fringe_phase, float(offset)


def slope_with_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Straight-line fit with a free intercept, returning slope, intercept and R².

    The intercept is free on purpose. Everything the tone does to the qubit outside the free
    evolution -- the phase picked up while the two pi/2 pulses play in a Stark-shifted qubit, the
    tilt of the rotation axis, any residual fill or ring-down -- is the same at every free evolution
    time, so it lands in the intercept and leaves the slope as the quantity being measured. Forcing
    the line through the origin puts all of it into the slope instead.

    Returns NaNs when fewer than three points survive or the x axis has no span, because a two
    parameter fit to two points has no residual to report.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 3:
        return _NAN, _NAN, _NAN
    x, y = x[usable], y[usable]
    if np.ptp(x) <= 0:
        return _NAN, _NAN, _NAN
    design = np.column_stack([x, np.ones_like(x)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(coefficients[0]), float(coefficients[1])
    residual = y - design @ coefficients
    total = y - np.mean(y)
    denominator = float(total @ total)
    r_squared = float(1 - (residual @ residual) / denominator) if denominator > 0 else _NAN
    return slope, intercept, r_squared


def power_dbm_to_mw(power_dbm: np.ndarray) -> np.ndarray:
    """Convert dBm to mW."""
    return 10 ** (np.asarray(power_dbm, dtype=float) / 10)


def tone_power_dbm(resonator, amplitude_scales: np.ndarray, operation: str = "readout") -> np.ndarray:
    """Power of the Stark tone at each amplitude scale, in dBm.

    The tone is the resonator's `const` operation played at the qubit's operating readout amplitude,
    so the power at scale 1.0 is the power of the readout operation. Reading it from the readout
    operation rather than from `const` keeps the result correct after the node has reverted the
    temporary changes it made to `const`. Power goes as amplitude squared, hence the factor of 20.
    """
    scales = np.asarray(amplitude_scales, dtype=float)
    base_dbm = float(resonator.get_output_power(operation))
    with np.errstate(divide="ignore"):
        return base_dbm + 20 * np.log10(scales)


def photons_per_mw_fit(power_mw: np.ndarray, n_bar: np.ndarray) -> Tuple[float, float]:
    """Fit n_bar = c * P_mW through the origin.

    This line is forced through the origin, unlike the phase-against-time fit, because an empty
    resonator holds no photons and there is nothing for an intercept to absorb. Returns the constant
    and the R² of the fit.
    """
    power_mw = np.asarray(power_mw, dtype=float)
    n_bar = np.asarray(n_bar, dtype=float)
    usable = np.isfinite(power_mw) & np.isfinite(n_bar) & (power_mw > 0)
    if usable.sum() < 2:
        raise ValueError("at least 2 non-zero amplitude points are needed for the photon calibration")
    power_mw = power_mw[usable]
    n_bar = n_bar[usable]
    constant = float(np.sum(power_mw * n_bar) / np.sum(power_mw**2))
    residual = n_bar - constant * power_mw
    total = n_bar - np.mean(n_bar)
    r_squared = float(1 - np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) > 0 else 0.0
    return constant, r_squared


def kappa_tot_hz_from_extras(qubit) -> float:
    """Read kappa_tot in Hz from a resonator's extras, as node 23a wrote it.

    Raises
    ------
    MissingResonatorLinewidthError
        When either linewidth is missing, so that no photon number is ever derived from an assumed
        linewidth and no pad is ever sized from one.
    """
    extras = getattr(qubit.resonator, "extras", None) or {}
    missing = [key for key in ("kappa_ext_hz", "kappa_int_hz") if extras.get(key) is None]
    if missing:
        raise MissingResonatorLinewidthError(
            f"{qubit.name}: {' and '.join(missing)} missing from qubit.resonator.extras. "
            f"Run node 23a_resonator_linewidth first; this node will not assume a linewidth."
        )
    return float(extras["kappa_ext_hz"]) + float(extras["kappa_int_hz"])


# --------------------------------------------------------------------------------------------- #
# Dispersive physics, with no expansion in chi / kappa
# --------------------------------------------------------------------------------------------- #
def gamma_per_photon(chi_rad: float, kappa_rad: float, detuning_rad: float = 0.0) -> float:
    """Measurement-induced dephasing rate per unit mean photon number, in s^-1.

    The steady-state coherent states for the two qubit states are `alpha_s = eps / (Delta_s + i
    kappa / 2)` with `Delta_g = delta + chi` and `Delta_e = delta - chi`, where `delta` is the drive
    detuning from the midpoint of the two dressed resonances. The dephasing rate is
    `2 chi Im(alpha_g alpha_e*)`, and `eps` is normalised here so that the mean photon number
    `(n_g + n_e) / 2` is one.

    The familiar `8 chi^2 n / kappa` is the limit of this for `chi << kappa`. It is not used, because
    dividing this by the Stark shift per photon `2 chi` gives the consistency check the node relies
    on, and on a resonator with |chi|/kappa near 0.5 the two differ by tens of percent.
    """
    chi_rad = float(chi_rad)
    kappa_rad = float(kappa_rad)
    delta_g = detuning_rad + chi_rad
    delta_e = detuning_rad - chi_rad
    half_kappa_squared = kappa_rad**2 / 4
    n_ground = 1.0 / (delta_g**2 + half_kappa_squared)
    n_excited = 1.0 / (delta_e**2 + half_kappa_squared)
    denominator = (delta_g * delta_e + half_kappa_squared) ** 2 + (kappa_rad * chi_rad) ** 2
    if denominator <= 0 or (n_ground + n_excited) <= 0:
        return _NAN
    epsilon_squared = 2.0 / (n_ground + n_excited)
    return float(abs(2 * chi_rad**2 * kappa_rad * epsilon_squared / denominator))


def transmon_anharmonicity(qubit) -> Tuple[float, bool]:
    """The anharmonicity to use in the formulas, in Hz, and whether its sign was imposed.

    A transmon is negatively anharmonic: the 1-2 transition sits below the 0-1. Every qubit in this
    package's state stores the anharmonicity as a positive number, which makes `g^2` come out
    negative and invites the chi sign guard to blame the wrong input. So the magnitude is taken and
    the negative sign imposed, and the substitution is reported.
    """
    anharmonicity = getattr(qubit, "anharmonicity", None)
    if not isinstance(anharmonicity, (int, float)) or not np.isfinite(anharmonicity) or anharmonicity == 0:
        return _NAN, False
    return -abs(float(anharmonicity)), bool(anharmonicity > 0)


def critical_photon_number(qubit) -> Dict[str, float]:
    """The photon number at which the dispersive approximation breaks down, from stored state only.

    For a transmon in the dispersive limit the dispersive shift is chi = g^2 * alpha /
    (Delta * (Delta + alpha)), with Delta = omega_qubit - omega_resonator. Substituting that into
    n_crit = Delta^2 / (4 g^2) cancels the coupling, which no node measures directly:

        n_crit = Delta * alpha / (4 * chi * (Delta + alpha))

    So nothing has to be measured to get this number: the detuning, the anharmonicity and chi are all
    already in state. It is a derived quantity and goes stale when any of the three changes.

    The anharmonicity sign is settled first, and only then is the chi sign guard applied, so that a
    state problem in one is not blamed on the other.

    Treat the result as an order of magnitude. It is the scale on which the dispersive approximation
    fails, derived within that same approximation, and measurement-induced state transitions
    generally arrive at a lower power, so it is a ceiling on the ceiling.

    Returns
    -------
    Dict[str, float]
        Keys: critical_photon_number, coupling_g_hz, detuning_hz, chi_sign_flipped,
        anharmonicity_sign_flipped. Every value is NaN when any input is missing, because this
        number is a bonus and must never stop the photon calibration.
    """
    empty = {
        "critical_photon_number": _NAN,
        "coupling_g_hz": _NAN,
        "detuning_hz": _NAN,
        "chi_sign_flipped": False,
        "anharmonicity_sign_flipped": False,
    }

    chi = getattr(qubit, "chi", None)
    f_01 = getattr(qubit, "f_01", None)
    # The bare resonator frequency is the right one, but it differs from the dressed frequency by a
    # chi, which is a part in ten thousand of the detuning, so either will do.
    f_r = getattr(qubit.resonator, "frequency_bare", None) or getattr(qubit.resonator, "RF_frequency", None)
    alpha, alpha_flipped = transmon_anharmonicity(qubit)
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in (chi, f_01, f_r)):
        return empty
    if chi == 0 or not np.isfinite(alpha):
        return empty

    chi = float(chi)
    detuning = float(f_01) - float(f_r)
    if detuning + alpha == 0:
        return empty

    # g^2 must come out positive. The anharmonicity sign is already settled, so if it still does not,
    # the stored chi carries the opposite sign convention to the formula.
    g_squared = chi * detuning * (detuning + alpha) / alpha
    chi_flipped = g_squared < 0
    if chi_flipped:
        g_squared = -g_squared
    if g_squared == 0:
        return empty

    return {
        "critical_photon_number": float(detuning**2 / (4 * g_squared)),
        "coupling_g_hz": float(np.sqrt(g_squared)),
        "detuning_hz": detuning,
        "chi_sign_flipped": chi_flipped,
        "anharmonicity_sign_flipped": alpha_flipped,
    }


def dressed_resonances(qubit, chi_hz: float) -> Tuple[float, float, str]:
    """The two dressed resonator frequencies in Hz, and where they came from.

    The tone detuning has to be measured against something that is not the tone itself. Taking the
    readout frequency as the midpoint and adding chi to it, which an earlier version of this module
    did, makes the reported detuning exactly minus chi whenever the tone defaults to the readout
    frequency, so it carries no information at all.

    Preference order: the two resonances node 23a fitted, then a stored bare resonator frequency,
    then nothing.
    """
    extras = getattr(qubit.resonator, "extras", None) or {}
    ground = extras.get("f_r_ground_hz")
    excited = extras.get("f_r_excited_hz")
    if ground is not None and excited is not None and np.isfinite(ground) and np.isfinite(excited):
        return float(ground), float(excited), "measured by node 23a"
    bare = getattr(qubit.resonator, "frequency_bare", None)
    if isinstance(bare, (int, float)) and np.isfinite(bare) and bare > 0:
        return float(bare) - chi_hz, float(bare) + chi_hz, "stored bare frequency"
    return _NAN, _NAN, "unavailable"


# --------------------------------------------------------------------------------------------- #
# Dataset handling
# --------------------------------------------------------------------------------------------- #
def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert the raw quadratures to volts when state discrimination was not used."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def _signal(ds: xr.Dataset, node: QualibrationNode) -> xr.DataArray:
    if node.parameters.use_state_discrimination:
        return ds.state
    return ds.I


def _chi_hz(qubit, params) -> float:
    """Return the stored chi in Hz, or raise a ValueError naming why the qubit is unusable."""
    chi = getattr(qubit, "chi", None)
    if chi is None or not np.isfinite(chi) or chi == 0:
        raise ValueError("the stored chi is zero or missing")
    if not (params.min_abs_chi_in_hz <= abs(float(chi)) <= params.max_abs_chi_in_hz):
        raise ValueError(
            f"the stored |chi| of {1e-6 * abs(float(chi)):.3f} MHz is outside the plausible range "
            f"{1e-6 * params.min_abs_chi_in_hz:.3f} to {1e-6 * params.max_abs_chi_in_hz:.3f} MHz"
        )
    return float(chi)


def _tone_frequency_hz(qubit, node: QualibrationNode) -> float:
    """The Stark tone frequency, defaulting to the qubit's current readout frequency."""
    override = node.parameters.tone_frequency_in_ghz
    if override is None:
        return float(qubit.resonator.RF_frequency)
    return float(override) * 1e9


def _fringe_grid(signal: xr.DataArray, num_taus: int, num_amps: int, phases: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit one fringe per (free evolution time, amplitude) pair."""
    contrast = np.full((num_taus, num_amps), _NAN)
    raw_phase = np.full((num_taus, num_amps), _NAN)
    for t in range(num_taus):
        for a in range(num_amps):
            contrast[t, a], raw_phase[t, a], _ = fit_fringe(phases, signal.isel(tau=t, amp_scale=a).values)
    return {"contrast": contrast, "raw_phase": raw_phase}


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Wrap an angle, or an array of them, into (-pi, pi]."""
    return (np.asarray(values, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _unwrap_usable(values: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Unwrap a phase series along the free evolution axis, ignoring the points that were dropped.

    Unwrapping through a collapsed fringe would let its noise-dominated phase decide how many turns
    every later point is given, so dropped points are left as NaN rather than carried through.
    """
    out = np.full(values.shape, _NAN)
    index = np.flatnonzero(usable & np.isfinite(values))
    if index.size:
        out[index] = np.unwrap(values[index])
    return out


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Turn the measured fringes into a photon calibration for every qubit.

    A qubit with a zero or implausible chi is skipped with a logged reason and the rest of the run
    continues. A missing kappa stops the node, because a photon number derived from an assumed
    linewidth would be worse than no number at all.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset over the `tau`, `phase` and `amp_scale` axes.
    node : QualibrationNode
        Supplies `node.namespace["qubits"]`, the per-qubit sequence timings and the analysis
        parameters.

    Returns
    -------
    ds_fit : xr.Dataset
        The input dataset with the fringe results, the per-amplitude fits and the per-qubit fit added.
    fit_results : dict[str, FitParameters]
    """
    qubits = node.namespace["qubits"]
    params = node.parameters
    signal = _signal(ds, node)
    amplitude_scales = np.asarray(ds.amp_scale.values, dtype=float)
    phases = np.asarray(ds.phase.values, dtype=float)
    taus = np.asarray(ds.tau.values, dtype=float) * 1e-9
    sequence = node.namespace.get("sequence_info", {})

    fit_results: Dict[str, FitParameters] = {}
    per_amp: Dict[str, Dict[str, np.ndarray]] = {}
    per_grid: Dict[str, Dict[str, np.ndarray]] = {}

    for qubit in qubits:
        name = qubit.name
        # Reading kappa is deliberately outside the try: a missing kappa stops the node.
        kappa_tot_hz = kappa_tot_hz_from_extras(qubit)
        tone_frequency = _tone_frequency_hz(qubit, node)
        info = sequence.get(name, {})
        try:
            chi_hz = _chi_hz(qubit, params)
            grid = _fringe_grid(signal.sel(qubit=name), taus.size, amplitude_scales.size, phases)
            fit_results[name], per_amp[name], per_grid[name] = _photon_calibration(
                qubit=qubit,
                grid=grid,
                amplitude_scales=amplitude_scales,
                taus=taus,
                chi_hz=chi_hz,
                kappa_tot_hz=kappa_tot_hz,
                tone_frequency=tone_frequency,
                info=info,
                params=params,
            )
        except Exception as error:  # a bad qubit must not take the rest of the run with it
            fit_results[name] = _failed_fit(kappa_tot_hz, tone_frequency, qubit, info, str(error))
            per_amp[name] = _empty_per_amp(amplitude_scales.size)
            per_grid[name] = _empty_grid(taus.size, amplitude_scales.size)

    ds_fit = _attach_fit_to_dataset(ds, fit_results, per_amp, per_grid)
    return ds_fit, fit_results


def _photon_calibration(qubit, grid, amplitude_scales, taus, chi_hz, kappa_tot_hz, tone_frequency, info, params):
    """Turn one qubit's grid of fringes into its photon calibration."""
    num_taus, num_amps = grid["contrast"].shape
    reference = int(np.argmin(np.abs(amplitude_scales)))
    if amplitude_scales[reference] != 0:
        raise ValueError("the amplitude sweep has no zero-amplitude reference point")
    if num_taus < 3:
        raise ValueError("at least 3 free evolution times are needed to fit a slope with a free intercept")

    chi_rad = 2 * np.pi * chi_hz
    kappa_rad = 2 * np.pi * kappa_tot_hz

    contrast = grid["contrast"]
    reference_contrast = contrast[:, reference]
    # Each fringe phase comes out of its own fit wrapped into (-pi, pi], so their difference lands
    # anywhere in (-2pi, 2pi] and carries a spurious turn whenever the two fits wrapped differently.
    # Wrapping the difference back before unwrapping along the time axis removes that; skipping it
    # leaves a phase that can be a whole turn out at a single point and a photon number that comes
    # out negative.
    raw_shift = _wrap_to_pi(grid["raw_phase"] - grid["raw_phase"][:, [reference]])

    # A fringe whose contrast has collapsed carries no phase, so it is dropped rather than fitted.
    with np.errstate(invalid="ignore"):
        usable = contrast >= params.min_contrast_fraction * reference_contrast[:, None]

    # The Stark phase is only 2 chi n_bar tau while the qubit cannot resolve one photon from the
    # next. Once 2 chi tau approaches a radian the qubit is resolving photon number, the apparent
    # phase follows n_bar sin(2 chi tau) instead of growing, and past 2 chi tau of pi/2 it turns
    # round and runs backwards. Those times are dropped rather than fitted, because a straight line
    # through them returns a photon number that is simply too small.
    splitting_phase = 2 * abs(chi_rad) * taus
    within_linear_regime = splitting_phase <= params.max_number_splitting_phase_rad
    usable &= within_linear_regime[:, None]
    usable[:, reference] = True

    phase_shift = np.full((num_taus, num_amps), _NAN)
    for a in range(num_amps):
        phase_shift[:, a] = _unwrap_usable(raw_shift[:, a], usable[:, a])

    delta_omega = np.full(num_amps, _NAN)
    phase_intercept = np.full(num_amps, _NAN)
    phase_r2 = np.full(num_amps, _NAN)
    gamma_d = np.full(num_amps, _NAN)
    gamma_r2 = np.full(num_amps, _NAN)
    usable_count = np.zeros(num_amps, dtype=int)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(reference_contrast[:, None]) - np.log(contrast)
    for a in range(num_amps):
        mask = usable[:, a] & np.isfinite(phase_shift[:, a])
        usable_count[a] = int(mask.sum())
        delta_omega[a], phase_intercept[a], phase_r2[a] = slope_with_intercept(taus[mask], phase_shift[mask, a])
        gamma_mask = usable[:, a] & np.isfinite(log_ratio[:, a])
        gamma_d[a], _, gamma_r2[a] = slope_with_intercept(taus[gamma_mask], log_ratio[gamma_mask, a])

    n_bar = delta_omega / (2 * chi_rad)

    # The sign relating the fitted fringe phase to the accumulated qubit phase depends on the
    # hardware's phase convention, which no part of the stored state records. The photon number is
    # positive by definition, so a sweep that comes out negative is flipped and the flip is reported.
    inverted = bool(np.nansum(n_bar) < 0)
    if inverted:
        # The intercept is flipped with the slope it belongs to. Leaving it behind makes the fitted
        # line run opposite to the points it was fitted to, in the results and in the figure.
        n_bar, delta_omega = -n_bar, -delta_omega
        phase_shift, phase_intercept = -phase_shift, -phase_intercept

    # The tone detuning has to be measured against something other than the tone.
    f_ground, f_excited, detuning_source = dressed_resonances(qubit, chi_hz)
    midpoint = 0.5 * (f_ground + f_excited)
    detuning_hz = tone_frequency - midpoint if np.isfinite(midpoint) else _NAN
    detuning_rad = 2 * np.pi * detuning_hz if np.isfinite(detuning_hz) else 0.0

    gamma_one_photon = gamma_per_photon(chi_rad, kappa_rad, detuning_rad)
    n_bar_gamma = (
        gamma_d / gamma_one_photon if np.isfinite(gamma_one_photon) and gamma_one_photon > 0 else gamma_d * _NAN
    )

    power_dbm = tone_power_dbm(qubit.resonator, amplitude_scales)
    power_mw = power_dbm_to_mw(power_dbm)
    photons_per_mw, r_squared = photons_per_mw_fit(power_mw, n_bar)
    try:
        photons_per_mw_gamma, _ = photons_per_mw_fit(power_mw, n_bar_gamma)
    except ValueError:
        photons_per_mw_gamma = _NAN

    operating_power_dbm = float(qubit.resonator.get_output_power("readout"))
    operating_power_mw = float(power_dbm_to_mw(operating_power_dbm))
    n_bar_operating = photons_per_mw * operating_power_mw
    n_bar_operating_gamma = photons_per_mw_gamma * operating_power_mw

    measured_ratio = _gamma_ratio(delta_omega, gamma_d)
    expected_ratio = gamma_one_photon / (2 * abs(chi_rad)) if np.isfinite(gamma_one_photon) else _NAN
    disagreement = abs(measured_ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else _NAN

    # Linearity is judged on the amplitudes that carry real signal. At the bottom of the sweep the
    # phase is a few hundredths of a radian and its R^2 says more about the shot noise than about
    # whether the phase grew with time.
    finite_shift = np.abs(delta_omega[np.isfinite(delta_omega)])
    threshold = 0.25 * finite_shift.max() if finite_shift.size else 0.0
    carries_signal = np.isfinite(delta_omega) & (np.abs(delta_omega) >= threshold) & (usable_count >= 3)
    carries_signal[reference] = False
    phase_linearity = float(np.nanmedian(phase_r2[carries_signal])) if carries_signal.any() else _NAN
    gamma_linearity = float(np.nanmedian(gamma_r2[carries_signal])) if carries_signal.any() else _NAN

    photon_lifetime = 1.0 / kappa_rad
    longest_usable = 0.0
    for a in range(num_amps):
        if a == reference or usable_count[a] < 3:
            continue
        times = taus[usable[:, a] & np.isfinite(phase_shift[:, a])]
        if times.size:
            longest_usable = max(longest_usable, float(times.max()))
    max_tau_over_lifetime = longest_usable / photon_lifetime

    steps = np.abs(np.diff(phase_shift, axis=0))
    max_phase_step = float(np.nanmax(steps)) if np.isfinite(steps).any() else _NAN

    contributing = np.isfinite(n_bar) & (power_mw > 0)
    n_crit = critical_photon_number(qubit)

    # n_bar = c * P_mW inverts to P_crit_mW = n_crit / c, which in dBm is 10 log10 of the ratio.
    # Power goes as amplitude squared, so the equivalent amplitude scale carries a factor of 20.
    critical_power_dbm = _NAN
    critical_amplitude_scale = _NAN
    if photons_per_mw > 0 and np.isfinite(n_crit["critical_photon_number"]):
        critical_power_dbm = float(10 * np.log10(n_crit["critical_photon_number"] / photons_per_mw))
        critical_amplitude_scale = float(10 ** ((critical_power_dbm - operating_power_dbm) / 20))

    reasons = []
    if not (np.isfinite(phase_linearity) and phase_linearity >= params.min_phase_linearity_r_squared):
        reasons.append(
            f"the fringe phase grows against the free evolution time with R² of {phase_linearity:.2f}, "
            f"below {params.min_phase_linearity_r_squared}, so it is not a Stark phase"
        )
    if not (np.isfinite(r_squared) and r_squared >= params.min_linear_fit_r_squared):
        reasons.append(f"R² {r_squared:.2f} of n_bar against power is below {params.min_linear_fit_r_squared}")
    if int(contributing.sum()) < 3:
        reasons.append(f"only {int(contributing.sum())} amplitude points produced a photon number")
    if not np.isfinite(photons_per_mw) or photons_per_mw <= 0:
        reasons.append("the fitted photons per mW is not a positive number")

    result = FitParameters(
        photons_per_mw=float(photons_per_mw),
        single_photon_power_dbm=float(-10 * np.log10(photons_per_mw)) if photons_per_mw > 0 else _NAN,
        n_bar_at_operating_amplitude=float(n_bar_operating),
        operating_power_dbm=operating_power_dbm,
        chi_hz=chi_hz,
        kappa_tot_hz=kappa_tot_hz,
        chi_over_kappa=float(abs(chi_hz) / kappa_tot_hz),
        critical_photon_number=n_crit["critical_photon_number"],
        coupling_g_hz=n_crit["coupling_g_hz"],
        anharmonicity_sign_flipped=bool(n_crit["anharmonicity_sign_flipped"]),
        chi_sign_flipped=bool(n_crit["chi_sign_flipped"]),
        critical_power_dbm=critical_power_dbm,
        critical_amplitude_scale=critical_amplitude_scale,
        n_bar_over_critical=float(n_bar_operating / n_crit["critical_photon_number"]),
        gamma_d_over_delta_omega=measured_ratio,
        expected_gamma_ratio=expected_ratio,
        gamma_ratio_disagreement=disagreement,
        n_bar_from_gamma_at_operating=float(n_bar_operating_gamma),
        n_bar_gamma_over_phase=float(n_bar_operating_gamma / n_bar_operating) if n_bar_operating else _NAN,
        phase_linearity_r_squared=phase_linearity,
        gamma_linearity_r_squared=gamma_linearity,
        linear_fit_r_squared=r_squared,
        max_phase_step_rad=max_phase_step,
        min_usable_free_evolution_count=int(usable_count[carries_signal].min()) if carries_signal.any() else 0,
        number_splitting_phase_limit_ns=float(1e9 * params.max_number_splitting_phase_rad / (2 * abs(chi_rad))),
        num_times_within_linear_regime=int(within_linear_regime.sum()),
        max_tau_over_photon_lifetime=float(max_tau_over_lifetime),
        num_contributing_amplitudes=int(contributing.sum()),
        fringe_phase_inverted=inverted,
        tone_frequency_hz=tone_frequency,
        tone_detuning_hz=detuning_hz,
        detuning_source=detuning_source,
        steady_state_pad_ns=float(info.get("pad_ns", _NAN)),
        steady_state_fraction=float(info.get("steady_state_fraction", _NAN)),
        depletion_over_lifetime=float(info.get("depletion_over_lifetime", _NAN)),
        success=not reasons,
        failure_reason="; ".join(reasons),
    )
    amp_curves = {
        "delta_omega": delta_omega,
        "gamma_d": gamma_d,
        "n_bar": n_bar,
        "n_bar_from_gamma": n_bar_gamma,
        "phase_intercept": phase_intercept,
        "phase_fit_r_squared": phase_r2,
        "gamma_fit_r_squared": gamma_r2,
        "usable_time_count": usable_count.astype(float),
        "tone_power_dbm": power_dbm,
    }
    grid_curves = {
        "fringe_contrast": contrast,
        "fringe_phase_shift": phase_shift,
        "fringe_usable": usable.astype(float),
    }
    return result, amp_curves, grid_curves


def _gamma_ratio(delta_omega: np.ndarray, gamma_d: np.ndarray) -> float:
    """Slope of Gamma_d against Delta_omega, fitted through the origin.

    Both vanish with the tone, so this line does go through the origin. Only points with a finite,
    non-zero shift take part.
    """
    usable = np.isfinite(delta_omega) & np.isfinite(gamma_d) & (np.abs(delta_omega) > 0)
    if usable.sum() < 2:
        return _NAN
    x = np.abs(delta_omega[usable])
    y = gamma_d[usable]
    return float(np.sum(x * y) / np.sum(x**2))


def _empty_per_amp(num_amps: int) -> Dict[str, np.ndarray]:
    keys = (
        "delta_omega",
        "gamma_d",
        "n_bar",
        "n_bar_from_gamma",
        "phase_intercept",
        "phase_fit_r_squared",
        "gamma_fit_r_squared",
        "usable_time_count",
        "tone_power_dbm",
    )
    return {key: np.full(num_amps, _NAN) for key in keys}


def _empty_grid(num_taus: int, num_amps: int) -> Dict[str, np.ndarray]:
    keys = ("fringe_contrast", "fringe_phase_shift", "fringe_usable")
    return {key: np.full((num_taus, num_amps), _NAN) for key in keys}


def _failed_fit(kappa_tot_hz: float, tone_frequency: float, qubit, info: Dict, reason: str) -> FitParameters:
    # The critical photon number is derived from stored state rather than from this node's data, so it
    # is still reported for a qubit whose fringe fit failed.
    n_crit = critical_photon_number(qubit)
    lifetime_ns = 1e9 / (2 * np.pi * kappa_tot_hz)
    return FitParameters(
        photons_per_mw=_NAN,
        single_photon_power_dbm=_NAN,
        n_bar_at_operating_amplitude=_NAN,
        operating_power_dbm=_NAN,
        chi_hz=_NAN,
        kappa_tot_hz=kappa_tot_hz,
        chi_over_kappa=_NAN,
        critical_photon_number=n_crit["critical_photon_number"],
        coupling_g_hz=n_crit["coupling_g_hz"],
        anharmonicity_sign_flipped=bool(n_crit["anharmonicity_sign_flipped"]),
        chi_sign_flipped=bool(n_crit["chi_sign_flipped"]),
        # The power at which the critical photon number is reached needs the measured calibration, so
        # unlike the critical photon number itself it does not survive a failed fit.
        critical_power_dbm=_NAN,
        critical_amplitude_scale=_NAN,
        n_bar_over_critical=_NAN,
        gamma_d_over_delta_omega=_NAN,
        expected_gamma_ratio=_NAN,
        gamma_ratio_disagreement=_NAN,
        n_bar_from_gamma_at_operating=_NAN,
        n_bar_gamma_over_phase=_NAN,
        phase_linearity_r_squared=_NAN,
        gamma_linearity_r_squared=_NAN,
        linear_fit_r_squared=_NAN,
        max_phase_step_rad=_NAN,
        min_usable_free_evolution_count=0,
        number_splitting_phase_limit_ns=_NAN,
        num_times_within_linear_regime=0,
        max_tau_over_photon_lifetime=_NAN,
        num_contributing_amplitudes=0,
        fringe_phase_inverted=False,
        tone_frequency_hz=tone_frequency,
        tone_detuning_hz=_NAN,
        detuning_source="unavailable",
        steady_state_pad_ns=float(info.get("pad_ns", _NAN)),
        steady_state_fraction=float(info.get("steady_state_fraction", _NAN)),
        depletion_over_lifetime=float(
            info.get("depletion_over_lifetime", float(qubit.resonator.depletion_time) / lifetime_ns)
        ),
        success=False,
        failure_reason=reason,
    )


_GRID_FIELDS = ("fringe_contrast", "fringe_phase_shift", "fringe_usable")
_AMP_FIELDS = (
    "delta_omega",
    "gamma_d",
    "n_bar",
    "n_bar_from_gamma",
    "phase_intercept",
    "phase_fit_r_squared",
    "gamma_fit_r_squared",
    "usable_time_count",
    "tone_power_dbm",
)
_SCALAR_FIELDS = (
    "photons_per_mw",
    "single_photon_power_dbm",
    "n_bar_at_operating_amplitude",
    "operating_power_dbm",
    "chi_hz",
    "kappa_tot_hz",
    "chi_over_kappa",
    "critical_photon_number",
    "coupling_g_hz",
    "critical_power_dbm",
    "critical_amplitude_scale",
    "n_bar_over_critical",
    "gamma_d_over_delta_omega",
    "expected_gamma_ratio",
    "gamma_ratio_disagreement",
    "n_bar_from_gamma_at_operating",
    "n_bar_gamma_over_phase",
    "phase_linearity_r_squared",
    "gamma_linearity_r_squared",
    "linear_fit_r_squared",
    "max_phase_step_rad",
    "min_usable_free_evolution_count",
    "number_splitting_phase_limit_ns",
    "num_times_within_linear_regime",
    "max_tau_over_photon_lifetime",
    "num_contributing_amplitudes",
    "tone_frequency_hz",
    "tone_detuning_hz",
    "steady_state_pad_ns",
    "steady_state_fraction",
    "depletion_over_lifetime",
)


def _attach_fit_to_dataset(ds, fit_results, per_amp, per_grid) -> xr.Dataset:
    """Add the fringe grid, the per-amplitude fits and the per-qubit scalars to the dataset."""
    names = [str(name) for name in ds.qubit.values]
    ds_fit = ds.copy()

    for field in _GRID_FIELDS:
        ds_fit[field] = xr.DataArray(
            np.array([per_grid[name][field] for name in names]),
            coords={"qubit": ds.qubit, "tau": ds.tau, "amp_scale": ds.amp_scale},
            dims=["qubit", "tau", "amp_scale"],
        )
    for field in _AMP_FIELDS:
        ds_fit[field] = xr.DataArray(
            np.array([per_amp[name][field] for name in names]),
            coords={"qubit": ds.qubit, "amp_scale": ds.amp_scale},
            dims=["qubit", "amp_scale"],
        )
    ds_fit["n_bar"].attrs = {"long_name": "photon number from the Stark phase", "units": "photons"}
    ds_fit["n_bar_from_gamma"].attrs = {"long_name": "photon number from the dephasing", "units": "photons"}
    ds_fit["tone_power_dbm"].attrs = {"long_name": "Stark tone power", "units": "dBm"}
    ds_fit["gamma_d"].attrs = {"long_name": "induced dephasing", "units": "1/s"}
    ds_fit["delta_omega"].attrs = {"long_name": "Stark shift", "units": "rad/s"}

    for field in _SCALAR_FIELDS:
        ds_fit[field] = xr.DataArray(
            [float(getattr(fit_results[name], field)) for name in names], coords={"qubit": ds.qubit}, dims="qubit"
        )
    for field in ("success", "fringe_phase_inverted", "anharmonicity_sign_flipped", "chi_sign_flipped"):
        ds_fit[field] = xr.DataArray(
            [bool(getattr(fit_results[name], field)) for name in names], coords={"qubit": ds.qubit}, dims="qubit"
        )
    return ds_fit


def _critical_photon_line(result: Dict) -> str:
    """The critical photon number line, or a note saying why there is none."""
    if not np.isfinite(result["critical_photon_number"]):
        return (
            "\tno critical photon number: it needs the stored chi, f_01 and anharmonicity, "
            "and at least one of them is missing"
        )
    lines = [
        f"\tcritical photon number: {result['critical_photon_number']:.1f} "
        f"(implied g = {1e-6 * result['coupling_g_hz']:.1f} MHz)"
    ]
    if np.isfinite(result["n_bar_over_critical"]):
        lines[0] += f", so the readout runs at {100 * result['n_bar_over_critical']:.0f}% of it"
    if np.isfinite(result["critical_power_dbm"]):
        margin_db = result["critical_power_dbm"] - result["operating_power_dbm"]
        lines.append(
            f"\tit is reached at {result['critical_power_dbm']:.1f} dBm, which is amplitude scale "
            f"{result['critical_amplitude_scale']:.2f} and {margin_db:+.1f} dB from the operating power"
        )
    return "\n".join(lines)


def log_fitted_results(fit_results: Dict, log_callable=None) -> None:
    """Log the photon calibration, the linearity that makes it falsifiable, and the cross-checks."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for name, result in fit_results.items():
        result = result if isinstance(result, dict) else result.__dict__
        if not result["success"]:
            log_callable(f"Results for qubit {name}: FAIL! {result['failure_reason']}")
            continue
        detuning = result["tone_detuning_hz"]
        detuning_text = (
            f"{1e-6 * detuning:+.3f} MHz from the midpoint of the dressed pair ({result['detuning_source']})"
            if np.isfinite(detuning)
            else f"detuning unknown ({result['detuning_source']}), so the expected ratio assumes zero"
        )
        lines = [
            f"Results for qubit {name}: SUCCESS!",
            f"\tphotons per mW: {result['photons_per_mw']:.4g} | "
            f"power giving n̄ = 1: {result['single_photon_power_dbm']:.1f} dBm",
            f"\tn̄ at the operating readout amplitude ({result['operating_power_dbm']:.1f} dBm): "
            f"{result['n_bar_at_operating_amplitude']:.2f} from the phase, "
            f"{result['n_bar_from_gamma_at_operating']:.2f} from the dephasing "
            f"(ratio {result['n_bar_gamma_over_phase']:.2f})",
            f"\tphase against free evolution time: R² {result['phase_linearity_r_squared']:.3f} over "
            f"{result['num_contributing_amplitudes']} amplitudes, at least "
            f"{result['min_usable_free_evolution_count']:.0f} times surviving each, reaching "
            f"{result['max_tau_over_photon_lifetime']:.1f} photon lifetimes | largest step between "
            f"times {result['max_phase_step_rad'] / np.pi:.2f} π",
            f"\tphoton-number-splitting limit 2χτ: times up to "
            f"{result['number_splitting_phase_limit_ns']:.0f} ns are in the linear regime, "
            f"{result['num_times_within_linear_regime']:.0f} of the swept times qualify",
            f"\tΓ_d / Δω: {result['gamma_d_over_delta_omega']:.3f} against an expected "
            f"{result['expected_gamma_ratio']:.3f} ({100 * result['gamma_ratio_disagreement']:.0f}% apart)",
            _critical_photon_line(result),
            f"\ttone at {1e-9 * result['tone_frequency_hz']:.6f} GHz, {detuning_text}",
            f"\t|χ|/κ = {result['chi_over_kappa']:.3f} | steady-state pad "
            f"{result['steady_state_pad_ns']:.0f} ns reaching {100 * result['steady_state_fraction']:.0f}% "
            f"of steady state | depletion {result['depletion_over_lifetime']:.1f} photon lifetimes",
            f"\tR² of n̄ against power: {result['linear_fit_r_squared']:.3f}",
        ]
        if result["anharmonicity_sign_flipped"]:
            lines.append(
                "\tthe stored anharmonicity was positive and was taken as negative; a transmon "
                "is negatively anharmonic and the populate script should be storing it signed"
            )
        if result["chi_sign_flipped"]:
            lines.append(
                "\tafter the anharmonicity sign was settled, χ still implied a negative g², so its sign was "
                "flipped to get a positive critical photon number"
            )
        if result["fringe_phase_inverted"]:
            lines.append(
                "\tthe fringe phase ran opposite to the assumed sign convention and was flipped so that "
                "n̄ comes out positive"
            )
        log_callable("\n".join(lines))
