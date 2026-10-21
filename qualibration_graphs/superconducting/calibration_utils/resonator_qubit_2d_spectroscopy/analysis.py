import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.data import convert_IQ_to_V

__all__ = [
    "FitParameters",
    "StateFitParameters",
    "stark_shift_lorentzian",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
]


def stark_shift_lorentzian(f, depth, center, kappa, offset):
    """
    The ac Stark shifted qubit frequency as a function of the resonator drive frequency.

    This is Eq. (7) of https://link.aps.org/doi/10.1103/PhysRevApplied.23.024055,

        w*_d,q = w_q,0 + 2 * chi * kappa * |A|^2 / ((w_d,r - (w_r,m -/+ chi))^2 + (kappa / 2)^2),

    rewritten in terms of the quantities that are actually fitted. The upper sign is for the qubit prepared in
    |0> and the lower sign for |1>, so the two curves are a pair of Lorentzians with means w_r,m -/+ chi, depths
    (2 * chi)(4 |A|^2 / kappa) and widths kappa.

    Parameters
    ----------
    f
        The resonator drive frequency.
    depth
        The ac Stark shift on resonance, (2 * chi)(4 |A|^2 / kappa) = 2 * chi * n_bar_max. It carries the sign of
        chi, so it is negative for a transmon below its resonator.
    center
        The mean of the Lorentzian, w_r,m - chi when the qubit is prepared in "g" and w_r,m + chi when it is
        prepared in "e".
    kappa
        The resonator linewidth (FWHM), in the same units as `f`.
    offset
        The unshifted qubit frequency w_q,0, as a detuning, nominally zero.

    Returns
    -------
    The ac Stark shifted qubit detuning at the drive frequency `f`.
    """
    return offset + depth / (1 + ((f - center) / (0.5 * kappa)) ** 2)


@dataclass
class StateFitParameters:
    """The ac Stark shift Lorentzian fitted for one qubit prepared in one state."""

    success: bool
    initial_state: str
    center: float
    center_error: float
    resonator_frequency: float
    kappa: float
    kappa_error: float
    depth: float
    depth_error: float
    clipped_fraction: float
    relative_residual: float
    points_per_linewidth: float
    n_bar_max: Optional[float] = None


@dataclass
class FitParameters:
    """The dispersive shift, resonator linewidth and drive power of one qubit, from the "g" and "e" curves."""

    success: bool
    chi: Optional[float]
    chi_error: Optional[float]
    kappa: float
    kappa_error: float
    resonator_frequency: float
    photon_number: Optional[float]
    drive_amplitude_squared: Optional[float]
    states: Dict[str, dict] = field(default_factory=dict)
    n_bar: Optional[float] = None


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs chi, kappa and the drive power for all qubits, followed by the per-state Lorentzians they came from.

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results:
        r = fit_results[q]
        s = f"Results for qubit {q}: " + (" SUCCESS!\n" if r["success"] else " FAIL!\n")
        if r["chi"] is not None:
            s += f"\tchi: {1e-6 * r['chi']:.3f} +/- {1e-6 * r['chi_error']:.3f} MHz | "
        else:
            s += "\tchi: not available, both 'g' and 'e' are needed | "
        s += f"kappa: {1e-6 * r['kappa']:.3f} +/- {1e-6 * r['kappa_error']:.3f} MHz | "
        s += f"resonator frequency: {1e-9 * r['resonator_frequency']:.6f} GHz\n"
        if r["photon_number"] is not None:
            s += f"\tPeak photon number: {r['photon_number']:.1f} | " f"|A|^2: {r['drive_amplitude_squared']:.3e} Hz\n"
        for state, sf in r["states"].items():
            s += (
                f"\t  |{state}>: mean {1e-9 * sf['resonator_frequency']:.6f} GHz, width "
                f"{1e-6 * sf['kappa']:.3f} MHz, depth {1e-6 * sf['depth']:.2f} MHz"
                + ("" if sf["success"] else "  (FAILED)")
                + "\n"
            )
        undersampled = {
            state: sf["points_per_linewidth"] for state, sf in r["states"].items() if sf["points_per_linewidth"] < 5
        }
        if undersampled:
            worst = min(undersampled.values())
            s += (
                f"\tWARNING: only {worst:.1f} drive frequencies fall across the fitted linewidth, so kappa is set "
                f"by the sweep step rather than by the resonator and most points carry no information. Narrow "
                f"'resonator_frequency_span_in_mhz' to a few times kappa and shrink "
                f"'resonator_frequency_step_in_mhz' to about kappa/10, which costs nothing if the point count is "
                f"kept the same.\n"
            )
        overdriven = [
            state
            for state, sf in r["states"].items()
            if sf["clipped_fraction"] > 0.05 or sf["relative_residual"] > 0.25
        ]
        if overdriven:
            s += (
                f"\tWARNING: the resonator drive looks too strong for |{'>, |'.join(overdriven)}>. The qubit left "
                f"the probe window or the Lorentzian does not describe the data, so kappa and the depth are not "
                f"meaningful. Lower 'resonator_drive_amplitude_factor' until the ac Stark shift on resonance is "
                f"about -20 MHz, which keeps it inside the window and away from the power at which the transmon "
                f"stops behaving dispersively.\n"
            )
        log_callable(s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, adding the RF frequencies of both
    swept axes as coordinates and building the 'flip_signal' used by the analysis.

    'flip_signal' is the probability that the final qubit state differs from the prepared one, which is the
    quantity plotted in Fig. 2(b) and 2(c) of https://link.aps.org/doi/10.1103/PhysRevApplied.23.024055. Without
    state discrimination it is replaced by the rotated 'I' quadrature, oriented so that driving the qubit gives a
    positive peak.
    """
    qubits = node.namespace["qubits"]

    if node.parameters.use_state_discrimination:
        # The averaged 'state' is the probability of measuring the qubit in |e>, so the probability of having
        # flipped is that for a qubit prepared in 'g' and its complement for a qubit prepared in 'e'
        prepared_in_e = ds.initial_state == "e"
        flip_signal = xr.where(prepared_in_e, 1 - ds.state, ds.state)
        flip_signal.attrs = {"long_name": "flip probability", "units": ""}
    else:
        ds = convert_IQ_to_V(ds, qubits)
        ds = ds.assign({"IQ_abs": np.sqrt(ds.I**2 + ds.Q**2)})
        flip_signal = _rotate_to_flip_axis(ds)
        flip_signal.attrs = {"long_name": "rotated 'I' quadrature", "units": "V"}

    ds = ds.assign({"flip_signal": flip_signal})

    # Add the absolute RF frequency of both swept axes as coordinates of the raw dataset
    qubit_RF = np.array([ds.qubit_detuning + q.xy.RF_frequency for q in qubits])
    ds = ds.assign_coords(qubit_RF_frequency=(["qubit", "qubit_detuning"], qubit_RF))
    ds.qubit_RF_frequency.attrs = {"long_name": "qubit pulse frequency", "units": "Hz"}
    resonator_RF = np.array([ds.resonator_detuning + q.resonator.RF_frequency for q in qubits])
    ds = ds.assign_coords(resonator_RF_frequency=(["qubit", "resonator_detuning"], resonator_RF))
    ds.resonator_RF_frequency.attrs = {"long_name": "resonator drive frequency", "units": "Hz"}
    return ds


def _rotate_to_flip_axis(ds: xr.Dataset) -> xr.DataArray:
    """
    Rotates the IQ plane so that the qubit response lies along the returned quadrature, as a positive peak.

    The rotation angle is taken, for each qubit and each prepared state, from the point of the 2D scan that is
    farthest away from the mean IQ value, which is where the qubit has been driven the most efficiently.
    """
    sweep_dims = ["resonator_detuning", "qubit_detuning"]
    dI = ds.I - ds.I.mean(dim=sweep_dims)
    dQ = ds.Q - ds.Q.mean(dim=sweep_dims)
    # Vectorised 2D argmax: stack both swept axes into a single dimension and index it per qubit and state
    distance = np.sqrt(dI**2 + dQ**2).stack(point=sweep_dims)
    farthest = distance.argmax(dim="point")
    angle = np.arctan2(
        dQ.stack(point=sweep_dims).isel(point=farthest),
        dI.stack(point=sweep_dims).isel(point=farthest),
    )
    # Selecting along a stacked dimension leaves 'point' behind as an object coordinate holding one tuple per
    # element, together with the scalar sweep coordinates it was built from. Those would propagate into the
    # result and make the dataset impossible to write to netCDF, so drop them here.
    angle = angle.drop_vars(["point", *sweep_dims], errors="ignore")
    # Projecting along that angle makes the qubit response a positive peak by construction
    return dI * np.cos(angle) + dQ * np.sin(angle)


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extracts the ac Stark shifted qubit frequency at each resonator drive frequency and for each prepared state,
    fits both curves to Eq. (7) of https://link.aps.org/doi/10.1103/PhysRevApplied.23.024055, and combines them
    into chi, kappa and the incident drive power.

    chi is half the horizontal distance between the two means, kappa is their common width, and the depths then
    give the peak photon number and hence |A|^2 = n_bar_max * kappa / 4.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the processed data.
    node : QualibrationNode
        The node whose parameters and qubits are used for the analysis.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitParameters]]
        The dataset containing the fit results and the fitted parameters for each qubit.
    """
    # Locate the qubit line at each resonator drive frequency: this is the ac Stark shift used as a photon sensor
    peaks = peaks_dips(
        ds.flip_signal,
        dim="qubit_detuning",
        prominence_factor=node.parameters.peak_prominence_factor,
    )
    ds_fit = ds.assign(
        {
            "stark_shift": peaks.position,
            "qubit_linewidth": np.abs(peaks.width),
            "qubit_peak_amplitude": peaks.amplitude,
        }
    )
    ds_fit.stark_shift.attrs = {"long_name": "ac Stark shift", "units": "Hz"}
    ds_fit.qubit_linewidth.attrs = {"long_name": "qubit linewidth", "units": "Hz"}

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _guess_lorentzian(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns an initial guess (depth, center, kappa, offset) for the ac Stark shift Lorentzian."""
    offset = float(np.median(y))
    deviation = y - offset
    farthest = int(np.argmax(np.abs(deviation)))
    depth = float(deviation[farthest])
    center = float(x[farthest])
    # The linewidth is guessed from the extent of the region rising above half of the maximum deviation
    above_half = np.abs(deviation) > 0.5 * abs(depth)
    if np.count_nonzero(above_half) > 1:
        kappa = float(x[above_half].max() - x[above_half].min())
    else:
        kappa = float(x.max() - x.min()) / 10
    return depth, center, max(kappa, float(np.min(np.abs(np.diff(np.sort(x)))))), offset


def _fit_stark_shift(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits the ac Stark shift versus resonator drive frequency, returning the parameters and their errors.

    Most drive frequencies are far off resonance, where the qubit is barely shifted and the peak finder is
    picking the largest fluctuation of a flat column rather than a real line. Those points are outliers rather
    than noisy measurements, so a soft-L1 loss is used to stop a handful of them from dragging the Lorentzian
    around. The scale below which residuals are still treated quadratically is tied to the depth being fitted.
    """
    nan_free = np.isfinite(x) & np.isfinite(y)
    x, y = x[nan_free], y[nan_free]
    if x.size < 5:
        return np.full(4, np.nan), np.full(4, np.nan)

    span = float(x.max() - x.min())
    guess = _guess_lorentzian(x, y)
    f_scale = max(abs(guess[0]) / 10, float(np.std(y)) / 10, np.finfo(float).tiny)
    try:
        popt, pcov = curve_fit(
            stark_shift_lorentzian,
            x,
            y,
            p0=guess,
            loss="soft_l1",
            f_scale=f_scale,
            bounds=(
                [-np.inf, x.min() - span, 0.0, -np.inf],
                [np.inf, x.max() + span, 10 * span, np.inf],
            ),
        )
    except (RuntimeError, ValueError):
        return np.full(4, np.nan), np.full(4, np.nan)
    return popt, np.sqrt(np.diag(pcov))


def _extract_relevant_fit_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fits one Lorentzian per qubit and prepared state, then combines them into chi, kappa and the drive power."""
    qubits = node.namespace["qubits"]
    states = [str(s) for s in ds_fit.initial_state.values]
    drive_detunings = ds_fit.resonator_detuning.values
    qubit_detunings = ds_fit.qubit_detuning.values
    span = float(drive_detunings.max() - drive_detunings.min())
    qubit_step = float(np.abs(np.diff(qubit_detunings)).min())

    values = np.empty((len(qubits), len(states), 4))
    errors = np.empty_like(values)
    curves = np.empty((len(qubits), len(states), len(drive_detunings)))
    for i, q in enumerate(qubits):
        for j, state in enumerate(states):
            shift = ds_fit.stark_shift.sel(qubit=q.name, initial_state=state).values
            values[i, j], errors[i, j] = _fit_stark_shift(drive_detunings, shift)
            curves[i, j] = stark_shift_lorentzian(drive_detunings, *values[i, j])

    ds_fit = ds_fit.assign(
        {
            "fit_results": (["qubit", "initial_state", "fit_vals"], values),
            "fit_errors": (["qubit", "initial_state", "fit_vals"], errors),
            "stark_shift_fitted": (["qubit", "initial_state", "resonator_detuning"], curves),
        }
    )
    ds_fit = ds_fit.assign_coords(fit_vals=["depth", "center", "kappa", "offset"])
    ds_fit.stark_shift_fitted.attrs = {"long_name": "fitted ac Stark shift", "units": "Hz"}

    depth = ds_fit.fit_results.sel(fit_vals="depth")
    center = ds_fit.fit_results.sel(fit_vals="center")
    kappa = ds_fit.fit_results.sel(fit_vals="kappa")

    # Drive the resonator too hard and the ac Stark shift runs past the swept qubit window, while near resonance
    # the transmon is pushed out of the computational subspace altogether. Both clip the Lorentzian, which makes
    # its depth and width meaningless, so flag it rather than reporting a kappa that cannot be trusted.
    at_edge = (ds_fit.stark_shift <= qubit_detunings.min() + qubit_step) | (
        ds_fit.stark_shift >= qubit_detunings.max() - qubit_step
    )
    clipped_fraction = (at_edge | ds_fit.stark_shift.isnull()).sum(dim="resonator_detuning") / ds_fit.sizes[
        "resonator_detuning"
    ]
    ds_fit = ds_fit.assign({"clipped_fraction": clipped_fraction})
    # A saturated band of noise does not clip the window, but it does stop the Lorentzian describing the data
    residual = ds_fit.stark_shift - ds_fit.stark_shift_fitted
    relative_residual = np.sqrt((residual**2).mean(dim="resonator_detuning")) / np.abs(depth)
    ds_fit = ds_fit.assign({"relative_residual": relative_residual})
    # kappa is read off the width of the Lorentzian, so it is only as good as the sampling of that width. A step
    # comparable with kappa leaves a couple of points on the whole feature and the rest on the baseline.
    drive_step = float(np.abs(np.diff(drive_detunings)).min())
    points_per_linewidth = kappa / drive_step
    ds_fit = ds_fit.assign({"points_per_linewidth": points_per_linewidth})

    # A linewidth or a mean falling outside the swept window cannot be trusted, and the ac Stark shift must be
    # resolvable on the qubit frequency axis for the Lorentzian to be meaningful.
    finite = np.isfinite(ds_fit.fit_results).all(dim="fit_vals") & np.isfinite(ds_fit.fit_errors).all(dim="fit_vals")
    extremum = ds_fit.fit_results.sel(fit_vals="offset") + depth
    state_success = (
        finite
        & (kappa > 0)
        & (kappa < span)
        & (center > drive_detunings.min())
        & (center < drive_detunings.max())
        & (np.abs(depth) > qubit_step)
        & (clipped_fraction <= node.parameters.max_clipped_fraction)
        & (relative_residual <= node.parameters.max_relative_residual)
        & (points_per_linewidth >= node.parameters.min_points_per_linewidth)
        & (extremum > qubit_detunings.min())
        & (extremum < qubit_detunings.max())
    )
    ds_fit = ds_fit.assign({"success": state_success})

    fit_results = {}
    for q in qubits:
        per_state = {}
        for state in states:
            sel = ds_fit.sel(qubit=q.name, initial_state=state)
            per_state[state] = asdict(
                StateFitParameters(
                    success=bool(sel.success.values),
                    initial_state=state,
                    center=float(sel.fit_results.sel(fit_vals="center").values),
                    center_error=float(sel.fit_errors.sel(fit_vals="center").values),
                    resonator_frequency=float(sel.fit_results.sel(fit_vals="center").values) + q.resonator.RF_frequency,
                    kappa=float(sel.fit_results.sel(fit_vals="kappa").values),
                    kappa_error=float(sel.fit_errors.sel(fit_vals="kappa").values),
                    depth=float(sel.fit_results.sel(fit_vals="depth").values),
                    depth_error=float(sel.fit_errors.sel(fit_vals="depth").values),
                    clipped_fraction=float(sel.clipped_fraction.values),
                    relative_residual=float(sel.relative_residual.values),
                    points_per_linewidth=float(sel.points_per_linewidth.values),
                )
            )
        fit_results[q.name] = _combine_states(per_state, q)

    # Calculate chi, peak photon number n_bar_max, |A|^2 and photon number profile n_bar(w_d,r)
    # according to Eq. (6) and Eq. (7) of PRApplied 23, 024055:
    #   n_bar(w_d,r) = (4 |A|^2 / kappa) / (1 + 4 * delta^2 / kappa^2)
    #   depth = 2 * chi * n_bar_max  =>  n_bar_max = depth / (2 * chi)
    #   stark_shift = offset + 2 * chi * n_bar(w_d,r)
    chi_vals = np.full(len(qubits), np.nan)
    n_bar_max_vals = np.full((len(qubits), len(states)), np.nan)
    drive_amp_sq_vals = np.full(len(qubits), np.nan)
    n_bar_fitted = np.full((len(qubits), len(states), len(drive_detunings)), np.nan)
    n_bar_data = np.full((len(qubits), len(states), len(drive_detunings)), np.nan)

    for i, q in enumerate(qubits):
        fr = fit_results[q.name]
        chi = fr.chi
        # Fallback to calibrated qubit.chi if not extracted in this run (e.g. if single state run)
        if chi is None or chi == 0:
            if hasattr(q, "chi") and q.chi is not None and q.chi != 0:
                chi = float(q.chi)
        if chi is not None and chi != 0:
            chi_vals[i] = chi
            if fr.drive_amplitude_squared is not None:
                drive_amp_sq_vals[i] = fr.drive_amplitude_squared
            for j, state in enumerate(states):
                depth_val = values[i, j, 0]
                offset_val = values[i, j, 3]
                n_bar_max_val = depth_val / (2 * chi)
                n_bar_max_vals[i, j] = n_bar_max_val
                # Eq. (6) of PRApplied 23, 024055:
                # n_bar = (4 * |A|^2 / kappa) / (1 + 4 * delta^2 / kappa^2)
                #       = n_bar_max / (1 + ((f - center) / (0.5 * kappa)) ** 2)
                # which is identically (stark_shift_fitted - offset) / (2 * chi)
                n_bar_fitted[i, j] = (curves[i, j] - offset_val) / (2 * chi)
                shift_data = ds_fit.stark_shift.sel(qubit=q.name, initial_state=state).values
                n_bar_data[i, j] = (shift_data - offset_val) / (2 * chi)

    ds_fit = ds_fit.assign(
        {
            "chi": (["qubit"], chi_vals),
            "n_bar_max": (["qubit", "initial_state"], n_bar_max_vals),
            "drive_amplitude_squared": (["qubit"], drive_amp_sq_vals),
            "n_bar": (["qubit", "initial_state", "resonator_detuning"], n_bar_data),
            "n_bar_fitted": (["qubit", "initial_state", "resonator_detuning"], n_bar_fitted),
        }
    )
    ds_fit.chi.attrs = {"long_name": "dispersive shift chi", "units": "Hz"}
    ds_fit.n_bar_max.attrs = {"long_name": "peak mean photon number n_bar_max", "units": ""}
    ds_fit.n_bar.attrs = {"long_name": "mean photon number n_bar", "units": ""}
    ds_fit.n_bar_fitted.attrs = {"long_name": "fitted mean photon number n_bar (Eq. 6)", "units": ""}
    ds_fit.drive_amplitude_squared.attrs = {"long_name": "drive power |A|^2", "units": "Hz"}

    return ds_fit, fit_results


def _combine_states(per_state: Dict[str, dict], qubit) -> FitParameters:
    """
    Combines the "g" and "e" Lorentzians into chi, kappa and the incident drive power.

    The two means are w_r,m - chi and w_r,m + chi, so their half-difference is chi and their mean is the bare
    resonator frequency. Both curves share the linewidth kappa, and each depth is 2 * chi * n_bar_max, which
    gives the peak photon number once chi is known and hence |A|^2 = n_bar_max * kappa / 4.
    """
    g, e = per_state.get("g"), per_state.get("e")
    kappas = [s["kappa"] for s in per_state.values() if s["success"]]
    kappa_errors = [s["kappa_error"] for s in per_state.values() if s["success"]]
    all_successful = bool(per_state) and all(s["success"] for s in per_state.values())

    if g is not None and e is not None:
        chi = 0.5 * (e["center"] - g["center"])
        chi_error = 0.5 * float(np.hypot(e["center_error"], g["center_error"]))
        resonator_frequency = 0.5 * (g["resonator_frequency"] + e["resonator_frequency"])
    else:
        only = g if g is not None else e
        chi, chi_error = None, None
        resonator_frequency = only["resonator_frequency"] if only else float("nan")

    # Fallback to calibrated qubit.chi if single state was acquired
    if (chi is None or chi == 0) and hasattr(qubit, "chi") and qubit.chi is not None and qubit.chi != 0:
        chi = float(qubit.chi)

    kappa = float(np.mean(kappas)) if kappas else float("nan")
    kappa_error = (
        float(np.hypot(*kappa_errors) / len(kappa_errors))
        if len(kappa_errors) > 1
        else (kappa_errors[0] if kappa_errors else float("nan"))
    )

    # The depth of each Lorentzian is 2 * chi * n_bar_max, so chi converts it into a photon number
    photon_number, drive_amplitude_squared = None, None
    if chi:
        photons = [s["depth"] / (2 * chi) for s in per_state.values() if s["success"]]
        if photons:
            photon_number = float(np.mean(photons))
            drive_amplitude_squared = photon_number * kappa / 4
        for s in per_state.values():
            s["n_bar_max"] = s["depth"] / (2 * chi)

    return FitParameters(
        success=all_successful and (chi is None or chi != 0),
        chi=chi,
        chi_error=chi_error,
        kappa=kappa,
        kappa_error=kappa_error,
        resonator_frequency=resonator_frequency,
        photon_number=photon_number,
        drive_amplitude_squared=drive_amplitude_squared,
        states=per_state,
        n_bar=photon_number,
    )
