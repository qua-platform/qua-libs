"""Analysis module for the JAZZ-N CZ amplitude calibration.

The protocol measures the stationary qubit's |1> population as a function of the
CZ-pulse amplitude scale, for several echo repetitions N = 4k + 1. Ignoring
decoherence, equation (36) of arXiv:2402.18926v3 predicts:

    P_|1>(amp, N) = (1 - cos((2k+1) * theta_CZ(amp))) / 2,    N = 4k + 1.

Averaging this over all probed odd multipliers m = 2k + 1 collapses the
N-dependence into the central lobe of a sinc-like pattern centred on the
optimal amplitude. Using the Dirichlet identity

    sum_{k=k_min..k_max} cos((2k+1) x) = sin(2 K x) / (2 sin x)

with K the number of N values, and writing x = pi + delta around the
optimum (delta ~ alpha * (amp - amp*)), one finds

    <P_|1>>_N(amp) = 1/2 + (1 / (4 K)) * sin(2 K delta) / sin(delta)
                  ~ 1/2 + (1/2) * sinc(w * (amp - amp*) / pi)        (small delta)

with sinc(y) = sin(pi y) / (pi y). The fit model used here is therefore

    f(amp) = B + A * sinc(w * (amp - amp_0) / pi)

with theoretical (A, B) = (1/2, 1/2). The peak position amp_0 is the
optimal CZ amplitude.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.optimize import curve_fit


@dataclass
class FitResults:
    """JAZZ-N fit results for a single qubit pair."""

    optimal_amplitude: float
    """Absolute optimal CZ amplitude (Volts), i.e. amp_scale_optimal * stored amplitude."""
    optimal_amplitude_scale: float
    """Optimal amplitude scale factor (dimensionless; multiplied with stored amplitude)."""
    success: bool
    """Whether the sinc fit succeeded (with the parabolic fallback path counted as failed)."""
    fit_method: str = "sinc"
    """Either 'sinc' (primary) or 'parabolic' (fallback)."""
    sinc_params: Dict[str, float] = field(default_factory=dict)
    """Fitted sinc parameters {'A','B','amp_0','w'} when ``fit_method == 'sinc'``."""


def coerce_to_4k_plus_1(n: int) -> int:
    """Coerce an integer to the nearest value of the form 4k + 1 with k >= 0."""
    if n < 1:
        return 1
    k = round((n - 1) / 4)
    return 4 * max(0, k) + 1


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """Log the JAZZ-N fit results per qubit pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        header = f"Results for qubit pair {qp_name}: " + ("SUCCESS!\n" if fit_result.success else "FAIL!\n")
        body = (
            f"\tOptimal CZ amplitude: {fit_result.optimal_amplitude:.6f} V "
            f"(scale {fit_result.optimal_amplitude_scale:.6f}; method={fit_result.fit_method})"
        )
        log_callable(header + body)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Augment the raw dataset with an absolute-amplitude coordinate per qubit pair."""
    qubit_pairs = node.namespace["qubit_pairs"]
    operation = node.parameters.operation

    def abs_amp(qp, amp):
        return amp * qp.macros[operation].flux_pulse_qubit.amplitude

    ds = ds.assign_coords(
        {"amp_full": (["qubit_pair", "amp"], np.array([abs_amp(qp, ds.amp.values) for qp in qubit_pairs]))}
    )
    return ds


def _sinc_model(amp: np.ndarray, A: float, B: float, amp_0: float, w: float) -> np.ndarray:
    """Sinc model used to localise the central peak of <P>_N(amp).

    f(amp) = B + A * sin(w * (amp - amp_0)) / (w * (amp - amp_0))
           = B + A * np.sinc(w * (amp - amp_0) / pi).
    """
    return B + A * np.sinc(w * (amp - amp_0) / np.pi)


def _parabolic_refine(y: np.ndarray, i: int) -> float:
    """Three-point parabolic refinement around an extremum index, returning fractional index."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y1, y2, y3 = y[i - 1], y[i], y[i + 1]
    denom = y1 - 2.0 * y2 + y3
    if denom == 0:
        return float(i)
    delta = 0.5 * (y1 - y3) / denom
    return float(i) + float(np.clip(delta, -0.5, 0.5))


def _argmax_with_refine(amp_values: np.ndarray, y: np.ndarray) -> float:
    """Argmax of ``y`` over ``amp_values`` with 3-point parabolic refinement."""
    i = int(np.argmax(y))
    i_star = _parabolic_refine(y, i)
    return float(np.interp(i_star, np.arange(len(amp_values)), amp_values))


def _estimate_fwhm_around(amp_values: np.ndarray, y: np.ndarray, i_max: int) -> float:
    """Estimate the full-width at half-maximum of the central peak around ``i_max``.

    Uses ``half = y_max - (y_max - y_min) / 2`` as the half-maximum reference
    (i.e. half-prominence above the global minimum of ``y``).
    """
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return float("nan")
    y_max = float(y[i_max])
    y_min = float(np.min(finite))
    half = y_max - (y_max - y_min) / 2.0
    left = i_max
    while left > 0 and y[left] > half:
        left -= 1
    right = i_max
    while right < len(y) - 1 and y[right] > half:
        right += 1
    fwhm = float(amp_values[right] - amp_values[left])
    if fwhm > 0:
        return fwhm
    return float(amp_values[-1] - amp_values[0]) / 4.0


def _fit_one_pair(
    amp_values: np.ndarray, n_values: np.ndarray, p_curve: np.ndarray
) -> Tuple[float, bool, str, Dict[str, float], np.ndarray, np.ndarray]:
    """Average ``p_curve`` over N and fit a sinc model to the central peak.

    Parameters
    ----------
    amp_values : (n_amp,) amplitude-scale values (centred at 1.0).
    n_values : (n_N,) echo counts N (sorted ascending; not used by the fit but
        retained to keep the call signature consistent with the previous version).
    p_curve : (n_N, n_amp) stationary P_|1> values.

    Returns
    -------
    amp_seed : optimal amplitude scale (dimensionless).
    success : True if the sinc fit converged inside the swept window.
    method : 'sinc' if the primary fit succeeded, else 'parabolic'.
    params : fitted sinc parameters (empty dict if parabolic fallback was used).
    p_avg : (n_amp,) the averaged-over-N curve.
    fit_curve : (n_amp,) the fitted-sinc evaluated at ``amp_values``; NaN if
        parabolic fallback was used.
    """
    if p_curve.ndim != 2 or p_curve.shape != (len(n_values), len(amp_values)):
        return float("nan"), False, "none", {}, np.full_like(amp_values, np.nan), np.full_like(amp_values, np.nan)

    p_avg = np.nanmean(p_curve, axis=0)
    if not np.any(np.isfinite(p_avg)):
        return float("nan"), False, "none", {}, p_avg, np.full_like(amp_values, np.nan)

    p_avg_finite = np.where(np.isfinite(p_avg), p_avg, -np.inf)
    i_max = int(np.argmax(p_avg_finite))

    # Initial guesses for the sinc fit.
    A_init = float(np.nanmax(p_avg) - np.nanmedian(p_avg))
    if not np.isfinite(A_init) or A_init <= 0:
        A_init = 0.5
    B_init = float(np.nanmedian(p_avg))
    if not np.isfinite(B_init):
        B_init = 0.5
    amp_seed_init = float(amp_values[i_max])
    fwhm = _estimate_fwhm_around(amp_values, p_avg, i_max)
    # For f(amp) = B + A * sinc(w (amp - amp_0) / pi), sin(x)/x = 1/2 at x ~ 1.8955,
    # so FWHM ~ 2 * 1.8955 / w ~ 3.79 / w, hence w ~ 3.79 / FWHM.
    if np.isfinite(fwhm) and fwhm > 0:
        w_init = 3.79 / fwhm
    else:
        w_init = 3.79 / max(float(amp_values[-1] - amp_values[0]) / 4.0, 1e-9)

    amp_min, amp_max = float(np.min(amp_values)), float(np.max(amp_values))

    try:
        popt, _ = curve_fit(
            _sinc_model,
            amp_values,
            p_avg,
            p0=[A_init, B_init, amp_seed_init, w_init],
            bounds=(
                [0.0, -0.5, amp_min, w_init / 50.0],
                [2.0, 1.5, amp_max, w_init * 50.0],
            ),
            maxfev=10000,
        )
        A_fit, B_fit, amp_0_fit, w_fit = map(float, popt)
        in_range = amp_min <= amp_0_fit <= amp_max
        if not in_range:
            raise RuntimeError(f"Fitted amp_0 = {amp_0_fit} outside swept window.")
        fit_curve = _sinc_model(amp_values, A_fit, B_fit, amp_0_fit, w_fit)
        return (
            amp_0_fit,
            True,
            "sinc",
            {"A": A_fit, "B": B_fit, "amp_0": amp_0_fit, "w": w_fit},
            p_avg,
            fit_curve,
        )
    except Exception:  # pylint: disable=broad-except
        amp_seed_refined = _argmax_with_refine(amp_values, p_avg_finite)
        return (
            amp_seed_refined,
            bool(np.isfinite(amp_seed_refined)),
            "parabolic",
            {},
            p_avg,
            np.full_like(amp_values, np.nan),
        )


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Fit the JAZZ-N data per qubit pair.

    The dataset is augmented with three new data variables (per qubit pair, on
    the amp axis): ``p_avg`` (mean over N of the stationary |1> population) and
    ``sinc_fit`` (the fitted sinc model evaluated on the amp grid), plus
    ``optimal_amplitude`` / ``optimal_amplitude_scale`` / ``success`` /
    ``fit_method`` as coordinates.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    operation = node.parameters.operation

    if "state_stationary" not in ds:
        raise RuntimeError("JAZZ-N analysis requires 'state_stationary' in the dataset (state discrimination).")

    amp_values = ds.amp.values
    n_values = ds.N.values

    opt_amps_abs = []
    opt_amps_scale = []
    successes = []
    methods = []
    p_avg_rows = []
    fit_rows = []
    qp_names = ds.qubit_pair.values
    fit_results: Dict[str, FitResults] = {}

    for qp_name in qp_names:
        qp = next(qp for qp in qubit_pairs if qp.name == qp_name)
        p = ds.state_stationary.sel(qubit_pair=qp_name).transpose("N", "amp").values
        amp_scale, success, method, params, p_avg, fit_curve = _fit_one_pair(
            amp_values, np.asarray(n_values), np.asarray(p)
        )
        stored_amp = qp.macros[operation].flux_pulse_qubit.amplitude
        amp_abs = amp_scale * stored_amp if np.isfinite(amp_scale) else np.nan

        opt_amps_abs.append(amp_abs)
        opt_amps_scale.append(amp_scale)
        successes.append(success)
        methods.append(method)
        p_avg_rows.append(p_avg)
        fit_rows.append(fit_curve)
        fit_results[str(qp_name)] = FitResults(
            optimal_amplitude=float(amp_abs),
            optimal_amplitude_scale=float(amp_scale),
            success=bool(success),
            fit_method=str(method),
            sinc_params=params,
        )

    ds_fit = ds.assign(
        {
            "p_avg": xr.DataArray(
                np.asarray(p_avg_rows),
                dims=("qubit_pair", "amp"),
                coords={"qubit_pair": qp_names, "amp": amp_values},
                name="p_avg",
            ),
            "sinc_fit": xr.DataArray(
                np.asarray(fit_rows),
                dims=("qubit_pair", "amp"),
                coords={"qubit_pair": qp_names, "amp": amp_values},
                name="sinc_fit",
            ),
        }
    )
    ds_fit = ds_fit.assign_coords(
        {
            "optimal_amplitude": ("qubit_pair", np.array(opt_amps_abs, dtype=float)),
            "optimal_amplitude_scale": ("qubit_pair", np.array(opt_amps_scale, dtype=float)),
            "success": ("qubit_pair", np.array(successes, dtype=bool)),
            "fit_method": ("qubit_pair", np.array(methods, dtype=object)),
        }
    )
    ds_fit.optimal_amplitude.attrs = {"long_name": "optimal CZ amplitude", "units": "V"}
    ds_fit.optimal_amplitude_scale.attrs = {"long_name": "optimal CZ amplitude scale", "units": "a.u."}
    ds_fit.p_avg.attrs = {"long_name": "<P_|1>>_N", "units": "a.u."}
    ds_fit.sinc_fit.attrs = {"long_name": "sinc fit", "units": "a.u."}
    return ds_fit, fit_results
