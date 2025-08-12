from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from typing import List, Tuple, Sequence
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from functools import reduce
from numpy.polynomial import Polynomial as P

@dataclass
class PiFluxParameters:
    fit_successful: bool
    best_fractions: List[float]
    best_components: List[Tuple[float, float]]
    best_a_dc: float
    best_rms: float


def log_fitted_results(fit_results: Dict[str, PiFluxParameters], log_callable=print) -> None:
    for qb, res in fit_results.items():
        if res.fit_successful:
            log_callable(
                f"{qb}: SUCCESS, a_dc={res.best_a_dc:.3e}, rms={res.best_rms:.3e}, comps={res.best_components}"
            )
        else:
            log_callable(f"{qb}: FAILED")


def process_raw_dataset(ds: xr.Dataset, node) -> xr.Dataset:
    """Convert IQ to V, add amplitude/phase, add freq_full if detuning present."""
    if "I" in ds or "Q" in ds:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])  # type: ignore
        ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    if "detuning" in ds.coords:
        full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
        ds = ds.assign_coords(full_freq=( ["qubit", "detuning"], full_freq))
        ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset


def fit_gaussian(freqs, states):
    p0 = [
        float(np.max(states) - np.min(states)),
        float(freqs[np.argmax(states)]),
        float((freqs[-1] - freqs[0]) / 10),
        float(np.min(states)),
    ]
    try:
        popt, _ = curve_fit(gaussian, freqs, states, p0=p0)
        return float(popt[1])
    except Exception:
        return float("nan")


def extract_center_freqs_state(ds: xr.Dataset, freqs: np.ndarray) -> xr.DataArray:
    """Extract center frequencies from state discrimination dataset as a DataArray.

    Ensures we operate on the 'state' DataArray (not the whole Dataset) so apply_ufunc
    returns a DataArray, avoiding assignment errors downstream.
    """
    freq_dim = "detuning" if "detuning" in ds.dims else "freq"
    state_da = ds["state"].transpose("qubit", "time", freq_dim)
    center_freqs = xr.apply_ufunc(
        lambda states: fit_gaussian(freqs, states),
        state_da,
        input_core_dims=[[freq_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return center_freqs.rename("center_frequency")


def extract_center_freqs_iq(ds: xr.Dataset, freqs: np.ndarray) -> xr.DataArray:
    iq_data = ds["IQ_abs"] if "IQ_abs" in ds.data_vars else ds.get("I", None)
    if iq_data is None:
        raise ValueError("Dataset is missing IQ_abs and I data variables for IQ analysis")
    stacked = iq_data.transpose("qubit", "time", "detuning")
    center_freqs = xr.apply_ufunc(
        lambda iq_slice: fit_gaussian(freqs, iq_slice),
        stacked,
        input_core_dims=[["detuning"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return center_freqs


def single_exp_decay(t, amp, tau):
    return amp * np.exp(-t / tau)


def sequential_exp_fit(t: np.ndarray, y: np.ndarray, start_fractions: List[float]):
    components: List[Tuple[float, float]] = []
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t_offset = t - t[0]

    # Estimate constant term from tail
    tail_start = max(1, int(0.9 * t.size))
    a_dc = float(np.nanmean(y[tail_start:])) if t.size > 2 else float(y[-1])
    y_res = y - a_dc

    for start_frac in start_fractions:
        start_idx = int(len(t) * start_frac)
        if start_idx >= len(t):
            break
        try:
            p0 = [float(y_res[start_idx]), max(0.1, (t_offset[-1] - t_offset[start_idx]) / 3)]
            bounds = ([-np.inf, 0.1], [np.inf, np.inf])
            popt, _ = curve_fit(single_exp_decay, t_offset[start_idx:], y_res[start_idx:], p0=p0, bounds=bounds)
            amp, tau = float(popt[0]), float(popt[1])
            components.append((amp, tau))
            y_res = y_res - amp * np.exp(-t_offset / tau)
        except Exception:
            break

    return components, a_dc, y_res


def optimize_start_fractions(t: np.ndarray, y: np.ndarray, base_fractions: List[float], bounds_scale: float = 0.5):
    from scipy.optimize import minimize

    def objective(x):
        if not np.all(np.diff(x) < 0):
            return 1e6
        components, _, residual = sequential_exp_fit(t, y, list(x))
        if len(components) != len(base_fractions):
            return 1e6
        return float(np.sqrt(np.nanmean(residual ** 2)))

    bounds = []
    for base in base_fractions:
        bmin = base * (1 - bounds_scale)
        bmax = base * (1 + bounds_scale)
        bounds.append((bmin, bmax))

    result = minimize(objective, x0=np.array(base_fractions, dtype=float), bounds=bounds, method="Nelder-Mead", options={"disp": False, "maxiter": 200})

    if result.success:
        best_fractions = list(result.x)
        components, a_dc, residual = sequential_exp_fit(t, y, best_fractions)
        best_rms = float(np.sqrt(np.nanmean(residual ** 2)))
        return True, best_fractions, components, float(a_dc), best_rms
    else:
        components, a_dc, residual = sequential_exp_fit(t, y, base_fractions)
        best_rms = float(np.sqrt(np.nanmean(residual ** 2)))
        return False, base_fractions, components, float(a_dc), best_rms


def fit_raw_data(ds: xr.Dataset, node) -> tuple[xr.Dataset, Dict[str, PiFluxParameters]]:
    """Compute center_freqs, detuning/freq_full/flux coords, flux_response, and fit cascade."""
    qubits = node.namespace["qubits"]

    # Frequency points array
    dfs = (
        node.namespace.get("sweep_axes", {}).get("detuning").values
        if "sweep_axes" in node.namespace
        else (ds.get("detuning").values if "detuning" in ds.dims else ds.get("freq").values)
    )

    # Ensure detuning naming aligns
    if "detuning" not in ds.dims and "freq" in ds.dims:
        ds = ds.rename({"freq": "detuning"})

    # Center frequencies from either state or IQ
    if node.parameters.use_state_discrimination and "state" in ds.data_vars:
        center_freqs = extract_center_freqs_state(ds, dfs)
    else:
        center_freqs = extract_center_freqs_iq(ds, dfs)

    # Add example-style coords
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "detuning"],
                np.array([dfs + q.xy.RF_frequency + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp ** 2 for q in qubits]),
            ),
            "detuning": (
                ["qubit", "detuning"],
                np.array([dfs + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp ** 2 for q in qubits]),
            ),
            "flux": (
                ["qubit", "detuning"],
                np.array([
                    (
                        np.sqrt(np.maximum(0.0, dfs / q.freq_vs_flux_01_quad_term + node.parameters.flux_amp ** 2))
                        if q.freq_vs_flux_01_quad_term != 0
                        else np.full_like(dfs, np.nan, dtype=float)
                    )
                    for q in qubits
                ]),
            ),
        }
    )

    # Add flux-induced static shift as in example
    times = ds["time"].values
    center_freqs = center_freqs + np.array(
        [q.freq_vs_flux_01_quad_term * node.parameters.flux_amp ** 2 * np.ones_like(times) for q in qubits]
    )

    quad_terms = xr.DataArray([q.freq_vs_flux_01_quad_term for q in qubits], coords={"qubit": [q.name for q in qubits]}, dims=["qubit"])
    flux_response = xr.where(quad_terms != 0, np.sqrt(center_freqs / quad_terms), np.nan)

    ds = ds.copy()
    ds["center_freqs"] = center_freqs
    ds["flux_response"] = flux_response

    fit_results: Dict[str, PiFluxParameters] = {}
    for q in qubits:
        t_data = flux_response.sel(qubit=q.name).time.values
        y_data = flux_response.sel(qubit=q.name).values
        fit_successful, best_fractions, best_components, best_a_dc, best_rms = optimize_start_fractions(
            t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5
        )
        fit_results[q.name] = PiFluxParameters(
            fit_successful=fit_successful,
            best_fractions=best_fractions,
            best_components=best_components,
            best_a_dc=best_a_dc,
            best_rms=best_rms,
        )

    return ds, fit_results

def decompose_exp_sum_to_cascade(A: Sequence, tau: Sequence, A_dc: float=1.,
                             compensate_v34_fpga_scale: bool=True, Ts: float=0.5) -> \
        tuple[np.ndarray, np.ndarray, float]:
    """decompose_exp_sum_to_cascade
    Translate from filters configuration as defined in QUA for version 3.5 (sum of exponents) to the
    definition of version 3.4.1 (cascade of single exponents filters).
    In v3.5, the analog linear distortion H is characterized by step response:
    s_H(t) = (A_dc + sum(A[i] * exp(-t/tau[i]), for i in 0...(N-1)))*u(t)
    In v3.4.1, it is a cascade of single exponent filters, each with step response:
    s_H_i(t) = (1 + A_c[i] * exp(-t/tau_c[i]))*u(t)
    The parameters [(A_c[0], tau_c[0]), ...] are the definitions of the filters (under "exponents")
    in 3.4.1.
    To make the filters equivalent, the 3.4.1 cascade needs to scaled by the parameter scale.
    This scaling can be done by multiplying the FIR coefficients by scale, or by scaling the waverform
    amp accordingly.
    :return A_c, tau_c, scale
    """

    assert A_dc > 0.2, "HPF mode is currently not supported"

    ba_sum = [get_rational_filter_single_exp_cont_time(A_i, tau_i) for A_i, tau_i in zip(A, tau)]
    ba_sum += [([A_dc], [1])]

    b, a = add_rational_terms(ba_sum)

    zeros = np.sort(np.roots(b))
    poles = np.sort(np.roots(a))

    assert np.all(np.isreal(zeros)) and np.all(np.isreal(poles)), \
        "Got complex zeros; this configuration can't be inverted or decomposed to cascade of single pole stages"

    tau_c = -1 / poles
    A_c = poles/zeros - 1

    scale = 1/A_dc

    if compensate_v34_fpga_scale:
        scale *= get_scaling_of_v34_fpga_filter(A_c, tau_c, Ts)

    return A_c, tau_c, scale


def get_scaling_of_v34_fpga_filter(A_c: np.ndarray, tau_c: np.ndarray, Ts) -> float:
    """get_scaling_of_v34_fpga_filter
    Calculate the scaling factor for the V3.4 FPGA filter implementation.
    This scaling is necessary to make the cascade of single exponent filters equivalent to the sum of exponents.
    :param A_c: Amplitudes of the cascade filters
    :param tau_c: Time constants of the cascade filters
    :param Ts: Sampling period
    :return: scale
    """
    return float(np.prod((Ts + 2*tau_c) / (Ts + 2*tau_c*(1+A_c))))


def get_rational_filter_single_exp_cont_time(A: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1, 1/tau])
    b = np.array([A])
    return b, a

# The functions below are used to decompose the sum of exponentials to a cascade of single 
# exponent filters, as implemented in QOP 3.4.    
def add_rational_terms(terms: List[Tuple[np.array, np.array]]) -> Tuple[np.array, np.array]:
    # Convert to Polynomial objects
    rational_terms = [(P(num), P(den)) for num, den in terms]

    # Compute common denominator
    common_den = reduce(lambda acc, t: acc * t[1], rational_terms, P([1]))

    # Adjust numerators to have the common denominator
    adjusted_numerators = []
    for num, den in rational_terms:
        multiplier = common_den // den
        adjusted_numerators.append(num * multiplier)

    # Sum all adjusted numerators
    final_numerator = sum(adjusted_numerators, P([0]))

    # Return as coefficient lists
    return final_numerator.coef, common_den.coef

