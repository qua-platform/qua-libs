import logging
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from qualibrate.core import QualibrationNode

logger = logging.getLogger(__name__)


@dataclass
class LorentzianPeak:
    """Parameters of a single fitted Lorentzian."""

    amplitude: float
    center: float
    hwhm: float

    @property
    def fwhm(self) -> float:
        return 2.0 * abs(self.hwhm)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude / (1.0 + ((x - self.center) / self.hwhm) ** 2)


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy parity-diff fit parameters for a single qubit."""

    frequency: float
    relative_freq: float
    fwhm: float
    success: bool
    num_peaks: int = 1
    readout_qubit_frequency: Optional[float] = None
    readout_qubit_relative_freq: Optional[float] = None
    peaks: List[dict] = field(default_factory=list)


def _lorentzian_model(x: np.ndarray, offset: float, *peak_params) -> np.ndarray:
    """Evaluate a multi-Lorentzian model: offset + sum of Lorentzians.

    Each peak is parameterised by (amplitude, center, hwhm).
    Amplitude can be positive (peak) or negative (dip).
    """
    y = np.full_like(x, offset, dtype=float)
    for i in range(0, len(peak_params), 3):
        amp, cen, hwhm = peak_params[i], peak_params[i + 1], peak_params[i + 2]
        y = y + amp / (1.0 + ((x - cen) / hwhm) ** 2)
    return y


def _bic(rss: float, n: int, k: int) -> float:
    """Bayesian Information Criterion (lower is better).

    BIC = n * ln(RSS/n) + k * ln(n)
    """
    return n * np.log(rss / n) + k * np.log(n)


def _guess_peaks(x: np.ndarray, y: np.ndarray, n: int):
    """Return up to *n* initial guesses as (amplitude, center, hwhm) tuples.

    Detects both peaks and dips by running ``find_peaks`` on the signal
    and its negation, then returns the most prominent candidates.
    """
    dx = abs(x[1] - x[0]) if len(x) > 1 else 1.0
    baseline = float(np.median(y))
    candidates = []

    for sign_label, data in [("peak", y - baseline), ("dip", -(y - baseline))]:
        idxs, props = find_peaks(data, prominence=0)
        for j, idx in enumerate(idxs):
            amp_sign = 1.0 if sign_label == "peak" else -1.0
            w = peak_widths(data, [idx], rel_height=0.5)[0][0] * dx
            candidates.append((
                amp_sign * data[idx],
                x[idx],
                max(w / 2.0, dx),
                props["prominences"][j],
            ))

    candidates.sort(key=lambda c: c[3], reverse=True)
    return [(a, c, h) for a, c, h, _ in candidates[:n]]


def _fit_lorentzians(
    x: np.ndarray, y: np.ndarray, n_peaks: int
) -> Tuple[np.ndarray, float]:
    """Fit *n_peaks* Lorentzians + offset using ``curve_fit`` with
    initial guesses from ``find_peaks``.

    Returns (best_params, rss).
    """
    guesses = _guess_peaks(x, y, n_peaks)
    baseline = float(np.median(y))

    p0 = [baseline]
    for amp, cen, hwhm in guesses:
        p0.extend([amp, cen, hwhm])
    while len(p0) < 1 + 3 * n_peaks:
        p0.extend([0.0, float(np.mean(x)), float(np.ptp(x)) / 4.0])

    x_range = float(np.ptp(x))
    y_range = max(float(np.ptp(y)), 1e-10)  # guard against flat signal
    dx = abs(x[1] - x[0]) if len(x) > 1 else 1.0
    lo = [float(np.min(y)) - y_range]
    hi = [float(np.max(y)) + y_range]
    for _ in range(n_peaks):
        lo.extend([-3.0 * y_range, float(x.min()) - 0.1 * x_range, dx])
        hi.extend([3.0 * y_range, float(x.max()) + 0.1 * x_range, max(x_range, 2 * dx)])

    def model(xv, *params):
        return _lorentzian_model(xv, params[0], *params[1:])

    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=10000)
    except RuntimeError:
        popt = np.array(p0)

    rss = float(np.sum((y - model(x, *popt)) ** 2))
    return popt, rss


def _select_model(
    x: np.ndarray, y: np.ndarray,
    min_secondary_amp_ratio: float = 0.5,
) -> Tuple[int, np.ndarray, float]:
    """Fit 1-peak and 2-peak models and select by BIC.

    The 2-peak model is only accepted when BIC prefers it *and* the
    smaller peak has at least ``min_secondary_amp_ratio`` times the
    amplitude of the larger peak.

    Returns (n_peaks, params, rss).
    """
    n = len(x)

    params_1, rss_1 = _fit_lorentzians(x, y, 1)
    bic_1 = _bic(rss_1, n, k=4)

    params_2, rss_2 = _fit_lorentzians(x, y, 2)
    bic_2 = _bic(rss_2, n, k=7)

    logger.info("BIC 1-peak: %.2f, BIC 2-peak: %.2f", bic_1, bic_2)

    if bic_2 < bic_1:
        _, peaks = _parse_peaks(params_2, 2)
        amps = sorted([abs(p.amplitude) for p in peaks], reverse=True)
        if amps[1] >= min_secondary_amp_ratio * amps[0]:
            return 2, params_2, rss_2

    return 1, params_1, rss_1


def _parse_peaks(params: np.ndarray, n_peaks: int) -> Tuple[float, list]:
    """Extract (offset, [LorentzianPeak, ...]) from flat parameter array."""
    offset = params[0]
    peaks = []
    for i in range(n_peaks):
        idx = 1 + 3 * i
        peaks.append(LorentzianPeak(
            amplitude=params[idx],
            center=params[idx + 1],
            hwhm=params[idx + 2],
        ))
    return offset, peaks


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Logs the node-specific fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        r = fit_results[q]
        status = "SUCCESS" if r["success"] else "FAIL"
        s = (
            f"Results for qubit {q}:  {status}!\n"
            f"\tQubit frequency: {1e-9 * r['frequency']:.3f} GHz | "
            f"FWHM: {1e-3 * r['fwhm']:.1f} kHz | "
            f"Peaks found: {r['num_peaks']}"
        )
        if r.get("readout_qubit_frequency") is not None:
            s += f"\n\tReadout qubit frequency: {1e-9 * r['readout_qubit_frequency']:.3f} GHz"
        log_callable(s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Restructure the per-qubit pdiff variables into a single DataArray with a qubit dimension."""
    qubits = node.namespace["qubits"]
    qubit_names = [q.name for q in qubits]

    pdiff_vars = sorted([v for v in ds.data_vars if v.startswith("pdiff_")])

    first = ds[pdiff_vars[0]]
    if "qubit" in first.dims:
        pdiff = first.assign_coords(qubit=qubit_names)
    else:
        pdiff = xr.DataArray(
            np.array([ds[v].values for v in pdiff_vars]),
            dims=["qubit", "detuning"],
            coords={"qubit": qubit_names, "detuning": ds.detuning},
        )
    ds = ds.assign({"pdiff": pdiff})

    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in qubits])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit the qubit Larmor frequency for each qubit using Lorentzian model selection.

    Fits 1 and 2 Lorentzian models via differential evolution and selects the
    better fit using the Bayesian Information Criterion (BIC).

    For a single-peak result the peak position is the qubit frequency.
    For a two-peak result the peak closest to the sweep centre (detuning = 0)
    is assigned to the qubit, and the other to its preferred readout qubit.
    """
    qubits = node.namespace["qubits"]
    analysis_signal = node.parameters.analysis_signal
    qubit_names = [q.name for q in qubits]

    arrays = []
    for qname in qubit_names:
        var = f"{analysis_signal}_{qname}"
        if var not in ds.data_vars:
            raise KeyError(
                f"Expected variable {var!r} not found in dataset. "
                "Did you call process_parity_streams before fit_raw_data?"
            )
        arrays.append(ds[var].values)

    pdiff = xr.DataArray(
        np.array(arrays),
        dims=["qubit", "detuning"],
        coords={"qubit": qubit_names, "detuning": ds.detuning},
    )

    ds_fit = ds.assign({"pdiff": pdiff})

    detunings = ds.detuning.values.astype(float)

    rf_freqs = np.array([q.xy.RF_frequency for q in qubits])
    full_freq = detunings[np.newaxis, :] + rf_freqs[:, np.newaxis]
    ds_fit = ds_fit.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds_fit.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    fit_results: dict[str, FitParameters] = {}
    fit_curve_data = np.empty((len(qubit_names), len(detunings)), dtype=float)
    positions = []
    widths = []

    for qi, (qname, qubit) in enumerate(zip(qubit_names, qubits)):
        y = pdiff.sel(qubit=qname).values.astype(float)

        n_peaks, params, rss = _select_model(detunings, y)
        offset, peaks = _parse_peaks(params, n_peaks)

        fit_curve_data[qi] = _lorentzian_model(detunings, offset, *params[1:])

        if n_peaks == 1:
            primary = peaks[0]
            readout_peak = None
        else:
            sorted_by_dist = sorted(peaks, key=lambda p: abs(p.center))
            primary = sorted_by_dist[0]
            readout_peak = sorted_by_dist[1]

        qubit_detuning = primary.center
        qubit_freq = qubit_detuning + qubit.xy.RF_frequency

        readout_freq = None
        readout_rel = None
        if readout_peak is not None:
            readout_rel = readout_peak.center
            readout_freq = readout_peak.center + qubit.xy.RF_frequency

        freq_span_hz = node.parameters.frequency_span_in_mhz * 1e6
        success = bool(abs(qubit_detuning) < freq_span_hz / 2.0)

        positions.append(qubit_detuning)
        widths.append(primary.fwhm)

        peak_dicts = [
            {"amplitude": p.amplitude, "center": p.center, "hwhm": p.hwhm}
            for p in peaks
        ]

        fit_results[qname] = FitParameters(
            frequency=qubit_freq,
            relative_freq=qubit_detuning,
            fwhm=primary.fwhm,
            success=success,
            num_peaks=n_peaks,
            readout_qubit_frequency=readout_freq,
            readout_qubit_relative_freq=readout_rel,
            peaks=peak_dicts,
        )

    ds_fit = ds_fit.assign({
        "fit_curve": (["qubit", "detuning"], fit_curve_data),
        "position": ("qubit", np.array(positions)),
        "width": ("qubit", np.array(widths)),
    })
    ds_fit.attrs = {"long_name": "frequency", "units": "Hz"}

    return ds_fit, fit_results
