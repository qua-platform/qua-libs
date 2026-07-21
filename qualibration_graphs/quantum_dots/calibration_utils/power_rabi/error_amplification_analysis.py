"""Error-amplified power-Rabi analysis: mean-signal resonance finding.

Strategy (adapted from the Ramsey chevron analysis)
----------------------------------------------------
The 2-D data ``signal(amp, n_pulses)`` is analogous to a Ramsey chevron
with ``amp ↔ detuning`` and ``n_pulses ↔ idle_time``.

1. **Optimal-amplitude finding** — The n_pulses-averaged signal vs
   amplitude is fitted to the exact finite-window model using
   ``differential_evolution`` (global optimizer).  The model sums
   over the (even-only) n_pulses grid with a combined exponential +
   Gaussian decay envelope:

   .. math::

       \\langle P \\rangle(a) = bg + \\frac{A}{K}
           \\sum_i \\exp\\!\\bigl(-\\gamma\\,n_i - (\\sigma_g\\,n_i)^2\\bigr)\\,
           \\cos\\!\\bigl(2\\pi\\,(a - a_0)\\,n_i \\, s\\bigr)

   where *s* (``scale``) is the Rabi frequency in cycles per unit
   amplitude per pulse.

2. **Decay validation** — The effective pulse-number coherence is
   the 1/e point of ``exp(-γn - (σ_g n)²)``.  If the fitted value
   far exceeds the measurement window, a fallback exponential-decay
   fit to the near-resonance trace is attempted.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, differential_evolution

from qualibrate.core import QualibrationNode
from calibration_utils.common_utils.parity_streams import get_parity_item_names

_logger = logging.getLogger(__name__)


# ── Mean-signal resonance model ──────────────────────────────────────────────


def _make_mean_rabi_model(n_pulses: np.ndarray):
    r"""Return a model for the n_pulses-averaged signal vs amplitude.

    Parameters (of the returned callable)
    --------------------------------------
    x : array — amplitude prefactor
    amp : float — peak contrast (positive = peak, negative = dip)
    x0 : float — optimal amplitude prefactor (a_π)
    gamma : float — exponential decay rate per pulse
    sigma_g : float — Gaussian decay rate per pulse
    bg : float — off-resonance baseline
    scale : float — Rabi frequency (cycles / unit_amp / pulse)
    """
    n = np.asarray(n_pulses, dtype=float)

    def _model(x, amp, x0, gamma, sigma_g, bg, scale):
        da = np.atleast_1d(np.asarray(x, dtype=float)) - x0
        phase = 2.0 * np.pi * da[:, None] * n[None, :] * scale
        envelope = np.exp(-gamma * n - (sigma_g * n) ** 2)[None, :]
        result = bg + amp * np.mean(envelope * np.cos(phase), axis=1)
        return result if np.ndim(x) > 0 else float(result[0])

    return _model


# ── Effective N_eff from combined envelope ───────────────────────────────────


def _effective_n_eff(gamma: float, sigma_g: float) -> float:
    r"""Compute the 1/e pulse-count of exp(-γn - (σ_g n)²).

    Solves  σ_g²n² + γn − 1 = 0  for the positive root.
    """
    if sigma_g < 1e-12:
        return 1.0 / gamma if gamma > 1e-12 else np.nan
    discriminant = gamma**2 + 4.0 * sigma_g**2
    return (-gamma + np.sqrt(discriminant)) / (2.0 * sigma_g**2)


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Extracted parameters from an error-amplified power-Rabi measurement."""

    opt_amp: float
    rabi_frequency: float
    decay_rate: float
    gauss_decay_rate: float
    n_eff: float
    success: bool


# ── Decay validation and fallback ────────────────────────────────────────────


def _fit_exponential_decay(
    n_pulses: np.ndarray,
    trace: np.ndarray,
    n_span: float,
) -> float | None:
    """Fit ``bg + A·exp(-γn)`` to a near-resonance pulse-count trace."""
    t = n_pulses - n_pulses[0]
    y = np.asarray(trace, dtype=float)

    bg0 = float(np.mean(y[-len(y) // 4 :]))
    amp0 = float(y[0]) - bg0
    if abs(amp0) < 0.01:
        return None
    gamma0 = 2.0 / n_span

    def _model(t, bg, amp, gamma):
        return bg + amp * np.exp(-gamma * t)

    try:
        popt, pcov = curve_fit(
            _model,
            t,
            y,
            p0=[bg0, amp0, gamma0],
            bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 10.0 / n_span]),
            maxfev=3000,
        )
        perr = np.sqrt(np.diag(pcov))
        gamma = float(popt[2])
        if gamma < 2e-6 or (perr[2] > gamma * 5):
            return None
        return gamma
    except Exception:
        return None


def _validate_n_eff(
    signal_2d: np.ndarray,
    n_pulses: np.ndarray,
    opt_amp: float,  # noqa: ARG001
    amps: np.ndarray,  # noqa: ARG001
    resonance_idx: int,
    decay_rate_de: float,
    sigma_g_de: float,
    n_eff_de: float,
) -> tuple[float, float, float]:
    """Validate N_eff from the DE fit, fallback if poorly constrained."""
    n_span = float(n_pulses[-1] - n_pulses[0]) if len(n_pulses) > 1 else 1.0

    if np.isfinite(n_eff_de) and 0 < n_eff_de < 5.0 * n_span:
        return decay_rate_de, sigma_g_de, n_eff_de

    n_amp = signal_2d.shape[1]
    half_w = max(1, n_amp // 20)
    lo = max(0, resonance_idx - half_w)
    hi = min(n_amp, resonance_idx + half_w + 1)
    near_trace = np.mean(signal_2d[:, lo:hi], axis=1)
    gamma_exp = _fit_exponential_decay(n_pulses, near_trace, n_span)
    if gamma_exp is not None:
        return gamma_exp, 0.0, 1.0 / gamma_exp

    return decay_rate_de, sigma_g_de, n_eff_de


# ── Single-qubit 2D analysis ────────────────────────────────────────────────


def _analyse_single_qubit(
    signal_2d: np.ndarray,
    amps: np.ndarray,
    n_pulses: np.ndarray,
) -> Dict[str, Any]:
    """Analyse one qubit's 2-D error-amplified Rabi and extract key parameters.

    Parameters
    ----------
    signal_2d : 2-D array (n_n_pulses, n_amp)
    amps : 1-D array (n_amp,)
    n_pulses : 1-D array (n_n_pulses,)

    Returns
    -------
    dict with ``opt_amp``, ``rabi_frequency``, ``decay_rate``,
    ``gauss_decay_rate``, ``n_eff``, ``success``, and ``_diag``.
    """
    n_np = signal_2d.shape[0]

    # ── Step 1: Fit mean signal vs amplitude ─────────────────────────────
    mean_signal = np.mean(signal_2d, axis=0)
    model = _make_mean_rabi_model(n_pulses)

    median_val = float(np.median(mean_signal))
    abs_dev = np.abs(mean_signal - median_val)
    extremum_idx = int(np.argmax(abs_dev))

    opt_amp = float(amps[extremum_idx])
    resonance_idx = extremum_idx
    mean_signal_fit = None
    decay_rate = np.nan
    sigma_g = 0.0
    n_eff = np.nan
    scale = np.nan

    try:
        ptp = float(np.ptp(mean_signal))
        amp_min, amp_max = float(amps.min()), float(amps.max())
        n_span = float(n_pulses[-1] - n_pulses[0]) if n_np > 1 else 1.0
        y_min, y_max = float(mean_signal.min()), float(mean_signal.max())
        amp_range = amp_max - amp_min

        # scale bounds: physically, scale = 1/(2*a_π) for a single-pulse
        # pi-condition.  For amplitude range ~0.2, we need scale such that
        # the highest-N slice shows a few cycles: N_max * scale * amp_range ~ 1–10.
        n_max = float(n_pulses[-1])
        scale_min = 0.1 / (n_max * amp_range) if n_max * amp_range > 0 else 0.01
        scale_max = (
            50.0 / (float(n_pulses[0]) * amp_range)
            if float(n_pulses[0]) * amp_range > 0
            else 100.0
        )

        # Constrain amp sign to match the observed peak/trough polarity so the
        # optimizer cannot converge to a wrong local minimum (e.g. fitting a
        # trough as a positive-amp peak displaced in x0).
        extremum_sign = float(np.sign(mean_signal[extremum_idx] - median_val))
        if extremum_sign > 0:
            amp_de_bounds = (0.0, ptp * 3)
        elif extremum_sign < 0:
            amp_de_bounds = (-ptp * 3, 0.0)
        else:
            amp_de_bounds = (-ptp * 3, ptp * 3)

        de_bounds = [
            amp_de_bounds,  # amp — sign locked to peak/trough polarity
            (amp_min, amp_max),  # x0 (opt_amp)
            (0.0, 10.0 / n_span),  # gamma (per pulse)
            (0.0, 10.0 / n_span),  # sigma_g (per pulse)
            (y_min - ptp, y_max + ptp),  # bg
            (scale_min, scale_max),  # scale
        ]

        def _objective(params):
            return np.sum((model(amps, *params) - mean_signal) ** 2)

        de_result = differential_evolution(
            _objective,
            de_bounds,
            seed=42,
            maxiter=2000,
            tol=1e-10,
            polish=True,
            popsize=25,
        )
        popt = de_result.x
        opt_amp = float(popt[1])
        decay_rate = float(popt[2])
        sigma_g = float(popt[3])
        scale = float(popt[5])
        resonance_idx = int(np.argmin(np.abs(amps - opt_amp)))
        n_eff = _effective_n_eff(decay_rate, sigma_g)
        mean_signal_fit = model(amps, *popt)
        _logger.debug(
            "Error-amp Rabi mean-signal fit (DE): a_π=%.4f, "
            "scale=%.4f c/u.a./pulse, gamma=%.5f, sigma_g=%.5f, N_eff=%.1f",
            opt_amp,
            scale,
            decay_rate,
            sigma_g,
            n_eff if np.isfinite(n_eff) else -1,
        )
    except Exception:
        _logger.debug(
            "Mean-signal fit failed; using raw extremum at a=%.4f",
            opt_amp,
        )

    # ── Step 2: Validate / refine N_eff ──────────────────────────────────
    decay_rate, sigma_g, n_eff = _validate_n_eff(
        signal_2d,
        n_pulses,
        opt_amp,
        amps,
        resonance_idx,
        decay_rate,
        sigma_g,
        n_eff,
    )

    rabi_frequency = 2.0 * np.pi * scale if np.isfinite(scale) else np.nan
    success = np.isfinite(opt_amp) and opt_amp > 0

    return {
        "opt_amp": opt_amp,
        "rabi_frequency": float(rabi_frequency),
        "decay_rate": float(decay_rate),
        "gauss_decay_rate": float(sigma_g),
        "n_eff": float(n_eff),
        "success": success,
        "_diag": {
            "mean_signal": mean_signal,
            "mean_signal_fit": mean_signal_fit,
            "resonance_idx": resonance_idx,
        },
    }


# ── Public API ───────────────────────────────────────────────────────────────


def _error_amp_qubit_names(
    ds: xr.Dataset,
    analysis_signal: str,
    qubits,
) -> list[str]:
    return get_parity_item_names(
        ds,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )


def _as_n_pulses_amp_signal(da: xr.DataArray, qname: str) -> np.ndarray:
    if "qubit" in da.dims:
        qubit_coord = da.coords.get("qubit")
        if qubit_coord is not None and qname in set(qubit_coord.values.tolist()):
            da = da.sel(qubit=qname, drop=True)
        elif da.sizes["qubit"] == 1:
            da = da.isel(qubit=0, drop=True)
        else:
            raise ValueError(
                f"{da.name!r} for {qname!r} still has a non-singleton qubit "
                f"dimension. Run process_raw_data before fit_raw_data_error_amplified."
            )

    expected_dims = ("n_pulses", "amp_prefactor")
    if all(dim in da.dims for dim in expected_dims):
        for dim in list(da.dims):
            if dim in expected_dims:
                continue
            if da.sizes[dim] != 1:
                raise ValueError(
                    f"{da.name!r} for {qname!r} has unexpected non-singleton "
                    f"dimension {dim!r} with size {da.sizes[dim]}."
                )
            da = da.isel({dim: 0}, drop=True)
        return da.transpose(*expected_dims).values.astype(float)

    data = np.asarray(da.values, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"{da.name!r} for {qname!r} must be 2-D over {expected_dims}; "
            f"shape is {data.shape}."
        )
    return data


def fit_raw_data_error_amplified(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit optimal amplitude per qubit from error-amplified power-Rabi data.

    Expects joint-outcome streams processed by
    :func:`~calibration_utils.common_utils.parity_streams.process_joint_streams`,
    so the analysis uses ``{analysis_signal}_{qubit}`` of shape
    ``(n_n_pulses, n_amp)``.

    Parameters
    ----------
    ds : xr.Dataset
        Measurement data with coordinates ``n_pulses`` and
        ``amp_prefactor``.
    node : QualibrationNode
        Calibration node (provides qubit list and ``analysis_signal``).

    Returns
    -------
    (ds_fit, fit_results) : tuple
    """
    qubits = node.namespace["qubits"]
    analysis_signal = getattr(node.parameters, "analysis_signal", "E_p2_given_p1_0")
    qubit_names = _error_amp_qubit_names(ds, analysis_signal, qubits)

    amps = np.asarray(ds.amp_prefactor.values, dtype=float)
    n_pulses_array = np.asarray(ds.n_pulses.values, dtype=float)

    fit_results: Dict[str, Dict[str, Any]] = {}

    for qname in qubit_names:
        signal_var = f"{analysis_signal}_{qname}"
        if signal_var not in ds.data_vars and f"p_{qname}" in ds.data_vars:
            signal_var = f"p_{qname}"
        if signal_var not in ds.data_vars:
            fp = FitParameters(
                opt_amp=np.nan,
                rabi_frequency=np.nan,
                decay_rate=np.nan,
                gauss_decay_rate=0.0,
                n_eff=np.nan,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        signal_2d = _as_n_pulses_amp_signal(ds[signal_var], qname)
        result = _analyse_single_qubit(signal_2d, amps, n_pulses_array)

        fp = FitParameters(
            opt_amp=result["opt_amp"],
            rabi_frequency=result["rabi_frequency"],
            decay_rate=result["decay_rate"],
            gauss_decay_rate=result["gauss_decay_rate"],
            n_eff=result["n_eff"],
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)
        fit_results[qname]["_diag"] = result.get("_diag")

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results_error_amplified(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qname, r in fit_results.items():
        a_pi = r.get("opt_amp", 0)
        omega = r.get("rabi_frequency", 0)
        gamma = r.get("decay_rate", 0)
        sigma_g = r.get("gauss_decay_rate", 0)
        n_eff = r.get("n_eff", 0)
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"a_π={a_pi:.4f}, "
            f"Ω={omega:.3f} rad/u.a./pulse, "
            f"γ={gamma:.5f}/pulse, "
            f"σ_g={sigma_g:.5f}/pulse, "
            f"N_eff={n_eff:.0f}, "
            f"success={success}"
        )
        log_callable(msg)
