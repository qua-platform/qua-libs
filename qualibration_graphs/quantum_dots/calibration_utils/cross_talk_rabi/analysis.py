"""Cross-talk Rabi analysis.

For a fixed measured qubit pair, the swept ``drive_qubit`` is a *different* qubit
outside that pair. If the drive bleeds onto the measured pair, its conditional
expectation traces out a (usually low-contrast) Rabi-like curve as the drive
amplitude is increased; the *height* of that curve is the cross-talk figure of
merit.

Each (measured pair, drive qubit) combination is identified by a combo name of
the form ``"<pair>__drive_<drive>"`` (e.g. ``"q2_q3__drive_q1"``). The fitting
machinery is shared with the standard power-Rabi node: an FFT seeds a
damped-cosine fit in the amplitude domain.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode

from calibration_utils.power_rabi.analysis import (
    _analyse_single_qubit,
    _as_amp_trace,
)

_logger = logging.getLogger(__name__)

# Separator embedded in a combo name: "<pair>__drive_<drive>".
COMBO_SEP = "__drive_"


# ── Combo identifier ─────────────────────────────────────────────────────────


@dataclass
class CrossTalkCombo:
    """A single (measured pair, drive qubit) experiment unit.

    ``name`` is used as the per-item key for the parity streams and dataset
    variables. ``pair`` / ``drive_qubit`` hold the live QuAM objects when the
    combo is built inside the node (they are not serialised).
    """

    pair_name: str
    drive_name: str
    pair: Any = None
    drive_qubit: Any = None

    @property
    def name(self) -> str:
        return f"{self.pair_name}{COMBO_SEP}{self.drive_name}"


def parse_combo_name(name: str) -> Tuple[str, str]:
    """Split a combo name into ``(pair_name, drive_name)``."""
    pair_name, _, drive_name = name.partition(COMBO_SEP)
    return pair_name, drive_name


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Fit parameters for a single (pair, drive) cross-talk Rabi sweep."""

    pair: str
    drive_qubit: str
    crosstalk_contrast: float  # raw peak-to-peak of the trace (0..1) — headline metric
    crosstalk_amplitude: float  # fitted oscillation amplitude (0..1), if the fit converged
    rabi_frequency: float  # rad per unit drive amplitude
    decay_rate: float  # 1 / unit drive amplitude
    success: bool  # valid measured trace (finite contrast)
    fit_success: bool  # damped-sinusoid fit converged (controls the curve overlay)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _combo_names_from_ds(ds: xr.Dataset, analysis_signal: str) -> List[str]:
    """Resolve combo names from the processed conditional-expectation variables."""
    prefix = f"{analysis_signal}_"
    names = [
        v[len(prefix):]
        for v in sorted(ds.data_vars)
        if v.startswith(prefix) and not v.endswith("_fit")
    ]
    if names:
        return names

    p0_p0_vars = [v for v in ds.data_vars if v.startswith("p0_p0_")]
    if p0_p0_vars:
        return [v.replace("p0_p0_", "", 1) for v in sorted(p0_p0_vars)]

    return [
        v[2:]
        for v in sorted(ds.data_vars)
        if v.startswith("p_") and not v.startswith(("p0_", "p1_", "pdiff_", "E_"))
    ]


# ── Public API ────────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit the cross-talk Rabi response for each (pair, drive) combo."""
    analysis_signal = getattr(node.parameters, "analysis_signal", "E_p2_given_p1_0")
    combo_names = _combo_names_from_ds(ds, analysis_signal)

    amps = np.asarray(ds.amp_prefactor.values, dtype=float)

    fit_results: Dict[str, Dict[str, Any]] = {}

    for cname in combo_names:
        pair_name, drive_name = parse_combo_name(cname)

        signal_var = f"{analysis_signal}_{cname}"
        if signal_var not in ds.data_vars and f"p_{cname}" in ds.data_vars:
            signal_var = f"p_{cname}"
        if signal_var not in ds.data_vars:
            fit_results[cname] = asdict(
                FitParameters(
                    pair=pair_name,
                    drive_qubit=drive_name,
                    crosstalk_contrast=np.nan,
                    crosstalk_amplitude=np.nan,
                    rabi_frequency=np.nan,
                    decay_rate=np.nan,
                    success=False,
                    fit_success=False,
                )
            )
            continue

        trace = _as_amp_trace(ds[signal_var], cname)
        result = _analyse_single_qubit(trace, amps)

        # Headline metric: raw peak-to-peak height of the response. This does not
        # depend on the (often fragile) damped-sinusoid fit converging.
        contrast = float(np.ptp(trace)) if trace.size else np.nan

        sinusoid = result.get("sinusoid_fit")
        if sinusoid is not None:
            amplitude = float(abs(sinusoid["amplitude"]))
        elif np.isfinite(contrast):
            amplitude = contrast / 2.0
        else:
            amplitude = np.nan

        fp = FitParameters(
            pair=pair_name,
            drive_qubit=drive_name,
            crosstalk_contrast=contrast,
            crosstalk_amplitude=amplitude,
            rabi_frequency=result["rabi_frequency"],
            decay_rate=result["decay_rate"],
            success=bool(np.isfinite(contrast)),
            fit_success=bool(result["success"]),
        )
        fit_results[cname] = asdict(fp)

        fit_results[cname]["_fft_diag"] = {
            "fft_freqs": result["fft_freqs"],
            "fft_magnitude": result["fft_magnitude"],
            "peak_curve": result["peak_curve"],
        }
        fit_results[cname]["_sinusoid_fit"] = result.get("sinusoid_fit")

    ds_fit = ds.copy()
    return ds_fit, fit_results


def build_crosstalk_matrix(
    fit_results: Dict[str, Dict[str, Any]],
    metric: str = "crosstalk_contrast",
) -> Dict[str, Any]:
    """Assemble a measured-pair x drive-qubit matrix of the chosen metric.

    Returns a dict with sorted ``pairs`` (rows), ``drives`` (columns) and a 2-D
    ``matrix`` (NaN where a combination was not measured). The default metric is
    the raw curve height (contrast), which is robust to fit convergence.
    """
    pairs = sorted({r["pair"] for r in fit_results.values()})
    drives = sorted({r["drive_qubit"] for r in fit_results.values()})

    pair_idx = {p: i for i, p in enumerate(pairs)}
    drive_idx = {d: j for j, d in enumerate(drives)}

    matrix = np.full((len(pairs), len(drives)), np.nan, dtype=float)
    for r in fit_results.values():
        value = r.get(metric, np.nan)
        if value is not None and np.isfinite(value):
            matrix[pair_idx[r["pair"]], drive_idx[r["drive_qubit"]]] = value

    return {
        "pairs": pairs,
        "drives": drives,
        "matrix": matrix,
        "metric": metric,
    }


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted cross-talk results for all (pair, drive) combos."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for cname, r in fit_results.items():
        msg = (
            f"Cross-talk drive {r.get('drive_qubit')} -> pair {r.get('pair')}: "
            f"contrast={r.get('crosstalk_contrast', float('nan')):.4f} (height), "
            f"fitted_amplitude={r.get('crosstalk_amplitude', float('nan')):.4f}, "
            f"Ω={r.get('rabi_frequency', float('nan')):.3f} rad/u.a., "
            f"fit_success={r.get('fit_success', False)}"
        )
        log_callable(msg)
