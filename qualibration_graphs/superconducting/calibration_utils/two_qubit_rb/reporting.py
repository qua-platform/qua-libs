"""Typed RB fit result containers and logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from calibration_utils.common_utils.rb import LeakageFit
from calibration_utils.two_qubit_rb.fidelity import ImpliedCZResult, compute_srb_fidelity


@dataclass
class BaseFitResult:
    """Fields common to both Standard and Interleaved RB fits."""

    alpha: float
    alpha_stderr: float | None
    fidelity: float
    fidelity_stderr: float | None
    fit_amplitude: float
    fit_amplitude_stderr: float | None
    fit_offset: float
    fit_offset_stderr: float | None
    success: bool
    fit_issues: tuple[str, ...] = field(default_factory=tuple)
    fit_warnings: tuple[str, ...] = field(default_factory=tuple)
    coherence_limit_epg: float | None = None


@dataclass
class SRBFitResult(BaseFitResult):
    """Standard (37a) RB fit result: 2Q Clifford fidelity, EPC, EPG."""

    epc: float | None = None
    epc_stderr: float | None = None
    epg: float | None = None
    epg_stderr: float | None = None
    average_gate_fidelity: float | None = None
    average_gates_per_clifford: float | None = None
    implied_cz: ImpliedCZResult | None = None


@dataclass
class IRBFitResult(BaseFitResult):
    """Interleaved (37b) RB fit result: direct CZ gate fidelity vs. reference."""

    epc: float | None = None
    epc_stderr: float | None = None
    epg: float | None = None
    epg_stderr: float | None = None
    standard_rb_alpha: float | None = None
    standard_rb_alpha_stderr: float | None = None


@dataclass
class LeakagePairResult:
    """Leakage RB result of one qubit pair: a :class:`LeakageFit` per series (pair, control, target).

    ``failure`` is set when the g/e/f readout looks wrong (nothing is then written to the state).
    The interleaved node (37b) adds ``per_cz`` (``l1``, ``l1_stderr``, ``l2``, ``l2_stderr`` per
    series) and the time the 37a reference was stored (``reference_updated_at``), or ``per_cz_note``
    when no reference was stored.
    """

    control_qubit: str
    target_qubit: str
    confusion_corrected: dict[str, bool]
    per_clifford: dict[str, LeakageFit]
    success: bool
    failure: str = ""
    per_cz: dict[str, dict[str, float | None]] | None = None
    reference_updated_at: str = ""
    per_cz_note: str = ""


def leakage_reference_entry(leakage_result: dict) -> dict:
    """37a's ``macros[operation].extras["leakage_per_2q_clifford"]``: L1 and L2 per 2Q Clifford for each
    series, and when they were stored (``updated_at``, so 37b can tell how old its reference is).

    ``leakage_result`` is this pair's ``node.results["leakage_results"]`` entry; the fit details stay
    in the run's saved results.
    """
    entry = {
        series: {key: rates[key] for key in ("l1", "l1_stderr", "l2", "l2_stderr")}
        for series, rates in leakage_result["per_clifford"].items()
    }
    entry["updated_at"] = _now()
    return entry


def leakage_per_cz_entry(leakage_result: dict) -> dict | None:
    """37b's ``macros[operation].extras["leakage_per_cz"]``: L1 per CZ for each series, or None without a
    37a reference."""
    per_cz = leakage_result["per_cz"]
    if per_cz is None:
        return None
    entry = {series: {"l1": rates["l1"], "l1_stderr": rates["l1_stderr"]} for series, rates in per_cz.items()}
    entry["updated_at"] = _now()
    return entry


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_fraction_pm(
    value: float | None,
    stderr: float | None = None,
    *,
    as_error_rate: bool = False,
) -> str:
    """Format a fraction in [0, 1] as a percentage, optionally with ±1σ."""
    if value is None or not np.isfinite(value):
        return "n/a"
    if stderr is None or not np.isfinite(stderr) or stderr <= 0:
        if as_error_rate and value < 0.01:
            return f"{value * 1e3:.2f} × 10⁻³"
        return f"{100 * value:.2f}%"
    if as_error_rate and value < 0.01:
        return f"{value * 1e3:.2f} ± {stderr * 1e3:.2f} × 10⁻³"
    return f"{100 * value:.2f} ± {100 * stderr:.2f}%"


def _coeff_pm(value: float, stderr: float | None) -> str:
    if stderr is not None and np.isfinite(stderr) and stderr > 0:
        return f"{value:.6f} ± {stderr:.6f}"
    return f"{value:.6f}"


def _format_issues_and_warnings(result: BaseFitResult) -> list[str]:
    lines = [f"\t- {issue}" for issue in result.fit_issues]
    lines += [f"\tWarning: {warning}" for warning in result.fit_warnings]
    return lines


def _format_coherence_limit(result: BaseFitResult, *, epg: float | None, epg_stderr: float | None) -> list[str]:
    if result.coherence_limit_epg is None:
        return []
    lines = [
        f"\tCoherence-limited EPG (T1/T2 floor) = "
        f"{format_fraction_pm(result.coherence_limit_epg, as_error_rate=True)}"
    ]
    if epg is not None and result.coherence_limit_epg > 0:
        ratio = epg / result.coherence_limit_epg
        verdict = (
            "Not coherence-limited (EPG exceeds T1/T2 floor)"
            if epg > result.coherence_limit_epg
            else "Coherence-limited (EPG at or below T1/T2 floor)"
        )
        lines.append(
            f"\t{verdict} ({ratio:.1f}× floor: "
            f"{format_fraction_pm(epg, epg_stderr, as_error_rate=True)} vs "
            f"{format_fraction_pm(result.coherence_limit_epg, as_error_rate=True)})"
        )
    return lines


def _format_srb_result(qp_name: str, r: SRBFitResult) -> str:
    lines = [f"Results for qubit pair {qp_name}: {'SUCCESS!' if r.success else 'FAIL!'}"]
    if r.success:
        lines.append(
            f"\tDecay model: P(m) = A * alpha^m + B "
            f"(A = {_coeff_pm(r.fit_amplitude, r.fit_amplitude_stderr)}, "
            f"alpha = {_coeff_pm(r.alpha, r.alpha_stderr)}, "
            f"B = {_coeff_pm(r.fit_offset, r.fit_offset_stderr)})"
        )
        lines.extend(
            [
                f"\t2Q Clifford Fidelity = {format_fraction_pm(r.fidelity, r.fidelity_stderr)}",
                f"\tError Per Clifford (EPC): 1 - 2Q Clifford Fidelity = "
                f"{format_fraction_pm(r.epc, r.epc_stderr, as_error_rate=True)}",
                f"\tError Per Gate (EPG) = EPC / N_gates_per_Clifford = "
                f"{format_fraction_pm(r.epc, r.epc_stderr, as_error_rate=True)} / "
                f"{r.average_gates_per_clifford:.2f} = "
                f"{format_fraction_pm(r.epg, r.epg_stderr, as_error_rate=True)}",
                f"\tAvg. Gate Fidelity (1-EPG) = " f"{format_fraction_pm(r.average_gate_fidelity, r.epg_stderr)}",
            ]
        )
        if r.implied_cz is not None and r.implied_cz.alpha_01 is not None:
            icz = r.implied_cz
            lines.extend(
                [
                    "",
                    "\tImplied CZ gate fidelity:",
                    f"\tImplied CZ Error Per Gate (CZ_EPG) = "
                    f"{format_fraction_pm(icz.epg_cz, icz.epg_cz_stderr, as_error_rate=True)} "
                    "(inferred by factoring out 1Q contributions)",
                    f"\tImplied CZ gate fidelity = 1 - CZ_EPG = "
                    f"{format_fraction_pm(None if icz.epg_cz is None else 1 - icz.epg_cz, icz.epg_cz_stderr)}",
                    "\tNote: Interleaved RB (37b) measures CZ EPG directly.",
                ]
            )
        lines.extend(_format_coherence_limit(r, epg=r.epg, epg_stderr=r.epg_stderr))
    lines.extend(_format_issues_and_warnings(r))
    return "\n".join(lines)


def _format_irb_result(qp_name: str, r: IRBFitResult) -> str:
    lines = [f"Results for qubit pair {qp_name}: {'SUCCESS!' if r.success else 'FAIL!'}"]
    if r.success:
        lines.append(
            f"\tDecay model: P(m) = A * alpha^m + B "
            f"(A = {_coeff_pm(r.fit_amplitude, r.fit_amplitude_stderr)}, "
            f"alpha_IRB = {_coeff_pm(r.alpha, r.alpha_stderr)}, "
            f"B = {_coeff_pm(r.fit_offset, r.fit_offset_stderr)})"
        )
        alpha_srb_text = (
            _coeff_pm(r.standard_rb_alpha, r.standard_rb_alpha_stderr) if r.standard_rb_alpha is not None else "n/a"
        )
        srb_ref = (
            compute_srb_fidelity(r.standard_rb_alpha, r.standard_rb_alpha_stderr, None, n_qubits=2)
            if r.standard_rb_alpha is not None
            else None
        )
        srb_fidelity = srb_ref.fidelity if srb_ref is not None else np.nan
        srb_fidelity_stderr = srb_ref.fidelity_stderr if srb_ref is not None else None
        lines.extend(
            [
                "",
                "\tStandard RB reference (37a):",
                f"\t2Q Clifford Fidelity = {format_fraction_pm(srb_fidelity, srb_fidelity_stderr)}",
                f"\tError Per Clifford (EPC) = 1 - 2Q Clifford Fidelity = "
                f"{format_fraction_pm(srb_ref.epc if srb_ref is not None else r.epc, srb_ref.epc_stderr if srb_ref is not None else r.epc_stderr, as_error_rate=True)}",
                f"\talpha_SRB = {alpha_srb_text}",
                "",
                "\tInterleaved RB (this run):",
                f"\tCZ_Error Per Gate (CZ_EPG) =  (d-1)/d * (1 - alpha_IRB/alpha_SRB) = "
                f"{format_fraction_pm(r.epg, r.epg_stderr, as_error_rate=True)}",
                f"\tCZ gate fidelity = 1 - CZ_EPG = " f"{format_fraction_pm(r.fidelity, r.fidelity_stderr)}",
            ]
        )
        lines.extend(_format_coherence_limit(r, epg=r.epg, epg_stderr=r.epg_stderr))
    lines.extend(_format_issues_and_warnings(r))
    return "\n".join(lines)


def log_srb_results(fit_results: dict[str, SRBFitResult], log_callable=None) -> None:
    """Log fitted Standard RB results for all qubit pairs."""
    log_callable = log_callable or logging.getLogger(__name__).info
    for qp_name, r in fit_results.items():
        log_callable(_format_srb_result(qp_name, r))


def log_irb_results(fit_results: dict[str, IRBFitResult], log_callable=None) -> None:
    """Log fitted Interleaved RB results for all qubit pairs."""
    log_callable = log_callable or logging.getLogger(__name__).info
    for qp_name, r in fit_results.items():
        log_callable(_format_irb_result(qp_name, r))


# --- Leakage RB (measure_leakage=True) -------------------------------------------------------


def format_leakage_rate(l1: float | None, l1_stderr: float | None = None, *, pm: str = "+/-") -> str:
    """Format a leakage rate as ``(value +/- stderr)e-07``, both on one exponent so a rate consistent with
    zero reads as such, or ``n/a``; plots pass ``pm="±"`` and get ``(value ± stderr) × 10⁻⁷``."""
    if l1 is None or not np.isfinite(l1):
        return "n/a"
    if l1_stderr is None or not np.isfinite(l1_stderr):
        return f"{l1:.2e}"
    scale = max(abs(l1), l1_stderr)
    exponent = int(np.floor(np.log10(scale))) if scale > 0 else 0
    mantissa = f"({l1 / 10**exponent:.2f} {pm} {l1_stderr / 10**exponent:.2f})"
    if pm == "±":
        return f"{mantissa} × 10{str(exponent).translate(_SUPERSCRIPT)}"
    return f"{mantissa}e{exponent:+03d}"


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def leakage_series_labels(control_qubit: str, target_qubit: str) -> dict[str, str]:
    """Display names of the leakage series: either qubit, then each qubit by name."""
    return {"pair": "either qubit", "control": control_qubit, "target": target_qubit}


def log_leakage_results(results: dict[str, LeakagePairResult], *, interleaved: bool, log_callable=print) -> None:
    """Log L1 per pair: per Clifford, and per CZ for the interleaved node."""
    for qp_name, result in results.items():
        labels = leakage_series_labels(result.control_qubit, result.target_qubit)
        unit = "interleaved Clifford" if interleaved else "2Q Clifford"
        lines = [f"Qubit pair {qp_name} leakage L1 per {unit}:"]
        if result.failure:
            lines.append(f"  FAILED: {result.failure}")
        for series, fit in result.per_clifford.items():
            status = "" if fit.success else " FAILED"
            lines.append(f"  {labels[series]}: {format_leakage_rate(fit.l1, fit.l1_stderr)}{status}")
            lines += [f"\t- {issue}" for issue in fit.fit_issues]
            lines += [f"\tWarning: {warning}" for warning in fit.fit_warnings]
        if interleaved:
            if result.per_cz is None:
                lines.append(f"  per CZ: {result.per_cz_note}")
            else:
                rates = ", ".join(
                    f"{labels[series]} {format_leakage_rate(rate['l1'], rate['l1_stderr'])}"
                    for series, rate in result.per_cz.items()
                )
                lines.append(f"  per CZ: {rates}")
                lines.append(
                    f"  (minus the 37a reference stored {result.reference_updated_at or 'at an unknown time'})"
                )
        log_callable("\n".join(lines))
