"""No-dip span-escalation planning for resonator spectroscopy bring-up.

Pure functions (no hardware, no node state) so the retry logic is unit-testable
offline. The node applies the plan: widen the sweep for the qubits whose fit
found no significant dip, re-centering the readout LO when the wider sweep
would push the intermediate frequency past its limit (same tracked-LO pattern
as node 09b's flux-detuned window).

Fresh-from-fab context: the design resonator frequency can be off by hundreds
of MHz, so the standard ±15 MHz scan sees nothing. The ladder doubles the span
per retry up to ``max_span_hz`` (default ±400 MHz). With the LO re-centered on
the expected frequency the IF sweep is symmetric (±span/2), which for the
800 MHz ceiling sits exactly at the MW-FEM ±400 MHz IF reach.
"""

from dataclasses import dataclass

from .analysis import FitParameters

# MW-FEM upconverter (LO) reach per band — used to refuse an LO move that the
# hardware cannot express. Matches the guards in node 09b (lower bounds) plus
# the band upper edges.
_BAND_LO_RANGE = {
    1: (0.05e9, 5.5e9),
    2: (4.5e9, 7.5e9),
    3: (6.5e9, 10.5e9),
}
_IF_LIMIT_HZ = 400e6


@dataclass
class SpanEscalationPlan:
    """Decision produced by :func:`plan_span_escalation` for one no-dip retry rung."""

    retry: bool
    """Whether another (wider) measurement pass should be attempted."""

    qubits: list[str]
    """Names of the qubits with no significant dip (the retry's participants, if any)."""

    new_span_hz: float
    """Span (Hz) to sweep next; equal to `current_span_hz` when `retry` is False."""


def plan_span_escalation(
    fit_results: dict[str, FitParameters],
    current_span_hz: float,
    max_span_hz: float,
    *,
    growth: float = 2.0,
) -> SpanEscalationPlan:
    """Decide whether/how to widen the sweep after a no-dip analysis pass.

    ``fit_results`` maps qubit name -> :class:`FitParameters`. Only
    frequency-failures (no significant dip) participate; shape-poor fits
    already delivered a frequency and are not re-measured.
    """
    failed = [q for q, r in fit_results.items() if not r.success]
    if not failed or current_span_hz >= max_span_hz:
        return SpanEscalationPlan(retry=False, qubits=failed, new_span_hz=current_span_hz)
    new_span = min(current_span_hz * growth, max_span_hz)
    return SpanEscalationPlan(retry=True, qubits=failed, new_span_hz=float(new_span))


@dataclass
class LoRecenterPlan:
    """Decision produced by :func:`plan_lo_recenter` for one qubit's readout LO."""

    shift: bool
    """Whether the LO needs to move to fit the wider IF sweep."""

    new_lo_hz: float
    """LO frequency (Hz) to use; unchanged from the input when `shift` is False."""

    error: str | None
    """Reason the plan could not honour the requested span/band, or None on success."""


def plan_lo_recenter(
    rf_hz: float,
    lo_hz: float,
    span_hz: float,
    band: int | None,
    *,
    if_limit_hz: float = _IF_LIMIT_HZ,
) -> LoRecenterPlan:
    """LO plan for a symmetric ±span/2 sweep around ``rf_hz``.

    If the sweep fits within the IF limit at the current LO, no move is needed.
    Otherwise the LO is re-centered ON the expected resonator frequency
    (IF -> 0), making the required IF reach exactly ±span/2; if even that
    exceeds the IF limit, or the new LO leaves the band, the plan reports an
    error string instead (caller should cap the span or skip the qubit).
    """
    if0 = rf_hz - lo_hz
    lo_min, lo_max = _BAND_LO_RANGE.get(band, (0.0, float("inf")))
    if abs(if0) + span_hz / 2.0 <= if_limit_hz:
        return LoRecenterPlan(shift=False, new_lo_hz=float(lo_hz), error=None)
    if span_hz / 2.0 > if_limit_hz:
        return LoRecenterPlan(
            shift=False,
            new_lo_hz=float(lo_hz),
            error=f"span/2 = {span_hz / 2 / 1e6:.0f} MHz exceeds the ±{if_limit_hz / 1e6:.0f} MHz IF reach",
        )
    new_lo = float(rf_hz)  # IF -> 0 at the expected center
    if not (lo_min <= new_lo <= lo_max):
        return LoRecenterPlan(
            shift=False,
            new_lo_hz=float(lo_hz),
            error=f"re-centered LO {new_lo / 1e9:.3f} GHz outside band-{band} range "
            f"[{lo_min / 1e9:.2f}, {lo_max / 1e9:.2f}] GHz",
        )
    return LoRecenterPlan(shift=True, new_lo_hz=new_lo, error=None)
