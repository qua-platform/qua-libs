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

from typing import Any

# MW-FEM upconverter (LO) reach per band — used to refuse an LO move that the
# hardware cannot express. Matches the guards in node 09b (lower bounds) plus
# the band upper edges.
_BAND_LO_RANGE = {
    1: (0.05e9, 5.5e9),
    2: (4.5e9, 7.5e9),
    3: (6.5e9, 10.5e9),
}
_IF_LIMIT_HZ = 400e6


def plan_span_escalation(
    fit_results: dict[str, dict[str, Any]],
    current_span_hz: float,
    max_span_hz: float,
    *,
    growth: float = 2.0,
) -> dict[str, Any]:
    """Decide whether/how to widen the sweep after a no-dip analysis pass.

    ``fit_results`` maps qubit name -> dict with at least ``success`` (bool).
    Only frequency-failures (no significant dip) participate; shape-poor fits
    already delivered a frequency and are not re-measured.

    Returns dict(retry: bool, qubits: [names], new_span_hz: float).
    """
    failed = [q for q, r in fit_results.items() if not r.get("success", False)]
    if not failed or current_span_hz >= max_span_hz:
        return dict(retry=False, qubits=failed, new_span_hz=current_span_hz)
    new_span = min(current_span_hz * growth, max_span_hz)
    return dict(retry=True, qubits=failed, new_span_hz=float(new_span))


def plan_lo_recenter(
    rf_hz: float,
    lo_hz: float,
    span_hz: float,
    band: int | None,
    *,
    if_limit_hz: float = _IF_LIMIT_HZ,
) -> dict[str, Any]:
    """LO plan for a symmetric ±span/2 sweep around ``rf_hz``.

    If the sweep fits within the IF limit at the current LO, no move is needed.
    Otherwise the LO is re-centered ON the expected resonator frequency
    (IF -> 0), making the required IF reach exactly ±span/2; if even that
    exceeds the IF limit, or the new LO leaves the band, the plan reports an
    error string instead (caller should cap the span or skip the qubit).

    Returns dict(shift: bool, new_lo_hz: float, error: str|None).
    """
    if0 = rf_hz - lo_hz
    lo_min, lo_max = _BAND_LO_RANGE.get(band, (0.0, float("inf")))
    if abs(if0) + span_hz / 2.0 <= if_limit_hz:
        return dict(shift=False, new_lo_hz=float(lo_hz), error=None)
    if span_hz / 2.0 > if_limit_hz:
        return dict(
            shift=False,
            new_lo_hz=float(lo_hz),
            error=f"span/2 = {span_hz / 2 / 1e6:.0f} MHz exceeds the ±{if_limit_hz / 1e6:.0f} MHz IF reach",
        )
    new_lo = float(rf_hz)  # IF -> 0 at the expected center
    if not (lo_min <= new_lo <= lo_max):
        return dict(
            shift=False,
            new_lo_hz=float(lo_hz),
            error=f"re-centered LO {new_lo / 1e9:.3f} GHz outside band-{band} range "
            f"[{lo_min / 1e9:.2f}, {lo_max / 1e9:.2f}] GHz",
        )
    return dict(shift=True, new_lo_hz=new_lo, error=None)
