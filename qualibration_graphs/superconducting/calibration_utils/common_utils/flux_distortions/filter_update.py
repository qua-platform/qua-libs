"""Apply flux-distortion IIR/FIR fit results to QUAM Z-line output state."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Set

LogCallable = Callable[[str], None]

IIR_MAX_BY_PORT_TYPE = {
    "LFFEMAnalogOutputPort": 6,
    "OPXPlusAnalogOutputPort": 3,
}


def _init_exponential_filters(qubits: Iterable[Any], machine: Any) -> None:
    for q in qubits:
        z_out = machine.qubits[q.name].z.opx_output
        if z_out.exponential_filter is None:
            z_out.exponential_filter = []


def _append_iir_taps_from_fit(
    z_out: Any,
    fit_result: Optional[dict[str, Any]],
    *,
    qubit_name: str,
    log_callable: LogCallable,
) -> bool:
    if fit_result is None or not fit_result.get("success"):
        if fit_result is not None and not fit_result.get("success"):
            log_callable(f"{qubit_name}: skip IIR update — fit did not succeed")
        return False

    a_dc = fit_result["a_dc"]
    new_taps = [(amp / a_dc, tau) for amp, tau in fit_result.get("a_tau_tuple") or []]
    if not new_taps:
        return False

    iir_max = IIR_MAX_BY_PORT_TYPE.get(type(z_out).__name__)
    existing = len(z_out.exponential_filter or [])
    if iir_max is None or existing + len(new_taps) > iir_max:
        log_callable(
            f"{qubit_name}: skip IIR update — {existing} existing + {len(new_taps)} new"
            + (f" > {iir_max} max" if iir_max else f", unsupported port {type(z_out).__name__}")
        )
        return False

    z_out.exponential_filter.extend(new_taps)
    log_callable(
        f"{qubit_name}: updated IIR ({len(z_out.exponential_filter)}/{iir_max}): "
        f"{z_out.exponential_filter}"
    )
    return True


def _set_feedforward_filter_from_fit(
    z_out: Any,
    fir_result: Optional[dict[str, Any]],
    *,
    qubit_name: str,
    log_callable: LogCallable,
) -> bool:
    if fir_result is not None and fir_result.get("success"):
        z_out.feedforward_filter = fir_result["inverse_fir"]
        return True

    log_callable(f"{qubit_name}: skip FIR update — analysis unavailable or did not succeed")
    return False


def update_filters(
    qubits: Iterable[Any],
    machine: Any,
    fit_results: dict[str, Any],
    *,
    update_iir: bool = False,
    update_fir: bool = False,
    fir_results: Optional[dict[str, Any]] = None,
    skip_qubits: Optional[Set[str]] = None,
    log_callable: LogCallable = print,
) -> None:
    """Write flux-line filters to each qubit's Z ``opx_output``.

    Call inside ``node.record_state_updates()`` after the node has checked
    ``update_state``. ``update_iir`` and/or ``update_fir`` select which
    filters are written.
    """
    if not update_iir and not update_fir:
        return

    skip = skip_qubits or set()
    fir_by_qubit = fir_results or {}

    if update_iir:
        _init_exponential_filters(qubits, machine)

    for q in qubits:
        if q.name in skip:
            continue
        z_out = machine.qubits[q.name].z.opx_output
        if update_iir:
            _append_iir_taps_from_fit(
                z_out,
                fit_results.get(q.name),
                qubit_name=q.name,
                log_callable=log_callable,
            )
        if update_fir:
            _set_feedforward_filter_from_fit(
                z_out,
                fir_by_qubit.get(q.name),
                qubit_name=q.name,
                log_callable=log_callable,
            )
