"""Resolve nominal pulse lengths for timestamp interval reporting."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from _timestamp_loops import _literal_value
from _timestamp_proto import _get_attribute


def resolve_operation_duration_ns(
    *,
    operation_type: str,
    pulse: str,
    element: str,
    config: Optional[Mapping[str, Any]],
    duration_cycles: Optional[int],
    clock_cycle_ns: float,
) -> Optional[float]:
    """Return nominal operation length in nanoseconds."""
    if duration_cycles is not None:
        return float(duration_cycles) * clock_cycle_ns
    if config is None or pulse in {"<dynamic>", "<unknown>"}:
        return None
    return pulse_length_ns_from_config(config, element, pulse)


def pulse_length_ns_from_config(config: Mapping[str, Any], element_name: str, operation_name: str) -> Optional[float]:
    elements = config.get("elements", {})
    element = elements.get(element_name, {})
    if not isinstance(element, dict):
        return None

    operations = element.get("operations", {})
    if not isinstance(operations, dict):
        return None

    pulse_key = operations.get(operation_name)
    if pulse_key is None:
        return None

    pulses = config.get("pulses", {})
    if not isinstance(pulses, dict):
        return None

    pulse = pulses.get(pulse_key, {})
    if not isinstance(pulse, dict):
        return None

    length = pulse.get("length")
    if length is None:
        return None
    return float(length)


def element_time_of_flight_ns(config: Mapping[str, Any], element_name: str) -> Optional[float]:
    elements = config.get("elements", {})
    element = elements.get(element_name, {})
    if not isinstance(element, dict):
        return None
    time_of_flight = element.get("time_of_flight")
    if time_of_flight is None:
        return None
    return float(time_of_flight)


def play_duration_cycles(statement_body: Any) -> Optional[int]:
    duration_expr = _get_attribute(statement_body, "duration", default=None)
    if duration_expr is None:
        return None
    if hasattr(duration_expr, "WhichOneof"):
        if duration_expr.WhichOneof("expression_oneof") != "literal":
            return None
    literal = _literal_value(duration_expr)
    if literal is None:
        return None
    return int(literal)
