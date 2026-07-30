"""
Automatic play/measure timestamps for existing QUA programs.

Physicist usage (see ``08g_timestamp_lazy_minimal.py``)::

    timing = TimestampRecorder(my_program, config=config)
    job = qm.execute(timing.program)
    result = timing.fetch(job, wait_for_all=False, timeout_s=120)
    result.print_shot()          # first sweep point
    result.print_shot((1, 2))    # avg index 1, inner-axis index 2

Hardware timestamps mark operation **starts**. Pass ``config`` so ``print_shot``
can add nominal pulse lengths from the QUA config and report end times and
gaps from pulse end to the next operation start.

Your QUA source is not edited. The recorder clones the compiled program and
adds timestamp streams to every ``play`` and ``measure``. Loop structure is
inferred so one sweep point can be selected even when the program runs a full
2D/3D sweep.

Requires QOP 2.2+ (``command_timestamps`` capability).
"""

import copy
import re
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from qm import QopCaps
from qm.program.program import Program
from qm.qua import play, program

from _timestamp_loops import (
    LoopIndexMapper,
    LoopIndices,
    OperationLoopLayout,
    Reference,
)
from _timestamp_durations import (
    element_time_of_flight_ns,
    play_duration_cycles,
    resolve_operation_duration_ns,
)
from _timestamp_proto import (
    _all_strings,
    _clone_message,
    _get_attribute,
    _set_attribute,
    _statement_type,
    _walk_messages,
)


@dataclass(frozen=True)
class _OperationSpec:
    name: str
    operation_type: str
    pulse: str
    element: str
    handle: str
    statement_path: str
    duration_ns: Optional[float] = None
    time_of_flight_ns: Optional[float] = None


@dataclass(frozen=True)
class OperationTimestamps:
    """Fetched timestamps for one static QUA play or measure statement."""

    name: str
    operation_type: str
    pulse: str
    element: str
    handle: str
    statement_path: str
    clock_cycles: np.ndarray
    nanoseconds: np.ndarray
    duration_ns: Optional[float] = None
    time_of_flight_ns: Optional[float] = None

    @property
    def occurrences(self) -> int:
        """Number of times this statement executed, including loop iterations."""
        return self.clock_cycles.size


class TimestampResults:
    """Timestamp data indexed by generated operation name or statement index."""

    def __init__(
        self,
        operations: List[OperationTimestamps],
        clock_cycle_ns: Real,
        loop_mapper: Optional[LoopIndexMapper] = None,
    ):
        self._operations = tuple(operations)
        self._by_name = {operation.name: operation for operation in operations}
        self.clock_cycle_ns = clock_cycle_ns
        self.loop_mapper = loop_mapper

    def __getitem__(self, reference: Reference) -> OperationTimestamps:
        if isinstance(reference, int):
            return self._operations[reference]
        return self._by_name[reference]

    def __iter__(self) -> Iterator[OperationTimestamps]:
        return iter(self._operations)

    def __len__(self) -> int:
        return len(self._operations)

    @property
    def names(self) -> Tuple[str, ...]:
        """Generated operation names in QUA statement order."""
        return tuple(operation.name for operation in self._operations)

    @property
    def clock_cycles(self) -> Dict[str, np.ndarray]:
        """Raw timestamp arrays keyed by generated operation name."""
        return {operation.name: operation.clock_cycles for operation in self._operations}

    @property
    def nanoseconds(self) -> Dict[str, np.ndarray]:
        """Timestamp arrays converted to nanoseconds."""
        return {operation.name: operation.nanoseconds for operation in self._operations}

    def relative_to(self, reference: Reference, occurrence: int = 0) -> Dict[str, np.ndarray]:
        """Return every timestamp relative to one occurrence of an operation."""
        reference_time = self._reference_time(reference, occurrence)
        return {operation.name: operation.nanoseconds - reference_time for operation in self._operations}

    def loop_layout(self, reference: Reference) -> OperationLoopLayout:
        if self.loop_mapper is None:
            raise ValueError("Loop layouts are only available when TimestampRecorder infers loop structure.")
        return self.loop_mapper.layout(reference)

    def expected_occurrences(self, reference: Reference) -> int:
        if self.loop_mapper is None:
            raise ValueError("Loop layouts are only available when TimestampRecorder infers loop structure.")
        return self.loop_mapper.expected_occurrences(reference)

    def occurrence_at(self, reference: Reference, loop_indices: LoopIndices) -> int:
        if self.loop_mapper is None:
            raise ValueError("Loop layouts are only available when TimestampRecorder infers loop structure.")
        return self.loop_mapper.coords_to_flat(reference, loop_indices)

    def coords_at(self, reference: Reference, occurrence: int) -> Dict[int, int]:
        if self.loop_mapper is None:
            raise ValueError("Loop layouts are only available when TimestampRecorder infers loop structure.")
        return self.loop_mapper.flat_to_coords(reference, occurrence)

    def select_shot(self, loop_indices: LoopIndices, *, reference: Optional[Reference] = None) -> Dict[str, Dict[str, Any]]:
        if self.loop_mapper is None:
            raise ValueError("Loop layouts are only available when TimestampRecorder infers loop structure.")
        return self.loop_mapper.select_shot(self, loop_indices, reference=reference)

    def print_shot(
        self,
        shot: LoopIndices = 0,
        *,
        reference: Optional[Reference] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Print play/measure timing for one sweep point and return the selected data."""
        return print_timing_summary(self, shot, reference=reference)

    def as_rows(
        self,
        reference: Optional[Reference] = None,
        reference_occurrence: int = 0,
        *,
        include_loop_coords: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return one row per executed occurrence, sorted by absolute timestamp.

        An integer ``reference`` selects an operation by its statement order.
        A string selects it by its generated name.
        """
        reference_time = None
        if reference is not None:
            reference_time = self._reference_time(reference, reference_occurrence)

        rows = []
        for statement_index, operation in enumerate(self._operations):
            for occurrence, (clock_cycle, time_ns) in enumerate(zip(operation.clock_cycles, operation.nanoseconds)):
                rows.append(
                    {
                        "name": operation.name,
                        "operation_type": operation.operation_type,
                        "pulse": operation.pulse,
                        "element": operation.element,
                        "statement_path": operation.statement_path,
                        "occurrence": occurrence,
                        "clock_cycle": _python_scalar(clock_cycle),
                        "time_ns": _python_scalar(time_ns),
                        "relative_ns": (None if reference_time is None else _python_scalar(time_ns - reference_time)),
                        "_statement_index": statement_index,
                    }
                )
                if include_loop_coords and self.loop_mapper is not None:
                    try:
                        rows[-1]["loop_coords"] = self.loop_mapper.flat_to_coords(operation.name, occurrence)
                    except (IndexError, KeyError, ValueError):
                        rows[-1]["loop_coords"] = None

        rows.sort(key=lambda row: (row["time_ns"], row["_statement_index"], row["occurrence"]))
        for row in rows:
            del row["_statement_index"]
        return rows

    def _reference_time(self, reference: Reference, occurrence: int) -> Real:
        operation = self[reference]
        if operation.occurrences == 0:
            raise ValueError(f"Operation '{operation.name}' has no timestamp values.")
        try:
            return operation.nanoseconds[occurrence]
        except IndexError as exc:
            raise IndexError(
                f"Occurrence {occurrence} does not exist for operation '{operation.name}' "
                f"with {operation.occurrences} occurrence(s)."
            ) from exc


class TimestampRecorder:
    """
    Instrument a completed QUA program without changing its source statements.

    ``program`` is an instrumented clone; the input program is not mutated.
    Every nested play and measure statement receives an automatically generated
    timestamp stream and operation name.

    Example:
        timing = TimestampRecorder(power_rabi, config=config)
        job = qm.execute(timing.program)
        result = timing.fetch(job, wait_for_all=False, timeout_s=120)
        result.print_shot()
    """

    def __init__(
        self,
        qua_program: Program,
        clock_cycle_ns: Real = 4,
        stream_prefix: str = "qua_timestamps",
        config: Optional[Mapping[str, Any]] = None,
    ):
        if not isinstance(qua_program, Program):
            raise TypeError("qua_program must be a completed qm.program.Program.")
        if qua_program.is_in_scope():
            raise RuntimeError("Finish the QUA program context before creating TimestampRecorder.")
        if not isinstance(clock_cycle_ns, Real) or clock_cycle_ns <= 0:
            raise ValueError("clock_cycle_ns must be a positive number.")
        if not isinstance(stream_prefix, str) or not stream_prefix:
            raise ValueError("stream_prefix must be a non-empty string.")

        self.clock_cycle_ns = clock_cycle_ns
        self.stream_prefix = stream_prefix
        self.config = config
        self._operations: List[_OperationSpec] = []
        self._loop_mapper = LoopIndexMapper.from_program(qua_program)
        self.program = self._instrument_program(qua_program)
        self._loop_mapper = self._loop_mapper.attach_operation_names(self._operations)

    @property
    def loop_mapper(self) -> LoopIndexMapper:
        return self._loop_mapper

    @property
    def names(self) -> Tuple[str, ...]:
        """Generated operation names in QUA statement order."""
        return tuple(operation.name for operation in self._operations)

    @property
    def handles(self) -> Dict[str, str]:
        """Map generated operation names to QUA result handles."""
        return {operation.name: operation.handle for operation in self._operations}

    def fetch(
        self,
        job: Any,
        wait_for_all: bool = True,
        timeout_s: Optional[Real] = None,
    ) -> TimestampResults:
        """Fetch every automatically generated timestamp result handle."""
        result_handles = job.result_handles
        if wait_for_all:
            result_handles.wait_for_all_values()
        elif timeout_s is not None:
            self.wait_for_timestamps(job, timeout_s=timeout_s)

        return self._read_timestamp_results(result_handles)

    def wait_for_timestamps(self, job: Any, timeout_s: Real = 60, poll_interval_s: Real = 0.1) -> None:
        """Block until every timestamp handle has at least one value."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._timestamp_handles_ready(job.result_handles):
                return
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Timed out after {timeout_s} s waiting for timestamp handles: {tuple(self.handles.values())}"
        )

    def _timestamp_handles_ready(self, result_handles: Any) -> bool:
        for operation in self._operations:
            result_handle = result_handles.get(operation.handle)
            if result_handle is None:
                return False
            try:
                values = _timestamp_values(result_handle.fetch_all())
            except Exception:
                return False
            if values.size == 0:
                return False
        return True

    def _read_timestamp_results(self, result_handles: Any) -> TimestampResults:
        operations = []
        for operation in self._operations:
            result_handle = result_handles.get(operation.handle)
            if result_handle is None:
                raise KeyError(
                    f"Timestamp result handle '{operation.handle}' for operation " f"'{operation.name}' was not found."
                )

            clock_cycles = _timestamp_values(result_handle.fetch_all())
            nanoseconds = clock_cycles * self.clock_cycle_ns
            clock_cycles.setflags(write=False)
            nanoseconds.setflags(write=False)
            operations.append(
                OperationTimestamps(
                    name=operation.name,
                    operation_type=operation.operation_type,
                    pulse=operation.pulse,
                    element=operation.element,
                    handle=operation.handle,
                    statement_path=operation.statement_path,
                    clock_cycles=clock_cycles,
                    nanoseconds=nanoseconds,
                    duration_ns=operation.duration_ns,
                    time_of_flight_ns=operation.time_of_flight_ns,
                )
            )
        return TimestampResults(operations, self.clock_cycle_ns, loop_mapper=self._loop_mapper)

    def _instrument_program(self, qua_program: Program) -> Program:
        instrumented_proto = _clone_message(qua_program.qua_program)
        statements = [
            (path, statement, statement_type)
            for path, statement in _walk_messages(instrumented_proto)
            if (statement_type := _statement_type(statement)) in {"play", "measure"}
        ]

        template_stream, template_model = _timestamp_stream_template()
        existing_strings = set(_all_strings(instrumented_proto))
        stream_index = _next_stream_index(existing_strings)

        for operation_index, (path, statement, statement_type) in enumerate(statements):
            statement_body = getattr(statement, statement_type)
            existing_timestamp = _get_attribute(statement_body, "timestamp_label", "timestampLabel")
            if existing_timestamp:
                raise ValueError(
                    f"Statement at '{path}' already has timestamp stream '{existing_timestamp}'. "
                    "TimestampRecorder expects a program without manual timestamps."
                )

            stream_name = f"r{stream_index}"
            stream_index += 1
            handle = _unique_handle(f"{self.stream_prefix}_{operation_index}", existing_strings)
            existing_strings.update({stream_name, handle})

            _set_attribute(statement_body, stream_name, "timestamp_label", "timestampLabel")
            result_model = _clone_message(template_model)
            _replace_result_model_values(result_model, handle, template_stream, stream_name)
            _append_result_model(instrumented_proto, result_model)

            pulse = _operation_pulse(statement_body, statement_type)
            element = _nested_name(statement_body, "qe")
            duration_cycles = play_duration_cycles(statement_body) if statement_type == "play" else None
            duration_ns = resolve_operation_duration_ns(
                operation_type=statement_type,
                pulse=pulse,
                element=element,
                config=self.config,
                duration_cycles=duration_cycles,
                clock_cycle_ns=float(self.clock_cycle_ns),
            )
            time_of_flight_ns = (
                element_time_of_flight_ns(self.config, element)
                if statement_type == "measure" and self.config is not None
                else None
            )
            self._operations.append(
                _OperationSpec(
                    name=_generated_operation_name(statement_type, operation_index, pulse, element),
                    operation_type=statement_type,
                    pulse=pulse,
                    element=element,
                    handle=handle,
                    statement_path=path,
                    duration_ns=duration_ns,
                    time_of_flight_ns=time_of_flight_ns,
                )
            )

        capabilities = set(qua_program.used_capabilities)
        capabilities.add(QopCaps.command_timestamps)
        instrumented_program = Program()
        instrumented_program._set_and_exit(instrumented_proto, capabilities)
        return instrumented_program


def print_timing_summary(
    results: TimestampResults,
    shot: LoopIndices = 0,
    *,
    reference: Optional[Reference] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Print the timing of one sweep point from a multi-loop program.

    ``shot=0`` (or ``shot='first'``) selects the first point on every static axis.
    Pass a tuple for explicit indices, e.g. ``shot=(2, 4)`` for avg index 2 and
    amplitude index 4. Dynamic inner loops return every play at that point.

    When operation durations are known (via ``config`` on ``TimestampRecorder`` or
    an inline ``duration=`` on ``play``), the report also shows pulse ends and
    gaps from one pulse end to the next operation start.
    """
    if results.loop_mapper is None:
        raise ValueError("Loop layouts are only available when TimestampRecorder infers loop structure.")

    measure_name = reference
    if measure_name is None or (
        isinstance(measure_name, str) and not measure_name.startswith("measure_")
    ):
        measure_name = next(name for name in results.names if name.startswith("measure_"))

    selected = results.select_shot(shot, reference=measure_name)
    play_names = sorted(name for name in selected if name.startswith("play_"))
    measure_data = selected[measure_name]
    axis_indices = tuple(
        measure_data["loop_indices"][axis.variable]
        for axis in results.loop_mapper.layout(measure_name).axes
        if axis.kind != "dynamic_range"
    )
    events = _shot_events(results, selected, play_names, measure_name, measure_data)

    print(f"\nTiming at sweep point {axis_indices}")
    print("-------------------------------------------------------------")
    for index, event in enumerate(events):
        _print_shot_event(event)
        if index + 1 < len(events):
            gap_ns = _gap_to_next_start(event, events[index + 1]["start_ns"])
            if gap_ns is not None:
                print(f"  gap (end -> next start): {gap_ns:.0f} ns")
    print("-------------------------------------------------------------")

    selected["_intervals"] = events
    return selected


def _shot_events(
    results: TimestampResults,
    selected: Dict[str, Dict[str, Any]],
    play_names: Sequence[str],
    measure_name: str,
    measure_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for play_name in play_names:
        operation = results[play_name]
        play_data = selected[play_name]
        if "nanoseconds" in play_data:
            for occurrence, start_ns in enumerate(np.asarray(play_data["nanoseconds"], dtype=float)):
                events.append(
                    {
                        "kind": "play",
                        "name": play_name,
                        "label": f"{operation.pulse} @ {operation.element}",
                        "occurrence": occurrence,
                        "start_ns": float(start_ns),
                        "duration_ns": operation.duration_ns,
                    }
                )
        else:
            events.append(
                {
                    "kind": "play",
                    "name": play_name,
                    "label": f"{operation.pulse} @ {operation.element}",
                    "occurrence": 0,
                    "start_ns": float(play_data["time_ns"]),
                    "duration_ns": operation.duration_ns,
                }
            )

    measure_operation = results[measure_name]
    events.append(
        {
            "kind": "measure",
            "name": measure_name,
            "label": f"{measure_operation.pulse} @ {measure_operation.element}",
            "occurrence": 0,
            "start_ns": float(measure_data["time_ns"]),
            "duration_ns": measure_operation.duration_ns,
            "time_of_flight_ns": measure_operation.time_of_flight_ns,
        }
    )
    return events


def _gap_to_next_start(event: Dict[str, Any], next_start_ns: float) -> Optional[float]:
    duration_ns = event.get("duration_ns")
    if duration_ns is None:
        return None
    return next_start_ns - (event["start_ns"] + duration_ns)


def _print_shot_event(event: Dict[str, Any]) -> None:
    label = event["label"]
    if event.get("occurrence", 0) > 0:
        label = f"{label}[{event['occurrence']}]"
    duration_ns = event.get("duration_ns")
    start_ns = event["start_ns"]
    if duration_ns is None:
        print(f"  {label}: start {start_ns:.0f} ns, length ?")
        return
    end_ns = start_ns + duration_ns
    print(f"  {label}: start {start_ns:.0f} ns, length {duration_ns:.0f} ns, end {end_ns:.0f} ns")
    if event.get("kind") == "measure" and event.get("time_of_flight_ns") is not None:
        adc_start_ns = start_ns + float(event["time_of_flight_ns"])
        print(f"    ADC integration start (start + ToF): {adc_start_ns:.0f} ns")


def _timestamp_stream_template() -> Tuple[str, Any]:
    """Let the installed SDK generate one valid timestamp result model."""
    with program() as template_program:
        play(
            "__qua_timestamp_template_pulse__",
            "__qua_timestamp_template_element__",
            timestamp_stream="__qua_timestamp_template_handle__",
        )

    template_statement = next(
        statement
        for _, statement in _walk_messages(template_program.qua_program)
        if _statement_type(statement) == "play"
    )
    template_stream = _get_attribute(
        getattr(template_statement, "play"),
        "timestamp_label",
        "timestampLabel",
    )
    result_analysis = _get_attribute(template_program.qua_program, "result_analysis", "resultAnalysis")
    return template_stream, result_analysis.model[0]


def _append_result_model(qua_proto: Any, result_model: Any) -> None:
    result_analysis = _get_attribute(qua_proto, "result_analysis", "resultAnalysis")
    result_analysis.model.append(copy.deepcopy(result_model))


def _replace_result_model_values(
    result_model: Any,
    handle: str,
    old_stream: str,
    new_stream: str,
) -> None:
    replaced_handle = False
    replaced_stream = False
    for value_message, value in _string_value_fields(result_model):
        if value == "__qua_timestamp_template_handle__":
            _set_attribute(value_message, handle, "string_value", "stringValue")
            replaced_handle = True
        elif value == old_stream:
            _set_attribute(value_message, new_stream, "string_value", "stringValue")
            replaced_stream = True
    if not replaced_handle or not replaced_stream:
        raise RuntimeError("The QUA timestamp result model has an unsupported structure.")


def _string_value_fields(message: Any) -> Iterator[Tuple[Any, str]]:
    for _, candidate in _walk_messages(message):
        value = _get_attribute(candidate, "string_value", "stringValue", default=None)
        if isinstance(value, str) and value:
            yield candidate, value


def _next_stream_index(existing_strings: Set[str]) -> int:
    used_indices = [
        int(match.group(1)) for value in existing_strings if (match := re.fullmatch(r"r(\d+)", value)) is not None
    ]
    return max(used_indices, default=0) + 1


def _unique_handle(base_handle: str, existing_strings: Set[str]) -> str:
    handle = base_handle
    suffix = 1
    while handle in existing_strings:
        handle = f"{base_handle}_{suffix}"
        suffix += 1
    return handle


def _operation_pulse(statement_body: Any, statement_type: str) -> str:
    if statement_type == "play":
        pulse_reference = _get_attribute(statement_body, "named_pulse", "namedPulse", default=None)
        if pulse_reference is None or not _get_attribute(pulse_reference, "name", default=""):
            return "<dynamic>"
    else:
        pulse_reference = _get_attribute(statement_body, "pulse")
    return _get_attribute(pulse_reference, "name", default="<unknown>")


def _nested_name(message: Any, attribute: str) -> str:
    nested = _get_attribute(message, attribute, default=None)
    return "<unknown>" if nested is None else _get_attribute(nested, "name", default="<unknown>")


def _generated_operation_name(
    statement_type: str,
    operation_index: int,
    pulse: str,
    element: str,
) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]+", "_", f"{pulse}_{element}").strip("_")
    return f"{statement_type}_{operation_index}_{suffix}" if suffix else f"{statement_type}_{operation_index}"


def _timestamp_values(raw_result: Any) -> np.ndarray:
    if isinstance(raw_result, Mapping) and "value" in raw_result:
        raw_result = raw_result["value"]
    elif getattr(getattr(raw_result, "dtype", None), "names", None) and "value" in raw_result.dtype.names:
        raw_result = raw_result["value"]
    return np.array(raw_result, dtype=np.int64, copy=True).reshape(-1)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


# Physicists only need TimestampRecorder; everything else is internal or advanced.
__all__ = ["TimestampRecorder"]
