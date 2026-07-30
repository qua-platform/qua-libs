"""
Local prototype for automatic QUA play/measure timestamp instrumentation.

This module intentionally lives beside the experiment so it can be inspected,
modified and tested without installing a development version of
``qualang_tools``. If the API proves stable on hardware, it can later be
upstreamed into that package.

The QUA SDK currently has no public post-build instrumentation API. This
prototype therefore isolates one internal dependency: it clones the completed
QUA protobuf, recursively adds timestamp streams to every play and measure
statement, and preserves the original program unchanged.
"""

import copy
import dataclasses
import re
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from betterproto import Message
from qm import QopCaps
from qm.program.program import Program
from qm.qua import play, program


Reference = Union[str, int]
LoopIndices = Union[int, str, Mapping[int, int], Mapping[str, int], Sequence[int]]


@dataclass(frozen=True)
class LoopAxis:
    """One enclosing loop axis in outer-to-inner order."""

    variable: str
    kind: str
    size: Optional[int] = None
    values: Optional[Tuple[Any, ...]] = None
    start: Real = 0
    step: Real = 1
    stop: Optional[Real] = None
    stop_variable: Optional[str] = None

    @property
    def iteration_count(self) -> Optional[int]:
        if self.kind == "for_each":
            return len(self.values) if self.values is not None else None
        return self.size


@dataclass(frozen=True)
class OperationLoopLayout:
    """Loop axes that determine how many times one statement executes."""

    operation_name: str
    statement_path: str
    axes: Tuple[LoopAxis, ...]

    @property
    def axis_variables(self) -> Tuple[str, ...]:
        return tuple(axis.variable for axis in self.axes)

    def expected_occurrences(self, variable_arrays: Mapping[str, Tuple[Any, ...]]) -> int:
        return LoopIndexMapper._expected_occurrences(self.axes, variable_arrays)


class LoopIndexMapper:
    """Infer nested QUA loop structure and map flat timestamp indices to loop coordinates."""

    def __init__(self, layouts: Mapping[str, OperationLoopLayout], variable_arrays: Mapping[str, Tuple[Any, ...]]):
        self._layouts = dict(layouts)
        self.variable_arrays = dict(variable_arrays)
        self._layouts_by_path = {layout.statement_path: layout for layout in self._layouts.values()}

    @classmethod
    def from_program(cls, qua_program: Program) -> "LoopIndexMapper":
        variable_arrays = _collect_variable_arrays(qua_program.qua_program)
        layouts: Dict[str, OperationLoopLayout] = {}
        _walk_statement_tree(
            qua_program.qua_program.script.body.statements,
            [],
            "program.script.body",
            variable_arrays,
            layouts,
        )
        return cls(layouts, variable_arrays)

    def attach_operation_names(self, operations: Sequence[Any]) -> "LoopIndexMapper":
        layouts = {}
        for operation in operations:
            layout = self._layouts_by_path.get(operation.statement_path)
            if layout is None:
                continue
            layouts[operation.name] = OperationLoopLayout(
                operation.name,
                operation.statement_path,
                layout.axes,
            )
        return LoopIndexMapper(layouts, self.variable_arrays)

    def layout(self, reference: Reference) -> OperationLoopLayout:
        return self._resolve_layout(reference)

    def expected_occurrences(self, reference: Reference) -> int:
        layout = self._resolve_layout(reference)
        return self._expected_occurrences(layout.axes, self.variable_arrays)

    def coords_to_flat(self, reference: Reference, loop_indices: LoopIndices) -> int:
        layout = self._resolve_layout(reference)
        indices = self._normalize_indices(layout, loop_indices)
        return self._coords_to_flat(layout.axes, indices, self.variable_arrays)

    def flat_to_coords(self, reference: Reference, occurrence: int) -> Dict[int, int]:
        layout = self._resolve_layout(reference)
        return self._flat_to_coords(layout.axes, occurrence, self.variable_arrays)

    def select_shot(
        self,
        results: "TimestampResults",
        loop_indices: LoopIndices,
        *,
        reference: Optional[Reference] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return one sweep-point worth of timestamps for every operation.

        ``loop_indices`` selects coordinates on the static / for_each axes shared by
        all operations. Dynamic inner loops (for example ``n2 < n_rabi``) return all
        play timestamps at that sweep point.
        """
        layouts = list(self._layouts.values())
        if not layouts:
            raise ValueError("No loop layouts are available for this program.")

        anchor_layout = self._resolve_layout(reference or layouts[0].operation_name)
        anchor_indices = self._normalize_indices(anchor_layout, loop_indices)
        self._validate_shared_indices(anchor_layout, anchor_indices)
        anchor_occurrence = self._coords_to_flat(anchor_layout.axes, anchor_indices, self.variable_arrays)

        shot: Dict[str, Dict[str, Any]] = {}
        for layout in self._layouts.values():
            indices = self._indices_for_layout(layout, anchor_layout, anchor_indices)
            if self._has_dynamic_axes(layout.axes):
                start, stop = self._dynamic_play_range(layout, anchor_layout, anchor_occurrence)
                shot[layout.operation_name] = {
                    "loop_indices": self._coords_dict(layout, indices),
                    "occurrence": slice(start, stop),
                    "clock_cycles": results[layout.operation_name].clock_cycles[start:stop],
                    "nanoseconds": results[layout.operation_name].nanoseconds[start:stop],
                }
            else:
                occurrence = self._coords_to_flat(layout.axes, indices, self.variable_arrays)
                shot[layout.operation_name] = {
                    "loop_indices": self._coords_dict(layout, indices),
                    "occurrence": occurrence,
                    "clock_cycle": int(results[layout.operation_name].clock_cycles[occurrence]),
                    "time_ns": float(results[layout.operation_name].nanoseconds[occurrence]),
                }
        return shot

    def _dynamic_play_range(
        self,
        play_layout: OperationLoopLayout,
        measure_layout: OperationLoopLayout,
        measure_occurrence: int,
    ) -> Tuple[int, int]:
        dynamic_axis = next(axis for axis in play_layout.axes if axis.kind == "dynamic_range")
        plays_before = 0
        for occurrence in range(measure_occurrence):
            measure_coords = self._flat_to_coords(measure_layout.axes, occurrence, self.variable_arrays)
            play_coords = self._indices_for_layout(play_layout, measure_layout, measure_coords)
            plays_before += self._axis_iteration_count(
                dynamic_axis,
                play_coords,
                play_layout.axes,
                self.variable_arrays,
            )

        measure_coords = self._flat_to_coords(measure_layout.axes, measure_occurrence, self.variable_arrays)
        play_coords = self._indices_for_layout(play_layout, measure_layout, measure_coords)
        plays_in_shot = self._axis_iteration_count(
            dynamic_axis,
            play_coords,
            play_layout.axes,
            self.variable_arrays,
        )
        return plays_before, plays_before + plays_in_shot

    def _resolve_layout(self, reference: Reference) -> OperationLoopLayout:
        if isinstance(reference, int):
            raise TypeError("Loop layouts are keyed by operation name or statement path.")
        if reference in self._layouts:
            return self._layouts[reference]
        if reference in self._layouts_by_path:
            return self._layouts_by_path[reference]
        raise KeyError(f"No loop layout found for {reference!r}.")

    def _normalize_indices(self, layout: OperationLoopLayout, loop_indices: LoopIndices) -> Dict[int, int]:
        if loop_indices in (0, "first"):
            return {
                index: 0
                for index, axis in enumerate(layout.axes)
                if axis.kind != "dynamic_range"
            }
        if isinstance(loop_indices, int):
            raise ValueError("Use shot=0 or shot='first' for the first sweep point.")
        if isinstance(loop_indices, Mapping):
            normalized: Dict[int, int] = {}
            name_to_index = {axis.variable: index for index, axis in enumerate(layout.axes)}
            for key, value in loop_indices.items():
                if isinstance(key, str):
                    if key not in name_to_index:
                        raise KeyError(f"Loop variable '{key}' is not part of {layout.operation_name}.")
                    normalized[name_to_index[key]] = int(value)
                else:
                    normalized[int(key)] = int(value)
            return normalized
        return {index: int(value) for index, value in enumerate(loop_indices)}

    def _coords_dict(self, layout: OperationLoopLayout, indices: Mapping[int, int]) -> Dict[str, int]:
        return {layout.axes[index].variable: indices[index] for index in sorted(indices)}

    def _validate_shared_indices(self, anchor_layout: OperationLoopLayout, indices: Mapping[int, int]) -> None:
        for axis_index, axis in enumerate(anchor_layout.axes):
            if axis.kind == "dynamic_range":
                continue
            if axis_index not in indices:
                raise KeyError(f"Missing loop index for axis '{axis.variable}'.")
            if axis.kind == "for_each":
                if not 0 <= indices[axis_index] < len(axis.values or ()):
                    raise IndexError(f"Axis '{axis.variable}' index {indices[axis_index]} is out of range.")
            elif axis.size is not None and not 0 <= indices[axis_index] < axis.size:
                raise IndexError(f"Axis '{axis.variable}' index {indices[axis_index]} is out of range.")

    def _indices_for_layout(
        self,
        layout: OperationLoopLayout,
        anchor_layout: OperationLoopLayout,
        anchor_indices: Mapping[int, int],
    ) -> Dict[int, int]:
        indices: Dict[int, int] = {}
        anchor_by_var = {axis.variable: axis for axis in anchor_layout.axes}
        for axis_index, axis in enumerate(layout.axes):
            if axis.kind == "dynamic_range":
                indices[axis_index] = 0
                continue
            if axis.variable not in anchor_by_var:
                raise ValueError(
                    f"Operation '{layout.operation_name}' has sweep axis '{axis.variable}' "
                    f"that is not shared with '{anchor_layout.operation_name}'."
                )
            anchor_index = anchor_layout.axes.index(anchor_by_var[axis.variable])
            indices[axis_index] = anchor_indices[anchor_index]
        return indices

    @staticmethod
    def _has_dynamic_axes(axes: Sequence[LoopAxis]) -> bool:
        return any(axis.kind == "dynamic_range" for axis in axes)

    @classmethod
    def _expected_occurrences(cls, axes: Sequence[LoopAxis], variable_arrays: Mapping[str, Tuple[Any, ...]]) -> int:
        if not cls._has_dynamic_axes(axes):
            total = 1
            for axis in axes:
                if axis.iteration_count is None:
                    raise ValueError(f"Cannot infer iteration count for axis '{axis.variable}'.")
                total *= axis.iteration_count
            return total

        def walk(axis_index: int, indices: Dict[int, int]) -> int:
            if axis_index == len(axes):
                return 1
            axis = axes[axis_index]
            if axis.kind == "dynamic_range":
                return cls._axis_iteration_count(axis, indices, axes, variable_arrays)
            total = 0
            axis_size = cls._axis_iteration_count(axis, indices, axes, variable_arrays)
            for idx in range(axis_size):
                child_indices = dict(indices)
                child_indices[axis_index] = idx
                total += walk(axis_index + 1, child_indices)
            return total

        return walk(0, {})

    @classmethod
    def _coords_to_flat(
        cls,
        axes: Sequence[LoopAxis],
        indices: Mapping[int, int],
        variable_arrays: Mapping[str, Tuple[Any, ...]],
    ) -> int:
        flat = 0
        stride = 1
        for axis_index in reversed(range(len(axes))):
            axis = axes[axis_index]
            if axis_index not in indices:
                raise KeyError(f"Missing loop index for axis '{axis.variable}'.")
            size = cls._axis_iteration_count(axis, indices, axes, variable_arrays)
            idx = indices[axis_index]
            if idx < 0 or idx >= size:
                raise IndexError(f"Axis '{axis.variable}' index {idx} is out of range (size {size}).")
            flat += idx * stride
            stride *= size
        return flat

    @classmethod
    def _flat_to_coords(
        cls,
        axes: Sequence[LoopAxis],
        occurrence: int,
        variable_arrays: Mapping[str, Tuple[Any, ...]],
    ) -> Dict[int, int]:
        if occurrence < 0:
            raise IndexError(f"Occurrence {occurrence} is negative.")
        indices: Dict[int, int] = {}
        remaining = occurrence
        for axis_index in reversed(range(len(axes))):
            axis = axes[axis_index]
            size = cls._axis_iteration_count(axis, indices, axes, variable_arrays)
            if size <= 0:
                raise ValueError(f"Axis '{axis.variable}' has non-positive iteration count.")
            indices[axis_index] = remaining % size
            remaining //= size
        if remaining != 0:
            raise IndexError(f"Occurrence {occurrence} is outside the inferred loop layout.")
        return indices

    @classmethod
    def _dynamic_block_size(
        cls,
        axes: Sequence[LoopAxis],
        indices: Mapping[int, int],
        variable_arrays: Mapping[str, Tuple[Any, ...]],
    ) -> int:
        for axis_index in reversed(range(len(axes))):
            axis = axes[axis_index]
            if axis.kind == "dynamic_range":
                return cls._axis_iteration_count(axis, indices, axes, variable_arrays)
        return 1

    @classmethod
    def _axis_iteration_count(
        cls,
        axis: LoopAxis,
        indices: Mapping[int, int],
        axes: Sequence[LoopAxis],
        variable_arrays: Mapping[str, Tuple[Any, ...]],
    ) -> int:
        if axis.kind == "static_range":
            if axis.size is None:
                raise ValueError(f"Static axis '{axis.variable}' is missing a size.")
            return axis.size
        if axis.kind == "for_each":
            if axis.values is None:
                raise ValueError(f"for_each axis '{axis.variable}' is missing values.")
            return len(axis.values)
        if axis.kind != "dynamic_range" or axis.stop_variable is None:
            raise ValueError(f"Unsupported axis '{axis.variable}' of kind '{axis.kind}'.")

        stop_value = cls._resolve_variable_value(axis.stop_variable, indices, axes, variable_arrays)
        start = axis.start
        step = axis.step or 1
        if stop_value < start:
            return 0
        if step == 1:
            return stop_value - start
        return (stop_value - start + step - 1) // step

    @classmethod
    def _resolve_variable_value(
        cls,
        variable: str,
        indices: Mapping[int, int],
        axes: Sequence[LoopAxis],
        variable_arrays: Mapping[str, Tuple[Any, ...]],
    ) -> int:
        for axis_index, axis in enumerate(axes):
            if axis.variable != variable:
                continue
            if axis_index not in indices:
                raise KeyError(f"Loop index for '{variable}' is required before resolving dynamic axes.")
            if axis.kind == "for_each":
                return int(axis.values[indices[axis_index]])  # type: ignore[index]
            if axis.kind == "static_range":
                return int(axis.start + indices[axis_index] * axis.step)
            return int(indices[axis_index])
        if variable in variable_arrays:
            raise ValueError(f"Variable '{variable}' is an array, not a scalar loop index.")
        raise KeyError(f"Could not resolve loop variable '{variable}'.")


@dataclass(frozen=True)
class _OperationSpec:
    name: str
    operation_type: str
    pulse: str
    element: str
    handle: str
    statement_path: str


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
        with program() as natural_program:
            play("x180", "qubit")
            measure("readout", "resonator")

        timing = TimestampRecorder(natural_program)
        job = qm.execute(timing.program)
        result = timing.fetch(job)
        print(result.as_rows(reference=0))
    """

    def __init__(
        self,
        qua_program: Program,
        clock_cycle_ns: Real = 4,
        stream_prefix: str = "qua_timestamps",
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
            self._operations.append(
                _OperationSpec(
                    name=_generated_operation_name(statement_type, operation_index, pulse, element),
                    operation_type=statement_type,
                    pulse=pulse,
                    element=element,
                    handle=handle,
                    statement_path=path,
                )
            )

        capabilities = set(qua_program.used_capabilities)
        capabilities.add(QopCaps.command_timestamps)
        instrumented_program = Program()
        instrumented_program._set_and_exit(instrumented_proto, capabilities)
        return instrumented_program


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


def _is_message(message: Any) -> bool:
    return isinstance(message, Message)


def _active_oneof_fields(message: Message) -> Dict[str, str]:
    return dict(getattr(message, "_group_current", {}))


def _field_is_set(message: Message, field_name: str) -> bool:
    oneof_group = message._betterproto.oneof_group_by_field.get(field_name)
    if oneof_group is None:
        return True
    return _active_oneof_fields(message).get(oneof_group) == field_name


def _message_field_values(message: Message) -> Iterator[Tuple[str, Any]]:
    for field in dataclasses.fields(message):
        field_name = field.name
        if field_name.startswith("_"):
            continue
        if not _field_is_set(message, field_name):
            continue
        try:
            value = getattr(message, field_name)
        except AttributeError:
            continue
        if value is None:
            continue
        yield field_name, value


def _walk_messages(message: Any, path: str = "program") -> Iterator[Tuple[str, Any]]:
    """Depth-first traversal through every populated QUA protobuf message."""
    yield path, message
    if not _is_message(message):
        return

    for field_name, value in _message_field_values(message):
        if isinstance(value, list):
            for index, item in enumerate(value):
                yield from _walk_messages(item, f"{path}.{field_name}[{index}]")
        elif _is_message(value):
            yield from _walk_messages(value, f"{path}.{field_name}")


def _statement_type(message: Any) -> Optional[str]:
    if not _is_message(message):
        return None
    active_oneofs = _active_oneof_fields(message)
    for oneof_name in ("statement", "statement_oneof"):
        if oneof_name in active_oneofs:
            return active_oneofs[oneof_name]
    return None


def _all_strings(message: Any) -> Iterator[str]:
    if isinstance(message, str):
        yield message
        return
    if isinstance(message, list):
        for item in message:
            yield from _all_strings(item)
        return
    if not _is_message(message):
        return

    for field_name, value in _message_field_values(message):
        if isinstance(value, str):
            yield value
        else:
            yield from _all_strings(value)


def _clone_message(message: Any) -> Any:
    return copy.deepcopy(message)


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


def _get_attribute(message: Any, *names: str, default: Any = ...):
    for name in names:
        if hasattr(message, name):
            return getattr(message, name)
    if default is not ...:
        return default
    raise AttributeError(f"{type(message).__name__} has none of the attributes {names!r}.")


def _set_attribute(message: Any, value: Any, *names: str) -> None:
    for name in names:
        if hasattr(message, name):
            setattr(message, name, value)
            return
    raise AttributeError(f"{type(message).__name__} has none of the attributes {names!r}.")


def _timestamp_values(raw_result: Any) -> np.ndarray:
    if isinstance(raw_result, Mapping) and "value" in raw_result:
        raw_result = raw_result["value"]
    elif getattr(getattr(raw_result, "dtype", None), "names", None) and "value" in raw_result.dtype.names:
        raw_result = raw_result["value"]
    return np.array(raw_result, dtype=np.int64, copy=True).reshape(-1)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _collect_variable_arrays(qua_program: Any) -> Dict[str, Tuple[Any, ...]]:
    arrays: Dict[str, Tuple[Any, ...]] = {}
    script = _get_attribute(qua_program, "script", default=None)
    if script is None:
        return arrays
    for declaration in _get_attribute(script, "variables", default=[]):
        values = tuple(_literal_value(item) for item in _get_attribute(declaration, "value", default=[]))
        if values:
            arrays[declaration.name] = values
    return arrays


def _walk_statement_tree(
    statements: Sequence[Any],
    loop_stack: Sequence[LoopAxis],
    path_prefix: str,
    variable_arrays: Mapping[str, Tuple[Any, ...]],
    layouts: Dict[str, OperationLoopLayout],
) -> None:
    for index, statement in enumerate(statements):
        path = f"{path_prefix}.statements[{index}]"
        statement_type = _statement_type(statement)
        if statement_type == "for_":
            axis = _parse_for_loop(statement.for_)
            _walk_statement_tree(
                statement.for_.body.statements,
                (*loop_stack, axis),
                f"{path}.for_.body",
                variable_arrays,
                layouts,
            )
        elif statement_type == "for_each":
            axis = _parse_for_each_loop(statement.for_each, variable_arrays)
            _walk_statement_tree(
                statement.for_each.body.statements,
                (*loop_stack, axis),
                f"{path}.for_each.body",
                variable_arrays,
                layouts,
            )
        elif statement_type in {"play", "measure"}:
            layouts[path] = OperationLoopLayout("", path, tuple(loop_stack))
        elif statement_type == "if_" and hasattr(statement, "if_"):
            if_body = _get_attribute(statement.if_, "then", "body", default=None)
            if if_body is not None:
                _walk_statement_tree(if_body.statements, loop_stack, f"{path}.if_.then.body", variable_arrays, layouts)


def _iter_operation_statements(qua_program: Any) -> Iterator[Tuple[str, Any, str]]:
    for path, statement in _walk_messages(qua_program):
        statement_type = _statement_type(statement)
        if statement_type in {"play", "measure"}:
            yield path, statement, statement_type


def _parse_for_each_loop(for_each_loop: Any, variable_arrays: Mapping[str, Tuple[Any, ...]]) -> LoopAxis:
    iterator = for_each_loop.iterator[0]
    variable = iterator.variable.name
    array_name = iterator.array.name
    values = variable_arrays.get(array_name)
    if values is None:
        raise ValueError(f"Could not resolve for_each array '{array_name}' for variable '{variable}'.")
    return LoopAxis(variable=variable, kind="for_each", values=values)


def _parse_for_loop(for_loop: Any) -> LoopAxis:
    init_assign = for_loop.init.statements[0].assign
    variable = init_assign.target.variable.name
    start = _literal_value(init_assign.expression)
    if start is None:
        start = 0
    condition = for_loop.condition.binary_operation
    stop_literal = _literal_value(condition.right)
    stop_variable = _variable_name(condition.right)
    step = _parse_update_step(for_loop.update.statements[0].assign, variable)

    if stop_literal is not None:
        op_name = str(condition.op).split(".")[-1]
        if op_name == "LT":
            if isinstance(start, float) or isinstance(stop_literal, float) or isinstance(step, float):
                stop_effective = float(stop_literal) - float(step) / 2
                size = int(round((stop_effective - float(start)) / float(step))) + 1
            else:
                size = int(stop_literal) - int(start)
        elif isinstance(start, float) or isinstance(stop_literal, float) or isinstance(step, float):
            size = int(round((float(stop_literal) - float(start)) / float(step))) + 1
        else:
            size = int((int(stop_literal) - int(start)) / int(step)) + 1
        return LoopAxis(
            variable=variable,
            kind="static_range",
            size=size,
            start=start,
            step=step,
            stop=stop_literal,
        )
    if stop_variable is not None:
        return LoopAxis(
            variable=variable,
            kind="dynamic_range",
            start=start,
            step=step,
            stop_variable=stop_variable,
        )
    raise ValueError(f"Could not infer loop bounds for variable '{variable}'.")


def _parse_update_step(assign: Any, variable: str) -> int:
    expression = assign.expression
    binary = expression.binary_operation
    op_name = str(binary.op).split(".")[-1] if binary.op is not None else ""
    if op_name == "ADD":
        value = _literal_value(binary.right)
        if value is None:
            raise ValueError(f"Unsupported loop update for variable '{variable}'.")
        return value
    raise ValueError(f"Unsupported loop update for variable '{variable}'.")


def _literal_value(expression: Any) -> Any:
    if expression is None:
        return None
    if hasattr(expression, "WhichOneof"):
        return None
    if hasattr(expression, "literal") and expression.literal is not None:
        literal = expression.literal
        type_name = str(literal.type).split(".")[-1]
        if type_name == "INT":
            return int(literal.value)
        if type_name in {"FIXED", "REAL"}:
            return float(literal.value)
        return literal.value
    if hasattr(expression, "binary_operation") and expression.binary_operation is not None:
        return _literal_value(expression.binary_operation.right)
    if hasattr(expression, "variable") and expression.variable is not None:
        return None
    return None


def _variable_name(expression: Any) -> Optional[str]:
    if expression is None:
        return None
    if hasattr(expression, "variable") and expression.variable is not None:
        return expression.variable.name
    return None


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
    loop_coords = measure_data["loop_indices"]

    print("\nTiming at sweep point", loop_coords)
    print("-------------------------------------------------------------")
    last_play_ns: Optional[float] = None
    for play_name in play_names:
        play_data = selected[play_name]
        if "nanoseconds" in play_data:
            play_times = np.asarray(play_data["nanoseconds"], dtype=float)
            for index, time_ns in enumerate(play_times):
                print(f"  {play_name}[{index}]: {time_ns:.0f} ns")
            if play_times.size:
                last_play_ns = float(play_times[-1])
        else:
            last_play_ns = float(play_data["time_ns"])
            print(f"  {play_name}: {last_play_ns:.0f} ns")

    measure_ns = float(measure_data["time_ns"])
    print(f"  {measure_name}: {measure_ns:.0f} ns")
    if last_play_ns is not None:
        print(f"  last play -> measure gap: {measure_ns - last_play_ns:.0f} ns")
    print("-------------------------------------------------------------")
    return selected


__all__ = [
    "LoopAxis",
    "LoopIndexMapper",
    "OperationLoopLayout",
    "OperationTimestamps",
    "TimestampRecorder",
    "TimestampResults",
    "print_timing_summary",
]
