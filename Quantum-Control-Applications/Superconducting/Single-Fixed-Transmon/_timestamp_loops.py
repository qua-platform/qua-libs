"""Private loop-structure inference for timestamp_tools (not part of the public API)."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from qm.program.program import Program

from _timestamp_proto import _get_attribute, _statement_type

if TYPE_CHECKING:
    from timestamp_tools import TimestampResults

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
    def from_program(cls, qua_program: Program) -> LoopIndexMapper:
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

    def attach_operation_names(self, operations: Sequence[Any]) -> LoopIndexMapper:
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
        results: TimestampResults,
        loop_indices: LoopIndices,
        *,
        reference: Optional[Reference] = None,
    ) -> Dict[str, Dict[str, Any]]:
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
        if statement_type in {"for_", "for"}:
            for_loop = _get_attribute(statement, "for_", "for")
            axis = _parse_for_loop(for_loop)
            _walk_statement_tree(
                for_loop.body.statements,
                (*loop_stack, axis),
                f"{path}.{statement_type}.body",
                variable_arrays,
                layouts,
            )
        elif statement_type in {"for_each", "forEach"}:
            for_each_loop = _get_attribute(statement, "for_each", "forEach")
            axis = _parse_for_each_loop(for_each_loop, variable_arrays)
            _walk_statement_tree(
                for_each_loop.body.statements,
                (*loop_stack, axis),
                f"{path}.{statement_type}.body",
                variable_arrays,
                layouts,
            )
        elif statement_type in {"play", "measure"}:
            layouts[path] = OperationLoopLayout("", path, tuple(loop_stack))
        elif statement_type in {"if_", "if"}:
            if_statement = _get_attribute(statement, "if_", "if")
            if_body = _get_attribute(if_statement, "then", "body", default=None)
            if if_body is not None:
                _walk_statement_tree(
                    if_body.statements,
                    loop_stack,
                    f"{path}.{statement_type}.then.body",
                    variable_arrays,
                    layouts,
                )


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
    condition = _get_attribute(for_loop.condition, "binary_operation", "binaryOperation")
    stop_literal = _literal_value(condition.right)
    stop_variable = _variable_name(condition.right)
    step = _parse_update_step(for_loop.update.statements[0].assign, variable)

    if stop_literal is not None:
        op_name = _binary_op_name(condition.op)
        if op_name == "LT":
            if isinstance(start, float) or isinstance(stop_literal, float) or isinstance(step, float):
                stop_effective = float(stop_literal) - float(step) / 2
                size = int(round((stop_effective - float(start)) / float(step))) + 1
            else:
                size = int(float(stop_literal)) - int(float(start))
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


_BINARY_OP_NAMES = {
    0: "ADD",
    7: "LT",
}


def _binary_op_name(op: Any) -> str:
    if isinstance(op, int):
        return _BINARY_OP_NAMES.get(op, str(op))
    name = str(op)
    return name.split(".")[-1] if "." in name else name


def _parse_update_step(assign: Any, variable: str) -> int:
    expression = assign.expression
    binary = _get_attribute(expression, "binary_operation", "binaryOperation")
    op_name = _binary_op_name(binary.op)
    if op_name == "ADD":
        value = _literal_value(binary.right)
        if value is None:
            raise ValueError(f"Unsupported loop update for variable '{variable}'.")
        return value
    raise ValueError(f"Unsupported loop update for variable '{variable}'.")


def _expression_kind(expression: Any) -> Optional[str]:
    if hasattr(expression, "WhichOneof"):
        for oneof_name in ("expression_oneof", "expression"):
            try:
                active = expression.WhichOneof(oneof_name)
            except ValueError:
                continue
            if active:
                return active
    return None


def _literal_value(expression: Any) -> Any:
    if expression is None:
        return None
    kind = _expression_kind(expression)
    if kind == "variable":
        return None
    if kind in {"binaryOperation", "binary_operation"}:
        binary = _get_attribute(expression, "binary_operation", "binaryOperation")
        return _literal_value(binary.right)
    if kind == "literal" or (hasattr(expression, "literal") and expression.literal is not None and expression.literal.value != ""):
        literal = expression.literal
        type_name = str(literal.type).split(".")[-1]
        value = literal.value
        if type_name == "INT":
            return int(value)
        if type_name in {"FIXED", "REAL"}:
            return float(value)
        if isinstance(value, str) and value != "":
            return float(value) if "." in value or "e" in value.lower() else int(value)
        return None
    if hasattr(expression, "variable") and expression.variable is not None and _variable_name(expression) is not None:
        return None
    return None


def _variable_name(expression: Any) -> Optional[str]:
    if expression is None:
        return None
    if hasattr(expression, "variable") and expression.variable is not None:
        return expression.variable.name
    return None
