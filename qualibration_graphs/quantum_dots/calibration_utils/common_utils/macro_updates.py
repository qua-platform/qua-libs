from typing import Any, Iterable, Sequence

from qualibrate.core import QualibrationNode
from qualibration_libs.core import tracked_updates

from quam_builder.architecture.quantum_dots.operations.default_macros.single_qubit_macros import (
    _resolve_qubit_pair,
    _state_macro_field_names,
)

__all__ = [
    "change_macro_tracked", 
    "revert_tracked_macro",
]

DEFAULT_TRACKED_STATE_MACROS = ("initialize", "measure")

def _resolve_quantum_dot_pair(element: Any):
    if hasattr(element, "preferred_readout_quantum_dot"):
        return _resolve_qubit_pair(element).quantum_dot_pair

    quantum_dot_pair = getattr(element, "quantum_dot_pair", None)
    if quantum_dot_pair is not None:
        return quantum_dot_pair
    return element

def _node_parameter_overrides(node: QualibrationNode, macro: Any) -> dict[str, Any]:
    if hasattr(node.parameters, "model_dump"):
        parameter_values = node.parameters.model_dump(exclude_none=True)
    else:
        parameter_values = {
            name: value
            for name, value in vars(node.parameters).items()
            if not name.startswith("_") and value is not None
        }

    macro_field_names = _state_macro_field_names(type(macro))
    return {
        name: value
        for name, value in parameter_values.items()
        if name in macro_field_names
    }

def change_macro_tracked(
    node: QualibrationNode,
    elements: Sequence[Any],
    macro_names: Iterable[str] = DEFAULT_TRACKED_STATE_MACROS,
) -> list:
    tracked_pair_macros = []
    for el in elements:
        pair = _resolve_quantum_dot_pair(el)
        for macro_name in macro_names:
            macro = pair.macros.get(macro_name)
            if macro is None:
                raise ValueError(f"Macro {macro_name} does not exist for pair {pair.name}")
            override_kwargs = _node_parameter_overrides(node, macro)
            if not override_kwargs:
                continue

            with tracked_updates(macro, auto_revert=False, dont_assign_to_none=True) as tracked_macro:
                tracked_macro.update(**override_kwargs)
                tracked_pair_macros.append(tracked_macro)

    node.namespace["tracked_pair_macros"] = tracked_pair_macros
    return tracked_pair_macros


def revert_tracked_macros(node: QualibrationNode) -> None:
    for tracked_macro in node.namespace.pop("tracked_pair_macros", []):
        tracked_macro.revert_changes()