"""Keep grouped QUAlibrate parameters at the values submitted for the last run.

Clicking Run immediately reloads the node schema. The form keeps an edited value
only when it sits on the parameter itself. A group is replaced wholesale by the
schema, so its fields fall back to the class defaults even though the run used
the submitted values.

This module rebuilds group fields from the live values, and writes those values
onto the library node before the submit request returns, so the reload that
follows shows them.
"""

import sys
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import Any, cast

from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined
from qualibrate.core.parameters import GroupParameters
from qualibrate.core.q_runnnable import QRunnable
from qualibrate.core.utils.logger_m import logger

_BASE_CLASS_ATTR = "_group_defaults_base_class"


def _build_parameters_class_from_instance(parameters: BaseModel, use_passed_as_base: bool = False) -> type[BaseModel]:
    """Same as QRunnable.build_parameters_class_from_instance, but a group field is rebuilt too."""
    klass = parameters.__class__
    base = (klass,) if use_passed_as_base else klass.__bases__
    fields: dict[str, Any] = {}
    for name, field_info in klass.model_fields.items():
        field_info_copy = copy(field_info)
        value = getattr(parameters, name)
        if isinstance(value, GroupParameters):
            # The GUI reads each field inside the group, not the group object.
            nested_cls = _build_parameters_class_from_instance(value, True)
            field_info_copy.annotation = nested_cls
            field_info_copy.default_factory = nested_cls
            field_info_copy.default = PydanticUndefined
        else:
            field_info_copy.default = deepcopy(value)
        fields[name] = field_info_copy

    model = create_model(  # type: ignore[call-overload]
        klass.__name__,
        __doc__=klass.__doc__,
        __base__=base,
        **{name: (info.annotation, info) for name, info in fields.items()},
    )
    if hasattr(parameters, "targets_name"):
        model.targets_name = parameters.targets_name
    return cast(type[BaseModel], model)


def _apply_submitted_parameters(node: Any, submitted: Mapping[str, Any]) -> None:
    """Point the library node at the values just submitted, including fields inside groups."""
    from qualibrate.core.qualibration_library import QualibrationLibrary

    library = QualibrationLibrary.get_active_library(create=False)
    original = library.get_nodes().get_nocopy(node.name)
    if original is None or getattr(original, "parameters", None) is None:
        return
    base_class = getattr(original, _BASE_CLASS_ATTR, None) or original.parameters.__class__
    setattr(original, _BASE_CLASS_ATTR, base_class)
    updated = base_class.model_validate(submitted)
    original._parameters = updated
    original.parameters_class = _build_parameters_class_from_instance(updated, True)


def _walk_routes(routes: Any):
    for route in routes:
        yield route
        nested = getattr(route, "routes", None)
        if nested:
            yield from _walk_routes(nested)


def _install_submit_hook() -> None:
    """Publish submitted parameters before /submit/node responds.

    The form refetches the node as soon as that response arrives. The hook has to
    be on the route object itself: replacing the function in the module does not
    change the endpoint FastAPI already registered.
    """
    app_module = sys.modules.get("qualibrate.runner.app")
    if app_module is None:
        return
    submit_module = sys.modules.get("qualibrate.runner.api.routes.submit")
    if submit_module is None:
        return
    original = submit_module.submit_node_run

    def submit_node_run(input_parameters, state, node, background_tasks):  # noqa: ANN001
        try:
            _apply_submitted_parameters(node, input_parameters)
        except Exception:
            logger.exception("Could not keep grouped parameter values for %s", getattr(node, "name", None))
        return original(input_parameters, state, node, background_tasks)

    for route in _walk_routes(app_module.app.routes):
        dependant = getattr(route, "dependant", None)
        if dependant is None or dependant.call is not original:
            continue
        if getattr(dependant, "_group_defaults_wrapped", False):
            continue
        dependant.call = submit_node_run
        dependant._group_defaults_wrapped = True


QRunnable.build_parameters_class_from_instance = staticmethod(_build_parameters_class_from_instance)
_install_submit_hook()
