"""
Define node-parameter mixins for custom state macros.

This module is the companion to ``my_macros.py``. The parameter
classes defined here should mirror the configurable fields of your custom
initialize and measure macros so that Qualibration nodes can expose those
values in their ``parameters.py`` files.

The main export is ``MacroParameters``, which bundles the macro-related
parameter mixins into a single class for convenient reuse across nodes.

When customizing this file, keep it aligned with the macros exported from
``my_macros.py`` and wired by ``populate_macros.py``. If you add, remove, or
rename configurable macro fields, you should update the corresponding
parameter classes here to match.

This example exposes parameters for a heralded / active-reset initialize
macro, while leaving the measure-macro parameter mixin empty until custom
measure fields are needed.
"""

from typing import Optional, Literal

from qualibrate.core.parameters import RunnableParameters

__all__ = ["MacroParameters"]


class InitializeMacroParameters(RunnableParameters):
    target_state: Optional[int] = None
    """The state you want to initialize into for heralded initialization."""
    max_loops: int = 100
    """Maximum number of initialization loops for heralded initialization."""
    return_n_loops: bool = False
    """Whether to return the number of times it has looped over the initialise sequence to achieve the desired result."""
    qubit_role: Literal["target", "control"] = "control"
    """Specify which qubit, related to the qubit_pair, to pulse. """


class MeasureMacroParameters(RunnableParameters):
    pass


class MacroParameters(InitializeMacroParameters, MeasureMacroParameters):
    """Batch all the macro related parameters to export in a single class"""
    pass
