"""
This script is made in order to create Node Parameters based on any custom macros that you have built.

This example populates the InitializeMacroParameters to be HeraldedBalancedInitializeMacro parameters.
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
