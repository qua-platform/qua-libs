"""
A library of measure macros. 

For each macro, create an Attributes class and mix it with the CustomMacro class from quam-builder.
"""

from quam_builder.architecture.quantum_dots import CustomMacro
from quam.core import quam_dataclass

__all__ = [
    "MeasureMacro", "MeasureMacroAttributes",
]

###########################
##### Example Measure #####
###########################

@quam_dataclass
class MeasureMacroAttributes: 
    point_duration: int = 1000
    """Hold duration of the Initialize voltage point."""

@quam_dataclass
class MeasureMacro(CustomMacro, MeasureMacroAttributes):
    @property
    def inferred_duration(self) -> float | None:
        return 0

    def apply(
        self,
        **kwargs,
    ):
        owner = self.owner
        params = self.resolve_params(**kwargs)
        pass
