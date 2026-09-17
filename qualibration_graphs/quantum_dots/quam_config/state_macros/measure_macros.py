"""Example measure macros for quantum-dot QUAM state readout.

Reference implementation details for the underlying macro infrastructure can
be found in the
[`quam-builder` operations package](https://github.com/qua-platform/quam-builder/tree/main/quam_builder/architecture/quantum_dots/operations).

Use the same pattern as in ``initialize_macros.py``:

1. Subclass ``CustomMacro``.
2. Put configurable fields directly on the macro dataclass.
3. Let ``CustomMacro.Parameters`` automatically expose those fields to
   ``quam_config/my_macros.py`` as an optional parameter model.
4. Implement ``inferred_duration`` when you can estimate how long the macro
   takes to run. Return the duration in seconds. This is especially useful
   when another macro builds on the measure macro and needs a timing estimate.
   If the duration cannot be known ahead of time, returning ``None`` is fine.

This file currently contains a minimal placeholder measure macro. Replace
or extend it when your lab needs custom readout behaviour beyond the
default measure macro provided by ``quam-builder``.
"""

from quam_builder.architecture.quantum_dots import CustomMacro
from quam.core import quam_dataclass

__all__ = [
    "MeasureMacro",
]

###########################
##### Example Measure #####
###########################

@quam_dataclass
class MeasureMacro(CustomMacro):
    """Minimal example measure macro."""
    point_duration: int = 1000
    """Example hold duration for a custom measure point."""

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
