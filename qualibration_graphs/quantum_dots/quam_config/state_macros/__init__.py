"""Library of reusable state-macro implementations.

The files in this package contain the actual initialize and measure macro
classes. A typical workflow is:

1. Implement or customize a macro in ``initialize_macros.py`` or
   ``measure_macros.py``.
2. Declare the configurable fields directly on the macro quam_dataclass.
3. Let ``CustomMacro.Parameters`` auto-generate the corresponding Qualibrate
   parameter model from those fields.
4. Import the macro in ``quam_config/my_macros.py`` and select it as the
   active macro for the project.

Keeping the implementations here and the project-specific selection in
``my_macros.py`` makes it easier to provide multiple examples without
forcing users to edit the same large file repeatedly.
"""

from .initialize_macros import *
from .measure_macros import *

__all__ = [
    *initialize_macros.__all__, 
    *measure_macros.__all__, 
]