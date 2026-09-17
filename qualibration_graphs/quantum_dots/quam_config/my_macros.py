"""Select, expose, and wire the state macros used by this QUAM.

This file is the main user-facing entry point for custom state macros.

It has three roles:

1. Choose which initialize and measure macro classes are considered
   "active" for this project.
2. Export ``MacroParameters``, the Qualibrate parameter mixin that exposes
   the active macro fields in node ``parameters.py`` files.
3. Provide a small ``main()`` helper that wires the selected macros onto the
   machine and saves the updated QUAM state.

The actual macro implementations live in ``state_macros/``. Each macro class
inherits from ``CustomMacro``, which auto-generates a ``Parameters`` model
from the macro's own dataclass fields. ``MacroParameters`` below simply mixes
the selected macros' generated parameter models together.

State macros are wired at the ``QuantumDotPair`` level. In practice this
means calls such as ``qubit.initialize()``, ``qubit_pair.initialize()``, and
``dot_pair.initialize()`` all resolve to the same underlying pair-level
macro. If you customize the pair macro here, those higher-level calls will
use it automatically.
"""

#########################
# %%     Imports ########
#########################

from quam_builder.architecture.quantum_dots.macro_engine import wire_machine_macros
from quam_builder.architecture.quantum_dots.operations.names import SingleQubitMacroName

from qualibrate.core.parameters import RunnableParameters
from quam_config import Quam
from quam_config.state_macros import (
    HeraldedInitializeMacro, 
    MeasureMacro,
)

# Select the active state macros for this project here.
initialize_macro = HeraldedInitializeMacro
measure_macro = MeasureMacro

__all__ = ["MacroParameters"]

class MacroParameters(
    RunnableParameters, 
    initialize_macro.Parameters, 
    measure_macro.Parameters
):
    """Expose the active macro fields to Qualibrate nodes."""
    pass


# Wiring the macros into the machine
def main():
    ##############################
    ######## Load machine ########
    ##############################

    machine = Quam.load()

    #################################################
    ######## Specify the dot pairs to update ########
    #################################################

    # Create a list of dot pair names here if you would like to update only a subset of dot pair macros
    dot_pairs = machine.quantum_dot_pairs.keys()

    #########################################################
    ######## Wire the custom macros into the machine ########
    #########################################################

    # Extract the name from the list of default macro names
    initialize_macro_name = SingleQubitMacroName.INITIALIZE
    measure_macro_name = SingleQubitMacroName.MEASURE

    wire_machine_macros(
        machine=machine,
        instance_overrides={
            f"quantum_dot_pairs.{qdp}": {
                initialize_macro_name : initialize_macro,
                # measure_macro_name: measure_macro,
            }
            for qdp in dot_pairs
        },
    )

    ######################################
    ######## Save machine changes ########
    ######################################

    machine.save()

    ###############################################################
    ######## Test the state to see if the macros are there ########
    ###############################################################

    machine = Quam.load()

    for dot_pair_name in dot_pairs: 
        qdp = machine.quantum_dot_pairs[dot_pair_name]
        macro_object = qdp.macros[initialize_macro_name]
        print(f"Dot pair {dot_pair_name}'s {initialize_macro_name} is mapped to {type(macro_object)}.")
        assert isinstance(macro_object, initialize_macro), f"Dot pair {dot_pair_name}'s {initialize_macro_name} is mapped to {type(macro_object)} and not {initialize_macro}"

if __name__ == "__main__":
    main()
