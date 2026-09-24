"""Select, expose, and wire the state macros used by this QUAM.

This file is the main user-facing entry point for custom state macros.

It has three roles:

1. Choose which initialize and measure macro classes are considered
   "active" for this project.
2. Export ``MacroParameters``, the Qualibrate parameter mixin that exposes
   the fields of whichever custom macros are in ``selected_macros``.
3. Provide a small ``main()`` helper that wires the selected macros onto the
   machine and saves the updated QUAM state.

The actual macro implementations live in ``state_macros/``. Each macro class
inherits from ``CustomMacro``, which auto-generates a ``Parameters`` model
from the macro's own dataclass fields. ``MacroParameters`` below simply mixes
the generated parameter models of the macros listed in ``selected_macros``.

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

# Extract the name from the list of default macro names
initialize_macro_name = SingleQubitMacroName.INITIALIZE
measure_macro_name = SingleQubitMacroName.MEASURE

# If you want to wire the macros in, un-comment the dictionary entries here. 
selected_macros = {
    # initialize_macro_name : HeraldedInitializeMacro,
    # measure_macro_name : MeasureMacro,
}

__all__ = ["MacroParameters"]

class MacroParameters(
    RunnableParameters, 
    *(macro.Parameters for macro in selected_macros.values()),
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

    wire_machine_macros(
        machine=machine,
        instance_overrides={
            f"quantum_dot_pairs.{qdp}": selected_macros
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
        for macro_name, macro_cls in selected_macros.items():
            macro_object = qdp.macros[macro_name]
            print(f"Dot pair {dot_pair_name}'s {macro_name} is mapped to {type(macro_object)}.")
            assert isinstance(macro_object, macro_cls), (
                f"Dot pair {dot_pair_name}'s {macro_name} is mapped to {type(macro_object)} and not {macro_cls}"
            )

if __name__ == "__main__":
    main()
