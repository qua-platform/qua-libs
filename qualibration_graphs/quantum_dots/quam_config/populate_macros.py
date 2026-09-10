"""
Use this script to populate QuantumDotPair components with custom macros, as created in my_macros.py

These custom state macros operate at the level of the QuantumDotPair. This means that
qubit.initialize() and qubit_pair.initialize() are quantum_dot_pair.initialize() under the hood, and 
the same goes for the measure macro. 

Therefore, for any custom state macro you create in regards to the initialize or the meaasure sequence, 
you only need to update it at the QuantumDotPair level. Any call (qubit.initialize() or qubit_pair.initialize())
you see in the nodes will simply use these wired QuantumDotPair macros under the hood! 

This script currently only wires in the InitializeMacro. If you write your own MeasureMacro, then make sure to 
wire it up. 
"""

#########################
# %%     Imports ########
#########################

from typing import List
from quam_builder.architecture.quantum_dots.macro_engine import wire_machine_macros
from quam_builder.architecture.quantum_dots.operations.names import SingleQubitMacroName

from quam_config import QubitQuam as Quam
from quam_config import InitializeMacro, MeasureMacro

# ### Helper function which allows you to choose the dot pairs to update


def get_dot_pairs(machine: Quam, dot_pairs: List[str] | None = None) -> List[str]:
    """
    Given a spin qubit Quam machine, extracts a list of QuantumDotPair names as a list.

    Args:
        - machine: Quam - The Spin Quam Machine to extract the dot pairs from
        - dot_pairs: List[str] - The list of quantum dot pair string names. If None, then
            this function will return all the dot pairs associated with the machine.
    """
    if dot_pairs is None:
        # No list supplied, default to all existing dot pairs.
        return list(machine.quantum_dot_pairs.keys())
    else:
        # List supplied. Means that the user wants to update only a subset of the dot pairs.
        return dot_pairs


##############################
# %%     Load machine ########
##############################

machine = Quam.load()


#################################################
# %%     Specify the dot pairs to update ########
#################################################

# Create a list of dot pair names here if you would like to update only a subset of dot pair macros
dot_pairs = None
dot_pairs = get_dot_pairs(machine, dot_pairs)

#########################################################
# %%     Wire the custom macros into the machine ########
#########################################################

wire_machine_macros(
    machine=machine,
    instance_overrides={
        f"quantum_dot_pairs.{qdp}": {
            SingleQubitMacroName.INITIALIZE: InitializeMacro,
            # SingleQubitMacroName.MEASURE: MeasureMacro,
        }
        for qdp in dot_pairs
    },
)

######################################
# %%     Save machine changes ########
######################################

machine.save()
