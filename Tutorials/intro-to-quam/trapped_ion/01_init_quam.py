# %%
import numpy as np

from qm.qua import *
from quam.components.ports import (
    LFFEMAnalogInputPort,
    LFFEMAnalogOutputPort,
    MWFEMAnalogOutputPort,
    FEMDigitalOutputPort,
)
from quam.components.channels import (
    Channel,
    SingleChannel,
    DigitalOutputChannel,
    InOutSingleChannel,
    MWChannel,
)
from quam.components.pulses import (
    Pulse,
    SquareReadoutPulse,
    SquarePulse,
)

#############################################################
## Import custom components and macros
#############################################################
from trapped_ion.custom_components import (
    HyperfineQubit,
    GlobalOperations,
    Quam,
)
from trapped_ion.custom_macros import MeasureMacro, SingleXMacro, DoubleXMacro


#############################################################
## Generate QUAM object
#############################################################

machine = Quam()

n_qubits = 2
aom_position = np.linspace(200e6, 300e6, n_qubits)
mw_IF = 100e6
mw_LO = 3e9
mw_band = 1

# for each qubit
for i in range(n_qubits):
    qubit_id = f"q{i + 1}"
    qubit = HyperfineQubit(
        id=f"{qubit_id}",
        readout=InOutSingleChannel(
            opx_output=LFFEMAnalogOutputPort("con1", 1, 2),
            opx_input=LFFEMAnalogInputPort("con1", 1, 2),
            intermediate_frequency=aom_position[i],
        ),
        shelving=SingleChannel(
            opx_output=LFFEMAnalogOutputPort("con1", 1, 3),
            intermediate_frequency=aom_position[i],
        ),
    )

    # define pulse
    qubit.shelving.operations["const"] = SquarePulse(length=1_000, amplitude=0.1)
    qubit.readout.operations["const"] = SquareReadoutPulse(length=2_000, amplitude=0.1)

    # define macro
    qubit.macros["measure"] = MeasureMacro(threshold=10)

    # add to quam
    machine.qubits[qubit_id] = qubit

# set global properties
machine.global_op = GlobalOperations(
    global_mw=MWChannel(
        id="global_mw",
        opx_output=MWFEMAnalogOutputPort("con1", 8, 1, band=mw_band, upconverter_frequency=mw_LO),
        intermediate_frequency=mw_IF,
    ),
    ion_displacement=Channel(
        digital_outputs={
            "ttl": DigitalOutputChannel(opx_output=FEMDigitalOutputPort("con1", 8, 1), delay=136, buffer=0)
        },
    ),
)

# define pulse
machine.global_op.global_mw.operations["x180"] = SquarePulse(amplitude=0.2, length=1000)
machine.global_op.global_mw.operations["y180"] = SquarePulse(amplitude=0.2, length=1000, axis_angle=90)
machine.global_op.ion_displacement.operations["ttl"] = Pulse(length=1000, digital_marker=[(1, 500), (0, 0)])

# operation macro
machine.global_op.macros["X"] = SingleXMacro()
machine.global_op.macros["N_XX"] = DoubleXMacro()

# print the state
machine.print_summary()

# save the state
machine.save("state_before.json")
