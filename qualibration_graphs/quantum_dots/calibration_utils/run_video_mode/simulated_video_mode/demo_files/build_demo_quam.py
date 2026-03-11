"""

This is an example script on how to instantiate a QPU which contains Loss-DiVincenzo qubits, with other barrier gates and sensor dots.

Workflow:

1. Instantiate your machine.

2. Instantiate the base hardware channels for the machine.
    - In this example, arbitrary HW gates are created as VoltageGates. For QuantumDots and SensorDots, the base channel must be VoltageGate and sticky. They are instantiated in a mapping dictionary to be input into the machine

3. Create your VirtualGateSet. You do not need to manually add all the channels, the function create_virtual_gate_set should do it automatically.
    Ensure that the mapping of the desired virtual gate to the relevant HW channel is correct, as the QuantumDot names will be extracted from this input dict.

4. Register your components.
    - Register the relevant QuantumDots, SensorDots and BarrierGates, mapped correctly to the relevant output channel. As long as the channel is correctly mapped,
        the name of the element will be made consistent to that in the VirtualGateSet

5. Create your QUA programme
    - For simultaneous stepping/ramping, use either
        sequence = machine.voltage_sequences[gate_set_id]
        sequence.step_to_voltages({"virtual_dot_1": ..., "virtual_dot_2": ...})
    or use sequence.simultaneous:
        with sequence.simultaneous(duration = ...):
            machine.qubits["virtual_dot_1"].step_to_voltages(...)
            machine.qubits["virtual_dot_2"].step_to_voltages(...)

"""

from numpy import True_
from quam.components import (
    StickyChannelAddon,
    pulses,
    DigitalOutputChannel,
    Channel,
)
from quam.components.ports import (
    LFFEMAnalogOutputPort,
    LFFEMAnalogInputPort,
    MWFEMAnalogOutputPort,
    MWFEMAnalogInputPort,
)

from quam_builder.architecture.quantum_dots.components import (
    VoltageGate,
)
from quam_builder.architecture.quantum_dots.qubit import LDQubit
from quam_builder.architecture.quantum_dots.components.voltage_gate import VoltageGate, QdacSpec
from demo_quam_qd import DemoQuam

Quam = DemoQuam

from quam_builder.architecture.quantum_dots.components.readout_resonator import (
    ReadoutResonatorSingle,
)
from qm.qua import *


# Instantiate Quam
machine = DemoQuam()
lf_fem = 5
mw_fem = 1

machine.network = {"host": "172.16.33.115", "cluster_name": "CS_4"}

###########################################
###### Instantiate Physical Channels ######
###########################################

p1 = VoltageGate(
    id=f"plunger_1",
    opx_output=LFFEMAnalogOutputPort("con1", lf_fem, port_id=1, shareable = True),
    sticky=StickyChannelAddon(duration=16, digital=False),
)
p2 = VoltageGate(
    id=f"plunger_2",
    opx_output=LFFEMAnalogOutputPort("con1", lf_fem, port_id=2, shareable = True),
    sticky=StickyChannelAddon(duration=16, digital=False),
)
s1 = VoltageGate(
    id=f"sensor_1",
    opx_output=LFFEMAnalogOutputPort("con1", lf_fem, port_id=8, shareable = True),
    sticky=StickyChannelAddon(duration=16, digital=False),
)
s2 = VoltageGate(
    id=f"sensor_2",
    opx_output=LFFEMAnalogOutputPort("con1", lf_fem, port_id=8, shareable = True),
    sticky=StickyChannelAddon(duration=16, digital=False),
)

readout_pulse = pulses.SquareReadoutPulse(length=200, id="readout", amplitude=0.01)
resonator1 = ReadoutResonatorSingle(
    id="readout_resonator_1",
    frequency_bare=0,
    intermediate_frequency=500e6,
    operations={"readout": readout_pulse},
    opx_output=LFFEMAnalogOutputPort("con1", 5, port_id=1, upsampling_mode="mw", shareable = True),
    opx_input=LFFEMAnalogInputPort("con1", 5, port_id=2, shareable = True),
)

readout_pulse = pulses.SquareReadoutPulse(length=200, id="readout", amplitude=0.01)
resonator2= ReadoutResonatorSingle(
    id="readout_resonator_2",
    frequency_bare=0,
    intermediate_frequency=500e6,
    operations={"readout": readout_pulse},
    opx_output=LFFEMAnalogOutputPort("con1", 5, port_id=1, upsampling_mode="mw", shareable = True),
    opx_input=LFFEMAnalogInputPort("con1", 5, port_id=2, shareable = True),
)

#####################################
###### Create Virtual Gate Set ######
#####################################

# Create virtual gate set out of all the relevant HW channels.
# This function adds HW channels to machine.physical_channels, so no need to independently map
machine.create_virtual_gate_set(
    virtual_channel_mapping={
        "virtual_dot_1": p1,
        "virtual_dot_2": p2,
        "virtual_sensor_1": s1,
        "virtual_sensor_2": s2,
    },
    gate_set_id="main_qpu",
    compensation_matrix = [
        [
            1.0,
            0.0,
            0.020406,
            0.020406
        ],
        [
            0.0,
            1.0,
            0.029189,
            0.029189
        ],
        [
            0.020406,
            0.029189,
            1.0,
            0.0
        ],
        [
            0.020406,
            0.029189,
            0.0,
            1.0
        ]
    ],
)


#########################################################
###### Register Quantum Dots, Sensors and Barriers ######
#########################################################

# Shortcut function to register QuantumDots, SensorDots, BarrierGates
machine.register_channel_elements(
    plunger_channels=[p1, p2],
    barrier_channels=[],
    sensor_resonator_mappings={s1: resonator1, s2: resonator2},
)

##################################################################
###### Connect the physical channels to the external source ######
##################################################################

qdac_connect = True
if qdac_connect:
    # Set up the QDAC port specs
    for i, (ch_name, ch_obj) in enumerate(machine.physical_channels.items()):
        if isinstance(ch_obj, VoltageGate):
            ch_obj.qdac_spec = QdacSpec(
                opx_trigger_out=Channel(
                    id=f"{ch_name}_qdac_trigger",
                    digital_outputs={
                        "trigger": DigitalOutputChannel(
                            opx_output=("con1", lf_fem, i + 1), delay=0, buffer=0
                        )
                    },
                    operations={"trigger": pulses.Pulse(length=100, digital_marker="ON")},
                ),
                qdac_output_port=i + 1,
            )

    qdac_ip = "172.16.33.101"
    machine.network.update({"qdac_ip": qdac_ip})
    machine.connect_to_external_source(external_qdac=True)
    machine.create_virtual_dc_set("main_qpu")

machine.save("/Users/kalidu_laptop/QUA/qua-libs/qualibration_graphs/quantum_dots/calibration_utils/run_video_mode/simulated_video_mode/quam_state")
########################################
###### Register Quantum Dot Pairs ######
########################################

# Register the quantum dot pairs
# machine.register_quantum_dot_pair(
#     id="dot1_dot2_pair",
#     quantum_dot_ids=["virtual_dot_1", "virtual_dot_2"],
#     sensor_dot_ids=["virtual_sensor_1"],
#     barrier_gate_id="virtual_barrier_2",
# )

# machine.register_quantum_dot_pair(
#     id="dot3_dot4_pair",
#     quantum_dot_ids=["virtual_dot_3", "virtual_dot_4"],
#     sensor_dot_ids=["virtual_sensor_1"],
#     barrier_gate_id="virtual_barrier_3",
# )

# # Define the detuning axes for both QuantumDotPairs
# machine.quantum_dot_pairs["dot1_dot2_pair"].define_detuning_axis(
#     matrix=[[1, -1]],
#     detuning_axis_name="dot1_dot2_pair_epsilon",
#     set_dc_virtual_axis=False,
# )

# machine.quantum_dot_pairs["dot3_dot4_pair"].define_detuning_axis(
#     matrix=[[1, -1]],
#     detuning_axis_name="dot3_dot4_pair_epsilon",
#     set_dc_virtual_axis=False,
# )


##################################################
###### Update the Cross Compensation Matrix ######
##################################################

# Update Cross Capacitance matrix values
# machine.update_cross_compensation_submatrix(
#     virtual_names=["virtual_barrier_1", "virtual_barrier_2"],
#     channels=[p4],
#     matrix=[[0.1, 0.5]],
#     target="opx",
# )

# machine.update_cross_compensation_submatrix(
#     virtual_names=["virtual_dot_1", "virtual_dot_2", "virtual_dot_3", "virtual_dot_4"],
#     channels=[p1, p2, p3, p4],
#     matrix=[[1, 0.1, 0.1, 0.3], [0.2, 1, 0.6, 0.8], [0.1, 0.3, 1, 0.3], [0.2, 0.5, 0.1, 1]],
#     target="opx",
# )

###########################
###### Example Usage ######
###########################


# Let's define some example points.
# In this example, we would like to load virtual_dot_1 and virtual_dot_2 simultaneously. This will be performed in a sequence.simultaneous block.
# Remember that if these two dictionaries hold contradicting information about the voltage of a particular gate, the last one in the QUA programme wins.

# In this example, we purposefully keep all the barrier and sensor voltages identical, so that they can be initialised together, and no gate should hold two voltages at once.


# machine.quantum_dots["virtual_dot_1"].add_point(
#     point_name="loading",
#     voltages={
#         "virtual_dot_1": 0.1,
#         "virtual_barrier_1": 0.4,
#         "virtual_barrier_2": 0.45,
#         "virtual_barrier_3": 0.42,
#         "virtual_sensor_1": 0.15,
#     },
# )

# machine.quantum_dots["virtual_dot_2"].add_point(
#     point_name="loading",
#     voltages={
#         "virtual_dot_2": 0.15,
#         "virtual_barrier_1": 0.4,
#         "virtual_barrier_2": 0.45,
#         "virtual_barrier_3": 0.42,
#         "virtual_sensor_1": 0.15,
#     },
# )

# # We can also initialise a tuning point for a qubit pair:
# machine.quantum_dot_pairs["dot3_dot4_pair"].add_point(
#     point_name="some_detuning_points",
#     voltages={
#         "virtual_dot_3": 0.2,
#         "virtual_dot_4": 0.25,
#         "virtual_barrier_1": 0.4,
#         "virtual_barrier_2": 0.45,
#         "virtual_barrier_3": 0.42,
#         "virtual_sensor_1": 0.15,
#     },
# )
