"""Minimal QuAM factory for gate virtualization analysis tests."""

from __future__ import annotations

from quam.components import pulses
from quam.components.channels import StickyChannelAddon
from quam.components.ports import LFFEMAnalogInputPort, LFFEMAnalogOutputPort

from quam_builder.architecture.quantum_dots.components import VoltageGate
from quam_builder.architecture.quantum_dots.components.readout_resonator import ReadoutResonatorSingle
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam


def create_gate_virtualization_quam() -> LossDiVincenzoQuam:
    """Create a minimal Quam for gate virtualization analysis tests.

    Contains:
    - virtual_sensor_1, virtual_dot_1, virtual_dot_2 in the ``main_qpu``
      VirtualGateSet with a 3×3 identity compensation matrix.
    - One readout resonator coupled to the sensor.
    - One quantum dot pair (virtual_dot_1, virtual_dot_2) sensed by virtual_sensor_1.

    No qubits, XY drives, or QDAC connections are created — only the
    voltage-gate/virtual-gate infrastructure needed by the gate virtualization
    calibration nodes.
    """
    machine = LossDiVincenzoQuam()
    controller = "con1"
    fem_slot = 2

    plunger_1 = VoltageGate(
        id="plunger_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller, fem_id=fem_slot, port_id=1, output_mode="direct"
        ),
        sticky=StickyChannelAddon(duration=16, digital=False),
    )
    plunger_2 = VoltageGate(
        id="plunger_2",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller, fem_id=fem_slot, port_id=2, output_mode="direct"
        ),
        sticky=StickyChannelAddon(duration=16, digital=False),
    )
    sensor_dc = VoltageGate(
        id="sensor_DC_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller, fem_id=fem_slot, port_id=3, output_mode="direct"
        ),
        sticky=StickyChannelAddon(duration=16, digital=False),
    )
    readout_resonator = ReadoutResonatorSingle(
        id="sensor_resonator_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller, fem_id=fem_slot, port_id=4, output_mode="direct"
        ),
        opx_input=LFFEMAnalogInputPort(
            controller_id=controller, fem_id=fem_slot, port_id=1
        ),
        intermediate_frequency=50e6,
        operations={
            "readout": pulses.SquareReadoutPulse(
                length=1000, amplitude=0.1, integration_weights_angle=0.0
            )
        },
    )

    machine.create_virtual_gate_set(
        virtual_channel_mapping={
            "virtual_sensor_1": sensor_dc,
            "virtual_dot_1": plunger_1,
            "virtual_dot_2": plunger_2,
        },
        gate_set_id="main_qpu",
        compensation_matrix=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    machine.register_channel_elements(
        plunger_channels=[plunger_1, plunger_2],
        sensor_resonator_mappings={sensor_dc: readout_resonator},
        barrier_channels=[],
    )

    machine.register_quantum_dot_pair(
        quantum_dot_ids=["virtual_dot_1", "virtual_dot_2"],
        sensor_dot_ids=["virtual_sensor_1"],
        id="qd_pair_1_2",
    )

    sensor_dot = machine.sensor_dots["virtual_sensor_1"]
    sensor_dot._add_readout_params(quantum_dot_pair_id="qd_pair_1_2", threshold=0.5)

    return machine
