"""Dedicated QuAM factory for simulated video mode."""

from __future__ import annotations

from quam.components import pulses
from quam.components.channels import StickyChannelAddon
from quam.components.ports import LFFEMAnalogInputPort, LFFEMAnalogOutputPort
from quam_builder.architecture.quantum_dots.components import VoltageGate
from quam_builder.architecture.quantum_dots.components.readout_resonator import ReadoutResonatorSingle
from quam_builder.architecture.quantum_dots.components.voltage_gate import QdacSpec
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam


def create_minimal_quam() -> LossDiVincenzoQuam:
    """Create the minimal 2-dot/2-sensor QuAM used by simulated video mode."""
    machine = LossDiVincenzoQuam()

    controller = "con1"
    lf_fem_slot = 5

    plunger_1 = VoltageGate(
        id="plunger_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=1,
            output_mode="direct",
        ),
        sticky=StickyChannelAddon(duration=1000, digital=False),
        attenuation=10,
        current_external_voltage=-0.0048056,
        qdac_spec=QdacSpec(qdac_output_port=1),
    )
    plunger_2 = VoltageGate(
        id="plunger_2",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=2,
            output_mode="direct",
        ),
        sticky=StickyChannelAddon(duration=1000, digital=False),
        attenuation=10,
        current_external_voltage=-0.0047219,
        qdac_spec=QdacSpec(qdac_output_port=2),
    )
    sensor_dc_1 = VoltageGate(
        id="sensor_DC_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=4,
            output_mode="direct",
        ),
        sticky=StickyChannelAddon(duration=1000, digital=False),
        attenuation=10,
        current_external_voltage=-0.0047641,
        qdac_spec=QdacSpec(qdac_output_port=3),
    )
    sensor_dc_2 = VoltageGate(
        id="sensor_DC_2",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=5,
            output_mode="direct",
        ),
        sticky=StickyChannelAddon(duration=1000, digital=False),
        attenuation=10,
        current_external_voltage=-0.0047641,
        qdac_spec=QdacSpec(qdac_output_port=4),
    )

    readout_resonator_1 = ReadoutResonatorSingle(
        id="readout_resonator_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=6,
            output_mode="direct",
        ),
        opx_input=LFFEMAnalogInputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=1,
        ),
        intermediate_frequency=150e6,
        operations={
            "readout": pulses.SquareReadoutPulse(
                id="readout",
                length=1000,
                amplitude=0.1,
                integration_weights_angle=0.0,
            )
        },
    )
    readout_resonator_2 = ReadoutResonatorSingle(
        id="readout_resonator_2",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=7,
            output_mode="direct",
        ),
        opx_input=LFFEMAnalogInputPort(
            controller_id=controller,
            fem_id=lf_fem_slot,
            port_id=2,
        ),
        intermediate_frequency=250e6,
        operations={
            "readout": pulses.SquareReadoutPulse(
                id="readout",
                length=1000,
                amplitude=0.1,
                integration_weights_angle=0.0,
            )
        },
    )

    machine.create_virtual_gate_set(
        virtual_channel_mapping={
            "virtual_dot_1": plunger_1,
            "virtual_dot_2": plunger_2,
            "virtual_sensor_1": sensor_dc_1,
            "virtual_sensor_2": sensor_dc_2,
        },
        gate_set_id="main_qpu",
        compensation_matrix=[
            [1.0, 0.0, 0.020406, 0.020406],
            [0.0, 1.0, 0.029189, 0.029189],
            [0.020406, 0.029189, 1.0, 0.0],
            [0.020406, 0.029189, 0.0, 1.0],
        ],
    )

    machine.register_channel_elements(
        plunger_channels=[plunger_1, plunger_2],
        sensor_resonator_mappings={
            sensor_dc_1: readout_resonator_1,
            sensor_dc_2: readout_resonator_2,
        },
        barrier_channels=[],
    )

    quantum_dot_1 = machine.quantum_dots["virtual_dot_1"]
    quantum_dot_1.add_point_with_step_macro(
        "empty",
        voltages={"virtual_dot_1": -0.1},
        duration=500,
    )
    quantum_dot_1.add_point_with_step_macro(
        "initialize",
        voltages={"virtual_dot_1": 0.05},
        duration=500,
    )
    quantum_dot_1.add_point(
        "measure",
        voltages={"virtual_dot_1": -0.05},
    )

    quantum_dot_2 = machine.quantum_dots["virtual_dot_2"]
    quantum_dot_2.add_point_with_step_macro(
        "empty",
        voltages={"virtual_dot_2": -0.1},
        duration=500,
    )
    quantum_dot_2.add_point_with_step_macro(
        "initialize",
        voltages={"virtual_dot_2": 0.05},
        duration=500,
    )
    quantum_dot_2.add_point(
        "measure",
        voltages={"virtual_dot_2": -0.05},
    )

    return machine
