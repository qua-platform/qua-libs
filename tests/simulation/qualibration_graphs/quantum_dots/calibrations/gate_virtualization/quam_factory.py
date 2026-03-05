"""Programmatic QuAM factory for gate virtualization simulation tests."""

from __future__ import annotations

from quam.components import pulses  # type: ignore[import-not-found]
from quam.components.channels import StickyChannelAddon  # type: ignore[import-not-found]
from quam.components.ports import (  # type: ignore[import-not-found]
    LFFEMAnalogInputPort,
    LFFEMAnalogOutputPort,
)

from quam_builder.architecture.quantum_dots.components import (  # type: ignore[import-not-found]
    VoltageGate,
)
from quam_builder.architecture.quantum_dots.components.readout_resonator import (  # type: ignore[import-not-found]
    ReadoutResonatorSingle,
)
from quam_builder.architecture.quantum_dots.qpu import (  # type: ignore[import-not-found]
    LossDiVincenzoQuam,
)


def create_minimal_quam() -> LossDiVincenzoQuam:
    """Create a minimal QuAM with one sensor and two virtual dot gates.

    The machine contains only the gate/sensor/readout components required to
    compile and simulate gate-virtualization scans.
    """
    machine = LossDiVincenzoQuam()
    controller = "con1"
    fem_slot = 2

    plunger_1 = VoltageGate(
        id="plunger_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=fem_slot,
            port_id=1,
            output_mode="direct",
        ),
        sticky=StickyChannelAddon(duration=16, digital=False),
    )
    plunger_2 = VoltageGate(
        id="plunger_2",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=fem_slot,
            port_id=2,
            output_mode="direct",
        ),
        sticky=StickyChannelAddon(duration=16, digital=False),
    )
    sensor_dc = VoltageGate(
        id="sensor_DC_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=fem_slot,
            port_id=3,
            output_mode="direct",
        ),
        sticky=StickyChannelAddon(duration=16, digital=False),
    )

    readout_resonator = ReadoutResonatorSingle(
        id="sensor_resonator_1",
        opx_output=LFFEMAnalogOutputPort(
            controller_id=controller,
            fem_id=fem_slot,
            port_id=4,
            output_mode="direct",
        ),
        opx_input=LFFEMAnalogInputPort(
            controller_id=controller,
            fem_id=fem_slot,
            port_id=1,
        ),
        intermediate_frequency=50e6,
        operations={
            "readout": pulses.SquareReadoutPulse(
                length=1000,
                amplitude=0.1,
                integration_weights_angle=0.0,
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
