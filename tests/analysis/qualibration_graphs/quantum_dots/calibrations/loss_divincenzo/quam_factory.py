"""Programmatic QuAM factory for simulation tests."""

from __future__ import annotations

from typing import List, Tuple

from quam.components import pulses  # type: ignore[import-not-found]
from quam.components.channels import StickyChannelAddon  # type: ignore[import-not-found]
from quam.components.hardware import FrequencyConverter, LocalOscillator  # type: ignore[import-not-found]
from quam.components.ports import (  # type: ignore[import-not-found]
    LFFEMAnalogInputPort,
    LFFEMAnalogOutputPort,
    MWFEMAnalogOutputPort,
)

from quam_builder.architecture.quantum_dots.components import (  # type: ignore[import-not-found]
    VoltageGate,
)

from quam_builder.architecture.quantum_dots.components import XYDriveMW as XYDrive  # type: ignore[import-not-found]
from quam_builder.architecture.quantum_dots.components.readout_resonator import (  # type: ignore[import-not-found]
    ReadoutResonatorIQ,
    ReadoutResonatorSingle,
)
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam  # type: ignore[import-not-found]
from quam_builder.architecture.quantum_dots.qubit import LDQubit  # type: ignore[import-not-found]

from .macros import MeasureMacro, X180Macro, X90Macro  # type: ignore[import-not-found]

# Compatibility shim for quam-builder feat/quantum_dots: ReadoutResonatorIQ.__post_init__
# expects opx_output, but InOutIQChannel defines opx_output_I/Q only.
if not hasattr(ReadoutResonatorIQ, "opx_output"):
    ReadoutResonatorIQ.opx_output = property(lambda self: self.opx_output_I)


def _create_minimal_machine() -> Tuple[LossDiVincenzoQuam, dict]:
    """Create a machine configuration with 4 qubits in two pairs."""
    # pylint: disable=unexpected-keyword-arg
    machine = LossDiVincenzoQuam()

    controller = "con1"
    lf_fem_slot_1 = 2  # For qubit pair 1 (Q1, Q2)
    lf_fem_slot_2 = 3  # For qubit pair 2 (Q3, Q4)
    mw_fem_slot = 1

    plungers = {}
    for i in range(1, 3):
        plungers[i] = VoltageGate(
            id=f"plunger_{i}",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=i,
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        )
    for i in range(3, 5):
        plungers[i] = VoltageGate(
            id=f"plunger_{i}",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=i - 2,
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        )

    sensor_dcs = {
        1: VoltageGate(
            id="sensor_DC_1",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=3,
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        ),
        2: VoltageGate(
            id="sensor_DC_2",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=3,
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        ),
    }

    # readout_resonators = {
    #     1: ReadoutResonatorIQ(
    #         id="sensor_resonator_1",
    #         opx_output_I=LFFEMAnalogOutputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_1,
    #             port_id=4,
    #             output_mode="direct",
    #         ),
    #         opx_output_Q=LFFEMAnalogOutputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_1,
    #             port_id=5,
    #             output_mode="direct",
    #         ),
    #         opx_input_I=LFFEMAnalogInputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_1,
    #             port_id=1,
    #         ),
    #         opx_input_Q=LFFEMAnalogInputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_1,
    #             port_id=2,
    #         ),
    #         frequency_converter_up=FrequencyConverter(
    #             local_oscillator=LocalOscillator(frequency=5e9),
    #         ),
    #         intermediate_frequency=50e6,
    #         operations={
    #             "readout": pulses.SquareReadoutPulse(
    #                 length=1000,
    #                 amplitude=0.1,
    #                 integration_weights_angle=0.0,
    #             )
    #         },
    #     ),
    #     2: ReadoutResonatorIQ(
    #         id="sensor_resonator_2",
    #         opx_output_I=LFFEMAnalogOutputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_2,
    #             port_id=4,
    #             output_mode="direct",
    #         ),
    #         opx_output_Q=LFFEMAnalogOutputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_2,
    #             port_id=5,
    #             output_mode="direct",
    #         ),
    #         opx_input_I=LFFEMAnalogInputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_2,
    #             port_id=1,
    #         ),
    #         opx_input_Q=LFFEMAnalogInputPort(
    #             controller_id=controller,
    #             fem_id=lf_fem_slot_2,
    #             port_id=2,
    #         ),
    #         frequency_converter_up=FrequencyConverter(
    #             local_oscillator=LocalOscillator(frequency=5e9),
    #         ),
    #         intermediate_frequency=50e6,
    #         operations={
    #             "readout": pulses.SquareReadoutPulse(
    #                 length=1000,
    #                 amplitude=0.1,
    #                 integration_weights_angle=0.0,
    #             )
    #         },
    #     ),
    # }

    readout_resonators = {
        1: ReadoutResonatorSingle(
            id="sensor_resonator_1",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=4,
                output_mode="direct",
            ),
            opx_input=LFFEMAnalogInputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
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
        ),
        2: ReadoutResonatorSingle(
            id="sensor_resonator_2",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=4,
                output_mode="direct",
            ),
            opx_input=LFFEMAnalogInputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=2,
            ),
            intermediate_frequency=50e6,
            operations={
                "readout": pulses.SquareReadoutPulse(
                    length=1000,
                    amplitude=0.1,
                    integration_weights_angle=0.0,
                )
            },
        ),
    }

    xy_drives = {}
    for i in range(1, 5):
        _xy_kwargs = dict(
            id=f"Q{i}_xy",
            opx_output=MWFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=mw_fem_slot,
                port_id=i,
                upconverter_frequency=5e9,
                band=2,
                full_scale_power_dbm=10,
            ),
            intermediate_frequency=100e6,
        )
        xy_drives[i] = XYDrive(**_xy_kwargs)
        length = 100
        xy_drives[i].operations["x180"] = pulses.GaussianPulse(length=length, amplitude=0.2, sigma=length / 6)

    machine.create_virtual_gate_set(
        virtual_channel_mapping={
            "virtual_dot_1": plungers[1],
            "virtual_dot_2": plungers[2],
            "virtual_dot_3": plungers[3],
            "virtual_dot_4": plungers[4],
            "virtual_sensor_1": sensor_dcs[1],
            "virtual_sensor_2": sensor_dcs[2],
        },
        gate_set_id="main_qpu",
        compensation_matrix=[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    machine.register_channel_elements(
        plunger_channels=list(plungers.values()),
        sensor_resonator_mappings={
            sensor_dcs[1]: readout_resonators[1],
            sensor_dcs[2]: readout_resonators[2],
        },
        barrier_channels=[],
    )

    machine.register_quantum_dot_pair(
        quantum_dot_ids=["virtual_dot_1", "virtual_dot_2"],
        sensor_dot_ids=["virtual_sensor_1"],
        id="qd_pair_1_2",
    )
    machine.register_quantum_dot_pair(
        quantum_dot_ids=["virtual_dot_3", "virtual_dot_4"],
        sensor_dot_ids=["virtual_sensor_2"],
        id="qd_pair_3_4",
    )

    sensor_dot_1 = machine.sensor_dots["virtual_sensor_1"]  # pylint: disable=unsubscriptable-object
    sensor_dot_1._add_readout_params(  # pylint: disable=protected-access
        quantum_dot_pair_id="qd_pair_1_2",
        threshold=0.5,
    )

    sensor_dot_2 = machine.sensor_dots["virtual_sensor_2"]  # pylint: disable=unsubscriptable-object
    sensor_dot_2._add_readout_params(  # pylint: disable=protected-access
        quantum_dot_pair_id="qd_pair_3_4",
        threshold=0.5,
    )

    return machine, xy_drives


def _register_qubits_with_points(
    machine: LossDiVincenzoQuam,
    xy_drives: dict,
) -> List[LDQubit]:
    """Register 4 LDQubits with voltage points and custom macros."""
    qubit_configs = [
        ("Q1", "virtual_dot_1", "virtual_dot_2", 1),
        ("Q2", "virtual_dot_2", "virtual_dot_1", 2),
        ("Q3", "virtual_dot_3", "virtual_dot_4", 3),
        ("Q4", "virtual_dot_4", "virtual_dot_3", 4),
    ]

    qubits = []

    for qubit_name, dot_id, readout_dot_id, xy_idx in qubit_configs:
        machine.register_qubit(
            qubit_name=qubit_name,
            quantum_dot_id=dot_id,
            xy=xy_drives[xy_idx],
            readout_quantum_dot=readout_dot_id,
        )

        qubit = machine.qubits[qubit_name]  # pylint: disable=unsubscriptable-object

        qubit.add_point_with_step_macro(
            "empty",
            voltages={f"virtual_dot_{xy_idx}": -0.1},
            duration=500,
        )
        qubit.add_point_with_step_macro(
            "initialize",
            voltages={f"virtual_dot_{xy_idx}": 0.05},
            duration=500,
        )
        qubit.add_point(
            "measure",
            voltages={f"virtual_dot_{xy_idx}": -0.05},
        )

        qubit.macros["x180"] = X180Macro(pulse_name="x180", amplitude_scale=1.0)
        qubit.macros["x90"] = X90Macro(pulse_name="x180", amplitude_scale=0.5)
        qubit.macros["measure"] = MeasureMacro(
            pulse_name="readout",
            readout_duration=2000,
        )

        qubits.append(qubit)

    return qubits


def create_minimal_quam() -> LossDiVincenzoQuam:
    """Create a minimal LossDiVincenzoQuam for simulation tests."""
    machine, xy_drives = _create_minimal_machine()
    _register_qubits_with_points(machine, xy_drives)
    machine.active_qubit_names = list(machine.qubits.keys())  # pylint: disable=no-member
    return machine
