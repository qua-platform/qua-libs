"""Dedicated QuAM factory for simulated video mode."""

from __future__ import annotations

from quam.components import pulses
from quam.components.channels import StickyChannelAddon
from quam.components.hardware import FrequencyConverter, LocalOscillator
from quam.components.ports import LFFEMAnalogInputPort, LFFEMAnalogOutputPort
from quam_builder.architecture.quantum_dots.components import VoltageGate
from quam_builder.architecture.quantum_dots.components.readout_resonator import ReadoutResonatorSingle
from quam_builder.architecture.quantum_dots.components.voltage_gate import QdacSpec
from quam_builder.architecture.quantum_dots.components.xy_drive import XYDriveIQ

# from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam
from calibration_utils.run_video_mode.simulated_video_mode.demo_files.demo_quam_ld import (
    DemoQuamLD as LossDiVincenzoQuam,
)
from quam_builder.architecture.quantum_dots.qubit import LDQubit

from calibration_utils.run_video_mode.simulated_video_mode.quam_macros import (
    MeasureMacro,
    XGateMacro,
    YGateMacro,
    ZGateMacro,
)


def _create_xy_drives(controller: str, xy_fem_slot: int) -> dict[int, XYDriveIQ]:
    pulse_length_ns = 100
    pulse_sigma_ns = pulse_length_ns / 6
    x180_amplitude = 0.2
    x90_amplitude = x180_amplitude / 2

    xy_drives: dict[int, XYDriveIQ] = {}
    for qubit_index, (port_i, port_q) in {1: (1, 2), 2: (3, 4)}.items():
        xy_drive = XYDriveIQ(
            id=f"Q{qubit_index}_xy",
            opx_output_I=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=xy_fem_slot,
                port_id=port_i,
                output_mode="direct",
            ),
            opx_output_Q=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=xy_fem_slot,
                port_id=port_q,
                output_mode="direct",
            ),
            frequency_converter_up=FrequencyConverter(
                local_oscillator=LocalOscillator(frequency=0),
            ),
            intermediate_frequency=100e6,
        )
        xy_drive.operations["x180"] = pulses.GaussianPulse(
            length=pulse_length_ns,
            amplitude=x180_amplitude,
            sigma=pulse_sigma_ns,
        )
        xy_drive.operations["x90"] = pulses.GaussianPulse(
            length=pulse_length_ns,
            amplitude=x90_amplitude,
            sigma=pulse_sigma_ns,
        )
        xy_drive.operations["-x90"] = pulses.GaussianPulse(
            length=pulse_length_ns,
            amplitude=-x90_amplitude,
            sigma=pulse_sigma_ns,
        )
        xy_drive.operations["y90"] = pulses.GaussianPulse(
            length=pulse_length_ns,
            amplitude=x90_amplitude,
            sigma=pulse_sigma_ns,
        )
        xy_drive.operations["-y90"] = pulses.GaussianPulse(
            length=pulse_length_ns,
            amplitude=-x90_amplitude,
            sigma=pulse_sigma_ns,
        )
        xy_drives[qubit_index] = xy_drive

    return xy_drives


def _register_charge_stability_points(machine: LossDiVincenzoQuam) -> None:
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


def _register_qubit_readout_topology(machine: LossDiVincenzoQuam) -> None:
    machine.register_quantum_dot_pair(
        quantum_dot_ids=["virtual_dot_1", "virtual_dot_2"],
        sensor_dot_ids=["virtual_sensor_1"],
        id="qd_pair_1_2",
    )
    machine.sensor_dots["virtual_sensor_1"]._add_readout_params(  # pylint: disable=protected-access
        quantum_dot_pair_id="qd_pair_1_2",
        threshold=0.0,
    )


def _register_qubits(machine: LossDiVincenzoQuam, xy_drives: dict[int, XYDriveIQ]) -> list[LDQubit]:
    pulse_duration_cc = 25
    qubit_configs = [
        ("Q1", "virtual_dot_1", "virtual_dot_2", 1),
        ("Q2", "virtual_dot_2", "virtual_dot_1", 2),
    ]
    qubits: list[LDQubit] = []

    for qubit_name, quantum_dot_id, readout_quantum_dot, drive_index in qubit_configs:
        machine.register_qubit(
            qubit_name=qubit_name,
            quantum_dot_id=quantum_dot_id,
            xy=xy_drives[drive_index],
            readout_quantum_dot=readout_quantum_dot,
        )
        qubit = machine.qubits[qubit_name]
        qubit.id = qubit_name

        qubit.add_point_with_step_macro(
            "empty",
            voltages={quantum_dot_id: -0.1},
            duration=500,
        )
        qubit.add_point_with_step_macro(
            "initialize",
            voltages={quantum_dot_id: 0.05},
            duration=500,
        )
        qubit.add_point(
            "measure",
            voltages={quantum_dot_id: -0.05},
        )

        qubit.macros["x180"] = XGateMacro(
            pulse_name="x180",
            amplitude_scale=1.0,
            duration=pulse_duration_cc,
        )
        qubit.macros["x90"] = XGateMacro(
            pulse_name="x180",
            amplitude_scale=0.5,
            duration=pulse_duration_cc,
        )
        qubit.macros["xm90"] = XGateMacro(
            pulse_name="x180",
            amplitude_scale=-0.5,
            duration=pulse_duration_cc,
        )
        qubit.macros["y180"] = YGateMacro(
            pulse_name="x180",
            amplitude_scale=1.0,
            duration=pulse_duration_cc,
        )
        qubit.macros["y90"] = YGateMacro(
            pulse_name="x180",
            amplitude_scale=0.5,
            duration=pulse_duration_cc,
        )
        qubit.macros["ym90"] = YGateMacro(
            pulse_name="x180",
            amplitude_scale=-0.5,
            duration=pulse_duration_cc,
        )
        qubit.macros["z90"] = ZGateMacro(theta=90.0)
        qubit.macros["z180"] = ZGateMacro(theta=180.0)
        qubit.macros["zm90"] = ZGateMacro(theta=-90.0)
        qubit.macros["measure"] = MeasureMacro(
            pulse_name="readout",
            readout_duration=2000,
        )
        qubits.append(qubit)

    machine.active_qubit_names = list(machine.qubits.keys())
    return qubits


def create_minimal_quam(
    host_ip: str = "172.16.33.115",
    cluster_name: str = "CS_4",
    create_dc_set: bool = True,
    qdac_ip: str = "172.16.33.101",
) -> LossDiVincenzoQuam:
    """Create the 2-dot/2-sensor QuAM used by simulated video mode."""
    machine = LossDiVincenzoQuam()
    machine.network["host"] = host_ip
    machine.network["cluster_name"] = cluster_name

    controller = "con1"
    lf_fem_slot = 5
    xy_fem_slot = 6

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

    xy_drives = _create_xy_drives(controller, xy_fem_slot)

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
    if create_dc_set:
        machine.network["qdac_ip"] = qdac_ip
        machine.connect_to_external_source(external_qdac=True)
        machine.create_virtual_dc_set("main_qpu")

    machine.register_channel_elements(
        plunger_channels=[plunger_1, plunger_2],
        sensor_resonator_mappings={
            sensor_dc_1: readout_resonator_1,
            sensor_dc_2: readout_resonator_2,
        },
        barrier_channels=[],
    )

    _register_charge_stability_points(machine)
    _register_qubit_readout_topology(machine)
    _register_qubits(machine, xy_drives)

    return machine
