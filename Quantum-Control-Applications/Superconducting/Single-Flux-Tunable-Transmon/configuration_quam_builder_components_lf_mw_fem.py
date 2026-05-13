"""
QUAM configuration for OPX1000 + LF-FEM + MW-FEM using **quam-builder architecture
components only** (no ``qualang_tools.wirer`` / ``build_quam_wiring``).

Ports and FEM settings are declared explicitly (same layout as
:mod:`configuration_quam_lf_fem_and_mw_fem`), while the logical qubit uses
``FluxTunableTransmon`` with ``XYDriveMW``, ``ReadoutResonatorMW``, and ``FluxLine``
from ``quam_builder.architecture``.

State file: ``quam_state_builder_components_lf_mw_fem.json`` (single JSON next to this file).

Use this when you want quam-builder qubit helpers (e.g. ``machine.qubits['q1'].reset()``,
``set_output_power``) without the auto-wiring pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from qualang_tools.units import unit
from quam.components.pulses import SquarePulse, SquareReadoutPulse
from quam.components.ports import FEMPortsContainer
from quam.core import quam_dataclass
from quam.serialisation.json import JSONSerialiser

from quam_builder.architecture.superconducting.components.flux_line import FluxLine
from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorMW,
)
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam
from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from quam_builder.builder.superconducting.pulses import add_DragGaussian_pulses

u = unit(coerce_to_integer=True)

state_path = Path(__file__).parent / "quam_state_builder_components_lf_mw_fem.json"


@quam_dataclass
class FluxTunableTransmonQuamComponents(FluxTunableQuam):
    """Root QUAM with explicit FEM ports + a single ``FluxTunableTransmon``."""

    qop_ip: str = "127.0.0.1"
    cluster_name: Optional[str] = None
    qop_port: Optional[int] = None

    qubit_T1: int = 10_000
    thermalization_time: int = 50_000
    depletion_time: int = 2_000
    flux_settle_time: int = 100

    ge_threshold: float = 0.0
    rotation_angle: float = 0.0

    amplitude_fit: float = 0.0
    frequency_fit: float = 0.0
    phase_fit: float = 0.0
    offset_fit: float = 0.0

    @classmethod
    def get_serialiser(cls) -> JSONSerialiser:
        return JSONSerialiser(
            content_mapping={"wiring": "wiring.json", "network": "wiring.json"},
            state_path=state_path,
        )


def _populate_single_qubit(machine: FluxTunableTransmonQuamComponents) -> None:
    q = machine.qubits["q1"]

    machine.qubit_T1 = int(10 * u.us)
    machine.thermalization_time = 5 * machine.qubit_T1
    machine.depletion_time = int(2 * u.us)
    machine.flux_settle_time = int(100 * u.ns)
    machine.ge_threshold = 0.0
    machine.rotation_angle = 0.0

    q.T1 = machine.qubit_T1 * 1e-9
    q.anharmonicity = int(-200 * u.MHz)
    q.f_01 = float(int(7.4 * u.GHz) + int(110 * u.MHz))

    q.xy.opx_output.upconverter_frequency = int(7.4 * u.GHz)
    q.xy.opx_output.band = 2
    q.xy.opx_output.full_scale_power_dbm = 1
    q.xy.LO_frequency = None
    q.xy.LO_frequency = int(7.4 * u.GHz)
    q.xy.intermediate_frequency = None
    q.xy.intermediate_frequency = int(110 * u.MHz)
    q.xy.RF_frequency = None
    q.xy.RF_frequency = q.f_01

    rr_lo = int(5.5 * u.GHz)
    rr_if = int(60 * u.MHz)
    q.resonator.f_01 = float(rr_lo + rr_if)
    q.resonator.LO_frequency = None
    q.resonator.LO_frequency = rr_lo
    q.resonator.intermediate_frequency = None
    q.resonator.intermediate_frequency = rr_if
    q.resonator.RF_frequency = None
    q.resonator.RF_frequency = q.resonator.f_01
    q.resonator.opx_output.upconverter_frequency = rr_lo
    q.resonator.opx_output.band = 2
    q.resonator.opx_output.full_scale_power_dbm = 1
    q.resonator.opx_input.band = 2
    q.resonator.time_of_flight = 28
    q.resonator.depletion_time = machine.depletion_time

    if q.z is not None:
        q.z.opx_output.output_mode = "amplified"
        q.z.opx_output.upsampling_mode = "pulse"
        q.z.opx_output.sampling_rate = int(1e9)
        q.z.opx_output.delay = 141
        q.z.opx_output.shareable = True
        q.z.settle_time = machine.flux_settle_time
        q.z.flux_point = "joint"

    add_DragGaussian_pulses(
        q,
        amplitude=1.0,
        length=40,
        sigma=8,
        alpha=0.0,
        detuning=0.0,
        anharmonicity=int(-200 * u.MHz),
        subtracted=True,
        digital_marker=None,
    )
    q.set_gate_shape("DragGaussian")

    q.xy.operations["cw"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    q.xy.operations["saturation"] = SquarePulse(
        length=int(10 * u.us), amplitude=0.03, axis_angle=0.0
    )
    q.xy.operations["pi"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    q.xy.operations["pi_half"] = SquarePulse(length=100, amplitude=0.015, axis_angle=0.0)

    q.resonator.operations["cw"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    q.resonator.operations["readout"] = SquareReadoutPulse(
        length=5000,
        amplitude=0.6,
        digital_marker="ON",
        threshold=machine.ge_threshold,
    )

    if q.z is not None:
        q.z.operations["const"] = SquarePulse(length=200, amplitude=0.45)

    machine.save(state_path)


if state_path.exists():
    machine = FluxTunableTransmonQuamComponents.load(state_path)
else:
    machine = FluxTunableTransmonQuamComponents(ports=FEMPortsContainer())

    machine.network = {
        "host": "127.0.0.1",
        "port": None,
        "cluster_name": "Cluster_1",
    }
    machine.qop_ip = machine.network["host"]
    machine.cluster_name = machine.network["cluster_name"]
    machine.qop_port = machine.network["port"]

    mw_res_out = machine.ports.get_mw_output(
        "con1",
        1,
        1,
        create=True,
        band=2,
        upconverter_frequency=int(5.5 * u.GHz),
        full_scale_power_dbm=1,
    )
    mw_qubit_out = machine.ports.get_mw_output(
        "con1",
        1,
        2,
        create=True,
        band=2,
        upconverter_frequency=int(7.4 * u.GHz),
        full_scale_power_dbm=1,
    )
    mw_res_in = machine.ports.get_mw_input(
        "con1",
        1,
        1,
        create=True,
        band=2,
        downconverter_frequency=mw_res_out.get_reference("upconverter_frequency"),
    )
    lf_flux = machine.ports.get_analog_output(
        "con1",
        5,
        1,
        create=True,
        offset=0.0,
        output_mode="amplified",
        sampling_rate=int(1e9),
        upsampling_mode="pulse",
        delay=141,
        shareable=True,
    )

    qubit = FluxTunableTransmon(
        id="q1",
        xy=XYDriveMW(opx_output=mw_qubit_out.get_reference()),
        resonator=ReadoutResonatorMW(
            opx_output=mw_res_out.get_reference(),
            opx_input=mw_res_in.get_reference(),
        ),
        z=FluxLine(opx_output=lf_flux.get_reference()),
    )
    machine.qubits["q1"] = qubit
    machine.active_qubit_names = ["q1"]

    _populate_single_qubit(machine)

config = machine.generate_config()

qop_ip = machine.qop_ip
cluster_name = machine.cluster_name
qop_port = machine.qop_port

qubit_T1 = machine.qubit_T1
thermalization_time = machine.thermalization_time
depletion_time = machine.depletion_time
flux_settle_time = machine.flux_settle_time
ge_threshold = machine.ge_threshold
rotation_angle = machine.rotation_angle
