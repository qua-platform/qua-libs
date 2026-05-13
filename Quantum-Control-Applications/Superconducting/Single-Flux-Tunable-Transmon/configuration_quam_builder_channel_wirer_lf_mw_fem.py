"""
QUAM configuration for OPX1000 + LF-FEM + MW-FEM using **quam-builder channel
components** (``XYDriveMW``, ``ReadoutResonatorMW``, ``FluxLine``) as top-level
``machine.channels`` entries — no ``FluxTunableTransmon`` wrapper.

``qualang_tools.wirer`` allocates MW/LF FEM ports; ports are then created on a
``BasicFEMQuam`` root and wired to standalone builder components (same pattern as
:mod:`configuration_quam_lf_fem_and_mw_fem`, with a type upgrade).

State file: ``quam_state_builder_channel_wirer_lf_mw_fem.json`` (next to this file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from qualang_tools.units import unit
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring
from qualang_tools.wirer.connectivity.element import QubitReference, WiringLineType
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec
from quam.components import BasicFEMQuam
from quam.components.channels import StickyChannelAddon
from quam.components.ports import FEMPortsContainer
from quam.components.pulses import DragGaussianPulse, SquarePulse, SquareReadoutPulse
from quam.core import quam_dataclass
from quam.serialisation.json import JSONSerialiser

from quam_builder.architecture.superconducting.components.flux_line import FluxLine
from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorMW,
)
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW

u = unit(coerce_to_integer=True)

state_path = Path(__file__).parent / "quam_state_builder_channel_wirer_lf_mw_fem.json"


@quam_dataclass
class StandaloneChannelQuamWirer(BasicFEMQuam):
    """FEM QUAM root with flat ``channels`` using quam-builder MW/LF components."""

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
        return JSONSerialiser(state_path=state_path)


def _controller_id(con: int) -> str:
    return f"con{con}"


def _allocate_wired_ports(
    machine: StandaloneChannelQuamWirer, connectivity: Connectivity, qubit_index: int = 1
) -> Tuple[object, object, object, object]:
    """Create FEM ports on ``machine`` from wirer allocation for one qubit."""
    ref = QubitReference(index=qubit_index)
    if ref not in connectivity.elements:
        raise KeyError(f"No wired element for {ref!r} in connectivity.elements")
    element = connectivity.elements[ref]
    chmap = element.channels

    rr_chs = chmap[WiringLineType.RESONATOR]
    rr_in = next(c for c in rr_chs if c.io_type == "input")
    rr_out = next(c for c in rr_chs if c.io_type == "output")
    xy_out = chmap[WiringLineType.DRIVE][0]
    z_out = chmap[WiringLineType.FLUX][0]

    mw_res_out = machine.ports.get_mw_output(
        _controller_id(rr_out.con),
        rr_out.slot,
        rr_out.port,
        create=True,
        band=2,
        upconverter_frequency=int(5.5 * u.GHz),
        full_scale_power_dbm=1,
    )
    mw_qubit_out = machine.ports.get_mw_output(
        _controller_id(xy_out.con),
        xy_out.slot,
        xy_out.port,
        create=True,
        band=2,
        upconverter_frequency=int(7.4 * u.GHz),
        full_scale_power_dbm=1,
    )
    mw_res_in = machine.ports.get_mw_input(
        _controller_id(rr_in.con),
        rr_in.slot,
        rr_in.port,
        create=True,
        band=2,
        downconverter_frequency=mw_res_out.get_reference("upconverter_frequency"),
    )
    lf_flux = machine.ports.get_analog_output(
        _controller_id(z_out.con),
        z_out.slot,
        z_out.port,
        create=True,
        offset=0.0,
        output_mode="amplified",
        sampling_rate=int(1e9),
        upsampling_mode="pulse",
        delay=141,
        shareable=True,
    )
    return mw_res_out, mw_qubit_out, mw_res_in, lf_flux


def _populate_scalars_and_channels(machine: StandaloneChannelQuamWirer) -> None:
    """Match hardware/pulses to :mod:`configuration_quam_lf_fem_and_mw_fem`."""
    machine.qubit_T1 = int(10 * u.us)
    machine.thermalization_time = 5 * machine.qubit_T1
    machine.depletion_time = int(2 * u.us)
    machine.flux_settle_time = int(100 * u.ns)
    machine.ge_threshold = 0.0
    machine.rotation_angle = 0.0

    qubit = machine.channels["qubit"]
    res = machine.channels["resonator"]
    flux = machine.channels["flux_line"]
    flux_sticky = machine.channels["flux_line_sticky"]

    qubit.opx_output.upconverter_frequency = int(7.4 * u.GHz)
    qubit.opx_output.band = 2
    qubit.opx_output.full_scale_power_dbm = 1
    qubit.LO_frequency = None
    qubit.LO_frequency = int(7.4 * u.GHz)
    qubit.intermediate_frequency = None
    qubit.intermediate_frequency = int(110 * u.MHz)
    qubit.RF_frequency = None
    qubit.RF_frequency = float(int(7.4 * u.GHz) + int(110 * u.MHz))

    rr_lo = int(5.5 * u.GHz)
    rr_if = int(60 * u.MHz)
    res.f_01 = float(rr_lo + rr_if)
    res.LO_frequency = None
    res.LO_frequency = rr_lo
    res.intermediate_frequency = None
    res.intermediate_frequency = rr_if
    res.RF_frequency = None
    res.RF_frequency = res.f_01
    res.opx_output.upconverter_frequency = rr_lo
    res.opx_output.band = 2
    res.opx_output.full_scale_power_dbm = 1
    res.opx_input.band = 2
    res.time_of_flight = 28
    res.smearing = 0
    res.depletion_time = machine.depletion_time

    for zch in (flux, flux_sticky):
        zch.opx_output.output_mode = "amplified"
        zch.opx_output.upsampling_mode = "pulse"
        zch.opx_output.sampling_rate = int(1e9)
        zch.opx_output.delay = 141
        zch.opx_output.shareable = True
        zch.settle_time = machine.flux_settle_time
        zch.flux_point = "joint"

    qubit_ops = qubit.operations
    qubit_ops["cw"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    qubit_ops["saturation"] = SquarePulse(
        length=int(10 * u.us), amplitude=0.03, axis_angle=0.0
    )
    qubit_ops["pi"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    qubit_ops["pi_half"] = SquarePulse(length=100, amplitude=0.015, axis_angle=0.0)
    qubit_ops["x180"] = DragGaussianPulse(
        length=40,
        amplitude=1.0,
        sigma=8.0,
        alpha=0.0,
        anharmonicity=int(-200 * u.MHz),
        detuning=0.0,
        axis_angle=0.0,
    )
    qubit_ops["x90"] = DragGaussianPulse(
        length=40,
        amplitude=0.5,
        sigma=8.0,
        alpha=0.0,
        anharmonicity=int(-200 * u.MHz),
        detuning=0.0,
        axis_angle=0.0,
    )
    qubit_ops["-x90"] = DragGaussianPulse(
        length=40,
        amplitude=-0.5,
        sigma=8.0,
        alpha=0.0,
        anharmonicity=int(-200 * u.MHz),
        detuning=0.0,
        axis_angle=0.0,
    )
    qubit_ops["y180"] = DragGaussianPulse(
        length=40,
        amplitude=1.0,
        sigma=8.0,
        alpha=0.0,
        anharmonicity=int(-200 * u.MHz),
        detuning=0.0,
        axis_angle=np.pi / 2,
    )
    qubit_ops["y90"] = DragGaussianPulse(
        length=40,
        amplitude=0.5,
        sigma=8.0,
        alpha=0.0,
        anharmonicity=int(-200 * u.MHz),
        detuning=0.0,
        axis_angle=np.pi / 2,
    )
    qubit_ops["-y90"] = DragGaussianPulse(
        length=40,
        amplitude=-0.5,
        sigma=8.0,
        alpha=0.0,
        anharmonicity=int(-200 * u.MHz),
        detuning=0.0,
        axis_angle=np.pi / 2,
    )

    res_ops = res.operations
    res_ops["cw"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    res_ops["readout"] = SquareReadoutPulse(
        length=5000,
        amplitude=0.6,
        digital_marker="ON",
        threshold=machine.ge_threshold,
    )

    flux.operations["const"] = SquarePulse(length=200, amplitude=0.45)
    flux_sticky.operations["const"] = SquarePulse(length=200, amplitude=0.45)

    machine.save(state_path)


if state_path.exists():
    machine = StandaloneChannelQuamWirer.load(state_path)
else:
    host_ip = "127.0.0.1"
    cluster_name = "Cluster_1"
    port: Optional[int] = None

    instruments = Instruments()
    instruments.add_mw_fem(controller=1, slots=[1])
    instruments.add_lf_fem(controller=1, slots=[5])

    connectivity = Connectivity()
    connectivity.add_resonator_line(
        qubits=[1],
        constraints=mw_fem_spec(con=1, slot=1, in_port=1, out_port=1),
    )
    connectivity.add_qubit_drive_lines(
        qubits=[1],
        constraints=mw_fem_spec(con=1, slot=1, in_port=None, out_port=2),
    )
    connectivity.add_qubit_flux_lines(qubits=[1])
    allocate_wiring(connectivity, instruments)

    machine = StandaloneChannelQuamWirer(ports=FEMPortsContainer())
    machine.qop_ip = host_ip
    machine.cluster_name = cluster_name
    machine.qop_port = port

    mw_res_out, mw_qubit_out, mw_res_in, lf_flux = _allocate_wired_ports(
        machine, connectivity, qubit_index=1
    )

    machine.channels["qubit"] = XYDriveMW(opx_output=mw_qubit_out.get_reference())
    machine.channels["resonator"] = ReadoutResonatorMW(
        opx_output=mw_res_out.get_reference(),
        opx_input=mw_res_in.get_reference(),
    )
    machine.channels["flux_line"] = FluxLine(opx_output=lf_flux.get_reference())
    machine.channels["flux_line_sticky"] = FluxLine(
        opx_output=lf_flux.get_reference(),
        sticky=StickyChannelAddon(duration=20, analog=True, digital=True),
    )

    _populate_scalars_and_channels(machine)

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
