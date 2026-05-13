"""
QUAM configuration for OPX1000 + LF-FEM + MW-FEM using the **quam-builder full
pipeline** (``qualang_tools.wirer`` + ``build_quam_wiring`` + ``build_quam``).

This complements :mod:`configuration_quam_lf_fem_and_mw_fem` (manual ``quam.components``)
and :mod:`configuration_quam_builder_components_lf_mw_fem` (architecture only, no wirer).

On first import, wiring is allocated for a single qubit, ``build_quam_wiring`` and
``build_quam`` populate the QUAM tree, hardware/pulse defaults are applied to match the
manual LF/MW-FEM example, and state is saved to ``quam_state_builder_lf_mw_fem.json``
next to this file. Later imports load that JSON.

Experiments should read parameters from the structured machine, e.g.
``machine.qubits["q1"].xy.operations["x180"]`` (after ``set_gate_shape("DragGaussian")``,
aliases point ``x180`` at the DRAG pulse block).

Legacy shims at the bottom mirror the manual configuration module for host/T1 scalars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from qualang_tools.units import unit
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec
from quam.components.pulses import SquarePulse, SquareReadoutPulse
from quam.core import quam_dataclass
from quam.serialisation.json import JSONSerialiser

from quam_builder.architecture.superconducting.qpu import FluxTunableQuam
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_builder.builder.superconducting.pulses import add_DragGaussian_pulses

u = unit(coerce_to_integer=True)

#############################################
#         State file (single JSON)          #
#############################################
state_path = Path(__file__).parent / "quam_state_builder_lf_mw_fem.json"


#############################################
#    Root QUAM (serialiser → state_path)    #
#############################################
@quam_dataclass
class FluxTunableTransmonQuamBuilder(FluxTunableQuam):
    """Flux-tunable QUAM root with extra experiment scalars (same role as manual file)."""

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
        # Single .json path ⇒ full state in one file; default path for builder save() calls.
        return JSONSerialiser(
            content_mapping={"wiring": "wiring.json", "network": "wiring.json"},
            state_path=state_path,
        )


def _populate_after_build(machine: FluxTunableTransmonQuamBuilder) -> None:
    """Align frequencies/pulses with :mod:`configuration_quam_lf_fem_and_mw_fem`."""
    q = machine.qubits["q1"]

    machine.qop_ip = machine.network.get("host", machine.qop_ip)
    machine.cluster_name = machine.network.get("cluster_name", machine.cluster_name)
    machine.qop_port = machine.network.get("port", machine.qop_port)

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
    # Break MWChannel default ref cycle (RF <-> IF via inferred_*).
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


#############################################
#            Build OR Load machine          #
#############################################
if state_path.exists():
    machine = FluxTunableTransmonQuamBuilder.load(state_path)
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

    machine = FluxTunableTransmonQuamBuilder()
    machine.qop_ip = host_ip
    machine.cluster_name = cluster_name
    machine.qop_port = port

    build_quam_wiring(connectivity, host_ip, cluster_name, machine, port=port)
    machine = FluxTunableTransmonQuamBuilder.load(state_path)
    build_quam(machine)
    _populate_after_build(machine)


#############################################
#         Generate the QUA configuration    #
#############################################
config = machine.generate_config()


#############################################
#   Legacy shim (derived from machine)      #
#############################################
qop_ip = machine.qop_ip
cluster_name = machine.cluster_name
qop_port = machine.qop_port

qubit_T1 = machine.qubit_T1
thermalization_time = machine.thermalization_time
depletion_time = machine.depletion_time
flux_settle_time = machine.flux_settle_time
ge_threshold = machine.ge_threshold
rotation_angle = machine.rotation_angle
