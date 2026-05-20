"""
QUAM-based QUA configuration built from ``QuamRoot`` (OPX1000 w/ LF-FEM + MW-FEM).

This variant starts from bare ``QuamRoot`` — the lowest-level QuAM root class —
rather than inheriting ``BasicQuam`` or ``BasicFEMQuam``.  Every container
(``ports``) and every logical grouping (``qubits``) is declared explicitly
on the custom root class.  This gives maximum control over the object graph
and makes the saved JSON fully self-documenting.

The qubit is modelled as a ``FluxTunableTransmonQubit(Qubit)`` that groups
its sub-channels (``xy``, ``resonator``, ``z``, ``z_sticky``) under a single
logical qubit component.  All channel and pulse types are core ``quam``
components — no ``quam-builder`` dependency is required.

First import:  builds the machine and saves ``quam_state_quamroot_lf_mw_fem.json``.
Subsequent imports:  loads the saved JSON state directly.

There are two equivalent ways to change a parameter persistently:

1. Mutate the Python object and save::

       from configuration_quam_quamroot_lf_mw_fem import machine, state_path
       machine.qubits["q1"].xy.operations["x180"].amplitude = 0.95
       machine.save(state_path)

2. Edit ``quam_state_quamroot_lf_mw_fem.json`` directly.

Either path is valid — the JSON file is the source of truth.
"""

from dataclasses import field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from qualang_tools.units import unit

from quam.components import Qubit
from quam.components.channels import (
    InOutMWChannel,
    MWChannel,
    SingleChannel,
    StickyChannelAddon,
)
from quam.components.ports import FEMPortsContainer
from quam.components.pulses import (
    DragGaussianPulse,
    SquarePulse,
    SquareReadoutPulse,
)
from quam.core import QuamRoot, quam_dataclass


u = unit(coerce_to_integer=True)


# ──────────────────────────────────────────────
#  Custom QuAM components (core quam only)
# ──────────────────────────────────────────────
@quam_dataclass
class FluxTunableTransmonQubit(Qubit):
    """Single flux-tunable transmon with XY drive, readout resonator, and flux line.

    Each sub-channel is a core ``quam`` channel type (no quam-builder).
    The qubit groups them so experiments can access
    ``machine.qubits["q1"].xy``, ``.resonator``, ``.z``, ``.z_sticky``.
    """

    xy: MWChannel = None
    resonator: InOutMWChannel = None
    z: SingleChannel = None
    z_sticky: SingleChannel = None

    f_01: float = 0.0
    anharmonicity: int = int(-200e6)
    T1: int = 10_000
    T2ramsey: int = 0
    T2echo: int = 0


# ──────────────────────────────────────────────
#  Custom QuamRoot subclass
# ──────────────────────────────────────────────
@quam_dataclass
class FluxTunableTransmonQuam(QuamRoot):
    """Root QUAM built from ``QuamRoot`` (not ``BasicQuam`` / ``BasicFEMQuam``).

    Every container is declared explicitly:

    * ``ports``  — ``FEMPortsContainer`` (MW + LF port factory helpers)
    * ``qubits`` — dict of ``FluxTunableTransmonQubit`` (qubit-centric model)

    Scalar experiment parameters (network info, characteristic times, fit
    placeholders) are also declared here so they round-trip through
    ``save()`` / ``load()``.
    """

    ports: FEMPortsContainer = None
    qubits: Dict[str, FluxTunableTransmonQubit] = field(default_factory=dict)

    # Network
    qop_ip: str = "127.0.0.1"
    cluster_name: Optional[str] = None
    qop_port: Optional[int] = None

    # Timing / experiment scalars
    thermalization_time: int = 50_000
    depletion_time: int = 2_000
    flux_settle_time: int = 100

    # Discriminator fit results (updated by IQ-blob calibration)
    ge_threshold: float = 0.0
    rotation_angle: float = 0.0

    # Resonator-vs-flux fit placeholders (filled by spectroscopy scripts)
    amplitude_fit: float = 0.0
    frequency_fit: float = 0.0
    phase_fit: float = 0.0
    offset_fit: float = 0.0


# ──────────────────────────────────────────────
#  State file path
# ──────────────────────────────────────────────
state_path = Path(__file__).parent / "quam_state_quamroot_lf_mw_fem.json"


# ──────────────────────────────────────────────
#  Build OR Load machine
# ──────────────────────────────────────────────
if state_path.exists():
    machine = FluxTunableTransmonQuam.load(state_path)
else:
    machine = FluxTunableTransmonQuam(ports=FEMPortsContainer())

    # --- Network / experiment parameters ---
    machine.qop_ip = "127.0.0.1"
    machine.cluster_name = None
    machine.qop_port = None

    machine.thermalization_time = int(50 * u.us)
    machine.depletion_time = int(2 * u.us)
    machine.flux_settle_time = int(100 * u.ns)
    machine.ge_threshold = 0.0
    machine.rotation_angle = 0.0

    # ── MW-FEM ports (band 2: 4.5–7.5 GHz) ──
    mw_res_out = machine.ports.get_mw_output(
        "con1", 1, 1,
        create=True,
        band=2,
        upconverter_frequency=int(5.5 * u.GHz),
        full_scale_power_dbm=1,
    )
    mw_qubit_out = machine.ports.get_mw_output(
        "con1", 1, 2,
        create=True,
        band=2,
        upconverter_frequency=int(7.4 * u.GHz),
        full_scale_power_dbm=1,
    )
    mw_res_in = machine.ports.get_mw_input(
        "con1", 1, 1,
        create=True,
        band=2,
        downconverter_frequency=mw_res_out.get_reference("upconverter_frequency"),
    )

    # ── LF-FEM port (flux line) ──
    # 141 ns delay aligns LF outputs with MW outputs for bands 1 and 3 (use
    # 161 ns for band 2 if alignment is critical).
    lf_flux = machine.ports.get_analog_output(
        "con1", 5, 1,
        create=True,
        offset=0.0,
        output_mode="amplified",
        sampling_rate=int(1e9),
        upsampling_mode="pulse",
        delay=141,
        shareable=True,
    )

    # ── Build the qubit component ──
    q1 = FluxTunableTransmonQubit(
        id="q1",
        f_01=7.4e9 + 110e6,
        anharmonicity=int(-200 * u.MHz),
        T1=int(10 * u.us),
        xy=MWChannel(
            opx_output=mw_qubit_out.get_reference(),
            intermediate_frequency=int(110 * u.MHz),
        ),
        resonator=InOutMWChannel(
            opx_output=mw_res_out.get_reference(),
            opx_input=mw_res_in.get_reference(),
            intermediate_frequency=int(60 * u.MHz),
            time_of_flight=28,
            smearing=0,
        ),
        z=SingleChannel(
            opx_output=lf_flux.get_reference(),
        ),
        z_sticky=SingleChannel(
            opx_output=lf_flux.get_reference(),
            sticky=StickyChannelAddon(duration=20, analog=True, digital=True),
        ),
    )

    machine.qubits["q1"] = q1

    # ── Qubit XY operations ──
    xy_ops = q1.xy.operations
    xy_ops["cw"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    xy_ops["saturation"] = SquarePulse(
        length=int(10 * u.us), amplitude=0.03, axis_angle=0.0
    )
    xy_ops["pi"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    xy_ops["pi_half"] = SquarePulse(length=100, amplitude=0.015, axis_angle=0.0)
    xy_ops["x180"] = DragGaussianPulse(
        length=40, amplitude=1.0, sigma=8.0, alpha=0.0,
        anharmonicity=int(-200 * u.MHz), detuning=0.0, axis_angle=0.0,
    )
    xy_ops["x90"] = DragGaussianPulse(
        length=40, amplitude=0.5, sigma=8.0, alpha=0.0,
        anharmonicity=int(-200 * u.MHz), detuning=0.0, axis_angle=0.0,
    )
    xy_ops["-x90"] = DragGaussianPulse(
        length=40, amplitude=-0.5, sigma=8.0, alpha=0.0,
        anharmonicity=int(-200 * u.MHz), detuning=0.0, axis_angle=0.0,
    )
    xy_ops["y180"] = DragGaussianPulse(
        length=40, amplitude=1.0, sigma=8.0, alpha=0.0,
        anharmonicity=int(-200 * u.MHz), detuning=0.0, axis_angle=np.pi / 2,
    )
    xy_ops["y90"] = DragGaussianPulse(
        length=40, amplitude=0.5, sigma=8.0, alpha=0.0,
        anharmonicity=int(-200 * u.MHz), detuning=0.0, axis_angle=np.pi / 2,
    )
    xy_ops["-y90"] = DragGaussianPulse(
        length=40, amplitude=-0.5, sigma=8.0, alpha=0.0,
        anharmonicity=int(-200 * u.MHz), detuning=0.0, axis_angle=np.pi / 2,
    )

    # ── Resonator operations ──
    res_ops = q1.resonator.operations
    res_ops["cw"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    res_ops["readout"] = SquareReadoutPulse(
        length=5000, amplitude=0.6, digital_marker="ON",
    )

    # ── Flux operations ──
    q1.z.operations["const"] = SquarePulse(length=200, amplitude=0.45)
    q1.z_sticky.operations["const"] = SquarePulse(length=200, amplitude=0.45)

    machine.save(state_path)


# ──────────────────────────────────────────────
#  Generate the QUA configuration
# ──────────────────────────────────────────────
config = machine.generate_config()


# ──────────────────────────────────────────────
#  Legacy shim (derived from machine)
# ──────────────────────────────────────────────
# Experiment scripts may import these directly.  Pulse amplitudes / lengths /
# frequencies are deliberately NOT re-exported — read them from ``machine``
# to keep a single source of truth.
qop_ip = machine.qop_ip
cluster_name = machine.cluster_name
qop_port = machine.qop_port

qubit_T1 = machine.qubits["q1"].T1
thermalization_time = machine.thermalization_time
depletion_time = machine.depletion_time
flux_settle_time = machine.flux_settle_time
ge_threshold = machine.ge_threshold
rotation_angle = machine.rotation_angle
