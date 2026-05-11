"""
QUAM-based QUA configuration supporting OPX1000 w/ LF-FEM + MW-FEM.

Single flat configuration file: defines the machine inline, populates all
hardware/pulse parameters, persists state to JSON via ``machine.save(...)``,
and exposes the generated QUA ``config``. Experiments should read parameters
from the QUAM object (e.g. ``machine.channels["qubit"].operations["x180"].amplitude``)
rather than relying on duplicated Python globals.

First import: builds the machine from the defaults below and saves
``quam_state_lf_mw_fem.json`` next to this file.
Subsequent imports: load the saved JSON state directly.

There are two equivalent ways to change a parameter persistently:

1. Mutate the Python object and save it back:

       from configuration_quam_lf_fem_and_mw_fem import machine, state_path
       machine.channels["qubit"].operations["x180"].amplitude = 0.95
       machine.save(state_path)

2. Edit ``quam_state_lf_mw_fem.json`` directly with any text/JSON editor.
   The file is plain JSON and stores scalar parameters (amplitudes, lengths,
   frequencies, offsets, axis angles, ...), so changes you make there are
   picked up the next time this module is imported and the machine is
   reloaded from disk.

Either path is valid -- the JSON file is the source of truth; the Python
object is just one convenient way to inspect and mutate it.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from qualang_tools.units import unit

from quam.components import BasicFEMQuam
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
from quam.core import quam_dataclass


u = unit(coerce_to_integer=True)


#############################################
#         Custom QUAM root subclass         #
#############################################
@quam_dataclass
class FluxTunableTransmonQuam(BasicFEMQuam):
    """Root QUAM object for the single flux-tunable transmon setup.

    Holds non-channel scalars (network info, characteristic times, fit
    parameters) so they round-trip through ``save()`` / ``load()`` alongside
    ports and channels.
    """

    qop_ip: str = "127.0.0.1"
    cluster_name: Optional[str] = None
    qop_port: Optional[int] = None

    qubit_T1: int = 10_000
    thermalization_time: int = 50_000
    depletion_time: int = 2_000
    flux_settle_time: int = 100

    ge_threshold: float = 0.0
    rotation_angle: float = 0.0

    # Resonator-vs-flux fit placeholders (filled in by spectroscopy scripts).
    amplitude_fit: float = 0.0
    frequency_fit: float = 0.0
    phase_fit: float = 0.0
    offset_fit: float = 0.0


#############################################
#               State file path             #
#############################################
state_path = Path(__file__).parent / "quam_state_lf_mw_fem.json"


#############################################
#            Build OR Load machine          #
#############################################
if state_path.exists():
    machine = FluxTunableTransmonQuam.load(state_path)
else:
    machine = FluxTunableTransmonQuam(ports=FEMPortsContainer())

    # --- Network / experiment parameters ---
    machine.qop_ip = "127.0.0.1"
    machine.cluster_name = None
    machine.qop_port = None

    machine.qubit_T1 = int(10 * u.us)
    machine.thermalization_time = 5 * machine.qubit_T1
    machine.depletion_time = int(2 * u.us)
    machine.flux_settle_time = int(100 * u.ns)
    machine.ge_threshold = 0.0
    machine.rotation_angle = 0.0

    # --- MW-FEM ports (band 2: 4.5-7.5 GHz) ---
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

    # --- LF-FEM port (flux line) ---
    # 141 ns delay aligns LF outputs with MW outputs for bands 1 and 3 (use
    # 161 ns for band 2 if the resulting alignment is critical).
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

    # --- Channels ---
    machine.channels["qubit"] = MWChannel(
        opx_output=mw_qubit_out.get_reference(),
        intermediate_frequency=int(110 * u.MHz),
    )
    machine.channels["resonator"] = InOutMWChannel(
        opx_output=mw_res_out.get_reference(),
        opx_input=mw_res_in.get_reference(),
        intermediate_frequency=int(60 * u.MHz),
        time_of_flight=28,
        smearing=0,
    )
    machine.channels["flux_line"] = SingleChannel(
        opx_output=lf_flux.get_reference(),
    )
    machine.channels["flux_line_sticky"] = SingleChannel(
        opx_output=lf_flux.get_reference(),
        sticky=StickyChannelAddon(duration=20, analog=True, digital=True),
    )

    # --- Qubit operations ---
    qubit_ops = machine.channels["qubit"].operations
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

    # --- Resonator operations ---
    resonator_ops = machine.channels["resonator"].operations
    resonator_ops["cw"] = SquarePulse(length=100, amplitude=0.03, axis_angle=0.0)
    resonator_ops["readout"] = SquareReadoutPulse(
        length=5000,
        amplitude=0.6,
        digital_marker="ON",
    )

    # --- Flux operations ---
    machine.channels["flux_line"].operations["const"] = SquarePulse(
        length=200, amplitude=0.45
    )
    machine.channels["flux_line_sticky"].operations["const"] = SquarePulse(
        length=200, amplitude=0.45
    )

    machine.save(state_path)


#############################################
#         Generate the QUA configuration    #
#############################################
config = machine.generate_config()


#############################################
#   Legacy shim (derived from machine)      #
#############################################
# These are the few non-channel scalars that experiment scripts may import
# directly. Pulse amplitudes / lengths / frequencies are deliberately NOT
# re-exported here -- read them from `machine` to keep a single source of truth.
qop_ip = machine.qop_ip
cluster_name = machine.cluster_name
qop_port = machine.qop_port

qubit_T1 = machine.qubit_T1
thermalization_time = machine.thermalization_time
depletion_time = machine.depletion_time
flux_settle_time = machine.flux_settle_time
ge_threshold = machine.ge_threshold
rotation_angle = machine.rotation_angle
