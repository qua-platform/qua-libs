"""
Generate QUAM wiring and baseline state for CS_4: two qubits, q1 flux, tunable coupler flux,
then attach a composite coupler (flux + MW XY on MW-FEM slot 1, port 4).

Run from the ``superconducting`` environment with ``quam_config`` on PYTHONPATH, or
``pip install -e .`` for ``qualibration_graphs/superconducting``.

Non-interactive save: set environment variable ``QUAM_AUTO_SAVE=1`` (or ``yes``/``true``).
"""

import os

import matplotlib.pyplot as plt
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec
from quam.components.ports.analog_outputs import MWFEMAnalogOutputPort
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam

from quam_config.my_quam import Quam, TunableCouplerWithXY

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "172.16.33.114"  # QOP / cluster router (CS_4)
port = None  # QOP Port (optional)
cluster_name = "CS_4"  # Cluster name in QOP

# MW-FEM slot 1, port 4 — coupler XY (must share band rules with port 5 on same FEM)
COUPLER_XY_CONTROLLER = "con1"
COUPLER_XY_MW_FEM_SLOT = 1
COUPLER_XY_PORT = 4
# Placeholder LO on port until populate script overwrites (required by MWFEMAnalogOutputPort)
COUPLER_XY_LO_PLACEHOLDER_HZ = 8.0e9
COUPLER_XY_BAND = 2

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1])
instruments.add_lf_fem(controller=1, slots=[5])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [1, 2]
qubit_pairs = [(1, 2)]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# Multiplexed readout on MW-FEM 1 O1/I1; individual XY drives on O2 and O3
res_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
drive_ch = mw_fem_spec(con=1, slot=1, in_port=None, out_port=None)

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
connectivity.add_resonator_line(qubits=qubits, constraints=res_ch)
connectivity.add_qubit_drive_lines(qubits=qubits, constraints=drive_ch)
connectivity.add_qubit_flux_lines(qubits=[1])
connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs)
allocate_wiring(connectivity, instruments)

# View wiring schematic (headless-safe)
try:
    visualize(connectivity.elements, available_channels=instruments.available_channels)
    plt.show(block=False)
except Exception:
    pass

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################


def attach_coupler_xy(machine: Quam) -> None:
    """Wrap flux-only ``TunableCoupler`` in ``TunableCouplerWithXY`` and add MW-FEM XY port."""
    xy_port = MWFEMAnalogOutputPort(
        controller_id=COUPLER_XY_CONTROLLER,
        fem_id=COUPLER_XY_MW_FEM_SLOT,
        port_id=COUPLER_XY_PORT,
        band=COUPLER_XY_BAND,
        upconverter_frequency=COUPLER_XY_LO_PLACEHOLDER_HZ,
        full_scale_power_dbm=-11,
    )
    for pair in machine.qubit_pairs.values():
        old = pair.coupler
        if old is None:
            continue
        if isinstance(old, TunableCouplerWithXY):
            continue
        # Reparent flux coupler under the composite (QUAM forbids changing .parent while set).
        old.parent = None
        pair.coupler = TunableCouplerWithXY(
            id=old.id,
            z=old,
            xy=XYDriveMW(
                id=f"{old.id}_xy",
                opx_output=xy_port,
                intermediate_frequency=0.0,
            ),
        )


_auto = os.environ.get("QUAM_AUTO_SAVE", "").lower() in ("1", "true", "yes", "y")
if _auto:
    user_input = "y"
else:
    user_input = input("Do you want to save the updated QUAM? (y/n)")

if user_input.lower() == "y":
    machine = Quam()
    build_quam_wiring(connectivity, host_ip, cluster_name, machine)

    machine = Quam.load()
    build_quam(machine)
    attach_coupler_xy(machine)
    machine.save()
