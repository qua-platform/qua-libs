# %%
import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "127.0.0.1"  # QOP IP address
cluster_name = "Cluster_1"  # Name of the cluster

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1, 2])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [
    1, 2, 3, 4,
    5, 6, 7, 8,
]
qubit_idxes = {q: i for i, q in enumerate(qubits)}
qubit_pairs = [
    (1, 2), (2, 1),
    (2, 3), (3, 2),
    (3, 4), (4, 3),

    (5, 6), (6, 5),
    (6, 7), (7, 6),
    (7, 8), (8, 7),
]

# Flatten the pairs
flattened_qubits = {q for pair in qubit_pairs for q in pair}

# Check if all entries are in `qubits`
assert flattened_qubits.issubset(set(qubits))


########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
con = 1
rr_slots = [
    1, 1, 1, 1,
    2, 2, 2, 2,
]
rr_out_ports = [
    1, 1, 1, 1,
    1, 1, 1, 1,
]
rr_in_ports = [
    1, 1, 1, 1,
    1, 1, 1, 1,
]

assert len(rr_slots) == len(qubits)
assert len(rr_out_ports) == len(qubits)
assert len(rr_in_ports) == len(qubits)

xy_slots = [
    1, 1, 1, 1,
    2, 2, 2, 2,
]
xy_ports = [
    2, 3, 4, 5,
    2, 3, 4, 5,
]

assert len(xy_slots) == len(qubits)
assert len(xy_ports) == len(qubits)


########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# Single qubit individual drive and readout lines
for i, qb in enumerate(qubits):
    connectivity.add_resonator_line(
        qubits=qb,
        constraints=mw_fem_spec(con=con, slot=rr_slots[i], in_port=rr_in_ports[i], out_port=rr_out_ports[i]),
    )
    # Don't block the xy channels to connect the CR and ZZ drives to the same ports
    allocate_wiring(connectivity, instruments, block_used_channels=False)

    connectivity.add_qubit_drive_lines(
        qubits=qb,
        constraints=mw_fem_spec(con=con, slot=xy_slots[i], out_port=xy_ports[i]),
    )
    # Don't block the xy channels to connect the CR and ZZ drives to the same ports
    allocate_wiring(connectivity, instruments, block_used_channels=False)


# Two-qubit drives
for (qc, qt) in qubit_pairs:
    idc, idt = qubit_idxes[qc], qubit_idxes[qt]

    # Add CR lines
    connectivity.add_qubit_pair_cross_resonance_lines(
        qubit_pairs=(qc, qt),
        constraints=mw_fem_spec(con=con, slot=xy_slots[idc], out_port=xy_ports[idc]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)

    # Add ZZ lines
    connectivity.add_qubit_pair_zz_drive_lines(
        qubit_pairs=(qc, qt),
        constraints=mw_fem_spec(con=con, slot=xy_slots[idc], out_port=xy_ports[idc]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)

    # Add XY detuned for ZZ lines
    connectivity.add_qubit_detuned_drive_lines(
        qubits=qt,
        constraints=mw_fem_spec(con=con, slot=xy_slots[idt], out_port=xy_ports[idt]),
    )
    # Don't block the xy channels to connect the CR and ZZ drives to the same ports
    allocate_wiring(connectivity, instruments, block_used_channels=False)


# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=True)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################

machine = Quam()
# Build the wiring (wiring.json) and initiate the QUAM
build_quam_wiring(connectivity, host_ip, cluster_name, machine)

# Reload QUAM, build the QUAM object and save the state as state.json
machine = Quam.load()
build_quam(machine)


########################################################################################################################
# %%                                   Populate QUAM
########################################################################################################################

from pathlib import Path
import subprocess

script = "populate_quam_mw_fem.py"
path_config = Path.cwd()
print(f"Running: {script}")
subprocess.run(["python", path_config / script], check=True)


# %%
