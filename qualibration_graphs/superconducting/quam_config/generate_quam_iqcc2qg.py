# %%
import os
import json
import shutil
from pathlib import Path
import subprocess


# Get the state folder path from environment variable
quam_state_folder_path = Path(os.environ["QUAM_STATE_PATH"])

shutil.rmtree(quam_state_folder_path)
shutil.rmtree((quam_state_folder_path / "../quam_state_copy").resolve())

script = "generate_quam_iqcc.py"
print(f"Running: {script}")
subprocess.run(["python", (quam_state_folder_path / "../quam_config" / script).resolve()], check=True)

# Copy `quam_state` folder to `destination_folder`
shutil.copytree(
    quam_state_folder_path,
    (quam_state_folder_path / "../quam_state_copy").resolve(),
    dirs_exist_ok=True,
)

shutil.rmtree(quam_state_folder_path)
shutil.copytree(
    (quam_state_folder_path / "../quam_state_copy").resolve(),
    quam_state_folder_path,
    dirs_exist_ok=True,
)

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_builder.builder.superconducting.pulses import add_DragCosine_pulses, add_Square_pulses, add_default_transmon_pair_pulses
from quam_builder.architecture.superconducting.custom_gates import cross_resonance
from quam.components import pulses
from qualang_tools.units import unit
from quam_config import Quam


# Save the files
path_latest_state = (quam_state_folder_path / "../quam_state/state.json").resolve()
with open(path_latest_state, "r") as f:
    latest_state = json.load(f)

path_latest_wiring = (quam_state_folder_path / "../quam_state/wiring.json").resolve()
with open(path_latest_wiring, "r") as f:
    latest_wiring = json.load(f)



connectivity = Connectivity()
instruments = Instruments()
# Single qubit individual drive and readout lines

import re
pattern = r'/ports/\w+/con(\d+)/(\d+)/(\d+)'

qubit_pairs = [
    ("A1", "A2"), ("A2", "A1"),
    ("A3", "A4"), ("A4", "A3"),
]
qubits_in_pairs = unique_qubits = sorted(set(q for pair in qubit_pairs for q in pair))

mwfem_slots = {1: set()}
lffem_slots = {1: set()}
wires = {}

for qb, elems in latest_wiring["wiring"]["qubits"].items():
    if qb.startswith("qC"):
        continue

    print(qb, "=" * 100)
    qb_name = qb[1:]
    wires[qb_name] = {}

    for elem_name, elem_port in elems.items():
        match = re.search(pattern, elem_port["opx_output"])
        ao_con, ao_slot, ao_port = match.groups()
        ao_con, ao_slot, ao_port = int(ao_con), int(ao_slot), int(ao_port)
        print(f"{qb}.{elem_name}.out: con{ao_con}, slot{ao_slot}, port{ao_port}")

        if elem_name == "rr":
            match = re.search(pattern, elem_port["opx_input"])
            ai_con, ai_slot, ai_port = match.groups()
            ai_con, ai_slot, ai_port = int(ai_con), int(ai_slot), int(ai_port)
            print(f"{qb}.{elem_name}.in: con{ao_con}, slot{ao_slot}, port{ao_port}")
            assert ai_con == ao_con
            assert ai_slot == ao_slot

            mwfem_slots[ao_con].add(ao_slot)
            wires[qb_name]["rr"] = {
                "con": ao_con,
                "slot": ao_slot,
                "out_port": ao_port,
                "in_port": ai_port,
            }

        elif elem_name == "xy":
            mwfem_slots[ao_con].add(ao_slot)
            wires[qb_name]["xy"] = {
                "con": ao_con,
                "slot": ao_slot,
                "out_port": ao_port,
            }

        elif elem_name == "z":
            lffem_slots[ao_con].add(ao_slot)
            wires[qb_name]["z"] = {
                "con": ao_con,
                "slot": ao_slot,
                "out_port": ao_port,
            }



for con, slots in mwfem_slots.items():
    instruments.add_mw_fem(controller=con, slots=list(slots))

for con, slots in lffem_slots.items():
    instruments.add_lf_fem(controller=con, slots=list(slots))
    


for qb_name, elem_port in wires.items():

    rr = elem_port["rr"]
    connectivity.add_resonator_line(
        qubits=qb_name,
        constraints=mw_fem_spec(con=rr["con"], slot=rr["slot"], in_port=rr["in_port"], out_port=rr["out_port"]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)

    xy = elem_port["xy"]
    connectivity.add_qubit_drive_lines(
        qubits=qb_name,
        constraints=mw_fem_spec(con=xy["con"], slot=xy["slot"], out_port=xy["out_port"]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)

    z = elem_port["z"]
    connectivity.add_qubit_flux_lines(
        qubits=qb_name,
        constraints=lf_fem_spec(con=z["con"], out_slot=z["slot"], out_port=z["out_port"]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)


# Two-qubit drives
for (qc, qt) in qubit_pairs:
    idc, idt = wires[qc], wires[qt]
    kv = wires[qc]["xy"]

    # Add CR lines
    connectivity.add_qubit_pair_cross_resonance_lines(
        qubit_pairs=(qc, qt),
        constraints=mw_fem_spec(con=kv["con"], slot=kv["slot"], out_port=kv["out_port"]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)

    # Add ZZ lines
    connectivity.add_qubit_pair_zz_drive_lines(
        qubit_pairs=(qc, qt),
        constraints=mw_fem_spec(con=kv["con"], slot=kv["slot"], out_port=kv["out_port"]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)


# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=True)




########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################

machine = Quam()
host_ip = "10.1.1.6"
cluster_name = "galil_arbel"
# Build the wiring (wiring.json) and initiate the QUAM
build_quam_wiring(connectivity, host_ip, cluster_name, machine)

# Reload QUAM, build the QUAM object and save the state as state.json
machine = Quam.load()
machine.network.update({
    "cloud": "true",
    "quantum_computer_backend": "arbel",
    "octave_ips": [],
    "octave_ports": [],
})
build_quam(machine)




#######################
# %%
#######################

# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)

active_qubits = []

# Update qubit rr freq and power
for qb_name, qubit in machine.qubits.items():

    qb_name = qubit.name[1:]

    try:
        # qubit 
        state_qubit = latest_state["qubits"][qubit.name]
        qubit.chi = state_qubit.get("chi")
        qubit.T1 = state_qubit.get("T1")
        qubit.T2ramsey = state_qubit.get("T2ramsey")
        qubit.phi0_current = state_qubit.get("phi0_current")
        qubit.phi0_voltage = state_qubit.get("phi0_voltage")
        qubit.grid_location = state_qubit.get("grid_location")
        qubit.freq_vs_flux_01_quad_term = state_qubit.get("freq_vs_flux_01_quad_term")

        # rr
        rr = qubit.resonator
        kv = wires[qb_name]["rr"]
        # rr in
        con, slot, in_port, out_port = kv.get("con"), kv.get("slot"), kv.get("in_port"), kv.get("out_port")
        state_rr_in_port = latest_state["ports"]["mw_inputs"][f"con{con}"][str(slot)][str(in_port)]
        rr.opx_input.band = state_rr_in_port.get("band")
        # rr out
        state_rr_out_port = latest_state["ports"]["mw_outputs"][f"con{con}"][str(slot)][str(out_port)]
        rr.opx_output.band = state_rr_out_port.get("band")
        rr.opx_output.upconverter_frequency = state_rr_out_port.get("upconverter_frequency")
        rr.opx_output.full_scale_power_dbm = state_rr_out_port.get("full_scale_power_dbm", -11)
        if state_rr_out_port.get("upconverter_frequency") != state_rr_in_port.get("downconverter_frequency"):
            rr.opx_input.downconverter_frequency = None
            rr.opx_input.downconverter_frequency = state_rr_in_port.get("downconverter_frequency")
        # rr - qubit
        state_qubit_rr = latest_state["qubits"][qubit.name]["resonator"]
        rr.thread = f"{qubit.name}_{rr.opx_output.controller_id}_slot{rr.opx_output.fem_id}" # state_qubit_rr.get("thread")
        rr.time_of_flight = state_qubit_rr.get("time_of_flight")
        rr.confusion_matrix = state_qubit_rr.get("confusion_matrix")
        print(f"{qubit.name} - rr IF before: ", rr.intermediate_frequency)
        # rr.intermediate_frequency = None
        # rr.intermediate_frequency = state_qubit_rr.get("intermediate_frequency")
        rr.RF_frequency = rr.LO_frequency + state_qubit_rr.get("intermediate_frequency")
        print(f"{qubit.name} - rr IF after: ", rr.intermediate_frequency)
        # rr.operations["const"].length = state_qubit_rr["operations"]["const"]["length"]
        # rr.operations["const"].amplitude = state_qubit_rr["operations"]["const"]["amplitude"]
        rr.operations["readout"].length = state_qubit_rr["operations"]["readout"].get("length")
        rr.operations["readout"].amplitude = state_qubit_rr["operations"]["readout"].get("amplitude")
        rr.operations["readout"].threshold = state_qubit_rr["operations"]["readout"].get("threshold", 0)
        rr.operations["readout"].rus_exit_threshold = state_qubit_rr["operations"]["readout"].get("rus_exit_threshold", 0)
        rr.operations["readout"].integration_weights_angle = state_qubit_rr["operations"]["readout"].get("integration_weights_angle", 0)

        # xy out
        xy = qubit.xy
        kv = wires[qb_name]["xy"]
        con, slot, out_port = kv.get("con"), kv.get("slot"), kv.get("out_port")
        xy.thread = f"{qubit.name}_{xy.opx_output.controller_id}_slot{xy.opx_output.fem_id}" # state_qubit_xy.get("thread") 
        state_xy_out_port = latest_state["ports"]["mw_outputs"][f"con{con}"][str(slot)][str(out_port)]
        xy.opx_output.band = state_xy_out_port.get("band")
        print(f"{qubit.name} - xy port: {xy.opx_output.controller_id}-{xy.opx_output.fem_id}-{xy.opx_output.port_id}, band: {state_xy_out_port.get('band')}")
        xy.opx_output.full_scale_power_dbm = state_xy_out_port.get("full_scale_power_dbm", -11)
        state_qubit_xy = latest_state["qubits"][qubit.name]["xy"]
        if qb_name in qubits_in_pairs:
            xy.opx_output.upconverter_frequency = None
            # print(f"{qubit.name} - xy upconverter_frequency before: ", xy.opx_output.upconverter_frequency)
            # print(f"{qubit.name} - xy upconverters before:", xy.opx_output.upconverters)
            # xy.opx_output.upconverter_frequency = state_xy_out_port.get("upconverter_frequency")
            # xy.opx_output.upconverters = {1: {"frequency": state_xy_out_port.get("upconverter_frequency")}, 2: {"frequency": 123}}
            xy.opx_output.upconverters = {1: {"frequency": state_xy_out_port.get("upconverter_frequency")}, 2: {"frequency": 123.0}}
            xy.upconverter = 1
            print(f"{qubit.name} - {state_xy_out_port}")
            print(f"{qubit.name} - xy upconverters after: {xy.opx_output.upconverters}")
            print(f"{qubit.name} - xy upconverter: {xy.upconverter}")
            # xy - qubit
            # print(f"{qubit.name} - xy IF before: ", xy.intermediate_frequency)
            # xy.intermediate_frequency = None
            # xy.intermediate_frequency = state_qubit_xy.get("intermediate_frequency")
            print(f"{qubit.name} - xy type: {type(xy)}")
            print(f"{qubit.name} - xy LO_frequency: {xy.LO_frequency}")
            print(f"{qubit.name} - xy upconverter.frequency: {xy.opx_output.upconverters[1]['frequency']}")
            # xy.RF_frequency = xy.upconverter_frequency + state_qubit_xy.get("intermediate_frequency")
            # print(f"{qubit.name} - xy upconverter_frequency after: ", xy.opx_output.upconverters[1]["frequency"])
            # xy.LO_frequency = xy.opx_output.upconverters[1]["frequency"]
            xy.RF_frequency = xy.opx_output.upconverters[1]["frequency"] + state_qubit_xy.get("intermediate_frequency")
            # print(f"{qubit.name} - xy IF after: ", xy.intermediate_frequency)

        else:
            xy.opx_output.upconverter_frequency = state_xy_out_port.get("upconverter_frequency")
            xy.RF_frequency = xy.upconverter_frequency + state_qubit_xy.get("intermediate_frequency")
            print(f"{qubit.name} - xy RF_frequency: {xy.RF_frequency}")
            print(f"{qubit.name} - xy LO_frequency: {xy.LO_frequency}")
            print(f"{qubit.name} - xy upconverter_frequency: {xy.opx_output.upconverter_frequency}")

        xy.operations["saturation"].length = state_qubit_xy["operations"]["saturation"]["length"]
        xy.operations["saturation"].amplitude = state_qubit_xy["operations"]["saturation"]["amplitude"]

        # z out
        z = qubit.z
        kv = wires[qb_name]["z"]
        con, slot, out_port = kv.get("con"), kv.get("slot"), kv.get("out_port")
        state_z_out_port = latest_state["ports"]["analog_outputs"][f"con{con}"][str(slot)][str(out_port)]
        z.opx_output.output_mode = state_z_out_port.get("output_mode")
        z.opx_output.upsampling_mode = state_z_out_port.get("upsampling_mode")
        # z - qubit
        state_qubit_z = latest_state["qubits"][qubit.name]["z"]
        z.operations["const"].length = state_qubit_z["operations"]["const"]["length"]
        z.operations["const"].amplitude = state_qubit_z["operations"]["const"]["amplitude"]
        z.min_offset = state_qubit_z.get("min_offset")
        z.joint_offset = state_qubit_z.get("joint_offset")
        z.offset_settle_time = state_qubit_z.get("offset_settle_time")

        # Single qubit gates - DragCosine
        add_DragCosine_pulses(
            qubit,
            amplitude=0.5,
            length=40,
            anharmonicity=250 * u.MHz,
            alpha=0.0,
            detuning=0,
        )
        xy.operations["x180_DragCosine"].length = state_qubit_xy["operations"]["x180_DragCosine"].get("length")
        xy.operations["x180_DragCosine"].amplitude = state_qubit_xy["operations"]["x180_DragCosine"].get("amplitude", 1)
        xy.operations["x90_DragCosine"].length = state_qubit_xy["operations"]["x90_DragCosine"].get("length")
        xy.operations["x90_DragCosine"].amplitude = state_qubit_xy["operations"]["x90_DragCosine"].get("amplitude", 1)

        # resonator const
        qubit.resonator.operations["const"] = pulses.SquarePulse(
            length=1 * u.us,
            amplitude=0.5,
            axis_angle=0.0,
        )
        # Qubit const
        qubit.xy.operations["const"] = pulses.SquarePulse(
            length=1 * u.us,
            amplitude=0.5,
            axis_angle=0.0,
        )
        
        active_qubits.append(qubit.name)

    except:
        qubit.resonator.intermediate_frequency = None
        qubit.resonator.intermediate_frequency = 0
        qubit.xy.intermediate_frequency = None
        qubit.xy.intermediate_frequency = 0
        print(f"failed at {qubit.name}: {wires[qb_name]}")

    print()

machine.active_qubit_names = active_qubits


for qp_name, qb_pair in machine.qubit_pairs.items():
    print(qp_name, qb_pair)
    print(dir(qb_pair))



for qp_name, qb_pair in machine.qubit_pairs.items():
    add_default_transmon_pair_pulses(qb_pair)
    qbt = qb_pair.qubit_target
    qb_pair.cross_resonance.target_qubit_LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
    qb_pair.cross_resonance.target_qubit_IF_frequency = f"#/qubits/{qbt.name}/xy/intermediate_frequency"
    qb_pair.cross_resonance.LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
    qb_pair.cross_resonance.intermediate_frequency = f"#./inferred_intermediate_frequency"

    qc = qb_pair.qubit_control
    qb_pair.cross_resonance.core = f"{qc.name}_{qc.xy.opx_output.controller_id}_slot{qc.xy.opx_output.fem_id}"
    qc.xy.opx_output.upconverters.parent = None
    qc.xy.opx_output.upconverters[2]["frequency"] = qb_pair.cross_resonance.target_qubit_LO_frequency
    qb_pair.cross_resonance.upconverter = 2
    qb_pair.cross_resonance.opx_output.upconverter_frequency = None

    print(f"{qp_name} - CR LO: {qb_pair.cross_resonance.opx_output.upconverters[2]['frequency']}")
    print(f"{qp_name} - CR IF: {qb_pair.cross_resonance.intermediate_frequency}")
    print(f"{qp_name} - CR target_qubit_LO_frequency: {qb_pair.cross_resonance.target_qubit_LO_frequency}")
    print(f"{qp_name} - CR target_qubit_IF_frequency: {qb_pair.cross_resonance.target_qubit_IF_frequency}")
    print(f"{qp_name} - CR LO_frequency: {qb_pair.cross_resonance.LO_frequency}")


    qb_pair.zz_drive.target_qubit_LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
    qb_pair.zz_drive.target_qubit_IF_frequency = f"#/qubits/{qbt.name}/xy/intermediate_frequency"
    qb_pair.zz_drive.LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
    qb_pair.zz_drive.detuning = -10 * u.MHz
    qb_pair.zz_drive.intermediate_frequency = f"#./inferred_intermediate_frequency"

    qc = qb_pair.qubit_control
    qb_pair.zz_drive.core = f"{qc.name}_{qc.xy.opx_output.controller_id}_slot{qc.xy.opx_output.fem_id}"

    print(f"{qp_name} - ZZ LO: {qb_pair.zz_drive.opx_output.upconverters[2]['frequency']}")
    print(f"{qp_name} - ZZ IF: {qb_pair.zz_drive.intermediate_frequency}")
    print(f"{qp_name} - ZZ target_qubit_LO_frequency: {qb_pair.zz_drive.target_qubit_LO_frequency}")
    print(f"{qp_name} - ZZ target_qubit_IF_frequency: {qb_pair.zz_drive.target_qubit_IF_frequency}")
    print(f"{qp_name} - ZZ LO_frequency: {qb_pair.zz_drive.LO_frequency}")

    try:
        qb_pair.macros["cr"] = cross_resonance.CRGate(qc_correction_phase=0.0)

        # square
        qb_pair.cross_resonance.operations["square"] = pulses.SquarePulse(
            length=100,
            amplitude=1.0,
            axis_angle=0.0,
        )
        qb_pair.qubit_target.xy.operations[f"cr_square_{qb_pair.name}"] = pulses.SquarePulse(
            length=100,
            amplitude=1.0,
            axis_angle=0.0,
        )
        # cosine
        qb_pair.cross_resonance.operations["cosine"] = pulses.DragCosinePulse(
            length=100,
            amplitude=1.0,
            axis_angle=0.0,
            anharmonicity=260 * u.MHz,
            alpha=0.0,
            detuning=0,
            # correction_phase=0.0,
        )
        qb_pair.qubit_target.xy.operations[f"cr_cosine_{qb_pair.name}"]= pulses.DragCosinePulse(
            length=100,
            amplitude=1.0,
            axis_angle=0.0,
            anharmonicity=260 * u.MHz,
            alpha=0.0,
            detuning=0,
        )
        # gauss
        qb_pair.cross_resonance.operations["gauss"] = pulses.DragGaussianPulse(
            length=100,
            sigma=100/5,
            amplitude=1.0,
            axis_angle=0.0,
            anharmonicity=260 * u.MHz,
            alpha=0.0,
            detuning=0,
            # correction_phase=0.0,
        )
        qb_pair.qubit_target.xy.operations[f"cr_gauss_{qb_pair.name}"]= pulses.DragGaussianPulse(
            length=100,
            sigma=100/5,
            amplitude=1.0,
            axis_angle=0.0,
            anharmonicity=260 * u.MHz,
            alpha=0.0,
            detuning=0,
        )

        # flattop
        rise_fall_len = 8
        flattop_lens = np.arange(0, 80, 20).tolist() # must be python list (not numpy array)
        for flattop_len in flattop_lens:
            qb_pair.cross_resonance.operations[f"flattop_{flattop_len:04d}"] = pulses.FlatTopGaussianPulse(
                amplitude=1.0,
                length=rise_fall_len + flattop_len + rise_fall_len,
                flat_length=flattop_len,
                axis_angle=0.0, 
            )
            qb_pair.qubit_target.xy.operations[f"cr_flattop_{flattop_len:04d}"] = pulses.FlatTopGaussianPulse(
                amplitude=1.0,
                length=rise_fall_len + flattop_len + rise_fall_len,
                flat_length=flattop_len,
                axis_angle=0.0, 
            )

    except:
        print(f"failed at {qb_pair.name}")

    print()



########################################################################################################################
# %%                                         Save the updated QUAM
########################################################################################################################
from pprint import pprint
# save into state.json
machine.save()
# Visualize the QUA config and save it
pprint(machine.generate_config())
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)

# %%
