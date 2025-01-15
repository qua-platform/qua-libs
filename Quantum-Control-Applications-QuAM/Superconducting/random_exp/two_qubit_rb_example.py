# flake8: noqa
# %%
import matplotlib.pyplot as plt
from qm.qua import *
from qm import QuantumMachinesManager
from qualang_tools.bakery.bakery import Baking

from qualang_tools.characterization.two_qubit_rb import TwoQubitRb, TwoQubitRbDebugger
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

machine = QuAM.load()
config = machine.generate_config()
octave_config = machine.get_octave_config()
qmm = machine.connect()
QC = machine.qubits["qubitC3"]
QT = machine.qubits["qubitC4"]
Q_aux = machine.qubits["qubitC1"]
QP = machine.qubit_pairs["qC3-qC4"]
# %%
##############################
## General helper functions ##
##############################


##############################
##  Two-qubit RB functions  ##
##############################
# assign a string to a variable to be able to call them in the functions
q1_idx_str = "1"
q2_idx_str = "2"


# single qubit generic gate constructor Z^{z}Z^{a}X^{x}Z^{-a}
# that can reach any point on the Bloch sphere (starting from arbitrary points)
def bake_phased_xz(baker: Baking, q, x, z, a):
    if q == 1:
        element = QC.xy.name
    else:
        element = QT.xy.name

    baker.frame_rotation_2pi(a / 2, element)
    baker.play("x180", element, amp=x)
    baker.frame_rotation_2pi(-(a + z) / 2, element)


# single qubit phase corrections in units of 2pi applied after the CZ gate
qubit1_frame_update = 0.23  # example values, should be taken from QPU parameters
qubit2_frame_update = 0.12  # example values, should be taken from QPU parameters


# defines the CZ gate that realizes the mapping |00> -> |00>, |01> -> |01>, |10> -> |10>, |11> -> -|11>
def bake_cz(baker: Baking, q1, q2):
    baker.align()

    baker.play(
        "Cz.CZ_snz_qubitC4",QC.z.name
    )
    amp_scale = QP.gates["Cz"].compensations[0]["shift"] / Q_aux.z.operations["const"].amplitude
    baker.play("const" , Q_aux.z.name)

    baker.align()
    baker.frame_rotation_2pi(QP.gates["Cz_SNZ"].phase_shift_control, QC.xy.name)
    baker.frame_rotation_2pi(QP.gates["Cz_SNZ"].phase_shift_target, QT.xy.name)
    baker.align()


def prep():
    # wait(int(machine.thermalization_time))  # thermal preparation in clock cycles (time = 10 x T1 x 4ns)
    machine.set_all_fluxes(flux_point="joint", target=machine.qubits["qubitC3"])
    align()
    # active_reset(QC)
    # active_reset(QT)
    wait(machine.thermalization_time)
    align()


def meas():
    threshold1 = 0.3  # threshold for state discrimination 0 <-> 1 using the I quadrature
    threshold2 = 0.3  # threshold for state discrimination 0 <-> 1 using the I quadrature
    I1 = declare(fixed)
    I2 = declare(fixed)
    Q1 = declare(fixed)
    Q2 = declare(fixed)
    state1 = declare(int)
    state2 = declare(int)
    # qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))  # readout macro for multiplexed readout
    readout_state(QC, state1)
    readout_state(QT, state2)
    return state1, state2


##############################
##  Two-qubit RB execution  ##
##############################

# create RB experiment from configuration and defined functions
rb = TwoQubitRb(
    config=config,  # enter your QUA config here
    single_qubit_gate_generator=bake_phased_xz,
    two_qubit_gate_generators={"CZ": bake_cz},  # can also provide e.g. "CNOT": bake_cnot
    prep_func=prep,
    measure_func=meas,
    interleaving_gate=None,
    # interleaving_gate=[cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))],
    verify_generation=False,
)
# %%
# run simpler experiment to verify `bake_phased_xz`, `prep` and `meas`
# rb_debugger = TwoQubitRbDebugger(rb)
# state = rb_debugger.run_phased_xz_commands(qmm, 100, machine)
# plt.show()

# %%
# run 2Q-RB experiment
res = rb.run(
    qmm,
    circuit_depths=np.arange(0, 21, 1),
    num_circuits_per_depth=30,
    num_shots_per_circuit=100,
    # unsafe=True will minimize switch-case gaps, but can lead to unexpected behaviour
    unsafe=False,
    machine=machine,
)

# circuit_depths ~ how many consecutive Clifford gates within one executed circuit
# (https://qiskit.org/documentation/apidoc/circuit.html)
# num_circuits_per_depth ~ how many random circuits within one depth
# num_shots_per_circuit ~ repetitions of the same circuit (averaging)

res.plot_hist()
plt.show()

res.plot_with_fidelity()
plt.show()

# verify/save the random sequences created during the experiment
rb.save_sequences_to_file("sequences.txt")  # saves the gates used in each random sequence
rb.save_command_mapping_to_file("commands.txt")  # saves mapping from "command id" to sequence
# rb.print_sequence()
# rb.print_command_mapping()
# rb.verify_sequences()  # simulates random sequences to ensure they recover to ground state. takes a while...

# # get the interleaved gate fidelity
# from two_qubit_rb.RBResult import get_interleaved_gate_fidelity
# interleaved_gate_fidelity = get_interleaved_gate_fidelity(
#     num_qubits=2,
#     reference_alpha=0.12345,  # replace with value from prior, non-interleaved experiment
#     # interleaved_alpha=res.fit_exponential()[1],  # alpha from the interleaved experiment
# )
# print(f"Interleaved Gate Fidelity: {interleaved_gate_fidelity*100:.3f}")

# %%
