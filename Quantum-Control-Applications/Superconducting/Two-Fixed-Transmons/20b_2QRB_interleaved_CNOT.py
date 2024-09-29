# %%
from qm.qua import *
from qm import QuantumMachinesManager
from configuration_mw_fem import *
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout, active_reset
from qualang_tools.results.data_handler import DataHandler
import time
import warnings
import matplotlib
from macros import qua_declaration, multiplexed_readout
from qualang_tools.bakery.bakery import Baking
from two_qubit_rb import TwoQubitRb
import cirq


##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_avg = 100
cr_drive_amp = 1.0
cr_drive_phase = 0.25
cr_cancel_amp = 0.2 # ratio
cr_cancel_phase = 0.5 # in units of 2pi

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
cr_drive = f"cr_drive_c{qc}t{qt}"
cr_cancel = f"cr_cancel_c{qc}t{qt}"
qubits = [f"q{i}_xy" for i in [qc, qt]]
resonators = [f"q{i}_rr" for i in [qc, qt]]

# Assertion
assert n_avg <= 10_000, "revise your number of shots"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "qc_xy": qc_xy,
    "qt_xy": qt_xy,
    "cr_drive": cr_drive,
    "cr_cancel": cr_cancel,
    "cr_cancel_phase": cr_cancel_phase,
    "cr_drive_phase": cr_drive_phase,
    "n_avg": n_avg,
    "config": config,
}


##############################
##  Two-qubit RB functions  ##
##############################
# assign a string to a variable to be able to call them in the functions
q1_idx_str = f"{qc}"
q2_idx_str = f"{qt}"

# single qubit generic gate constructor Z^{z}Z^{a}X^{x}Z^{-a}
# that can reach any point on the Bloch sphere (starting from arbitrary points)
def bake_phased_xz(baker: Baking, q, x, z, a):
    if q == 1:
        element = f"q{q1_idx_str}_xy"
    else:
        element = f"q{q2_idx_str}_xy"

    baker.frame_rotation_2pi(a / 2, element)
    baker.play("x180", element, amp=x)
    baker.frame_rotation_2pi(-(a + z) / 2, element)
    
    if q == 2:
        # keep track the frame for cr and cr_cancel
        baker.frame_rotation_2pi(-z / 2, cr_drive)
        baker.frame_rotation_2pi(-z / 2, cr_cancel)


# defines the CNOT gate that realizes the mapping |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
def bake_cnot(baker: Baking, q1, q2):
    # Play ZI(-pi/2) and IX(-pi/2)
    baker.frame_rotation_2pi(+0.25, qc_xy) # +0.25 for Z(-pi/2)
    baker.play("-x90", qt_xy)
    # baker.align(qc_xy, cr_drive)
    # baker.align(qt_xy, cr_drive)
    # baker.align(qc_xy, cr_cancel)
    # baker.align(qt_xy, cr_cancel)
    baker.wait(PI_LEN, qc_xy)
    baker.wait(PI_LEN, cr_drive)
    baker.wait(PI_LEN, cr_cancel)

    # Shift frames to the calibrated phases
    baker.frame_rotation_2pi(cr_drive_phase, cr_drive)
    baker.frame_rotation_2pi(cr_cancel_phase, cr_cancel)

    # Play CR
    # main
    baker.play("square_positive", cr_drive)
    baker.play("square_positive", cr_cancel)
    baker.wait(CR_SQUARE_LEN, qc_xy)
    baker.wait(CR_SQUARE_LEN, qt_xy)
    # echo
    baker.play("x180", qc_xy)
    baker.wait(PI_LEN, qt_xy)
    baker.wait(PI_LEN, cr_drive)
    baker.wait(PI_LEN, cr_cancel)
    baker.play("square_negative", cr_drive)
    baker.play("square_negative", cr_cancel)
    baker.wait(CR_SQUARE_LEN, qc_xy)
    baker.wait(CR_SQUARE_LEN, qt_xy)
    baker.play("x180", qc_xy)
    baker.wait(PI_LEN, qt_xy)
    baker.wait(PI_LEN, cr_drive)
    baker.wait(PI_LEN, cr_cancel)
    # Shift back the phase of cr and cr cancel pulse so they won't be accumulated
    baker.frame_rotation_2pi(-cr_drive_phase, cr_drive)
    baker.frame_rotation_2pi(-cr_cancel_phase, cr_cancel)


def prep():
    wait(rr_reset_time * u.ns)  # thermal preparation in clock cycles (time = 10 x T1 x 4ns)
    align()


def meas():
    I, _, Q, _, n, _ = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    multiplexed_readout(I, None, Q, None, state, None, resonators=resonators, weights=weights)
    return state


##############################
##  Two-qubit RB execution  ##
##############################
# create RB experiment from configuration and defined functions
rb = TwoQubitRb(
    config=config,
    single_qubit_gate_generator=bake_phased_xz,
    two_qubit_gate_generators={"CNOT": bake_cnot},
    prep_func=prep,
    measure_func=meas,
    interleaving_gate=[cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))],
    verify_generation=False,
)

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config) # initialize qmm
res = rb.run(qmm, circuit_depths=[3, 4, 5], num_circuits_per_depth=4, num_shots_per_circuit=5)
# circuit_depths ~ how many consecutive Clifford gates within one executed circuit
# (https://qiskit.org/documentation/apidoc/circuit.html)
# num_circuits_per_depth ~ how many random circuits within one depth
# num_shots_per_circuit ~ repetitions of the same circuit (averaging)

res.plot_hist()
res.plot_fidelity()
plt.show()

# verify/save the random sequences created during the experiment
rb.save_sequences_to_file("sequences.txt")  # saves the gates used in each random sequence
# rb.save_command_mapping_to_file('commands.txt')  # saves mapping from "command id" to sequence
# rb.print_sequences()
# rb.print_command_mapping()
# rb.verify_sequences() # simulates random sequences to ensure they recover to ground state. takes a while...

# %%
