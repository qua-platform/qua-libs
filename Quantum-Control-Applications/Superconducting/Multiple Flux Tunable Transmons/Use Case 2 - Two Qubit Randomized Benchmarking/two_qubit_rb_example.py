# import tools 
import matplotlib.pylab as plt
from qm.qua import *
from qm import QuantumMachinesManager
from qualang_tools.bakery.bakery import Baking
from configuration import *
from two_qubit_rb import TwoQubitRb
import cirq
from macros import multiplexed_readout
# %matplotlib qt
# assign a string to a variable to be able to call them in the functions
q0 = '0'
q1 = '1'
# single qubit generic gate constructor Z^{z}Z^{a}X^{x}Z^{-a} that can reach any point on the Bloch sphere (starting from arbitrary points)
def bake_phased_xz(baker: Baking, q, x, z, a):
    element = f"q{q}_xy"
    baker.frame_rotation_2pi(-a, element)
    baker.play("x180", element, amp=x)
    baker.frame_rotation_2pi(a + z, element)
# single qubit phase corrections in units of 2pi applied after the CZ gate
qubit0_frame_update = 0.23  # example values, should be taken from QPU parameters
qubit1_frame_update = 0.12  # example values, should be taken from QPU parameters
# defines the CZ gate that realizes the mapping |00> -> |00>, |01> -> |01>, |10> -> |10>, |11> -> -|11>
def bake_cz(baker: Baking, q0, q1):
    q0_xy_element = f"q{q0}_xy" #
    q1_xy_element = f"q{q1}_xy"
    q0_z_element = f"q{q0}_z"
    baker.play("cz", q0_z_element)
    baker.align()
    baker.frame_rotation_2pi(qubit0_frame_update, q0_xy_element)
    baker.frame_rotation_2pi(qubit1_frame_update, q1_xy_element)
    baker.align()
def prep():
    T1 = 10000
    wait(int(10*T1))  # thermal preparation in clock cycles (time = 10 x T1 x 4ns)
    align()


def meas():
    threshold0 = 0.3 #threshold for state discrimination 0 <-> 1 using the I quadrature
    threshold1 = 0.3 #threshold for state discrimination 0 <-> 1 using the I quadrature
    I0 = declare(fixed)
    I1 = declare(fixed)
    Q0 = declare(fixed)
    Q1 = declare(fixed)
    state0 = declare(bool)
    state1 = declare(bool)
    multiplexed_readout([I0,I1], None, [Q0, Q1], None, resonators=[0, 1], weights="rotated_") #readout macro for multiplexed readout
    assign(state0, I0 > threshold0) #assume that all information is in I
    assign(state1, I1 > threshold1) #assume that all information is in I
    return state0, state1

rb = TwoQubitRb(config, bake_phased_xz, {"CZ": bake_cz}, prep, meas, verify_generation=False, interleaving_gate=None) #create RB experiment from configuration and defined funtions

qmm = QuantumMachinesManager('127.0.0.1',8080) #initialize qmm
res = rb.run(qmm, circuit_depths=[1, 2, 3, 4, 5], num_circuits_per_depth=5, num_shots_per_circuit=100)
# circuit_depths ~ how many consecutive Clifford gates within one executed circuit https://qiskit.org/documentation/apidoc/circuit.html
# num_circuits_per_depth ~ how many different randmon circuits within one depth
# num_shots_per_circuit ~ repetitions of the same circuit (averaging)

res.plot_hist()
plt.show()

res.plot_fidelity()
plt.show()


# %%
#############################################
############ Interleaved Example ############
#############################################
# q2, q3 = cirq.LineQubit.range(2)

# rb = TwoQubitRb(config, bake_phased_xz, {"CZ": bake_cz}, prep, meas, verify_generation=False,interleaving_gate= [cirq.CZ(q2,q3)])

# qmm = QuantumMachinesManager('127.0.0.1',8080)
# res = rb.run(qmm, circuit_depths=[1, 2, 3, 4, 5], num_circuits_per_depth=5, num_shots_per_circuit=100)
