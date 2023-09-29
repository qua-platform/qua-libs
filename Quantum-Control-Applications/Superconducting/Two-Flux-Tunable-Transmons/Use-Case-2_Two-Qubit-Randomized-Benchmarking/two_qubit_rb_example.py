import matplotlib.pylab as plt
from qm.qua import *
from qm import QuantumMachinesManager
from qualang_tools.bakery.bakery import Baking
from configuration import *
from two_qubit_rb import TwoQubitRb
from macros import multiplexed_readout

# assign a string to a variable to be able to call them in the functions
q1 = '1'
q2 = '2'
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
def bake_cz(baker: Baking, q1, q2):
    q1_xy_element = f"q{q1}_xy" #
    q2_xy_element = f"q{q2}_xy"
    q1_z_element = f"q{q1}_z"
    baker.play("cz", q1_z_element)
    baker.align()
    baker.frame_rotation_2pi(qubit0_frame_update, q1_xy_element)
    baker.frame_rotation_2pi(qubit1_frame_update, q2_xy_element)
    baker.align()
def prep():
    wait(int(10*qubit_T1))  # thermal preparation in clock cycles (time = 10 x T1 x 4ns)
    align()

def meas():
    threshold0 = 0.3 #threshold for state discrimination 0 <-> 1 using the I quadrature
    threshold1 = 0.3 #threshold for state discrimination 0 <-> 1 using the I quadrature
    I1 = declare(fixed)
    I2 = declare(fixed)
    Q1 = declare(fixed)
    Q2 = declare(fixed)
    state1 = declare(bool)
    state2 = declare(bool)
    multiplexed_readout([I1,I2], None, [Q1, Q2], None, resonators=[1, 2], weights="rotated_") #readout macro for multiplexed readout
    assign(state1, I1 > threshold0) #assume that all information is in I
    assign(state2, I2 > threshold1) #assume that all information is in I
    return state1, state2

rb = TwoQubitRb(config, bake_phased_xz, {"CZ": bake_cz}, prep, meas, verify_generation=False, interleaving_gate=None) #create RB experiment from configuration and defined funtions

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name) #initialize qmm
res = rb.run(qmm, circuit_depths=[1, 2, 3, 4, 5], num_circuits_per_depth=5, num_shots_per_circuit=100)
# circuit_depths ~ how many consecutive Clifford gates within one executed circuit https://qiskit.org/documentation/apidoc/circuit.html
# num_circuits_per_depth ~ how many different randmon circuits within one depth
# num_shots_per_circuit ~ repetitions of the same circuit (averaging)

res.plot_hist()
plt.show()

res.plot_fidelity()
plt.show()
