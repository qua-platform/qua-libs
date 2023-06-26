#%%
import matplotlib.pylab as plt
from qm.qua import *
from qm import QuantumMachinesManager
from qualang_tools.bakery.bakery import Baking
from configuration import *
from two_qubit_rb import TwoQubitRb
# %matplotlib qt
#%%

q0 = '0'
q1 = '1'
#%%
def bake_phased_xz(baker: Baking, q, x, z, a):
    element = f"qubit{q}_xy"
    baker.frame_rotation_2pi(-a, element)
    baker.play("x", element, amp=x)
    baker.frame_rotation_2pi(a + z, element)
#%%
qubit0_frame_update = 0.23  # examples, should be taken from QPU parameters
qubit1_frame_update = 0.12  # examples, should be taken from QPU parameters

def bake_cz(baker: Baking, q0, q1):
    q0_xy_element = f"qubit{q0}_xy"
    q1_xy_element = f"qubit{q1}_xy"
    q1_z_element = f"qubit{q1}_z"
    baker.play("cz", q1_z_element)
    baker.align()
    baker.frame_rotation_2pi(qubit0_frame_update, q0_xy_element)
    baker.frame_rotation_2pi(qubit1_frame_update, q1_xy_element)
    baker.align()
#%%
def prep():
    T1 = 10000
    wait(int(10*T1))  # thermal preparation
    align()

#%%
def meas():
    threshold0 = 0.3 #example value
    threshold1 = 0.3 #example value
    rr0_name = f"qubit{q0}_rr"
    rr1_name = f"qubit{q1}_rr"
    Iq0 = declare(fixed)
    Qq0 = declare(fixed)
    Iq1 = declare(fixed)
    Qq1 = declare(fixed)
    state0 = declare(bool)
    state1 = declare(bool)
    measure("readout", rr0_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq0),
            dual_demod.full("w3", "out1", "w1", "out2", Qq0)
            )
    measure("readout", rr1_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq1),
            dual_demod.full("w3", "out1", "w1", "out2", Qq1)
            )
    assign(state0, Iq0 > threshold0)
    assign(state1, Iq1 > threshold1)
    return state0, state1


#%%
rb = TwoQubitRb(config, bake_phased_xz, {"CZ": bake_cz}, prep, meas, verify_generation=True)
#%%

qmm = QuantumMachinesManager('127.0.0.1',8080)
res = rb.run(qmm, circuit_depths=[1, 2, 3, 4, 5], num_circuits_per_depth=50, num_shots_per_circuit=1000)

# %%

res.plot_hist()
plt.show()

res.plot_fidelity()
plt.show()

