# %%
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qualang_tools.bakery.bakery import Baking
from TwoQubitRB import TwoQubitRb

from two_qubit_rb_config import *


# %%


def bake_phased_xz(baker: Baking, q, x, z, a):
    qe = qubit0_qe if q == 0 else qubit1_qe
    pulse = qubit0_x_pulse if q == 0 else qubit1_x_pulse

    baker.frame_rotation_2pi(-a, qe)
    baker.play(pulse, qe, amp=x)
    baker.frame_rotation_2pi(a + z, qe)


def bake_sqrt_iswap(baker: Baking, q1, q2):
    baker.play(iswap_pulse, qubit0_flux_qe)


def bake_cnot(baker: Baking, q1, q2):
    baker.play(iswap_pulse, qubit0_flux_qe)
    baker.wait(int(const_flux_len//4), qubit1_flux_qe)


# def bake_cz(baker: Baking, q1, q2):
#     baker.play(cz_pulse, qubit0_flux_qe)


def prep():
    wait(4)


def meas():
    q0 = declare(bool)
    q1 = declare(bool)
    measure_qubit_0(q0)
    measure_qubit_1(q1)
    return q0, q1


# %%


qmm = QuantumMachinesManager(host="172.16.33.100", port=80)

rb = TwoQubitRb(config, bake_phased_xz, {"CNOT": bake_sqrt_iswap}, prep, meas)

res = rb.run(qmm, sequence_depths=[10, 15, 20, 25, 30], num_repeats=4, num_averages=10)

# %%

res.plot_hist()
plt.show()

res.plot_fidelity()
plt.show()
