"""
suppressing-dephasing-real-time.py: suppressing dephasing using real-time Hamiltonian estimation
Author: Gal Winer - Quantum Machines
Created: 02/02/2021
Created on QUA version: 0.8.439
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from configuration import *

qmManager = QuantumMachinesManager()
QM = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

N = 3  # 120
t_samp = 12 // 4
th = 0
fB_min = 50
fB_max = 70
Nf = 2  # 256
N_avg = 2
fB_vec = np.linspace(fB_min, fB_max, Nf)
dfB = np.diff(fB_vec)[0]
alpha = 0.25
beta = 0.67
elements = ["DET", "D1", "D2", "RF-QPC", "DET_RF"]


def prep_stage():
    play("prep", "D1")
    play("prep", "D2")
    play("ramp_down_evolve", "D1")
    play("ramp_up_evolve", "D2")


def readout_stage():
    play("ramp_up_read", "D1")
    play("ramp_down_read", "D2")

    wait(t_samp * k + prep_len // 4 + 2 * ramp_len // 4, "RF-QPC")
    play("readout", "D1")
    play("readout", "D2")
    measure("measure", "RF-QPC", None, demod.full("integW1", I))


with program() as dephasingProg:
    N_avg = declare(int)
    k = declare(int)
    I = declare(fixed)
    fB = declare(fixed)

    state = declare(bool, size=N)
    state_str = declare_stream()
    rabi_state = declare(bool)
    rabi_state_str = declare_stream()

    ind1 = declare(int, value=0)
    Pf = declare(fixed, value=[1 / len(fB_vec)] * int(len(fB_vec)))
    rk = declare(fixed)
    C = declare(fixed)
    norm = declare(fixed)

    #########################
    # Feedback and Estimate #
    #########################
    update_frequency("DET", 0)

    with for_(k, 2, k <= N, k + 1):

        play("prep", "D1")
        play("prep", "D2")
        play("evolve", "D1", duration=t_samp * k)
        play("evolve", "D2", duration=t_samp * k)
        play("readout", "D1")
        play("readout", "D2")
        wait(t_samp * k + prep_len // 4, "RF-QPC")

        measure("measure", "RF-QPC", None, demod.full("integW1", I))
        assign(state[k - 1], I > 0)
        save(state[k - 1], state_str)
        assign(rk, Cast.to_fixed(state[k - 1]) - 0.5)
        assign(ind1, 0)
        with for_(fB, fB_min, fB < fB_max, fB + dfB):
            assign(C, Math.cos2pi(Cast.mul_fixed_by_int(fB, t_samp * k)))
            assign(Pf[ind1], (0.5 + rk * (alpha + beta * C)) * Pf[ind1])
            assign(ind1, ind1 + 1)

        assign(norm, 1 / Math.sum(Pf))
        with for_(ind1, 0, ind1 < Pf.length(), ind1 + 1):
            assign(Pf[ind1], Pf[ind1] * norm)

    update_frequency("D1_RF", 1e6 * (fB_min + dfB * Math.argmax(Pf)))
    update_frequency("D2_RF", 1e6 * (fB_min + dfB * Math.argmax(Pf)))

    #########
    # Operate#
    #########

    with for_(k, 2, k <= N, k + 1):
        prep_stage()
        wait(prep_len // 4 + ramp_len // 4, "D1_RF")
        wait(prep_len // 4 + ramp_len // 4, "D2_RF")
        play("const" * amp(0.05), "D1_RF", duration=t_samp * k)
        play("const" * amp(0.05), "D2_RF", duration=t_samp * k)
        play("evolve", "D1", duration=t_samp * k)
        play("evolve", "D2", duration=t_samp * k)
        readout_stage()
        assign(rabi_state, I > 0)
        save(rabi_state, rabi_state_str)

    with stream_processing():
        state_str.save_all("state_str")
        rabi_state_str.save_all("rabi_state_str")

job = qmManager.simulate(config, dephasingProg, SimulationConfig(int(5000)))
# res = job.result_handles
# states = res.state_str.fetch_all()['value']
samples = job.get_simulated_samples()
samples.con1.plot()

plt.figure()
# G1=samples.con1.analog['1'][270:550]
# G2=samples.con1.analog['2'][270:550]
G1 = samples.con1.analog["1"][1050:1300]
G2 = samples.con1.analog["2"][1050:1300]
ax1 = plt.subplot(311)
plt.plot(G1)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel("G1[V]")
plt.yticks(np.linspace(-0.1, 0.2, 5))
plt.grid()
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(G2)
plt.xlabel("time [ns]")
plt.ylabel("G2[V]")
plt.setp(ax2.get_xticklabels(), visible=False)
plt.yticks(np.linspace(-0.2, 0.1, 5))
plt.grid()
ax3 = plt.subplot(313, sharex=ax1)
plt.plot(G1 - G2)
plt.xlabel("time [ns]")
plt.ylabel("$\epsilon [V]$")
plt.ylim([-0.3, 0.6])
plt.yticks(np.linspace(-0.1, 0.5, 5))
plt.text(40, 0.2, "Preparation stage")
plt.text(170, 0.2, "Readout stage")
plt.grid()
plt.show()
