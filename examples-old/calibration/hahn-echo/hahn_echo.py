"""
hahn_echo.py: hahn echo experiment
Author: Arthur Strauss - Quantum Machines
Created: 16/11/2020
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from configuration import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
π = np.pi


def Hadamard(tgt):
    U2(tgt, 0, π)


def U2(tgt, 𝜙=0, 𝜆=0):
    Rz(𝜆, tgt)
    Y90(tgt)
    Rz(𝜙, tgt)


def U3(tgt, 𝜃=0, 𝜙=0, 𝜆=0):
    Rz(𝜆 - π / 2, tgt)
    X90(tgt)
    Rz(π - 𝜃, tgt)
    X90(tgt)
    Rz(𝜙 - π / 2, tgt)


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def Ry(𝜆, tgt):
    U3(tgt, 𝜆, 0, 0)


def Rx(𝜆, tgt):
    U3(tgt, 𝜆, -π / 2, π / 2)


def X90(tgt):
    play("X90", tgt)


def Y90(tgt):
    play("Y90", tgt)


def Y180(tgt):
    play("Y180", tgt)


def measure_and_save_state(
    tgt,
):  # Perform measurement, and state discrimination, here done for a SC circuit with IQ values
    measure("meas_pulse", tgt, None, demod.full("integW1", I), demod.full("integW2", Q))
    save(I, I_res)
    save(Q, Q_res)
    # State discrimination : assume that the separation in IQ plane has been determined using the defined threshold th
    assign(state, I > th)
    save(state, state_res)


taumax = 300
dtau = 5
NAVG = 30
recovery_delay = 16  # wait to return to ground
N_tau = taumax // dtau

with program() as hahn_echo:
    I = declare(fixed)
    Q = declare(fixed)
    n = declare(int)
    tau = declare(int)
    th = declare(fixed, value=3.2)
    state = declare(bool)

    tau_vec = declare_stream()
    I_res = declare_stream()
    Q_res = declare_stream()
    state_res = declare_stream()
    with for_(tau, 4, tau < taumax, tau + dtau):
        with for_(n, 0, n < NAVG, n + 1):
            Hadamard("qubit")
            wait(tau, "qubit")
            Rx(π, "qubit")
            wait(tau, "qubit")
            Hadamard("qubit")
            align("rr", "qubit")
            measure_and_save_state("rr")
            wait(recovery_delay // 4, "qubit")
        save(tau, tau_vec)

    with stream_processing():
        tau_vec.save_all("tau_vec")
        I_res.buffer(NAVG).save_all("I_res")
        Q_res.buffer(NAVG).save_all("Q_res")
        state_res.buffer(NAVG).save_all("state_res")

job = qm.simulate(
    hahn_echo,
    SimulationConfig(int(300000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)

# samples = job.get_simulated_samples()
job.result_handles.wait_for_all_values()
res = job.result_handles
I = res.I_res.fetch_all()["value"]
Q = res.Q_res.fetch_all()["value"]
state = res.state_res.fetch_all()["value"]
tau_vec = res.tau_vec.fetch_all()["value"]

# What we would do if we had real experimental data : Determine experimental probability of measuring 1 for each tau
def decay(t, a, b):
    return a * np.exp(-t / b)


"""P_1 = []
counts=0
for i in range(len(tau_vec)):
    for j in range(NAVG):
        if state[i][j]:
            counts += 1
    P_1.append(counts / NAVG)
    counts = 0
param0 = [1, 10]
popt, pcov = curve_fit(decay, tau_vec, np.mean(P_1), param0, sigma=np.std(P_1))

plt.figure()
plt.errorbar(tau_vec, np.mean(P_1), yerr=np.std(P_1), label='measurement')
plt.plot(tau_vec, decay(tau_vec, *popt), '-r', label='fit')
plt.legend()
plt.title(f'T2 measurement T2={popt[1]:.1f} [ns]')
plt.xlabel('tau[ns]')
plt.ylabel('P(|1>)')
plt.show()"""

# What we simulate by knowing the theoretical law governing decoherence process
def estimate_state(v):
    t = v
    t2 = 40
    pr = np.exp(-t / t2)
    return np.random.choice([0, 1], p=[1 - pr, pr])


state_th = np.empty_like(state)
for i in range(len(tau_vec)):
    for j in range(NAVG):
        state_th[i][j] = estimate_state(tau_vec[i])


state_value_mean = state_th.mean(axis=1)
state_value_var = state_th.var(axis=1)


param0 = [1, 10]
popt, pcov = curve_fit(decay, tau_vec, state_value_mean, param0, sigma=state_value_var)

plt.figure()
plt.plot(tau_vec, state_value_mean, ".", label="measurement")
plt.plot(tau_vec, decay(tau_vec, *popt), "-r", label="fit")
plt.legend()
plt.title(f"T2 measurement T2={popt[1]:.1f} [ns]")
plt.xlabel("tau[ns]")
plt.ylabel("P(|1>)")
plt.show()
