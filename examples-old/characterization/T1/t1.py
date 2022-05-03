"""
t1.py: Single qubit T1 decay time
Author: Steven Frenkel, Gal Winer - Quantum Machines
Created: 7/11/2020
Created on QUA version: 0.5.138
"""

import matplotlib.pyplot as plt
from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import numpy as np
from scipy.optimize import curve_fit

QMm = QuantumMachinesManager()


# Create a quantum machine based on the configuration.
def measure_and_save_state(
    tgt,
):  # Perform measurement, and state discrimination, here done for a SC circuit with IQ values
    measure("readout", tgt, None, demod.full("integW1", I), demod.full("integW2", Q))
    save(I, I_res)
    save(Q, Q_res)
    # State discrimination : assume that the separation in IQ plane has been determined using the defined threshold th
    assign(state, I > th)
    save(state, state_res)


QM1 = QMm.open_qm(config)

taumax = 60  # in units of clock cycles, i.e., taumax = 60*4 ns = 240 ns
dtau = 1  # in units of clock cycles (4 ns)
NAVG = 60
recovery_delay = 100  # wait to return to ground
N_tau = taumax // dtau
with program() as T1:
    I = declare(fixed)
    Q = declare(fixed)
    th = declare(fixed, value=3.2)
    state = declare(bool)
    n = declare(int)
    tau = declare(int)
    tau_vec = declare_stream()
    I_res = declare_stream()
    Q_res = declare_stream()
    state_res = declare_stream()

    with for_(n, 0, n < NAVG, n + 1):
        with for_(tau, 4, tau < taumax, tau + dtau):
            play("X", "qubit")
            wait(tau, "qubit")
            align("rr", "qubit")
            measure_and_save_state("rr")
            wait(recovery_delay // 4, "qubit")
            save(tau, tau_vec)

    with stream_processing():
        tau_vec.buffer(N_tau).save("tau_vec")
        I_res.buffer(N_tau).average().save("I_res")
        Q_res.buffer(N_tau).average().save("Q_res")
        state_res.boolean_to_int().buffer(N_tau).save("state_res")

job = QM1.simulate(
    T1,
    SimulationConfig(int(100000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)

samples = job.get_simulated_samples()
job.result_handles.wait_for_all_values()
res = job.result_handles
I = res.I_res.fetch_all()
Q = res.Q_res.fetch_all()
state = res.state_res.fetch_all()
tau_vec = res.tau_vec.fetch_all()


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
plt.title(f'T1 measurement T1={popt[1]:.1f} [ns]')
plt.xlabel('tau[ns]')
plt.ylabel('P(|1>)')
plt.show()"""


# What we simulate by knowing the theoretical law governing decoherence process
def estimate_state(v):
    t = v
    t1 = 40
    pr = np.exp(-t / t1)
    return np.random.choice([0, 1], p=[1 - pr, pr])


state_th = [[None] * NAVG] * N_tau
state_value_mean = []
state_value_var = []

for i in range(N_tau):
    for j in range(NAVG):
        state_th[i][j] = estimate_state(tau_vec[i])
    state_value_mean.append(np.mean(state_th[i]))
    state_value_var.append(np.std(state_th[i]))

# tau_vec was defined in units of clock cycles (4 ns)
# so to properly obtain T1 in units of ns, we multiply tau_vec*4 below
param0 = [1, 20]
popt, pcov = curve_fit(decay, 4 * tau_vec, state_value_mean, param0, sigma=state_value_var)

plt.figure()
plt.plot(4 * tau_vec, state_value_mean, ".", label="measurement")
plt.plot(4 * tau_vec, decay(4 * tau_vec, *popt), "-r", label="fit")
plt.legend()
plt.title(f"T1 measurement T1={popt[1]:.1f} [ns]")
plt.xlabel("tau[ns]")
plt.ylabel("P(|1>)")
plt.show()
