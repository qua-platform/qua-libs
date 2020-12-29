"""
T2.py: Single qubit T2 decay time
Author: Steven Frankel, Gal Winer - Quantum Machines
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


def X(theta, q):
    play('XPulse' * amp(theta), q)


def Y(theta, q):
    play('YPulse' * amp(theta), q)


def Z(theta, q):
    frame_rotation(theta, q)


def H(q):
    X(2, q='qe1')
    Y(1, q='qe1')


# Create a quantum machine based on the configuration.

QM1 = QMm.open_qm(config)

taumax = 100
dtau = 5
NAVG = 20
recovery_delay = 100  # wait to return to ground
N_tau = taumax // dtau

def measure_and_save_state(tgt): #Perform measurement, and state discrimination, here done for a SC circuit with IQ values
    measure('readout', tgt, None, demod.full('integW1', I), demod.full('integW2', Q))
    save(I, I_res)
    save(Q, Q_res)
    #State discrimination : assume that the separation in IQ plane has been determined using the defined threshold th
    assign(state, I > th)
    save(state, state_res)

with program() as T2:
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


    # T2
    with for_(tau, 4, tau < taumax, tau + dtau):
        with for_(n, 0, n < NAVG, n + 1):
            play('H', 'qubit')
            wait(tau, 'qubit')
            play('H', 'qubit')
            align('rr', 'qubit')
            measure_and_save_state('rr')
            wait(recovery_delay // 4, 'qubit')
        save(tau, tau_vec)

    with stream_processing():
        tau_vec.save_all('tau_vec')
        I_res.buffer(NAVG).save_all('I_res')
        Q_res.buffer(NAVG).save_all('Q_res')
        state_res.buffer(NAVG).save_all('state_res')

job = QM1.simulate(T2,
                   SimulationConfig(int(100000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))

#samples = job.get_simulated_samples()
job.result_handles.wait_for_all_values()
res = job.result_handles
I = res.I_res.fetch_all()['value']
Q = res.Q_res.fetch_all()['value']
state = res.state_res.fetch_all()['value']
tau_vec = res.tau_vec.fetch_all()['value']

#What we would do if we had real experimental data : Determine experimental probability of measuring 1 for each tau
def decay(t, a, b, c):
    return 0.5 + a * np.cos(b * t * 0.5) * np.exp(-t / c)
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
    t2 = 40
    tp = t / t2
    pr = 0.5 * (1 + np.cos(t) * np.exp(-tp))
    return np.random.choice([0, 1], p=[1 - pr, pr])


state_th = np.empty_like(state)
for i in range(len(tau_vec)):
    for j in range(NAVG):
        state_th[i][j] = estimate_state(tau_vec[i])


state_value_mean = state_th.mean(axis=1)
state_value_var = state_th.var(axis=1)




param0 = [1, 10, 40]
popt, pcov = curve_fit(decay, tau_vec, state_value_mean, param0)

plt.figure()
plt.errorbar(tau_vec, state_value_mean, yerr=state_value_var, label='measurement')
plt.plot(tau_vec, decay(tau_vec, *popt), '-r', label='fit')
plt.legend()
plt.title(f'T2* measurement T2*={popt[1]:.1f} [ns]')
plt.xlabel('tau[ns]')
plt.ylabel('P(|0>)')
plt.show()
