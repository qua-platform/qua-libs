"""
XY-n.py: XY-n protocol
Author: Gal Winer - Quantum Machines
Created: 8/11/2020
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

def Y(theta,q):
    play('YPulse'*amp(theta), q)

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
with program() as XY8:
    I = declare(fixed)
    Q = declare(fixed)
    n = declare(int)
    tau = declare(int)
    tau_vec = declare_stream()


    with for_(n, 0, n < NAVG, n + 1):
        with for_(tau, 4, tau < taumax, tau + dtau):
            play('X', 'qubit')
            wait(tau, 'qubit')
            play('X', 'qubit')
            align('rr', 'qubit')
            measure('readout', 'rr', None, demod.full('integW1', I), demod.full('integW2', Q))
            save(I, 'I_res')
            save(Q, 'Q_res')
            wait(recovery_delay // 4, 'qubit')
            save(tau, tau_vec)

    with stream_processing():
        tau_vec.save_all('tau_vec')

job = QM1.simulate(T2,
                   SimulationConfig(int(100000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))

samples = job.get_simulated_samples()
job.result_handles.wait_for_all_values()
res = job.result_handles
I = res.I_res.fetch_all()['value']
Q = res.Q_res.fetch_all()['value']
tau_vec = res.tau_vec.fetch_all()['value']

# I=I.reshape(NAVG,N_tau)
# Q=Q.reshape(NAVG,N_tau)

def estimate_state(v):
    t = v[1]
    t2 = 40
    tp=t/t2
    pr = 0.5*(1 + np.cos(t)*np.exp(-tp))
    return np.random.choice([0, 1], p=[1 - pr, pr])

state_value = np.empty_like(tau_vec)
for pos, val in enumerate(zip(zip(I, Q), tau_vec)):
    state_value[pos] = estimate_state(val)

tau_vec = tau_vec.reshape(NAVG, N_tau)[0, :]
state_value = state_value.reshape(NAVG, N_tau)
state_value_mean = state_value.mean(axis=0)
state_value_var = state_value.var(axis=0)

def decay(t, a,b,c):
    return 0.5 + a*np.cos(b*t*0.5)*np.exp(-t/c)

param0=[1,10,40]
popt, pcov = curve_fit(decay, tau_vec, state_value_mean,param0)

plt.figure()
plt.errorbar(tau_vec, state_value_mean, yerr=state_value_var,label='measurement')
plt.plot(tau_vec,decay(tau_vec,*popt),'-r',label='fit')
plt.legend()
plt.title(f'T2 measurement T2={popt[1]:.1f} [ns]')
plt.xlabel('tau[ns]')
plt.ylabel('P(|0>)')
plt.show()
