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


# Create a quantum machine based on the configuration.

QM1 = QMm.open_qm(config)

# measurement parameters
#######################

taumax = 100  # max delay
dtau = 5  # delay step
NAVG = 50  # number of avraging steps
th = 0  # state estimate threshold
recovery_delay = 100  # wait for qubit return to ground state
N_tau = taumax // dtau  # number of delay steps

with program() as T2:
    n = declare(int)
    tau = declare(int)
    state = declare(bool)

    tau_vec = declare_stream()
    I_res = declare_stream()
    Q_res = declare_stream()
    state_res = declare_stream()

    # T2
    with for_(n, 0, n < NAVG, n + 1):
        with for_(tau, 4, tau < taumax, tau + dtau):

            play("pi2", "qubit")
            wait(tau, "qubit")
            play("pi2", "qubit")
            align("rr", "qubit")
            measure_and_save_state("rr", I_res, Q_res, state_res, th)
            wait(recovery_delay // 4, "qubit")
            save(tau, tau_vec)

    with stream_processing():
        tau_vec.buffer(N_tau).save("tau_vec")
        I_res.buffer(N_tau).average().save("I_res")
        Q_res.buffer(N_tau).average().save("Q_res")
        state_res.boolean_to_int().buffer(N_tau).average().save("state_res")

job = QM1.simulate(
    T2,
    SimulationConfig(int(500000), simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])),
)

job.result_handles.wait_for_all_values()
res = job.result_handles
I = res.I_res.fetch_all()
Q = res.Q_res.fetch_all()
state = res.state_res.fetch_all()
t = res.tau_vec.fetch_all()


state_th = np.empty_like(state)
for i in range(tau_vec.shape[0]):
    for j in range(tau_vec.shape[1]):
        state_th[i][j] = estimate_state(tau_vec[i][j])


state_value_mean = state_th.mean(axis=0)
state_value_var = state_th.var(axis=0)


param0 = [1, 10, 40]
popt, pcov = curve_fit(decay, t, state_value_mean, param0)

plt.figure()
plt.errorbar(t, state_value_mean, yerr=state_value_var, label="measurement")
plt.plot(t, decay(t, *popt), "-r", label="fit")
plt.legend()
plt.title(f"T2* measurement T2*={popt[1]:.1f} [ns]")
plt.xlabel("$\tau[ns]$")
plt.ylabel("P(|0>)")
plt.show()
