"""
hello_qua.py: Single qubit T1 decay time
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
from scipy import optimize

QMm = QuantumMachinesManager()

# Create a quantum machine based on the initial configuration.

QM1 = QMm.open_qm(config)

with program() as dragopt:
    data_vec = declare_stream(adc_trace=True)
    measure("readout", "rr", data_vec)

    with stream_processing():
        data_vec.input1().save_all("data")


def cost(x):
    config["waveforms"]["gauss_wf"]["samples"] = gauss(0.2, x[0], x[1], 0, 100)  # update the config
    QM1 = QMm.open_qm(config)  # reopen the QM using new config file e.g. new waveform
    job = QM1.simulate(
        dragopt,
        SimulationConfig(int(1000), simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])),
    )
    res = job.result_handles
    result_vector = res.data.fetch_all()["value"] / 2**12
    result_vector = -np.squeeze(result_vector, axis=0)
    optimal = np.array(gauss(0.2, 0, 3.0, 0, 100))
    answer = np.sqrt(np.sum((optimal - result_vector) ** 2))
    return answer


res = optimize.minimize(cost, x0=[1, 2], method="nelder-mead", options={"xatol": 1e-8, "disp": True})

config["waveforms"]["gauss_wf"]["samples"] = gauss(0.2, 1, 2, 0, 100)  # update the config
QM1 = QMm.open_qm(config)  # reopen the QM using new config file e.g. new waveform
job = QM1.simulate(
    dragopt,
    SimulationConfig(int(1000), simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])),
)
res_init = job.result_handles
result_vector = res_init.data.fetch_all()["value"] / 2**12
result_vector_init = -np.squeeze(result_vector, axis=0)

config["waveforms"]["gauss_wf"]["samples"] = gauss(0.2, res.x[0], res.x[1], 0, 100)  # update the config
QM1 = QMm.open_qm(config)  # reopen the QM using new config file e.g. new waveform
job = QM1.simulate(
    dragopt,
    SimulationConfig(int(1000), simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])),
)
res_opt = job.result_handles
result_vector = res_opt.data.fetch_all()["value"] / 2**12
result_vector_opt = -np.squeeze(result_vector, axis=0)

plt.figure()
plt.plot(result_vector_init, label="init")
plt.plot(result_vector_opt, label="opt")
plt.plot(np.array(gauss(0.2, 0, 3.0, 100)), label="target")
plt.legend()
plt.show()
