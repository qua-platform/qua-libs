"""
wigner-tomography.py: A template for performing Wigner tomography using a superconducting qubit
Author: Ilan Mitnikov - Quantum Machines
Created: 15/11/2020
Created on QUA version: 0.5.170
"""
from configuration import config
import configuration

from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

simulation_config = SimulationConfig(
    duration=int(2e5),  # need to run the simulation for long enough to get all points
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
        latency=200,
        noisePower=0.05**2,
    ),
)

revival_time = int(np.pi / configuration.chi / 4) * 4  # get revival time in multiples of 4 ns

# range to sample alpha
points = 10
alpha = np.linspace(-2, 2, points)
# scale alpha to get the required power amplitude for the pulse
amp_displace = list(-alpha / np.sqrt(2 * np.pi) / 4)
shots = 5


def wigner_prog():
    with program() as wigner_tomo:
        amp_dis = declare(fixed, value=amp_displace)
        n = declare(int)
        i = declare(int)
        r = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        I1 = declare(fixed)
        Q1 = declare(fixed)
        I2 = declare(fixed)
        Q2 = declare(fixed)
        with for_(r, 0, r < points, r + 1):
            with for_(i, 0, i < points, i + 1):
                with for_(n, 0, n < shots, n + 1):
                    align("cavity_I", "cavity_Q")
                    play("displace_I" * amp(amp_dis[r]), "cavity_I")
                    play("displace_Q" * amp(amp_dis[i]), "cavity_Q")

                    align("cavity_I", "cavity_Q", "qubit")
                    play("x_pi/2", "qubit")
                    wait(revival_time, "qubit")
                    play("x_pi/2", "qubit")

                    align("qubit", "rr")
                    measure(
                        "readout",
                        "rr",
                        "raw",
                        demod.full("integW_cos", I1, "out1"),
                        demod.full("integW_sin", Q1, "out1"),
                        demod.full("integW_cos", I2, "out2"),
                        demod.full("integW_sin", Q2, "out2"),
                    )
                    assign(I, I1 + Q2)
                    assign(Q, -Q1 + I2)
                    save(I, "I")
                    save(Q, "Q")

                    wait(10, "cavity_I", "cavity_Q", "qubit", "rr")  # wait and let all elements relax
    return wigner_tomo


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.simulate(wigner_prog(), simulation_config)
# job.get_simulated_samples().con1.plot()  # to see the output pulses
job.result_handles.wait_for_all_values()
I = job.result_handles.I.fetch_all()["value"]
Q = job.result_handles.Q.fetch_all()["value"]
phase = np.arctan2(I, Q).reshape(-1, shots)
ground = np.count_nonzero(phase < 0, axis=1) / shots  # count the number of ground state measurements
excited = np.count_nonzero(phase > 0, axis=1) / shots  # count the number of excited states measurements
parity = (excited - ground).reshape(points, points)
wigner = 2 / np.pi * parity
plt.figure()
ax = plt.subplot()
sns.heatmap(
    wigner,
    -2 / np.pi,
    2 / np.pi,
    xticklabels=alpha,
    yticklabels=alpha,
    ax=ax,
    cmap="Blues",
)
ax.set_xlabel("Im(alpha)")
ax.set_ylabel("Re(alpha)")
ax.set_title("Wigner function")
