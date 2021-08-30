"""
bit_flip_time_estimation.py: Script showcasing how the OPX can be used to generate Fig 3
Author: Arthur Strauss - Quantum Machines
Created: 24/03/2021
Created on QUA version: 0.80.769
"""

from configuration import *
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
        noisePower=0.05 ** 2,
    ),
)

revival_time = int(np.pi / chi_qa / 4) * 4  # get revival time in multiples of 4 ns

# range to sample alpha
points = 10
alpha = np.linspace(-2, 2, points)
# scale alpha to get the required power amplitude for the pulse
amp_displace = list(-alpha / np.sqrt(2 * np.pi) / 4)
shots = 1
t_max = 2e6 // 4
dt = 2e3 // 4
N = int(t_max // dt)

with program() as bit_flip_time_simu:
    amp_dis = declare(fixed, value=amp_displace)
    th = declare(fixed, value=1.0)  # threshold for state discrimination
    n = declare(int)
    t = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)
    state = declare(bool)
    I_stream = declare_stream()
    Q_stream = declare_stream()
    state_stream = declare_stream()

    with for_(t, 0, t < t_max, t + dt):  # Buffer drive duration
        with for_(n, 0, n < shots, n + 1):  # shots for averaging

            play("Pump_Op", "ATS", duration=t)
            play("probe_Op", "buffer", duration=t)
            align("ATS", "buffer", "cat_qubit", "transmon")

            # Measurement
            wait(t, 'cat_qubit', 'transmon')
            align("cat_qubit", "transmon")
            play("X90", "transmon")
            wait(revival_time, "transmon")
            play("X90", "transmon")

            align("transmon", "RR")
            measure(
                "Readout_Op",
                "RR",
                "raw",
                demod.full("integW_cos", I1, "out1"),
                demod.full("integW_sin", Q1, "out1"),
                demod.full("integW_cos", I2, "out2"),
                demod.full("integW_sin", Q2, "out2"),
            )
            assign(I, I1 + Q2)
            assign(Q, -Q1 + I2)
            assign(state, I > th)
            save(I, I_stream)
            save(Q, Q_stream)
            save(state, state_stream)
            wait(10, "cat_qubit", "transmon", "RR")  # wait and let all elements relax

    with stream_processing():
        I_stream.auto_reshape().save_all("I")
        Q_stream.auto_reshape().save_all("Q")
        state_stream.auto_reshape().boolean_to_int().save_all("state")


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.simulate(bit_flip_time_simu, simulation_config)
# job.get_simulated_samples().con1.plot()  # to see the output pulses
job.result_handles.wait_for_all_values()
I = job.result_handles.I.fetch_all()["value"]
Q = job.result_handles.Q.fetch_all()["value"]
state = job.result_handles.state.fetch_all()["value"]
ground, excited, parity = (
    np.zeros_like(state),
    np.zeros_like(state),
    np.zeros_like(state),
)
for t in range(N):
    excited[t] = np.count_nonzero(state[t]) / shots
    ground[t] = 1 - excited[t]
    parity[t] = excited[t] - ground[t]
