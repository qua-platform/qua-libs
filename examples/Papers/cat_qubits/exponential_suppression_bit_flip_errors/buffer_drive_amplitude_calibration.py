"""
buffer_drive_amplitude_calibration.py: Script showcasing how the OPX can be used to generate Fig S9
Author: Arthur Strauss - Quantum Machines
Created: 24/03/2021
Created on QUA version: 0.80.769
"""

from configuration import *
from qm import SimulationConfig, LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from Examples.Papers.cat_qubits.qua_macros import *

simulation_config = SimulationConfig(
    duration=int(2e5),  # need to run the simulation for long enough to get all points
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
        noisePower=0.05 ** 2,
    ),
)

revival_time = int(np.pi / chi_qa / 4) * 4  # get revival time in multiples of 4 ns
shots = 10

# Sweeping parameters for amplitude
eps_min = 0.3
eps_max = 1
d_eps = 0.3
eps_vec = np.arange(eps_min, eps_max - d_eps / 2, d_eps)
N_eps = len(eps_vec)
# Threshold for state discrimination
threshold = 0.0

# Grid for sweeping phase space
IQ_step = 0.5
grid_start = -1.0
grid_stop = 1
grid_vec = np.arange(grid_start, grid_stop - IQ_step / 2, IQ_step)
N_points = len(grid_vec)

with program() as buffer_drive_amp_cal:
    n = declare(int)
    i = declare(fixed)
    q = declare(fixed)
    a = declare(fixed)

    I = declare(fixed)
    pol = declare(int)
    state = declare(bool)

    parity_stream = declare_stream()

    with for_(n, 0, n < shots, n + 1):  # shots for averaging
        with for_(a, eps_min, a < eps_max - d_eps / 2, a + d_eps):  # Buffer drive amplitude sweep
            with for_(i, grid_start, i < grid_stop + IQ_step / 2, i + IQ_step):  # Real part of alpha
                with for_(q, grid_start, q < grid_stop + IQ_step / 2, q + IQ_step):  # Imaginary part of alpha
                    with for_(pol, 0, pol < 2, pol + 1):  # Iterate over X90 and -X90 for second Ramsey pulse
                        # State preparation
                        play("pump", "ATS", duration=2e2)
                        play("drive" * amp(a), "buffer", duration=2e2)
                        align()

                        # Displacement operation on storage mode
                        play("Displace_Op" * amp(i, 0, 0, q), "storage")
                        align()

                        # Parity measurement
                        Ramsey("transmon", "RR", revival_time, threshold, pol, state, I)

                        # Active reset of transmon and storage
                        align()
                        play("X", "transmon", condition=state)
                        # deflate(buffer_drive_on=False, buffer_amp=eps)
                        align("ATS", "storage", "transmon")
                        with if_(state):
                            # g_one_to_g_zero()
                            g_odd_to_g_even()

                        with if_(pol == 1):
                            assign(state, ~state)

                        save(state, parity_stream)

    with stream_processing():
        parity_stream.boolean_to_int().buffer(2).map(FUNCTIONS.average()).buffer(
            N_eps, N_points, N_points).average().save("parity")

qmm = QuantumMachinesManager()
job = qmm.simulate(
    config, buffer_drive_amp_cal, simulation_config, flags=["auto-element-thread"]
)
samples = job.get_simulated_samples()

plt.figure(1)
plt.plot(samples.con1.analog["1"])
plt.plot(samples.con1.analog["2"])
plt.plot(samples.con1.analog["5"] + 1)
plt.plot(samples.con1.analog["3"] + 2)
plt.plot(samples.con1.analog["4"] + 2)
plt.plot(samples.con1.analog["7"] + 3)
plt.plot(samples.con1.analog["8"] + 3)
plt.plot(samples.con1.analog["9"] + 4)
plt.plot(samples.con1.analog["10"] + 4)
plt.plot(samples.con2.analog["1"] + 5)
plt.plot(samples.con2.analog["2"] + 5)
plt.xlabel("Time [ns]")
plt.yticks(
    [0, 1, 2, 3, 4, 5],
    ["Transmon", "flux", "Readout", "storage", "ATS pump", "buffer drive"],
    rotation="vertical",
    va="center",
)
#

res = job.result_handles
res.wait_for_all_values()
P_excited = res.parity.fetch_all()
Wigner = np.zeros_like(P_excited)
for a in range(N_eps):
    for re in range(N_points):
        for im in range(N_points):
            Wigner[a][re][im] = (4 * P_excited[a][re][im] - 1) / np.pi

# for eps in range(eps):
#     plt.figure()
#     ax = plt.subplot()
#     sns.heatmap(
#         Wigner[eps],
#         -2 / np.pi,
#         2 / np.pi,
#         xticklabels=alpha,
#         yticklabels=alpha,
#         ax=ax,
#         cmap="Blues",
#     )
#     ax.set_xlabel("Re(alpha)")
#     ax.set_ylabel("Im(alpha)")
#     ax.set_title("Wigner function")

# Insert fitting tools here for 2D Gaussian to assess
# the distance between the two measured states
